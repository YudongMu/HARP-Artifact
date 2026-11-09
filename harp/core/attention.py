"""HARP Sec IV-A numerical reference: paired region-window / residual lifetime.

This implementation executes real selected-key attention but is a Python/PyTorch
reference, not the paper's block-sparse CUDA kernel. The default evaluates one
query region at a time. The opt-in masked_chunked quality backend computes dense
QK/PV with a frozen key mask in bounded query chunks. Neither retains NxN P.
The adapter supplies spatial token order and normalized/position-encoded Q/K/V.
"""
from dataclasses import dataclass
from typing import Hashable, Callable, Optional
import torch


def select_region_mask(mass: torch.Tensor, threshold: float) -> torch.Tensor:
    """Smallest prefix of descending region masses; ties use original index.

    ``mass`` is [..., key_regions] and must contain probability distributions.
    threshold=1 is an explicit dense control and retains every region.
    """
    if not 0 < threshold <= 1:
        raise ValueError('mass threshold must lie in (0,1]')
    if mass.ndim < 1 or mass.shape[-1] == 0:
        raise ValueError('at least one key region is required')
    if not bool(torch.isfinite(mass).all()) or bool((mass < 0).any()):
        raise ValueError('region masses must be finite and nonnegative')
    if not bool(torch.allclose(mass.sum(-1), torch.ones_like(mass.sum(-1)), atol=1e-5, rtol=1e-5)):
        raise ValueError('each region mass row must sum to one')
    if threshold == 1:
        return torch.ones_like(mass, dtype=torch.bool)
    values, indices = torch.sort(mass, dim=-1, descending=True, stable=True)
    cumulative = values.cumsum(-1)
    reached = cumulative >= threshold
    # Float32 accumulation may undershoot unity by a few ULP; retain all then.
    first = torch.where(reached.any(-1), reached.to(torch.int64).argmax(-1),
                        torch.full_like(reached[..., 0], mass.shape[-1]-1, dtype=torch.int64))
    selected_sorted = torch.arange(mass.shape[-1],device=mass.device) <= first.unsqueeze(-1)
    return torch.zeros_like(selected_sorted).scatter(-1,indices,selected_sorted)


@dataclass
class _WindowState:
    mask: torch.Tensor
    residual: torch.Tensor
    signature: tuple
    token_order: torch.Tensor
    refresh_step: int
    last_step: int


class ResidualAttention:
    """Stateful self-attention service, keyed by (trajectory, branch batch, site).

    Batch items including conditional/unconditional CFG branches have independent
    masks and residual values. The caller must use a fresh cache_key for each
    trajectory/layer, or call reset() between trajectories. Calls for a given key
    must progress through contiguous denoising *call indices*, starting at zero.
    """
    def __init__(self, region_tokens: int = 16, mass_threshold: float = .95,
                 refresh_interval: int = 5,cache_device=None,*,backend='selected',
                 query_chunk_tokens=64):
        if region_tokens <= 0 or refresh_interval <= 0:
            raise ValueError('region size and refresh interval must be positive')
        if not 0 < mass_threshold <= 1:
            raise ValueError('mass threshold must lie in (0,1]')
        if backend not in {'selected','masked_chunked'}:
            raise ValueError('Unknown reference attention backend')
        if type(query_chunk_tokens) is not int or query_chunk_tokens<=0:
            raise ValueError('query_chunk_tokens must be a positive integer')
        self.backend=backend
        self.query_chunk_tokens=query_chunk_tokens
        self.region_tokens = region_tokens
        self.mass_threshold = mass_threshold
        self.refresh_interval = refresh_interval
        self.cache_device = cache_device
        self._states = {}
        self.last_stats = {}
        self.counters = dict(refresh_calls=0,replay_calls=0,residual_reads=0,residual_writes=0)

    def reset(self, cache_key: Optional[Hashable] = None):
        if cache_key is None:
            self._states.clear()
            self.counters = dict(refresh_calls=0,replay_calls=0,residual_reads=0,residual_writes=0)
        else:
            self._states.pop(cache_key,None)
        self.last_stats = {}

    def selected_mask(self, cache_key: Hashable) -> torch.Tensor:
        """Return a detached copy for trace collection without exposing state mutation."""
        return self._states[cache_key].mask.clone()

    def _full_and_mass(self,q,k,v,region_lengths=None):
        # The caller runs BF16 linear layers under autocast. Casting operands to
        # float is not enough: autocast would cast these matmuls back to BF16.
        with torch.autocast(device_type=q.device.type,enabled=False):
            return self._full_and_mass_fp32(q,k,v,region_lengths)

    def _full_and_mass_fp32(self,q,k,v,region_lengths=None):
        b,h,n,d = q.shape
        r = self.region_tokens
        lengths = tuple(region_lengths) if region_lengths is not None else (r,)*(n//r)
        if self.backend=='masked_chunked':
            return self._chunked_fp32(q,k,v,lengths,collect_mass=True)
        groups = len(lengths)
        key_groups = torch.repeat_interleave(torch.arange(groups,device=q.device),torch.tensor(lengths,device=q.device))
        outputs, masses = [], []
        kt = k.float().transpose(-2,-1)
        vf = v.float()
        start = 0
        for length in lengths:
            p = (q[:,:,start:start+length].float()@kt/(d**.5)).softmax(-1)
            outputs.append(p@vf)
            # Sum key tokens, average query tokens and heads (paper region_mass).
            # Average over real queries only. Edge regions have fewer tokens;
            # no padded key may enter Softmax or acquire probability mass.
            mean = p.mean((1,2))
            masses.append(torch.zeros(b,groups,device=q.device).scatter_add_(1,key_groups.expand(b,-1),mean))
            start += length
        return torch.cat(outputs,dim=2),torch.stack(masses,dim=1)

    def _selected(self,q,k,v,mask,region_lengths=None):
        with torch.autocast(device_type=q.device.type,enabled=False):
            return self._selected_fp32(q,k,v,mask,region_lengths)

    def _selected_fp32(self,q,k,v,mask,region_lengths=None):
        b,h,n,d = q.shape
        r = self.region_tokens
        lengths = tuple(region_lengths) if region_lengths is not None else (r,)*(n//r)
        if self.backend=='masked_chunked':
            return self._chunked_fp32(q,k,v,lengths,mask=mask)[0]
        repeats = torch.tensor(lengths,device=q.device)
        batches = []
        for bi in range(b):
            regions = []
            start = 0
            for gi,length in enumerate(lengths):
                token_ids = mask[bi,gi].repeat_interleave(repeats).nonzero(as_tuple=False).flatten()
                keys = k[bi].index_select(1,token_ids).float()
                values = v[bi].index_select(1,token_ids).float()
                p = (q[bi,:,start:start+length].float()@keys.transpose(-2,-1)/(d**.5)).softmax(-1)
                regions.append(p@values)
                start += length
            batches.append(torch.cat(regions,dim=1))
        return torch.stack(batches,dim=0)

    def _chunked_fp32(self,q,k,v,lengths,mask=None,collect_mass=False):
        """Bounded dense QK/PV quality oracle, NOT selected-work execution.

        Excluded key regions have -inf logits before softmax. No padded query or
        key is introduced. Reduce head means into real key/query regions, then
        divide by each real query-region length; edge regions carry no padding.
        Called only within the wrappers' autocast-disabled FP32 context.
        """
        b,h,n,d=q.shape
        groups=len(lengths)
        sizes=torch.tensor(lengths,device=q.device)
        ids=torch.repeat_interleave(torch.arange(groups,device=q.device),sizes)
        kt=k.float().transpose(-2,-1)
        vf=v.float()
        output=torch.empty(b,h,n,v.shape[-1],device=q.device,dtype=torch.float32)
        mass=torch.zeros(b,groups,groups,device=q.device) if collect_mass else None
        for start in range(0,n,self.query_chunk_tokens):
            end=min(n,start+self.query_chunk_tokens)
            logits=q[:,:,start:end].float()@kt/(d**.5)
            if mask is not None:
                allowed=mask.index_select(1,ids[start:end]).index_select(2,ids)
                logits.masked_fill_(~allowed[:,None],-torch.inf)
            p=logits.softmax(-1)
            # A large-query SGEMM can accumulate a 32K-key PV dot sequentially,
            # amplifying roundoff on real Wan activations. Bound that reduction
            # by key blocks, keeping FP32 accumulation and the same softmax.
            accumulated=torch.zeros_like(output[:,:,start:end])
            for key_start in range(0,n,1024):
                key_end=min(n,key_start+1024)
                accumulated=accumulated+p[:,:,:,key_start:key_end]@vf[:,:,key_start:key_end]
            output[:,:,start:end]=accumulated
            if collect_mass:
                per_query=torch.zeros(b,end-start,groups,device=q.device)
                per_query.scatter_add_(2,ids[None,None].expand(b,end-start,-1),p.mean(1))
                mass.index_add_(1,ids[start:end],per_query)
            # Release the last chunk before the next logits allocation.
            del logits,p
        if collect_mass:mass=mass/sizes[None,:,None]
        return output,mass

    def forward(self,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,step:int,
                cache_key:Hashable,token_order:Optional[torch.Tensor]=None,
                active_transform:Optional[Callable]=None,region_lengths=None) -> torch.Tensor:
        if q.ndim!=4 or k.ndim!=4 or v.ndim!=4:
            raise ValueError('Q/K/V must have [batch,heads,tokens,channels] shape')
        if q.shape!=k.shape or q.shape[:3]!=v.shape[:3]:
            raise ValueError('only equal-length self-attention with matching Q/K is supported')
        if q.device!=k.device or q.device!=v.device or q.dtype!=k.dtype or q.dtype!=v.dtype:
            raise ValueError('Q/K/V device and dtype must match')
        if region_lengths is None:
            if q.shape[2]==0 or q.shape[2]%self.region_tokens:
                raise ValueError('token count must be a nonzero multiple of region_tokens')
            lengths=(self.region_tokens,)*(q.shape[2]//self.region_tokens)
        else:
            lengths=tuple(region_lengths)
            if not lengths or any(type(x) is not int or not 0<x<=self.region_tokens for x in lengths) or sum(lengths)!=q.shape[2]:
                raise ValueError('Invalid real-token region partition')
        if not isinstance(step,int) or step<0:
            raise ValueError('step must be a nonnegative denoising call index')
        n=q.shape[2]
        if token_order is None:
            order=torch.arange(n,device=q.device)
        else:
            order=torch.as_tensor(token_order,device=q.device)
            if order.dtype!=torch.int64 or order.shape!=(n,) or not torch.equal(torch.sort(order).values,torch.arange(n,device=q.device)):
                raise ValueError('token_order must be an int64 permutation of token indices')
        signature=(tuple(q.shape),tuple(v.shape),q.dtype,q.device,lengths)
        state=self._states.get(cache_key)
        if state is None:
            if step!=0:raise ValueError('new attention state must begin at step zero')
        else:
            if step!=state.last_step+1:raise ValueError('denoising calls must be contiguous; reset before reuse')
            if signature!=state.signature or not torch.equal(order,state.token_order):
                raise ValueError('shape/dtype/device/layout changed inside a cached trajectory')
        refresh=step%self.refresh_interval==0
        if not refresh and active_transform is not None:
            transformed=[]
            for site,x in [('q',q),('k',k),('v',v)]:
                y=active_transform(site,x)
                if y.shape!=x.shape or y.dtype!=x.dtype or y.device!=x.device:
                    raise ValueError('active_transform must preserve tensor shape, dtype and device')
                transformed.append(y)
            q,k,v=transformed
        qo,ko,vo=(x.index_select(2,order) for x in (q,k,v))
        if refresh:
            full,mass=self._full_and_mass(qo,ko,vo,lengths)
            mask=select_region_mask(mass,self.mass_threshold)
            selected=self._selected(qo,ko,vo,mask,lengths)
            # Cache in operand dtype (BF16 for HARP), never actively clear residual.
            residual=(full-selected).to(device=self.cache_device or q.device,dtype=q.dtype).detach().clone()
            state=_WindowState(mask.detach().clone(),residual,signature,order.clone(),step,step)
            self._states[cache_key]=state
            result=full
            self.counters['refresh_calls']+=1
            self.counters['replay_calls']+=1
            self.counters['residual_writes']+=residual.numel()
        else:
            result=self._selected(qo,ko,vo,state.mask,lengths)+state.residual.to(q.device).float()
            state.last_step=step
            self.counters['residual_reads']+=state.residual.numel()
        self.last_stats={
            'implementation':'pytorch_reference_selected_attention_not_fused_cuda',
            'execution_backend':('masked_chunked_dense_qk_pv_quality_only' if self.backend=='masked_chunked' else 'selected_key_gather_reference'),
            'query_chunk_tokens':self.query_chunk_tokens if self.backend=='masked_chunked' else None,
            'selected_region_pairs_are_logical_only':self.backend=='masked_chunked',
            'refresh':refresh,'step':step,'refresh_step':state.refresh_step,
            'selected_region_ratio':float(state.mask.float().mean()),
            'selected_region_pairs':int(state.mask.sum()),'candidate_region_pairs':state.mask.numel(),
            'residual_elements_read':0 if refresh else state.residual.numel(),
            'residual_elements_written':state.residual.numel() if refresh else 0,
            'residual_bytes':state.residual.numel()*state.residual.element_size(),
            'residual_cache_device':str(state.residual.device),
            'selected_region_replay':refresh,
            'active_clearing_disabled':refresh,
        }
        return result.to(q.dtype).index_select(2,torch.argsort(order))

    __call__=forward
