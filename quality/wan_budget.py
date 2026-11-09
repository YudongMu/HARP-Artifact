"""Wan-only short-protocol calibration semantics; process-local, no harp edits."""
import contextlib,hashlib,json,math,pathlib
from dataclasses import dataclass
import torch
from harp.runtime import HarpRuntime,ATTENTION_SITES
from harp.core.planes import clear_slices
from harp.calibration import BudgetTable

LAMBDAS=(0.,.005,.02)
PRIMARY_TARGETS=[dict(step=t,layer=l,site=s) for t,l,s in [(1,0,'q'),(10,20,'ffn2_input'),(39,39,'v')]]
SURVEY_TARGETS=PRIMARY_TARGETS+[dict(step=39,layer=l,site='v') for l in (35,30,25,20,10,5,0)]
SITES=('qkv_input','q','k','v','o_proj_input','ffn1_input','ffn2_input')
def key_of(t):return f"{t['step']}:{t['layer']}:{t['site']}"
def tensor_sha(x):return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def sidecar_sha():
    root=pathlib.Path(__file__).parent
    records={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'wan_budget.py',root/'wan_stage4.py') if p.exists()}
    return hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()

@dataclass(frozen=True)
class Protocol:
    frames:int=9
    @property
    def latent_frames(self):return (self.frames-1)//4+1
    def __post_init__(self):
        if self.frames not in (9,17):raise ValueError('Independent development protocol permits only 9 or17 frames, never matched81 transfer')
    def fields(self):
        return dict(wan_stage4_frames=self.frames,wan_stage4_sidecar_sha256=sidecar_sha(),
            selector_backend='process_local_cpu_fp32_stable_sort',gpu_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            zero_allowance_execution='wan_known_BF16_linear_bypass_no_fake_plane_stats',
            calibration_protocol='short_video_development_branch0_single_mask_both_branch_deployment')

def input_split():
    from .wan_semantic import PROMPTS
    prompts=tuple('a red car' if p=='a red bicycle' else p for p in PROMPTS)
    return {'calibration':[dict(id='cal-0',prompt='a dog and a horse',seed=510000)],
            'heldout':[dict(id=f'held-{i:02d}-s{j}',prompt_index=i,seed_index=j,prompt=p,seed=520000+2*i+j) for i,p in enumerate(prompts) for j in range(2)]}

def validate_targets(targets,*,read_only=False):
    seen=set()
    for t in targets:
        if type(t['step']) is not int or not 0<=t['step']<50 or type(t['layer']) is not int or not 0<=t['layer']<40 or t['site'] not in SITES:raise ValueError('Invalid Wan40-layer /50-step target')
        if not read_only and t['site'] in ATTENTION_SITES and t['step']%5==0:raise ValueError('Refresh attention perturbation is protected')
        if key_of(t) in seen:raise ValueError('Duplicate Wan target')
        seen.add(key_of(t))

def _coordinates(runtime,x,start,count):
    b,n,c=x.shape;g=len(runtime.region_lengths);cb=(c+31)//32;total=b*g*cb
    if n!=len(runtime.order) or start<0 or count<0 or start+count>total:raise ValueError('Invalid Slice coordinates')
    ids=torch.arange(start,start+count,device=x.device);batch=ids//(g*cb);region=(ids//cb)%g
    positions=runtime.tile_positions.to(x.device).reshape(g,16).index_select(0,region)
    tokens=runtime.order.to(x.device).index_select(0,positions.clamp_min(0).reshape(-1)).reshape(count,16)
    channels=(ids%cb)[:,None]*32+torch.arange(32,device=x.device)[None]
    return batch,tokens,channels,(positions>=0),channels<c

def slice_chunk(runtime,x,start,count):
    """Gather only requested [B,Region,Cblock] Slices; no whole-layer packing."""
    batch,tokens,channels,valid_t,valid_c=_coordinates(runtime,x,start,count)
    values=x[batch[:,None,None],tokens[:,:,None],channels.clamp_max(x.shape[-1]-1)[:,None,:]].bfloat16()
    return values.masked_fill(~(valid_t[:,:,None]&valid_c[:,None,:]),0)

def put_slice(runtime,x,index,value):
    if value.shape!=(1,16,32) or value.dtype!=torch.bfloat16:raise ValueError('Expected one BF16 Slice')
    batch,tokens,channels,valid_t,valid_c=_coordinates(runtime,x,index,1)
    result=x.clone();ti=tokens[0,valid_t[0]];ci=channels[0,valid_c[0]]
    result[int(batch[0]),ti[:,None],ci[None,:]]=value[0][valid_t[0]][:,valid_c[0]].to(x.dtype)
    return result

def candidates(sample):
    result=[]
    for lam in LAMBDAS:
        y,stats=clear_slices(sample,lam);c=float((sample.double().abs()-y.double().abs()).sum()/sample.double().abs().sum().clamp_min(1e-300))
        if not -1e-12<=c<=lam+1e-12:raise ValueError('Actual clearing violates allowance')
        result.append(dict(allowance=lam,cleared_mass=c,committed_slice_sha256=tensor_sha(y),plane_stats=stats))
    return result

class WanSurvey:
    def __init__(self,input_id,targets=SURVEY_TARGETS,chunk_size=2048):
        validate_targets(targets,read_only=True)
        self.input_id=input_id;self.targets={key_of(t) for t in targets};self.steps={t['step'] for t in targets}
        self.chunk_size=chunk_size;self.sites={};self.mass={};self.selected={}
    def record(self,runtime,x,layer,site):
        key=f'{runtime.step}:{layer}:{site}'
        if key not in self.targets or key in self.sites:return
        count=x.shape[0]*len(runtime.region_lengths)*((x.shape[-1]+31)//32);eligible=[]
        for start in range(0,count,self.chunk_size):
            flat=slice_chunk(runtime,x,start,min(self.chunk_size,count-start));raw=flat.view(torch.int16).int()&65535
            ex=(raw>>7)&255;normal=(ex>0)&(ex<255);special=(~normal)&((raw&32767)!=0)
            spread=torch.where(normal,ex,0).flatten(1).amax(-1)-torch.where(normal,ex,256).flatten(1).amin(-1)
            valid=(~special.flatten(1).any(-1))&normal.flatten(1).any(-1)&(spread<=8)
            eligible.extend((valid.nonzero().flatten()+start).cpu().tolist())
        row={'status':'missing_supported_nonzero_mask','slices_at_site':count,'supported_nonzero_slices':len(eligible),'actual_slice':None,
             'measurement_branch':0,'deployment_branch_multiplier':2,'input_dtype':str(x.dtype),
             'sampling':'SHA256(input,target) cyclic supported-index order; first with nonzero committed .02 mask; no E-based selection'}
        self.sites[key]=row
        if not eligible:return
        offset=int(hashlib.sha256((self.input_id+'|'+key).encode()).hexdigest(),16)%len(eligible)
        for position in range(len(eligible)):
            index=eligible[(offset+position)%len(eligible)];sample=slice_chunk(runtime,x,index,1).detach().cpu();grid=candidates(sample)
            if not any(r['cleared_mass']>0 for r in grid):continue
            row.update(status='sampled',actual_slice=index,input_slice_sha256=tensor_sha(sample),candidates=grid)
            self.selected[key]=sample;return

class WanProbe:
    def __init__(self,target,allowance,index,expected):
        validate_targets([target]);self.target=dict(target);self.allowance=allowance;self.index=index;self.expected=expected
        self.expected_input=tensor_sha(expected);self.expected_committed=tensor_sha(clear_slices(expected,allowance)[0])
        self.measurement=dict(target,allowance=allowance,actual_slice=index,hits=0,measurement_branch=0,deployment_branch_multiplier=2)
    def apply(self,runtime,x,layer,site):
        if (runtime.step,layer,site)!=(self.target['step'],self.target['layer'],self.target['site']):return x
        if getattr(runtime,'branch',0)!=0:raise ValueError('Single-Slice probe leaked into unconditional branch')
        if self.measurement['hits']:raise ValueError('Wan probe hit more than once')
        before=slice_chunk(runtime,x,self.index,1)
        if tensor_sha(before)!=self.expected_input:raise ValueError('Actual Wan prefix Slice differs from surveyed reference')
        after,stats=clear_slices(before,self.allowance)
        result=x if torch.equal(before.view(torch.int16),after.view(torch.int16)) else put_slice(runtime,x,self.index,after)
        committed=slice_chunk(runtime,result,self.index,1)
        if tensor_sha(committed)!=self.expected_committed:raise ValueError('Actual committed Wan bits differ from CPU mask')
        c=float((before.double().abs()-committed.double().abs()).sum()/before.double().abs().sum().clamp_min(1e-300))
        self.measurement.update(hits=1,cleared_mass=c,input_slice_sha256=tensor_sha(before),committed_slice_sha256=tensor_sha(committed),
            slices_at_site=x.shape[0]*len(runtime.region_lengths)*((x.shape[-1]+31)//32),plane_stats=stats)
        return result

class WanAttentionBridge:
    def __init__(self,base,runtime):self.base=base;self.runtime=runtime
    def __getattr__(self,name):return getattr(self.base,name)
    def __call__(self,q,k,v,step,cache_key,**kwargs):
        rt=self.runtime;layer=cache_key[-1];observer=rt.observer
        if observer is not None and step%rt.refresh_interval==0:
            for site,x in zip(('q','k','v'),(q,k,v)):observer.record(rt,x.reshape(x.shape[0]*x.shape[1],x.shape[2],x.shape[3]),layer,site)
        if rt.mode=='residual' and (observer is not None or rt.intervention is not None) and kwargs.get('active_transform') is None:
            kwargs['active_transform']=lambda site,x:rt.process_qkv(site,x,layer)
        return self.base(q,k,v,step,cache_key,**kwargs)

class WanBudgetRuntime(HarpRuntime):
    def __init__(self,*args,branch=0,**kwargs):
        self.branch=branch;super().__init__(*args,**kwargs);self.attention=WanAttentionBridge(self.attention,self)
        self.stats['wan_stage4_branch']=branch
    def process_tokens(self,x,layer,site):
        if self.step is None:raise RuntimeError('set_step is required before model forward')
        if x.ndim!=3 or x.shape[1]!=len(self.order):raise ValueError('Adapter token layout mismatch')
        if self.mode in {'planeclear','combined'} and not(site in ATTENTION_SITES and self.is_refresh()):
            allowance=self.uniform_value if self.uniform else self.table.allowance(self.step,layer,site)
            if allowance==0:
                if self.observer is not None or self.intervention is not None:raise RuntimeError('Zero fast bypass is joint replay only')
                if x.dtype!=torch.bfloat16 and (x.dtype!=torch.float32 or site not in {'qkv_input','o_proj_input','ffn1_input','ffn2_input'} or not torch.is_autocast_enabled(x.device.type) or torch.get_autocast_dtype(x.device.type)!=torch.bfloat16):raise RuntimeError('Wan zero bypass needs BF16 operand or known BF16-autocast Linear input')
                self.stats['zero_budget_bypass_calls']=self.stats.get('zero_budget_bypass_calls',0)+1;return x
        return super().process_tokens(x,layer,site)

class RuntimeFactory:
    def __init__(self,protocol,survey=None,probe=None):self.protocol=protocol;self.survey=survey;self.probe=probe;self.created=[]
    def attach_identity(self,identity):
        if identity.get('frames')!=self.protocol.frames:raise ValueError('Stage protocol frame mismatch')
        identity.update(self.protocol.fields());return identity
    def __call__(self,*args,**kwargs):
        branch=len(self.created)
        if branch>=2:raise ValueError('Wan calibration supports exactly one sample and two CFG branches')
        if kwargs.get('frames')!=self.protocol.latent_frames:raise ValueError('Wan latent-frame grid mismatch')
        self.attach_identity(kwargs['identity'])
        kwargs['observer']=self.survey if branch==0 else None;kwargs['intervention']=self.probe if branch==0 else None
        rt=WanBudgetRuntime(*args,branch=branch,**kwargs);self.created.append(rt);return rt

@contextlib.contextmanager
def patched_runtime(factory):
    import harp.adapters.wan as adapter
    import harp.wan_cli as cli
    import harp.core.attention as attention
    originals=(adapter.HarpRuntime,cli.BudgetTable,attention.select_region_mask)
    class BoundTable(BudgetTable):
        def validate(self,identity):
            factory.attach_identity(identity)
            if set(factory.protocol.fields())-set(self.identity):raise ValueError('Budget lacks Wan stage4 sidecar/short-protocol identity')
            return super().validate(identity)
    def stable(mass,threshold):return originals[2](mass.cpu(),threshold).to(mass.device)
    adapter.HarpRuntime=factory;cli.BudgetTable=BoundTable;attention.select_region_mask=stable
    try:yield
    finally:adapter.HarpRuntime,cli.BudgetTable,attention.select_region_mask=originals

def table_identity(reference,mode):
    if mode not in {'planeclear','combined'} or reference['frames'] not in (9,17):raise ValueError('No silent81 migration or unknown table mode')
    result=dict(reference,mode=mode,attention_execution_backend='masked_chunked' if mode=='combined' else 'selected',attention_query_chunk=128)
    return result

def risk_summary(observations):
    grouped={}
    for row in observations:
        c,e=row['cleared_mass'],row['final_error']
        if not all(math.isfinite(v) and v>=0 for v in (c,e,row['allowance'])) or (c==0 and e!=0):raise ValueError('Invalid actual C/E')
        grouped.setdefault(row['key'],[]).append(row)
    result={}
    for key,rows in grouped.items():
        counts={r['slices_at_site'] for r in rows}
        if len(counts)!=1:raise ValueError('Mixed physical Slice layouts')
        n=counts.pop();positive=[r for r in rows if r['cleared_mass']>0];s=max((r['final_error']/r['cleared_mass'] for r in positive),default=None)
        result[key]={'measurement_branch':0,'measurement_slices':n,'deployment_branch_multiplier':2,'deployment_slices':2*n,'worst_e_over_c':s,
            'identified':bool(positive),'candidates':[dict(allowance=lam,risk_proxy=0. if lam==0 else 2*n*s*lam if s is not None else None,
                observed_cleared_mass=max((r['cleared_mass'] for r in rows if r['allowance']==lam),default=0.),G=None) for lam in LAMBDAS]}
    return result

def proposals(sites):
    zeros={f'{step}:{layer}:{site}':0. for step in range(50) for layer in range(40) for site in SITES}
    identified=sorted((k,v) for k,v in sites.items() if v['identified'])
    ranked=sorted(identified,key=lambda kv:kv[1]['deployment_slices']*kv[1]['worst_e_over_c'])
    result=[];seen=set()
    for label in ('zero-control','conservative','mixed'):
        values=dict(zeros);risk=utility=0.
        for rank,(key,site) in enumerate(ranked):
            choices=[r for r in site['candidates'] if r['allowance']>0 and r['observed_cleared_mass']>0]
            if not choices or label=='zero-control' or (label=='conservative' and rank):continue
            chosen=min(choices,key=lambda x:x['allowance']) if label=='conservative' or rank else max(choices,key=lambda x:x['allowance'])
            values[key]=chosen['allowance'];risk+=chosen['risk_proxy'];utility+=site['deployment_slices']*chosen['observed_cleared_mass']
        digest=hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()
        if digest in seen:continue
        seen.add(digest);result.append(dict(id=label,allowances=values,risk=risk,utility=utility,G=None,frozen=False,cycle_optimal=False,
            policy='sparse_measured_sites_only_unmeasured_or_noC_zero; utility_is_cleared_mass_proxy_not_cycles'))
    return result

def select_policy(candidates,baseline,input_id,threshold=.02):
    if input_id!='cal-0':raise ValueError('Held-out inputs cannot select Wan policy')
    nonzero=[];accepted=[]
    for row in candidates:
        if not any(v>0 for v in row['allowances'].values()):continue
        scores=row['scores'];losses=[];valid=set(scores)==set(baseline)
        for dim,d in baseline.items():
            v=scores.get(dim)
            if d is None or v is None or not math.isfinite(d) or not math.isfinite(v):valid=False;continue
            if d>0:losses.append((d-v)/d)
            elif v<d:valid=False
        result=dict(row,worst_dimension_relative_loss=max(losses,default=0.),quality_rule='calibration combined vs matched dense; worst applicable raw-dimension relative loss<=.02; zero baseline cannot worsen')
        nonzero.append(result)
        if valid and result['worst_dimension_relative_loss']<=threshold:accepted.append(result)
    selected=max(accepted,key=lambda r:(r['utility'],-r['risk'],r['id'])) if accepted else None
    exploratory=min(nonzero,key=lambda r:(r['risk'],r['id'])) if nonzero else None
    return dict(selected=selected,exploratory=None if selected else exploratory,frozen_development_policy=selected is not None,
        paper_policy=False,G=None,reason='calibration_rule_met' if selected else 'no_feasible_nonzero; never replace optimization by all-zero table',heldout_used_for_selection=False)
