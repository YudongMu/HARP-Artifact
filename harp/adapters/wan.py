"""Instance adapter for official single-device Wan T2V, self-attention + FFN.

Cross-attention and text encoding remain dense. This is a BF16 algorithm
reference, not a sparse CUDA backend or complete paper-quality reproduction.
"""
import types
import torch
from ..runtime import HarpRuntime


class WanAdapter:
    def __init__(self, model, mode, **runtime_options):
        self.model = model
        self.mode = mode
        self.options = runtime_options
        self.original = []
        self.handles = []
        self.runtimes = {}
        self.forward_calls = 0
        self.current = []
        self.pair_timestep = None
        self.failed = False

    def begin_trajectory(self):
        for runtime in self.runtimes.values():
            runtime.reset()
        self.runtimes.clear()
        self.forward_calls = 0
        self.pair_timestep = None
        self.current = []
        self.failed = False

    def _prepare(self, x, t, seq_len):
        if self.failed:
            raise RuntimeError('Failed forward requires begin_trajectory before reuse')
        branch = self.forward_calls % 2
        step = self.forward_calls // 2
        timestep = t.detach().cpu()
        if branch and not torch.equal(timestep, self.pair_timestep):
            raise ValueError('CFG branches must use the same timestep')
        if not branch:
            self.pair_timestep = timestep.clone()
        self.current = []
        for sample, latent in enumerate(x):
            shape = tuple(latent.shape[1:])
            patch = self.model.patch_size
            if any(a % b for a, b in zip(shape, patch)):
                raise ValueError('Latent shape must divide into Wan patch sizes')
            grid = tuple(a // b for a, b in zip(shape, patch))
            n = grid[0] * grid[1] * grid[2]
            if n > seq_len:
                raise ValueError('seq_len does not cover all real tokens')
            key = (branch, sample)
            runtime = self.runtimes.get(key)
            if runtime is None:
                if step:
                    raise ValueError('Batch changed within trajectory')
                runtime = HarpRuntime(self.mode, grid[1], grid[2], frames=grid[0],
                    partial_regions=True, **self.options)
                runtime.video_grid = grid
                self.runtimes[key] = runtime
            elif runtime.video_grid != grid:
                raise ValueError('Video grid changed within trajectory')
            runtime.set_step(step)
            self.current.append(runtime)
        if step and len(self.current) != len([k for k in self.runtimes if k[0] == branch]):
            raise ValueError('Batch changed within trajectory')

    def _tokens(self, x, layer, site):
        if len(self.current) != x.shape[0]:
            raise ValueError('Wan model must be called through its attached forward')
        result = x.clone()
        for sample, runtime in enumerate(self.current):
            n = len(runtime.order)
            result[sample:sample+1, :n] = runtime.process_tokens(x[sample:sample+1, :n], layer, site)
        return result

    def attach(self):
        if self.original:
            raise RuntimeError('Adapter already attached')
        if self.model.model_type != 't2v':
            raise ValueError('Only official T2V is validated')
        if any(tuple(block.self_attn.window_size) != (-1, -1) for block in self.model.blocks):
            raise ValueError('Local attention requires separate validation')
        original = self.model.forward
        self.original.append((self.model, original))
        owner = self

        def model_forward(this, x, t, context, seq_len, **kwargs):
            owner._prepare(x, t, seq_len)
            try:
                output = original(x, t=t, context=context, seq_len=seq_len, **kwargs)
            except Exception:
                owner.failed = True
                raise
            owner.forward_calls += 1
            return output

        self.model.forward = types.MethodType(model_forward, self.model)
        # Dense leaves every upstream operator untouched.
        if self.mode == 'dense':
            return self
        from wan.modules.model import rope_apply
        from wan.modules import model as wan_ops
        for layer, block in enumerate(self.model.blocks):
            attn = block.self_attn
            self.original.append((attn, attn.forward))

            def attention_forward(this, x, seq_lens, grid_sizes, freqs, layer=layer):
                b, s, _ = x.shape
                if len(owner.current) != b:
                    raise ValueError('Wan batch layout mismatch')
                for sample, runtime in enumerate(owner.current):
                    if tuple(grid_sizes[sample].tolist()) != runtime.video_grid or int(seq_lens[sample]) != len(runtime.order):
                        raise ValueError('Wan valid-token layout mismatch')
                x = owner._tokens(x, layer, 'qkv_input')
                q = this.norm_q(this.q(x)).view(b, s, this.num_heads, this.head_dim)
                k = this.norm_k(this.k(x)).view(b, s, this.num_heads, this.head_dim)
                v = this.v(x).view(b, s, this.num_heads, this.head_dim)
                q, k = rope_apply(q, grid_sizes, freqs), rope_apply(k, grid_sizes, freqs)
                output_dtype = q.dtype  # official flash_attention returns RoPE Q dtype
                output = torch.zeros(b, s, this.num_heads, this.head_dim, device=x.device, dtype=output_dtype)
                for sample, runtime in enumerate(owner.current):
                    n = len(runtime.order)
                    qi, ki, vi = [z[sample:sample+1, :n].transpose(1, 2).to(torch.bfloat16) for z in (q, k, v)]
                    if owner.mode in {'residual', 'combined'}:
                        transform = (lambda site, z: runtime.process_qkv(site, z, layer)) if owner.mode == 'combined' else None
                        yi = runtime.attention(qi, ki, vi, runtime.step, ('wan', layer),
                            token_order=runtime.order, region_lengths=runtime.region_lengths, active_transform=transform)
                        runtime.stats['attention_calls'] += 1
                        runtime.stats['attention_counters'] = dict(runtime.attention.counters)
                        totals = runtime.stats.setdefault('attention_totals', {})
                        for key in ('selected_region_pairs', 'candidate_region_pairs', 'residual_elements_read', 'residual_elements_written'):
                            totals[key] = totals.get(key, 0) + runtime.attention.last_stats[key]
                        totals['selected_region_ratio_weighted'] = totals['selected_region_pairs'] / totals['candidate_region_pairs']
                    else:
                        qi, ki, vi = [runtime.process_qkv(site, z, layer) for site, z in zip(('q', 'k', 'v'), (qi, ki, vi))]
                        yi = wan_ops.flash_attention(q=qi.transpose(1,2), k=ki.transpose(1,2),
                            v=vi.transpose(1,2),window_size=this.window_size).transpose(1,2)
                    output[sample:sample+1, :n] = yi.transpose(1, 2).to(output_dtype)
                return this.o(owner._tokens(output.flatten(2), layer, 'o_proj_input'))

            attn.forward = types.MethodType(attention_forward, attn)
            for module, site in ((block.ffn[0], 'ffn1_input'), (block.ffn[2], 'ffn2_input')):
                def before(module, args, layer=layer, site=site):
                    return (owner._tokens(args[0], layer, site), *args[1:])
                self.handles.append(module.register_forward_pre_hook(before))
        return self

    def summary(self):
        return {'adapter': 'wan-self-ffn-reference-v1', 'forward_calls': self.forward_calls,
                'cross_attention': 'upstream dense; excluded from HARP approximation',
                'runtimes': {f'branch{branch}:sample{sample}': runtime.stats
                             for (branch, sample), runtime in self.runtimes.items()}}

    def close(self):
        for handle in self.handles:
            handle.remove()
        for module, original in reversed(self.original):
            module.forward = original
        self.handles.clear()
        self.original.clear()
        for runtime in self.runtimes.values():
            runtime.attention.reset()

    def __enter__(self):
        try:
            return self.attach()
        except Exception:
            self.close()
            raise

    def __exit__(self, *args):
        self.close()
