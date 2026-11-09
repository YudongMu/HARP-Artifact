"""Integration interfaces for SD3 Medium, FLUX.1-dev, Latte-1 and Open-Sora v1.3."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from harp.calibration import BudgetTable
from harp.core.attention import select_region_mask
from harp.runtime import HarpRuntime


class Adapter:
    """Bind one runtime per token layout/CFG stream, and call hooks at the upstream sites.

    Call set_step with the denoising call index. Use activation before the
    corresponding image-stream linear and attention after normalization/RoPE.
    Upstream retains projection, output splitting, conditioning and decoding.
    """

    def __init__(self, model, mode, table, *, identity, layouts):
        self.model = model
        self.mode = mode
        self.groups = dict(self.block_groups(model))
        self.offsets = {}
        offset = 0
        for group, blocks in self.groups.items():
            self.offsets[group] = offset
            offset += len(blocks)
        spec = json.loads(Path(table).read_text(encoding='utf-8'))
        if spec['model'] != self.model_id or spec['layers'] != offset or spec['steps'] != identity['steps']:
            raise ValueError('Starter table does not match model block inventory or denoising length')
        if set(layouts) != set(self.groups):
            raise ValueError('Provide the actual token layout for every block group')
        deployment = dict(identity, mode=mode)
        budget = BudgetTable(deployment, spec['allowances'], spec['provenance'])
        self.runtime = {group: HarpRuntime(mode, height, width, budget_table=budget,
                        identity=deployment, frames=frames, partial_regions=True)
                        for group, (frames, height, width) in layouts.items()}
        self.joint_cache = {}

    def set_step(self, step):
        for runtime in self.runtime.values():
            runtime.set_step(step)

    def reset(self):
        self.joint_cache.clear()
        for runtime in self.runtime.values():
            runtime.reset()

    def activation(self, group, layer, site, image_tokens):
        """Process only the image/video span before its projection or FFN linear."""
        return self.runtime[group].process_tokens(image_tokens, self.offsets[group] + layer, site)

    def attention(self, group, layer, q, k, v, *, image_start=0, image_tokens=None, attention_mask=None):
        """Post-RoPE [B,H,N,D] self/joint attention. Text keys and queries stay dense.

        Pass original cross-attention to upstream. For joint blocks the caller
        supplies the image span in the upstream concatenation order. Masked
        joint layouts use upstream attention, without window approximation.
        """
        rt = self.runtime[group]
        count = len(rt.order) if image_tokens is None else image_tokens
        if count != len(rt.order) or not 0 <= image_start <= q.shape[2] - count or q.shape != k.shape or q.shape[:3] != v.shape[:3]:
            raise ValueError('Unexpected attention shape or image span')
        image = slice(image_start, image_start + count)
        index = self.offsets[group] + layer

        def transform(site, x):
            return rt.process_qkv(site, x, index)

        residual = self.mode in ('residual', 'combined')
        if attention_mask is None and residual and count == q.shape[2]:
            return rt.attention(q, k, v, rt.step, ('additional-model', group, layer), token_order=rt.order,
                                region_lengths=rt.region_lengths,
                                active_transform=transform if self.mode == 'combined' else None)
        if self.mode in ('planeclear', 'combined') and not rt.is_refresh():
            def clear_span(site, x):
                return torch.cat((x[:, :, :image.start], transform(site, x[:, :, image]), x[:, :, image.stop:]), dim=2)
            q, k, v = (clear_span(site, x) for site, x in zip(('q', 'k', 'v'), (q, k, v)))
        if not residual or attention_mask is not None:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)

        # Joint attention keeps the original full softmax normalization over
        # selected image keys and all text keys. Only image outputs are cached.
        order = rt.order.to(q.device) + image_start
        lengths = rt.region_lengths
        region = torch.repeat_interleave(torch.arange(len(lengths), device=q.device),
                                        torch.tensor(lengths, device=q.device))
        logits = (q.float() @ k.float().transpose(-1, -2)) * (q.shape[-1] ** -.5)
        key = (group, layer)
        if rt.is_refresh():
            probabilities = logits.softmax(-1)
            image_mass = probabilities.index_select(2, order).index_select(3, order).sum(dim=1)
            mass = torch.stack([torch.stack([tile.sum(dim=(-1, -2)) for tile in part.split(lengths, dim=-1)], dim=-1)
                                for part in image_mass.split(lengths, dim=-2)], dim=1)
            mass = mass / mass.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(mass.dtype).tiny)
            mask = select_region_mask(mass.cpu(), rt.attention.mass_threshold).to(q.device)
        else:
            mask, cached = self.joint_cache[key]
        allowed = torch.ones((q.shape[0], q.shape[2], q.shape[2]), dtype=torch.bool, device=q.device)
        allowed[:, order[:, None], order[None, :]] = mask[:, region[:, None], region[None, :]]
        selected = logits.masked_fill(~allowed[:, None], -torch.inf).softmax(-1) @ v.float()
        if rt.is_refresh():
            full = probabilities @ v.float()
            self.joint_cache[key] = (mask, (full[:, :, image] - selected[:, :, image]).to(v.dtype).detach())
            return full.to(v.dtype)
        selected[:, :, image] += cached.float()
        return selected.to(v.dtype)


class SD3Adapter(Adapter):
    model_id = 'sd3-medium'

    @staticmethod
    def block_groups(model):
        return [('joint', model.transformer_blocks)]


class FluxAdapter(Adapter):
    model_id = 'flux-dev'

    @staticmethod
    def block_groups(model):
        return [('joint', model.transformer_blocks), ('single', model.single_transformer_blocks)]


class LatteAdapter(Adapter):
    model_id = 'latte-1'

    @staticmethod
    def block_groups(model):
        return [('spatial', model.transformer_blocks), ('temporal', model.temporal_transformer_blocks)]


class OpenSoraAdapter(Adapter):
    model_id = 'open-sora-v1.3'

    @staticmethod
    def block_groups(model):
        return [('spatial', model.spatial_blocks), ('temporal', model.temporal_blocks)]
