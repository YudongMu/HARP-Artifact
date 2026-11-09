"""Read-only BF16 operand observations, collected on a dense trajectory."""
import torch
from .core.attention import select_region_mask


def exponent_histogram(x):
    raw = x.contiguous().view(torch.int16).to(torch.int32).reshape(-1, 512) & 65535
    exponents = (raw >> 7) & 255
    normal = (exponents > 0) & (exponents < 255)
    zero = (raw & 32767) == 0
    special = (~normal & ~zero).any(-1)
    spread = torch.where(normal, exponents, 0).amax(-1) - torch.where(normal, exponents, 256).amin(-1)
    valid = ~special & normal.any(-1)
    counts = torch.bincount(spread[valid], minlength=255).cpu().tolist()
    return {'slices': len(raw), 'supported_slices': int((~special & (spread <= 8)).sum()),
            'special_slices': int(special.sum()), 'spread_gt8_slices': int((~special & (spread > 8)).sum()),
            'all_zero_slices': int(zero.all(-1).sum()),
            'normal_exponent_spread_histogram': {str(i): n for i, n in enumerate(counts) if n}}


class DenseObserver:
    def __init__(self, steps=(0, 5, 25, 49)):
        self.steps = set(steps)
        self.sites = {}
        self.mass = {}

    def record(self, runtime, x, layer, site):
        if runtime.step not in self.steps:
            return
        b, n, c = x.shape
        order = runtime.order.to(x.device)
        positions = runtime.tile_positions.to(x.device)
        padded = torch.nn.functional.pad(x.to(torch.bfloat16), (0, (-c) % 32))
        tiled = padded.index_select(1, order).index_select(1, positions.clamp_min(0))
        tiled = tiled.masked_fill((positions < 0)[None, :, None], 0)
        tiled = tiled.reshape(b, len(runtime.region_lengths), 16, -1, 32).permute(0, 1, 3, 2, 4)
        self.sites[f'{runtime.step}:{layer}:{site}'] = exponent_histogram(tiled)

    def attention(self, runtime, q, k, v, layer):
        if runtime.step not in self.steps:
            return
        order = runtime.order.to(q.device)
        _, mass = runtime.attention._full_and_mass(*(z.index_select(2, order) for z in (q, k, v)), runtime.region_lengths)
        mask = select_region_mask(mass, runtime.attention.mass_threshold)
        self.mass[f'{runtime.step}:{layer}'] = {
            'selected_region_pairs': int(mask.sum()), 'candidate_region_pairs': mask.numel(),
            'selected_fraction_by_batch_item': mask.float().mean((1, 2)).cpu().tolist(),
            'max_mass_normalization_error': float((mass.sum(-1) - 1).abs().max())}
