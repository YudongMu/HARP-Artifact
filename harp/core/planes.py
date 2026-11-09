"""Vectorized, exact-BF16 PlaneClear reference semantics (paper Section IV-B).

This executes Torch tensor operations, not the paper's fused numeric-plane CUDA
datapath. The paper does not specify guard bit widths; ``fraction_bits`` is an
explicit reconstruction choice. It quantizes lambda downward and verifies the
tentative mask with integer arithmetic. It does not guarantee generation quality.
"""
from __future__ import annotations

import torch


def clear_slices(x: torch.Tensor, allowance, fraction_bits: int = 24):
    """Clear complete numeric planes in BF16 ``[..., 16, 32]`` slices.

    ``allowance`` is a scalar or broadcasts to the leading slice dimensions,
    between zero and one inclusive. Zero and signed zero are supported; BF16
    subnormals, NaNs, infinities and exponent spread > 8 trigger bit-exact dense
    fallback of the complete slice. The return value is ``(tensor, stats)``;
    stats are JSON-safe aggregate counters and their extraction synchronizes a
    GPU. Callers must not use this reference path for claimed accelerator timing.

    Active BF16 components use common exponent ``emax`` and plane significance
    ``2**(emax-p)``. Integer masses are multiplied by ``2**15`` relative to the
    paper's normalized mass, permitting an exact int64 commit check. Natural
    zero-plane counters exclude unsupported/fallback slices.
    """
    if x.dtype != torch.bfloat16:
        raise TypeError('PlaneClear requires BF16 input; conversion belongs to the model adapter')
    if x.ndim < 2 or tuple(x.shape[-2:]) != (16, 32):
        raise ValueError('Expected slices with trailing dimensions [16, 32]')
    if isinstance(fraction_bits, bool) or not isinstance(fraction_bits, int) or not 0 <= fraction_bits <= 30:
        raise ValueError('fraction_bits must be an integer from 0 to 30')
    lead = x.shape[:-2]
    lam = torch.as_tensor(allowance, dtype=torch.float64, device=x.device)
    try:
        lam = torch.broadcast_to(lam, lead).reshape(-1)
    except RuntimeError as exc:
        raise ValueError('allowance must broadcast to the leading slice dimensions') from exc
    if not bool(torch.isfinite(lam).all()) or bool(((lam < 0) | (lam > 1)).any()):
        raise ValueError('allowance must be finite and in [0, 1]')
    flat = x.contiguous().reshape(-1, 512)
    count = flat.shape[0]
    if count == 0:
        return x.clone(), _empty_stats()

    raw = flat.view(torch.int16).to(torch.int32) & 0xffff
    exponent_field = (raw >> 7) & 255
    mantissa = raw & 127
    is_zero = (raw & 0x7fff) == 0
    normal = (exponent_field > 0) & (exponent_field < 255)
    special = (~normal) & (~is_zero)
    emax_field = torch.where(normal, exponent_field, 0).amax(dim=1)
    emin_field = torch.where(normal, exponent_field, 256).amin(dim=1)
    supported = (~special.any(dim=1)) & ((emax_field - emin_field) <= 8)

    # BF16 significand is 128+mantissa, and its least component is 2**(e-7).
    # Align to the 16-plane LSB, 2**(emax-15), by shifting e-emax+8.
    shifts = (exponent_field - emax_field[:, None] + 8).clamp(0, 8)
    aligned = torch.where(normal, (128 + mantissa) << shifts, 0)
    significance = torch.arange(15, -1, -1, device=x.device, dtype=torch.int32)
    bits = ((aligned[:, :, None] >> significance) & 1).bool()
    active_count = bits.sum(dim=1, dtype=torch.int64)
    plane_weights = (torch.ones(16, device=x.device, dtype=torch.int64) << significance)
    masses = active_count * plane_weights
    total_mass = masses.sum(dim=1)
    order = torch.argsort(masses, dim=1, stable=True)
    sorted_mass = torch.gather(masses, 1, order)
    prefix_mass = sorted_mass.cumsum(dim=1)
    tentative_sorted = (prefix_mass <= lam[:, None] * total_mass[:, None]) & (sorted_mass > 0)
    tentative = torch.zeros_like(tentative_sorted).scatter(1, order, tentative_sorted)
    tentative &= supported[:, None]
    tentative_mass = (masses * tentative).sum(dim=1)

    scale = 1 << fraction_bits
    fixed_lambda = torch.floor(lam * scale).to(torch.int64)
    # total_mass <= 512*65535; fraction_bits<=30 keeps products below int64.
    reject = supported & (tentative_mass * scale > fixed_lambda * total_mass)
    committed = tentative & (~reject[:, None])
    removed = ((bits & committed[:, None, :]).to(torch.int64) * plane_weights).sum(dim=2)
    retained = aligned.to(torch.int64) - removed
    sign = torch.where((raw & 0x8000) != 0, -1.0, 1.0)
    restored = torch.ldexp(retained.float() * sign, (emax_field - 127 - 15)[:, None]).bfloat16()
    changed = committed.any(dim=1)
    # Besides fallbacks, unchanged slices retain original signed-zero/NaN bits.
    y = torch.where(changed[:, None], restored, flat).reshape(x.shape)

    def integer(t):
        return int(t.sum().item())

    natural = (active_count == 0) & supported[:, None]
    actual_mass = (masses * committed).sum(dim=1)
    fractions = actual_mass.double() / total_mass.clamp_min(1).double()
    eligible_planes = integer(supported) * 16
    zero_planes = integer(natural)
    cleared = integer(committed)
    stats = {
        'slices': count,
        'supported_slices': integer(supported),
        'fallback_slices': integer(~supported),
        'guard_rejected_slices': integer(reject),
        'natural_zero_planes': zero_planes,
        'candidate_cleared_planes': integer(tentative),
        'cleared_planes': cleared,
        'retained_planes': eligible_planes - zero_planes - cleared,
        'eligible_planes': eligible_planes,
        'plane_skip_rate_supported': (zero_planes + cleared) / max(1, eligible_planes),
        'cleared_mass_fraction_max': float(fractions.max().item()),
        'guard_fraction_bits': fraction_bits,
        'implementation': 'torch_reference',
    }
    return y, stats


def _empty_stats():
    return {'slices': 0, 'supported_slices': 0, 'fallback_slices': 0,
            'guard_rejected_slices': 0, 'natural_zero_planes': 0,
            'candidate_cleared_planes': 0, 'cleared_planes': 0,
            'retained_planes': 0, 'eligible_planes': 0,
            'plane_skip_rate_supported': 0.0, 'cleared_mass_fraction_max': 0.0,
            'implementation': 'torch_reference'}


def clear_slices_chunked(x, allowance, *, chunk_size=4096, **kwargs):
    """Bound temporary plane tensors without changing any per-Slice decision."""
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer')
    if x.dtype != torch.bfloat16 or x.ndim < 2 or tuple(x.shape[-2:]) != (16,32):
        return clear_slices(x,allowance,**kwargs)  # retain canonical validation
    flat=x.reshape(-1,16,32)
    if len(flat) <= chunk_size:
        return clear_slices(x,allowance,**kwargs)
    lam=torch.as_tensor(allowance,dtype=torch.float64,device=x.device)
    lam=torch.broadcast_to(lam,x.shape[:-2]).reshape(-1)
    outputs=[];total=None
    counters=('slices','supported_slices','fallback_slices','guard_rejected_slices',
              'natural_zero_planes','candidate_cleared_planes','cleared_planes','retained_planes','eligible_planes')
    for start in range(0,len(flat),chunk_size):
        y,stats=clear_slices(flat[start:start+chunk_size],lam[start:start+chunk_size],**kwargs)
        outputs.append(y)
        if total is None:
            total=dict(stats)
        else:
            for name in counters:
                total[name]+=stats[name]
            total['cleared_mass_fraction_max']=max(total['cleared_mass_fraction_max'],stats['cleared_mass_fraction_max'])
    total['plane_skip_rate_supported']=(total['natural_zero_planes']+total['cleared_planes'])/max(1,total['eligible_planes'])
    return torch.cat(outputs).reshape(x.shape),total
