"""One hardware-realizable Slice perturbation on a residual-only trajectory."""
import math
import torch
from .core.planes import clear_slices


class SingleSliceIntervention:
    def __init__(self, step, layer, site, allowance, slice_index='auto'):
        if type(step) is not int or step < 0 or type(layer) is not int or layer < 0:
            raise ValueError('Invalid intervention step/layer')
        if not math.isfinite(allowance) or not 0 <= allowance <= 1:
            raise ValueError('Invalid intervention allowance')
        if site not in {'qkv_input','q','k','v','o_proj_input','ffn1_input','ffn2_input'}:
            raise ValueError('Unknown intervention site')
        if slice_index != 'auto' and (type(slice_index) is not int or slice_index < 0):
            raise ValueError('Invalid Slice index')
        self.step, self.layer, self.site, self.allowance, self.slice_index = step, layer, site, allowance, slice_index
        self.measurement = {'step':step,'layer':layer,'site':site,'allowance':allowance,'requested_slice':slice_index,'hits':0}

    def apply(self, runtime, x, layer, site):
        if (runtime.step,layer,site) != (self.step,self.layer,self.site):
            return x
        if site in {'qkv_input','q','k','v','o_proj_input'} and runtime.is_refresh():
            raise ValueError('Do not perturb protected refresh attention state')
        if self.measurement['hits']:
            raise ValueError('Single-Slice intervention site executed more than once')
        b,n,c = x.shape
        if n != len(runtime.order):
            raise ValueError('Intervention token layout mismatch')
        order = runtime.order.to(x.device)
        positions = runtime.tile_positions.to(x.device)
        cp = (c+31)//32*32
        padded = torch.nn.functional.pad(x.to(torch.bfloat16),(0,cp-c))
        packed = padded.index_select(1,order).index_select(1,positions.clamp_min(0)).masked_fill((positions<0)[None,:,None],0)
        tiles = packed.reshape(b,len(runtime.region_lengths),16,cp//32,32).permute(0,1,3,2,4).contiguous()
        flat = tiles.reshape(-1,16,32)
        index = self.slice_index
        if index == 'auto':
            raw = flat.view(torch.int16).int() & 65535
            exponents = (raw>>7)&255
            normal = (exponents>0)&(exponents<255)
            special = ~normal & ((raw&32767)!=0)
            spread = torch.where(normal,exponents,0).flatten(1).amax(-1)-torch.where(normal,exponents,256).flatten(1).amin(-1)
            supported = (~special.flatten(1).any(-1)) & normal.flatten(1).any(-1) & (spread<=8)
            indices = supported.nonzero().flatten()
            if not len(indices):
                raise ValueError('No supported nonzero Slice at requested intervention site')
            index = int(indices[0])
        if index >= len(flat):
            raise ValueError('Slice index outside operator')
        selected = flat[index:index+1].clone()
        cleared, stats = clear_slices(selected,self.allowance)
        mass = float((selected.double().abs()-cleared.double().abs()).sum()/selected.double().abs().sum().clamp_min(1e-300))
        self.measurement.update(hits=1,actual_slice=index,cleared_mass=mass,
                                slices_at_site=len(flat),plane_stats=stats)
        if torch.equal(selected,cleared):
            return x
        flat[index:index+1] = cleared
        values = tiles.permute(0,1,3,2,4).reshape(b,-1,cp).index_select(1,runtime.real_positions.to(x.device))[...,:c]
        mask = torch.zeros_like(tiles,dtype=torch.bool)
        mask.reshape(-1,16,32)[index:index+1] = True
        mask = mask.permute(0,1,3,2,4).reshape(b,-1,cp).index_select(1,runtime.real_positions.to(x.device))[...,:c]
        values = torch.where(mask,values.to(x.dtype),x.index_select(1,order))
        result = x.clone()
        result.index_copy_(1,order,values.to(x.dtype))
        return result
