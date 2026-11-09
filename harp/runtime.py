"""Shared runtime policies and model-neutral token/channel tiling."""
import torch
import torch.nn.functional as F
from .core.planes import clear_slices,clear_slices_chunked
from .core.attention import ResidualAttention

ATTENTION_SITES={'qkv_input','q','k','v','o_proj_input'}

def spatial_order(height,width,region=(4,4)):
 rh,rw=region
 if height%rh or width%rw:raise ValueError('Spatial grid must divide into Token Regions')
 return torch.arange(height*width).reshape(height//rh,rh,width//rw,rw).permute(0,2,1,3).reshape(-1)

def video_layout(frames,height,width):
 if any(type(x) is not int or x<=0 for x in (frames,height,width)):raise ValueError('Invalid video grid')
 grid=torch.arange(frames*height*width).reshape(frames,height,width)
 regions=[grid[f,y:y+4,x:x+4].reshape(-1) for f in range(frames) for y in range(0,height,4) for x in range(0,width,4)]
 return torch.cat(regions),tuple(len(x) for x in regions)

class HarpRuntime:
 def __init__(self,mode,height,width,*,budget_table=None,identity=None,experimental_uniform=False,allowance=None,mass_threshold=.95,refresh_interval=5,frames=1,partial_regions=False,observer=None,residual_cache_device=None,intervention=None,plane_chunk_size=None,attention_backend='selected',attention_query_chunk=64):
  if mode not in {'dense','residual','planeclear','combined'}:raise ValueError('Unknown HARP mode')
  self.mode=mode;self.step=None
  self.plane_chunk_size=plane_chunk_size
  if partial_regions:self.order,self.region_lengths=video_layout(frames,height,width)
  else:
   if frames!=1:raise ValueError('Video requires explicit partial_regions layout')
   self.order=spatial_order(height,width);self.region_lengths=(16,)*(height*width//16)
  positions=[];offset=0
  for length in self.region_lengths:
   positions.extend(range(offset,offset+length));positions.extend([-1]*(16-length));offset+=length
  self.tile_positions=torch.tensor(positions,dtype=torch.long)
  self.real_positions=(self.tile_positions>=0).nonzero().flatten()
  self.table=budget_table;self.uniform=experimental_uniform;self.uniform_value=allowance
  self.refresh_interval=refresh_interval
  self.attention=ResidualAttention(16,mass_threshold,refresh_interval,cache_device=residual_cache_device,backend=attention_backend,query_chunk_tokens=attention_query_chunk)
  self.stats={'implementation':'semantic_reference_dense_reconstructed_GEMM','mode':mode,'experimental_uniform':experimental_uniform,'site_calls':{},'plane_totals':{},'attention_calls':0}
  self.stats['attention_execution_backend']=attention_backend
  self.stats['attention_query_chunk']=attention_query_chunk
  self.observer=observer
  self.intervention=intervention
  if intervention is not None:
   if mode!='residual':raise ValueError('Single-slice calibration requires residual-only reference')
   self.stats['intervention']=intervention.measurement
  if observer is not None:self.stats['dense_observations']={'sites':observer.sites,'attention_mass':observer.mass,'steps':sorted(observer.steps)}
  if mode in {'planeclear','combined'}:
   if budget_table is None and not(experimental_uniform and allowance is not None):raise ValueError('PlaneClear requires a deployment-matched table or explicit experimental uniform allowance')
   if experimental_uniform and (allowance is None or not 0<=allowance<=1):raise ValueError('Invalid experimental allowance')
   if budget_table is not None:
    if experimental_uniform:raise ValueError('Choose table or experimental uniform, not both')
    budget_table.validate(identity or {})
 def set_step(self,step):
  if not isinstance(step,int) or step<0:raise ValueError('Expected denoising call index')
  self.step=step
 def is_refresh(self):return self.mode in {'residual','combined'} and self.step%self.refresh_interval==0
 def process_tokens(self,x,layer,site):
  if self.step is None:raise RuntimeError('set_step is required before model forward')
  if self.observer is not None:self.observer.record(self,x,layer,site)
  if self.mode not in {'planeclear','combined'} or (site in ATTENTION_SITES and self.is_refresh()):
   return self.intervention.apply(self,x,layer,site) if self.intervention is not None else x
  if x.ndim!=3 or x.shape[1]!=len(self.order):raise ValueError('Adapter token layout mismatch')
  allowance=self.uniform_value if self.uniform else self.table.allowance(self.step,layer,site)
  b,n,c=x.shape;order=self.order.to(x.device);padded=F.pad(x.to(torch.bfloat16),(0,(-c)%32));cp=padded.shape[-1]
  positions=self.tile_positions.to(x.device)
  tiled=padded.index_select(1,order).index_select(1,positions.clamp_min(0))
  tiled=tiled.masked_fill((positions<0)[None,:,None],0)
  tiled=tiled.reshape(b,len(self.region_lengths),16,cp//32,32).permute(0,1,3,2,4)
  cleared,stats=clear_slices(tiled,allowance) if self.plane_chunk_size is None else clear_slices_chunked(tiled,allowance,chunk_size=self.plane_chunk_size)
  values=cleared.permute(0,1,3,2,4).reshape(b,-1,cp).index_select(1,self.real_positions.to(x.device))[...,:c]
  result=torch.empty_like(x);result.index_copy_(1,order,values.to(x.dtype))
  key=f'{layer}:{site}';self.stats['site_calls'][key]=self.stats['site_calls'].get(key,0)+1
  per_site=self.stats.setdefault('plane_by_site',{}).setdefault(key,{})
  for name in ['slices','supported_slices','fallback_slices','guard_rejected_slices','natural_zero_planes','candidate_cleared_planes','cleared_planes','retained_planes','eligible_planes']:
   self.stats['plane_totals'][name]=self.stats['plane_totals'].get(name,0)+stats[name]
   per_site[name]=per_site.get(name,0)+stats[name]
  return result
 def process_qkv(self,site,x,layer):
  b,h,n,d=x.shape
  # QK/PV slices are head-local. Pad a 72-channel head to 3x32, never mix
  # adjacent heads into a shared-exponent Slice; strip zero padding afterward.
  per_head=x.reshape(b*h,n,d)
  return self.process_tokens(per_head,layer,site).reshape(b,h,n,d)
 def reset(self):
  self.attention.reset();self.step=None

