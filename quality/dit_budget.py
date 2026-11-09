"""Development calibration primitives; missing HARP cycles remain missing.

No changes to the production harp runtime. Risk-balanced proposals are unselected
experiments, not the paper's G-optimized water-filling policy or quality guarantee.
"""
import hashlib,math
import torch
import torch.nn.functional as F
from harp.runtime import HarpRuntime,ATTENTION_SITES
from harp.intervention import SingleSliceIntervention
from harp.core.planes import clear_slices

TARGETS=[dict(step=t,layer=l,site=s) for t,l,s in [
 (1,0,'qkv_input'),(1,0,'q'),(9,13,'k'),(17,27,'v'),
 (24,13,'o_proj_input'),(10,0,'ffn1_input'),(39,27,'ffn2_input')]]
LAMBDAS=(0.,.005,.02)

def key_of(target):return f"{target['step']}:{target['layer']}:{target['site']}"

def public_schedule_record(scheduler):
 def normalize(x):
  if isinstance(x,float) and not math.isfinite(x):return str(x)
  if isinstance(x,dict):return {k:normalize(v) for k,v in x.items() if not k.startswith('_')}
  if isinstance(x,(list,tuple)):return [normalize(v) for v in x]
  return x
 return {'config':normalize(dict(scheduler.config)),'timesteps':scheduler.timesteps.tolist()}

def input_split():
 return {'calibration':[dict(id=f'cal-{i:03d}',class_id=10*i,seed=310000+i) for i in range(100)],
         'heldout':[dict(id=f'held-{i:03d}',class_id=5+50*i,seed=320000+i) for i in range(20)],
         'warmup':[dict(id=f'warm-{i:03d}',class_id=991+2*i,seed=330000+i) for i in range(5)]}

def validate_targets(targets):
 seen=set()
 for t in targets:
  if type(t['step']) is not int or not 0<=t['step']<50 or type(t['layer']) is not int or not 0<=t['layer']<28:
   raise ValueError('Invalid DiT50 target')
  if t['site'] not in ATTENTION_SITES|{'ffn1_input','ffn2_input'}:raise ValueError('Unknown site')
  if t['site'] in ATTENTION_SITES and t['step']%5==0:raise ValueError('Refresh attention is protected')
  key=key_of(t)
  if key in seen:raise ValueError('Duplicate target')
  seen.add(key)

def tensor_sha(x):
 return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def pack_slices(runtime,x):
 b,n,c=x.shape;cp=(c+31)//32*32
 padded=F.pad(x.to(torch.bfloat16),(0,cp-c))
 pos=runtime.tile_positions.to(x.device)
 packed=padded.index_select(1,runtime.order.to(x.device)).index_select(1,pos.clamp_min(0))
 packed=packed.masked_fill((pos<0)[None,:,None],0)
 return packed.reshape(b,len(runtime.region_lengths),16,cp//32,32).permute(0,1,3,2,4).contiguous().reshape(-1,16,32)

class SliceSurvey:
 """Read-only reference survey, hash-sampling supported nonzero real Slices."""
 def __init__(self,input_id,targets=TARGETS):
  validate_targets(targets);self.input_id=input_id
  self.targets={key_of(t) for t in targets};self.steps={t['step'] for t in targets}
  self.sites={};self.mass={};self.selected={}
 def attention(self,*args):pass
 def record(self,runtime,x,layer,site):
  key=f'{runtime.step}:{layer}:{site}'
  if key not in self.targets or key in self.sites:return
  flat=pack_slices(runtime,x);raw=flat.view(torch.int16).int()&65535
  ex=(raw>>7)&255;normal=(ex>0)&(ex<255);special=~normal&((raw&32767)!=0)
  spread=torch.where(normal,ex,0).flatten(1).amax(-1)-torch.where(normal,ex,256).flatten(1).amin(-1)
  supported=~special.flatten(1).any(-1)&normal.flatten(1).any(-1)&(spread<=8)
  indices=supported.nonzero().flatten()
  row={'slices_at_site':len(flat),'supported_slices':len(indices),'sampling':'sha256(input_id,target) modulo supported nonzero indices; no output-error selection',
       'candidates':[],'actual_slice':None}
  self.sites[key]=row
  if not len(indices):row['status']='no_supported_nonzero_slice';return
  offset=int(hashlib.sha256((self.input_id+'|'+key).encode()).hexdigest(),16)%len(indices)
  index=int(indices[offset]);sample=flat[index:index+1].detach().cpu()
  self.selected[key]=sample
  row.update(status='sampled',actual_slice=index,input_slice_sha256=tensor_sha(sample))
  for allowance in LAMBDAS:
   y,stats=clear_slices(sample,allowance)
   c=float((sample.double().abs()-y.double().abs()).sum()/sample.double().abs().sum())
   assert -1e-12<=c<=allowance+1e-12
   row['candidates'].append(dict(allowance=allowance,cleared_mass=c,cleared_slice_sha256=tensor_sha(y),plane_stats=stats))

class AuditedIntervention(SingleSliceIntervention):
 """Validate the actual pre-intervention Slice against the reference survey."""
 def __init__(self,target,allowance,index,expected):
  super().__init__(target['step'],target['layer'],target['site'],allowance,index)
  self.expected=expected
  self.expected_input_sha=tensor_sha(expected)
  self.expected_committed_sha=tensor_sha(clear_slices(expected,allowance)[0])
 def apply(self,runtime,x,layer,site):
  hit=(runtime.step,layer,site)==(self.step,self.layer,self.site)
  if hit:
   selected=pack_slices(runtime,x)[self.slice_index:self.slice_index+1]
   if tensor_sha(selected)!=self.expected_input_sha:raise ValueError('Intervention prefix bits differ from reference Slice')
   self.measurement['input_slice_sha256']=tensor_sha(selected)
  result=super().apply(runtime,x,layer,site)
  if hit:
   committed=pack_slices(runtime,result)[self.slice_index:self.slice_index+1]
   actual=tensor_sha(committed)
   if actual!=self.expected_committed_sha:raise ValueError('Actual committed Slice differs from CPU candidate')
   self.measurement['committed_slice_sha256']=actual
  return result

class CalibrationRuntime(HarpRuntime):
 """Quality-only zero-budget shortcut before BF16 Linear autocast.

 This avoids forming planes for zero entries. It is not an accelerator cost
 model; skipped zero-site plane counts are deliberately not invented.
 """
 def process_tokens(self,x,layer,site):
  if self.mode in {'combined','planeclear'} and not(site in ATTENTION_SITES and self.is_refresh()):
   allowance=self.uniform_value if self.uniform else self.table.allowance(self.step,layer,site)
   if allowance==0:
    if self.observer is not None or self.intervention is not None:raise RuntimeError('Zero bypass is for joint quality replay only')
    if x.dtype!=torch.bfloat16:
     linear_sites={'qkv_input','o_proj_input','ffn1_input','ffn2_input'}
     if x.dtype!=torch.float32 or site not in linear_sites or not(torch.is_autocast_enabled(x.device.type) and torch.get_autocast_dtype(x.device.type)==torch.bfloat16):
      raise RuntimeError('FP32 zero bypass requires a known BF16-autocast Linear input; Q/K/V must already be BF16')
    self.stats['zero_budget_bypass_calls']=self.stats.get('zero_budget_bypass_calls',0)+1
    return x
  return super().process_tokens(x,layer,site)

def risk_summary(observations):
 grouped={}
 for row in observations:
  c,e=row['cleared_mass'],row['final_error']
  if not all(math.isfinite(x) and x>=0 for x in [c,e,row['allowance']]):raise ValueError('Invalid C/E measurement')
  if c==0 and e!=0:raise ValueError('Nonzero E with no actual clearing')
  grouped.setdefault(row['key'],[]).append(row)
 result={}
 for key,rows in grouped.items():
  counts={r['slices_at_site'] for r in rows}
  if len(counts)!=1:raise ValueError('Mixed tiling')
  n=counts.pop();positive=[r for r in rows if r['cleared_mass']>0]
  s=max((r['final_error']/r['cleared_mass'] for r in positive),default=None)
  result[key]={'worst_e_over_c':s,'sensitivity_identified':bool(positive),'slices_at_site':n,'observations':len(rows),'candidates':[
   {'allowance':lam,'risk_proxy':0. if lam==0 else n*s*lam if s is not None else None,'net_cycle_gain':None,
    'observed_max_final_error':max((r['final_error'] for r in rows if r['allowance']==lam),default=None)}
   for lam in LAMBDAS]}
 return result

def risk_balanced_proposals(sites):
 """Explicitly unselected feasible proposals; no fake G or cycle optimality.

 Allocate equal surrogate-risk caps to observed sites and sweep total caps.
 This is a development proposal generator, not paper water-filling selection.
 """
 allzero={f'{t}:{l}:{s}':0. for t in range(50) for l in range(28)
          for s in ['qkv_input','q','k','v','o_proj_input','ffn1_input','ffn2_input']}
 identified={k:v for k,v in sites.items() if v['sensitivity_identified']}
 anchor=sum(next(r['risk_proxy'] for r in v['candidates'] if r['allowance']==.005) for v in identified.values())
 proposals=[]
 for fraction in [0.,.5,1.,2.]:
  budget=anchor*fraction;quota=budget/max(1,len(identified));allowances=dict(allzero);used=0.
  if fraction:
   for key,site in identified.items():
    choices=[r for r in site['candidates'] if r['risk_proxy']<=quota+1e-12]
    chosen=max(choices,key=lambda r:r['allowance']);allowances[key]=chosen['allowance'];used+=chosen['risk_proxy']
  proposals.append(dict(id=f'risk-cap-{fraction:g}',policy='equal_site_surrogate_risk_caps_development_only',
   cycle_optimal=False,frozen=False,net_cycle_gain_available=False,risk_budget=budget,used_risk=used,
   measured_sites=len(sites),identified_sites=len(identified),unidentified_sites=[k for k in sites if k not in identified],
   unmeasured_allowances_zero=True,allowances=allowances))
 return proposals
