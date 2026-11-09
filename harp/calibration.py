"""Paper IV-C calibration primitives. Measured curves are required; none are fabricated."""
from dataclasses import dataclass
import json,math,pathlib

@dataclass
class BudgetTable:
 identity:dict
 allowances:dict
 provenance:dict=None
 def validate(self,identity):
  required={'checkpoint_sha256','adapter','sampler','noise_schedule_sha256','steps','precision','resolution','layout','mode','cfg_scale','cfg_channels','mass_threshold','refresh_interval'}
  if str(identity.get('adapter','')).startswith('wan-'):
   required |= {'frames','negative_prompt','cross_attention','implementation_sha256',
                'attention_execution_backend','attention_query_chunk'}
  missing=required-set(self.identity)
  if missing:raise ValueError('Budget identity missing required fields: '+','.join(sorted(missing)))
  for key,value in self.identity.items():
   if identity.get(key)!=value:raise ValueError(f'Budget deployment mismatch: {key}')
  for value in self.allowances.values():
   if not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid allowance')
 def allowance(self,step,layer,site):
  # Missing entries are errors, never silently replaced by dense/no-clearing values.
  return self.allowances[f'{step}:{layer}:{site}']
 def save(self,path):
  pathlib.Path(path).write_text(json.dumps({'identity':self.identity,'allowances':self.allowances,'provenance':self.provenance or {}},indent=2,allow_nan=False),encoding='utf-8')
 @classmethod
 def load(cls,path):return cls(**json.loads(pathlib.Path(path).read_text(encoding='utf-8')))

def build_curve(observations,tile_count):
 """Measured E/C worst sensitivity; risk remains a proxy, not a quality bound."""
 if not math.isfinite(tile_count) or tile_count<0:raise ValueError('tile_count must be finite and nonnegative')
 rows=list(observations);sensitivity=0.0
 for row in rows:
  if not all(math.isfinite(row[k]) for k in ['allowance','cleared_mass','final_error','cycle_saving']):raise ValueError('Nonfinite observation')
  c,e=row['cleared_mass'],row['final_error']
  if c<0 or e<0:raise ValueError('Negative observed mass/error')
  if c==0 and e!=0:raise ValueError('Nonzero error with no cleared mass')
  if c:sensitivity=max(sensitivity,e/c)
 values={0.0:(0.0,0.0,0.0)}
 for allowance in sorted({r['allowance'] for r in rows}):
  if not 0<=allowance<=1:raise ValueError('Invalid lambda')
  selected=[r for r in rows if r['allowance']==allowance]
  gain=sum(r['cycle_saving'] for r in selected)/len(selected)
  values[allowance]=(allowance,tile_count*sensitivity*allowance,gain)
 return [values[x] for x in sorted(values)]

def waterfill(curves,total_risk_budget):
 """Discrete marginal cycle/risk advances from Eq offline_allocation.

 Curves contain (allowance, proxy risk, measured/modelled net cycle saving).
 No claim of global optimality for arbitrary nonconcave measured curves.
 """
 if not math.isfinite(total_risk_budget) or total_risk_budget<0:raise ValueError('Invalid risk budget')
 points={k:list(v) for k,v in curves.items()};indices={k:0 for k in curves};spent=0.0
 for key,rows in points.items():
  if not rows or rows[0]!=(0,0,0):raise ValueError('Curves must start at (0,0,0)')
  if any(not all(math.isfinite(x) for x in row) for row in rows):raise ValueError('Nonfinite curve')
  if any(b[0]<a[0] or b[1]<a[1] for a,b in zip(rows,rows[1:])):raise ValueError('Unordered risk/allowance curve')
 while True:
  candidates=[]
  for key,rows in points.items():
   i=indices[key]
   if i+1==len(rows):continue
   _,risk,gain=rows[i];_,next_risk,next_gain=rows[i+1];dr=next_risk-risk;dg=next_gain-gain
   if dg<=0 or spent+dr>total_risk_budget+1e-12:continue
   candidates.append((float('inf') if dr==0 else dg/dr,key,dr))
  if not candidates:break
  _,key,dr=max(candidates,key=lambda x:(x[0],x[1]));indices[key]+=1;spent+=dr
 return {key:points[key][i][0] for key,i in indices.items()}

def select_joint(candidates,evaluate_calibration,calibration_ids,heldout_ids,quality_target=.02):
 """Select fastest jointly evaluated table on calibration only.

 Callback returns {'quality_loss':float,'cycles':float}. Caller subsequently
 evaluates frozen selection on held-out data; held-out scores never enter here.
 """
 if set(calibration_ids)&set(heldout_ids):raise ValueError('Calibration/held-out leakage')
 if not math.isfinite(quality_target) or quality_target<0:raise ValueError('Invalid quality target')
 accepted=[]
 for table in candidates:
  result=evaluate_calibration(table)
  if not all(math.isfinite(result[k]) for k in ['quality_loss','cycles']) or result['cycles']<0:raise ValueError('Invalid joint measurement')
  if result['quality_loss']<=quality_target:accepted.append((result['cycles'],table,result))
 if not accepted:raise ValueError('No jointly validated table meets quality target')
 _,table,result=min(accepted,key=lambda x:x[0])
 return table,result
