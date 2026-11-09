"""Independent stage4 image-quality sidecar; no changes to HARP production code."""
import hashlib,json,math,pathlib

CANDIDATES=('risk-cap-0','risk-cap-0.5','risk-cap-1','risk-cap-2')
NONZERO=CANDIDATES[1:]

def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
def file_sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write_json(path,value):
 path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 pending=path.with_name(path.name+'.incoming');pending.write_text(json.dumps(value,indent=2,allow_nan=False));pending.replace(path)
def all_legacy_inputs():
 return ([dict(id=f'cal-{i:03d}',class_id=10*i,seed=310000+i) for i in range(100)]+
 [dict(id=f'held-{i:03d}',class_id=5+50*i,seed=320000+i) for i in range(20)]+
 [dict(id=f'warm-{i:03d}',class_id=991+2*i,seed=330000+i) for i in range(5)])
def calibration_items():return all_legacy_inputs()[:10]
def heldout_items():
 rows=[dict(id=f'heldout256-{i:03d}',class_id=(999*i)//255,seed=400000+i) for i in range(256)]
 if {(x['class_id'],x['seed']) for x in rows}&{(x['class_id'],x['seed']) for x in all_legacy_inputs()}:raise ValueError('Heldout leakage')
 return rows
def group_names(phase):
 if phase=='calibration':return ['dense','residual']+[f'{m}-{c}' for c in CANDIDATES for m in ('pc','combined')]
 if phase=='heldout':return ['dense','residual','pc','combined']
 raise ValueError('Unknown phase')

def validate_candidates(tables):
 if set(tables)!=set(CANDIDATES):raise ValueError('Missing or extra candidate')
 base=tables[CANDIDATES[0]]['identity'];keys=set(tables[CANDIDATES[0]]['allowances'])
 if not keys:raise ValueError('Empty allowance table')
 previous={k:0. for k in keys}
 for index,c in enumerate(CANDIDATES):
  table=tables[c];values=table['allowances']
  if table['identity']!=base or set(values)!=keys:raise ValueError('Candidate deployment identities/keys differ')
  if any(not math.isfinite(v) or not 0<=v<=1 for v in values.values()):raise ValueError('Invalid candidate allowance')
  if index==0 and any(values.values()):raise ValueError('Zero control is not zero')
  if index and not any(values.values()):raise ValueError('Nonzero candidate is zero')
  if any(values[k]<previous[k] for k in keys):raise ValueError('Candidate strength order not nested')
  previous=values

def validate_smokes(root,phase,execution_sha,input_id):
 root=pathlib.Path(root)
 for group,reference in [('pc','dense'),('combined','residual')]:
  suffix=group+'-risk-cap-0' if phase=='calibration' else group
  smoke=json.loads((root/f'zero-smoke-{suffix}.json').read_text())
  ref=json.loads((root/reference/'cases'/input_id/'sample.json').read_text())
  if (smoke.get('status')!='PASS' or smoke.get('execution_sha256')!=execution_sha or
      smoke.get('latent_sha256')!=ref['latent_sha256'] or smoke.get('noise_sha256')!=ref['noise_sha256']):
   raise ValueError('Zero table control absent or source/reference bits differ')
def validate_generation(root,run,phase,group,ids):
 if run.get('status')!='complete' or run.get('phase')!=phase or run.get('group')!=group:raise ValueError('Incomplete or mixed generation')
 if run.get('count')!=len(ids) or run.get('completed')!=len(ids) or [s['id'] for s in run['samples']]!=ids:raise ValueError('Wrong input cohort')
 root=pathlib.Path(root).resolve()
 for row in run['samples']:
  p=(root/row['image']).resolve()
  if not p.is_relative_to(root) or not p.is_file() or file_sha(p)!=row['image_sha256']:raise ValueError('Image missing or changed')
def choose_candidate(report,limit_percent=2.):
 ids=[x['id'] for x in calibration_items()]
 if report.get('phase')!='calibration' or report.get('status')!='complete' or report.get('input_ids')!=ids:raise ValueError('Selection requires complete calibration only')
 groups=report['groups']
 if set(groups)!=set(group_names('calibration')):raise ValueError('Incomplete calibration groups')
 for row in groups.values():
  if row['count']!=10 or row['input_ids']!=ids or not math.isfinite(row['fid']):raise ValueError('Wrong or nonfinite group')
 base=groups['dense']['fid']
 if base<=0:raise ValueError('Dense FID denominator must be positive')
 losses={c:{m:100*(groups[f'{m}-{c}']['fid']-base)/base for m in ('pc','combined')} for c in CANDIDATES}
 feasible=[c for c in NONZERO if all(v<=limit_percent for v in losses[c].values())]
 selected=feasible[-1] if feasible else NONZERO[0]
 return {'candidate':selected,'calibration_passed':bool(feasible),'exploratory':not bool(feasible),
  'rule':'strongest preordered nonzero candidate with both PC and combined calibration FID loss <= limit; otherwise most conservative nonzero exploratory',
  'limit_percent':limit_percent,'calibration_input_ids':ids,'loss_percent':losses,'zero_is_never_selected':True,
  'formal_paper_quality':False,'G_available':False}
def freeze_selection(path,payload):
 path=pathlib.Path(path);body=dict(payload);body['freeze_sha256']=canonical_sha(payload)
 if path.exists():
  if load_freeze(path)!=body:raise ValueError('Frozen selection cannot be replaced')
  return body
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(body,f,indent=2,allow_nan=False)
 return body
def load_freeze(path):
 d=json.loads(pathlib.Path(path).read_text());expected=d.pop('freeze_sha256')
 if canonical_sha(d)!=expected or d.get('candidate') not in NONZERO:raise ValueError('Frozen selection changed')
 d['freeze_sha256']=expected;return d

def generate(args):
 """Resident-model generation, source-bound resumability and per-group outputs."""
 import os,sys,time,torch,importlib.metadata
 from PIL import Image
 from quality.dit import initial_noise,make_scheduler,denoise
 from quality.dit_budget import CalibrationRuntime,public_schedule_record,tensor_sha
 from harp.runtime import HarpRuntime
 from harp.adapters.dit import DiTAdapter
 from harp.calibration import BudgetTable
 import harp.core.attention as attention_module
 from diffusers import AutoencoderKL
 root=pathlib.Path('/opt/harp');out=pathlib.Path(args.output).resolve();phase=args.phase
 candidate_root=pathlib.Path(args.candidates).resolve()
 tables={c:json.loads((candidate_root/(c+'-table.json')).read_text()) for c in CANDIDATES}
 table_hashes={c:file_sha(candidate_root/(c+'-table.json')) for c in CANDIDATES}
 validate_candidates(tables)
 base=tables[CANDIDATES[0]]['identity']
 frozen=None
 if phase=='heldout':
  frozen=load_freeze(args.selection)
  if frozen.get('table_sha256')!=table_hashes:raise ValueError('Frozen table source changed')
  if file_sha(frozen['calibration_report'])!=frozen['calibration_report_sha256']:raise ValueError('Frozen calibration metric changed')
 items=calibration_items() if phase=='calibration' else heldout_items();groups=group_names(phase)
 source_files=sorted((root/'harp').rglob('*.py'))+[root/'quality/dit.py',root/'quality/dit_budget.py',pathlib.Path(__file__),root/'scripts/dit_stage4_generate.py',root/'src/dit-original/models.py']
 source_sha={str(p.relative_to(root)):file_sha(p) for p in source_files}
 execution=dict(phase=phase,items=items,source_sha256=source_sha,table_sha256=table_hashes,
                freeze_sha256=frozen['freeze_sha256'] if frozen else None,
                original_deployment_identity=base,pixel_protocol='uint8 clamp(decoded*127.5+128), RGB PNG; FP32 VAE',
                vae_source_sha256={str(p.relative_to(root)):file_sha(p) for p in sorted((root/'assets/weights/dit-vae').rglob('*')) if p.is_file()},
                environment=dict(torch=torch.__version__,cuda=torch.version.cuda,diffusers=importlib.metadata.version('diffusers'),numpy=importlib.metadata.version('numpy'),gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None),
                same_phase_group_pairing=True,batch_size=1,steps=50)
 config=out/'execution.json'
 if config.exists():
  if json.loads(config.read_text())!=execution:raise ValueError('Resume execution identity changed')
 elif out.exists() and any(out.iterdir()):raise ValueError('Unidentified nonempty output directory')
 else:write_json(config,execution)
 if (out/'progress.json').exists() and json.loads((out/'progress.json').read_text()).get('status')=='complete':
  for group in groups:validate_generation(out/group,json.loads((out/group/'generation.json').read_text()),phase,group,[i['id'] for i in items])
  validate_smokes(out,phase,canonical_sha(execution),items[0]['id'])
  print('Already complete; artifacts verified, no overwrite',flush=True);return
 if not torch.cuda.is_available():raise RuntimeError('GPU required for generation')
 if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('Fixed CuBLAS workspace required')
 torch.manual_seed(0);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 original=attention_module.select_region_mask
 attention_module.select_region_mask=lambda mass,threshold:original(mass.cpu(),threshold).to(mass.device)
 sys.path.insert(0,str(root/'src/dit-original'));from models import DiT_models
 ckpt=root/'assets/weights/dit/DiT-XL-2-512x512.pt'
 if file_sha(ckpt)!=base['checkpoint_sha256']:raise ValueError('Checkpoint changed')
 sch=make_scheduler();sch.set_timesteps(50);schedule=public_schedule_record(sch)
 from harp.protocol import schedule_digest
 if schedule_digest(schedule)!=base['noise_schedule_sha256']:raise ValueError('Schedule differs from calibrated tables')
 model=DiT_models['DiT-XL/2'](input_size=64,num_classes=1000).cuda().eval()
 model.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True),strict=True);model.to(torch.bfloat16)
 vae=AutoencoderKL.from_pretrained(root/'assets/weights/dit-vae',local_files_only=True).cuda().eval()
 actual_impl=canonical_sha(source_sha);start=time.monotonic();done=0
 def run(item,mode,candidate):
  identity=dict(base,mode=mode,implementation_sha256=actual_impl,
                attention_execution_backend='flash_attention' if mode in ('dense','planeclear') else 'masked_chunked',
                precision='BF16 transformer and Flash attention; FP32 latent/VAE' if mode in ('dense','planeclear') else 'BF16 transformer; FP32 residual attention and latent/VAE')
  options=dict(attention_backend='masked_chunked',attention_query_chunk=128)
  if candidate is None:rt=HarpRuntime(mode,32,32,**options)
  else:
   t=tables[candidate];table=BudgetTable(identity,t['allowances'],dict(source_table_sha256=table_hashes[candidate],derived_for_mode=mode,frozen_quality_selection=False))
   rt=CalibrationRuntime(mode,32,32,budget_table=table,identity=identity,**options)
  noise=initial_noise([item['seed']],device='cuda');clock=time.monotonic()
  with torch.inference_mode(),DiTAdapter(model,rt):z=denoise(model,make_scheduler(),noise,[item['class_id']],steps=50,runtime=rt)
  counters={key:rt.stats[key] for key in ['mode','plane_totals','plane_by_site','zero_budget_bypass_calls','attention_calls','attention_totals'] if key in rt.stats}
  return z.detach().cpu(),tensor_sha(noise),time.monotonic()-clock,identity,counters
 try:
  for item in items:
   for group in groups:
    target=out/group/'cases'/item['id'];receipt=target/'sample.json'
    if receipt.exists():
     row=json.loads(receipt.read_text())
     if row['item']!=item or row['execution_sha256']!=canonical_sha(execution) or file_sha(target/'sample.png')!=row['image_sha256'] or file_sha(target/'latent.pt')!=row['latent_file_sha256']:raise ValueError('Completed artifact changed')
     done+=1;continue
    mode='planeclear' if group.startswith('pc') else 'combined' if group.startswith('combined') else group
    candidate=(group.split('-',1)[1] if phase=='calibration' else frozen['candidate']) if mode in ('planeclear','combined') else None
    z,noise_sha,seconds,identity,counters=run(item,mode,candidate)
    if group!='dense':
     reference_row=json.loads((out/'dense'/'cases'/item['id']/'sample.json').read_text())
     if noise_sha!=reference_row['noise_sha256']:raise ValueError('Paired initial noise differs')
    if item==items[0] and (group in ('pc-risk-cap-0','combined-risk-cap-0') or (phase=='heldout' and group in ('pc','combined'))):
     if candidate=='risk-cap-0':zero,zero_noise=z,noise_sha
     else:zero,zero_noise,_,_,_=run(item,mode,'risk-cap-0')
     baseline=out/('dense' if mode=='planeclear' else 'residual')/'cases'/item['id']
     ref=torch.load(baseline/'latent.pt',weights_only=True)
     if tensor_sha(zero)!=tensor_sha(ref) or noise_sha!=zero_noise:raise ValueError('Zero table smoke differs from reference bits')
     write_json(out/f'zero-smoke-{group}.json',dict(status='PASS',execution_sha256=canonical_sha(execution),reference=str(baseline),latent_sha256=tensor_sha(zero),noise_sha256=zero_noise))
    target.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():decoded=vae.decode(z.cuda().float()/.18215).sample
    if not bool(torch.isfinite(decoded).all()):raise ValueError('Nonfinite decoded pixels')
    pixels=(decoded*127.5+128).clamp(0,255).permute(0,2,3,1).to(torch.uint8).cpu().numpy()[0]
    Image.fromarray(pixels).save(target/'sample.png');torch.save(z,target/'latent.pt')
    row=dict(item=item,execution_sha256=canonical_sha(execution),identity=identity,schedule=schedule,noise_sha256=noise_sha,
             latent_sha256=tensor_sha(z),latent_file_sha256=file_sha(target/'latent.pt'),image_sha256=file_sha(target/'sample.png'),seconds=seconds,
             runtime_counters=counters,candidate=candidate,source_table_sha256=table_hashes.get(candidate),effective_table_sha256=canonical_sha(dict(identity=identity,allowances=tables[candidate]['allowances'])) if candidate else None)
    write_json(receipt,row);done+=1
    write_json(out/'progress.json',dict(status='running',phase=phase,completed=done,total=len(items)*len(groups),last_input=item['id'],last_group=group,wall_seconds=time.monotonic()-start))
    print(phase,item['id'],group,round(seconds,3),flush=True)
  validate_smokes(out,phase,canonical_sha(execution),items[0]['id'])
  for group in groups:
   samples=[]
   for item in items:
    row=json.loads((out/group/'cases'/item['id']/'sample.json').read_text())
    samples.append(dict(item,image=f"cases/{item['id']}/sample.png",image_sha256=row['image_sha256'],noise_sha256=row['noise_sha256'],latent_sha256=row['latent_sha256']))
   run_manifest=dict(status='complete',phase=phase,group=group,count=len(items),completed=len(items),samples=samples,execution_sha256=canonical_sha(execution))
   manifest=out/group/'generation.json'
   if manifest.exists():
    if json.loads(manifest.read_text())!=run_manifest:raise ValueError('Completed group manifest cannot be replaced')
   else:write_json(manifest,run_manifest)
  write_json(out/'progress.json',dict(status='complete',phase=phase,completed=done,total=len(items)*len(groups),wall_seconds=time.monotonic()-start))
 except Exception as error:
  write_json(out/'progress.json',dict(status='failed',phase=phase,completed=done,error_type=type(error).__name__,error=str(error)));raise
