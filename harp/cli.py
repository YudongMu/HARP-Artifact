"""Unified entry point; only DiT adapter is currently implemented."""
import argparse,hashlib,json,pathlib,sys,time,math
import torch
from .runtime import HarpRuntime
from .calibration import BudgetTable
from .adapters.dit import DiTAdapter

def arguments():
 p=argparse.ArgumentParser(description='HARP functional reconstruction; not a paper speedup benchmark')
 p.add_argument('--model',choices=['dit'],default='dit')
 p.add_argument('--mode',choices=['dense','residual','planeclear','combined'],default='residual')
 p.add_argument('--steps',type=int,default=50);p.add_argument('--seed',type=int,default=0);p.add_argument('--class-id',type=int,default=281)
 p.add_argument('--checkpoint',default='/opt/harp/assets/weights/dit/DiT-XL-2-512x512.pt')
 p.add_argument('--vae',default='/opt/harp/assets/weights/dit-vae')
 p.add_argument('--budget-table');p.add_argument('--experimental-uniform',action='store_true');p.add_argument('--allowance',type=float)
 p.add_argument('--output');p.add_argument('--mass-threshold',type=float,default=.95);p.add_argument('--refresh-interval',type=int,default=5)
 return p.parse_args()

def main():
 a=arguments()
 if a.steps<=0 or not 0<=a.class_id<1000:raise ValueError('Invalid steps/class')
 if a.mode in {'planeclear','combined'} and not(a.budget_table or (a.experimental_uniform and a.allowance is not None)):
  raise ValueError('Use --budget-table, or explicitly --experimental-uniform --allowance VALUE for unvalidated functional experiments')
 if not torch.cuda.is_available():raise RuntimeError('Real DiT execution requires a CUDA GPU; CPU unit tests are separate')
 from diffusers import AutoencoderKL,DPMSolverMultistepScheduler,__version__ as diffusers_version
 from torchvision.utils import save_image
 from torch.nn.attention import sdpa_kernel,SDPBackend
 root=pathlib.Path(__file__).resolve().parents[1]
 sys.path.insert(0,str(root/'src/dit-original'))
 from models import DiT_models
 torch.manual_seed(a.seed);torch.set_grad_enabled(False)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 h=hashlib.sha256()
 with open(a.checkpoint,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 scheduler=DPMSolverMultistepScheduler(num_train_timesteps=1000,beta_start=.0001,beta_end=.02,beta_schedule='linear',prediction_type='epsilon',algorithm_type='dpmsolver',solver_order=2,solver_type='midpoint',lower_order_final=True,final_sigmas_type='sigma_min',timestep_spacing='linspace')
 scheduler.set_timesteps(a.steps,device='cuda')
 def normalize(value):
  if isinstance(value,float) and not math.isfinite(value):return str(value)
  if isinstance(value,dict):return {k:normalize(v) for k,v in value.items() if not k.startswith('_')}
  if isinstance(value,(list,tuple)):return [normalize(v) for v in value]
  return value
 schedule={'config':normalize(dict(scheduler.config)),'timesteps':scheduler.timesteps.tolist()}
 schedule_hash=hashlib.sha256(json.dumps(schedule,sort_keys=True,allow_nan=False).encode()).hexdigest()
 identity={'checkpoint_sha256':h.hexdigest(),'adapter':'dit-reference-v1','sampler':f'diffusers-{diffusers_version}-dpmsolver-order2-midpoint-linear1000-sigmamin-linspace','noise_schedule_sha256':schedule_hash,'steps':a.steps,'precision':'bf16','resolution':512,'layout':'spatial32x32-region4x4-linear_channels32-QKV_headlocal_padded32','mode':a.mode,'cfg_scale':4.0,'cfg_channels':'first3','mass_threshold':a.mass_threshold,'refresh_interval':a.refresh_interval}
 table=BudgetTable.load(a.budget_table) if a.budget_table else None
 runtime=HarpRuntime(a.mode,32,32,budget_table=table,identity=identity,experimental_uniform=a.experimental_uniform,allowance=a.allowance,mass_threshold=a.mass_threshold,refresh_interval=a.refresh_interval)
 model=DiT_models['DiT-XL/2'](input_size=64,num_classes=1000).cuda().eval()
 loaded=model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True),strict=True);model.to(torch.bfloat16)
 vae=AutoencoderKL.from_pretrained(a.vae,local_files_only=True).cuda().eval()
 # Draw noise after model initialization, matching the earlier dense restoration runner.
 latent=torch.randn(1,4,64,64,device='cuda');labels=torch.tensor([a.class_id,1000],device='cuda')
 out=pathlib.Path(a.output) if a.output else root/'results'/f'harp-dit-{a.mode}'
 out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
 torch.save(latent.cpu(),out/'initial-noise.pt')
 initial_noise_hash=hashlib.sha256(latent.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
 with DiTAdapter(model,runtime):
  for index,timestep in enumerate(scheduler.timesteps):
   runtime.set_step(index)
   with torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    prediction=model.forward_with_cfg(torch.cat([latent,latent]),timestep.expand(2),y=labels,cfg_scale=4.0)
   latent=scheduler.step(prediction[:1,:4].float(),timestep,latent).prev_sample
   if not torch.isfinite(latent).all():raise RuntimeError(f'Nonfinite latent at step {index}')
   if index%5==0:print(f'HARP reference step {index+1}/{a.steps}',flush=True)
 torch.cuda.synchronize();duration=time.perf_counter()-start
 image=vae.decode(latent.float()/.18215).sample
 if not torch.isfinite(image).all():raise RuntimeError('Nonfinite decoded image')
 save_image(image,out/'sample.png',normalize=True,value_range=(-1,1));torch.save(latent.cpu(),out/'latent.pt')
 result={'implementation':'shared HARP torch semantic reference, dense reconstructed GEMM; not fused plane CUDA','identity':identity,'seed':a.seed,'class_id':a.class_id,'strict_load':str(loaded),'experimental_uniform':a.experimental_uniform,'uniform_allowance':a.allowance,'budget_table':a.budget_table,'paper_exact_reproduction':False,'paper_quality_verified':False,'hardware_simulator_restored':False,'functional_wall_seconds':duration,'timing_note':'Reference includes Python/statistics synchronization; not benchmark or speedup evidence','peak_mib':torch.cuda.max_memory_allocated()/2**20,'finite':True,'image_sha256':hashlib.sha256((out/'sample.png').read_bytes()).hexdigest(),'runtime':runtime.stats}
 result['initial_noise_sha256']=initial_noise_hash;result['schedule']=schedule
 result['source_sha256']={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'harp').rglob('*.py')}
 (out/'run.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
