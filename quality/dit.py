"""Class-balanced DiT dense pilot; exact inputs and reconstructed solver recorded."""
import argparse,hashlib,json,pathlib,sys,time
from contextlib import nullcontext
import numpy as np
import torch
from PIL import Image
from quality.dit_paired import MODES, evaluation_items, paired_items


def pilot_items(count,seed_base=100000):
    return paired_items(count, seed_base)


def initial_noise(seeds,*,device,size=64):
    return torch.cat([torch.randn(1,4,size,size,device=device,
        generator=torch.Generator(device=device).manual_seed(seed)) for seed in seeds])


def make_scheduler():
    from diffusers import DPMSolverMultistepScheduler
    return DPMSolverMultistepScheduler(num_train_timesteps=1000,beta_start=.0001,beta_end=.02,
        beta_schedule='linear',prediction_type='epsilon',algorithm_type='dpmsolver',solver_order=2,
        solver_type='midpoint',lower_order_final=True,final_sigmas_type='sigma_min',timestep_spacing='linspace')


def denoise(model,scheduler,noise,class_ids,*,steps=50,runtime=None,cfg=4.):
    scheduler.set_timesteps(steps,device=noise.device)
    batch=len(class_ids)
    if len(noise)!=batch:raise ValueError('Noise and class batch disagree')
    labels=torch.tensor(list(class_ids)+[1000]*batch,device=noise.device)
    latent=noise.clone()
    from torch.nn.attention import sdpa_kernel,SDPBackend
    for index,timestep in enumerate(scheduler.timesteps):
        if runtime is not None:runtime.set_step(index)
        attention=sdpa_kernel(SDPBackend.FLASH_ATTENTION) if noise.is_cuda else nullcontext()
        with torch.autocast(noise.device.type,dtype=torch.bfloat16,enabled=noise.is_cuda),attention:
            prediction=model.forward_with_cfg(torch.cat([latent,latent]),timestep.expand(2*batch),y=labels,cfg_scale=cfg)
        latent=scheduler.step(prediction[:batch,:4].float(),timestep,latent).prev_sample
    if not bool(torch.isfinite(latent).all()):raise RuntimeError('Nonfinite dense latent')
    return latent


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def paired_runtime(mode, table):
    from harp.calibration import BudgetTable
    from quality.dit_budget import CalibrationRuntime

    identity = dict(table["identity"], mode=mode)
    budget = BudgetTable(identity, table["allowances"]) if mode in ("planeclear", "combined") else None
    return CalibrationRuntime(
        mode, 32, 32, budget_table=budget, identity=identity,
        mass_threshold=identity["mass_threshold"], refresh_interval=identity["refresh_interval"],
        attention_backend="masked_chunked", attention_query_chunk=128,
    )


def resume_samples(out, expected, rows, file_sha):
    report = json.loads((out / "generation.json").read_text())
    for key in ("mode", "count", "paired_contract"):
        if report[key] != expected[key]:
            raise ValueError(f"Cannot resume with a different {key}")
    samples = report["samples"]
    if (report["status"] not in ("running", "complete") or len(samples) != report["completed"]
            or len(samples) > len(rows) or (report["status"] == "complete" and len(samples) != len(rows))):
        raise ValueError("Invalid generation progress")
    if len(samples) != len(rows) and len(samples) % expected["batch_size"]:
        raise ValueError("Resume must start at a batch boundary")
    for item, sample in zip(rows, samples):
        filename = f"images/{item['index']:06d}-class{item['class_id']:04d}-seed{item['seed']}.png"
        if (any(sample[key] != value for key, value in item.items())
                or sample["image"] != filename
                or file_sha(out / filename) != sample["image_sha256"]):
            raise ValueError("Completed DiT sample differs from its receipt")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--count',type=int,default=1000);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--steps',type=int,default=50);p.add_argument('--cfg',type=float,default=4.)
    p.add_argument('--seed-base',type=int,default=100000);p.add_argument('--output',required=True)
    p.add_argument('--paired', choices=MODES)
    p.add_argument('--balanced-cohort', action='store_true')
    a=p.parse_args()
    if a.batch_size<=0 or a.steps<=0:raise ValueError('Positive batch size and steps required')
    if a.balanced_cohort and not a.paired:
        p.error("--balanced-cohort requires --paired")
    rows=evaluation_items(a.count,a.seed_base) if a.balanced_cohort else pilot_items(a.count,a.seed_base)
    root=pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'src/dit-original'))
    from models import DiT_models
    from diffusers import AutoencoderKL,__version__ as diffusers_version
    from torchvision.models._meta import _IMAGENET_CATEGORIES
    from harp.adapters.dit import DiTAdapter
    from harp.runtime import HarpRuntime
    out=pathlib.Path(a.output)
    torch.manual_seed(0);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    checkpoint=root/'assets/weights/dit/DiT-XL-2-512x512.pt'
    sha=hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):sha.update(block)
    scheduler=make_scheduler();scheduler.set_timesteps(a.steps)
    schedule={'config':{k:v for k,v in dict(scheduler.config).items() if not k.startswith('_')},'timesteps':scheduler.timesteps.tolist()}
    for key,value in schedule['config'].items():
        if isinstance(value,float) and not np.isfinite(value):schedule['config'][key]=str(value)
    report={'status':'running','model':'DiT-XL/2-512','checkpoint_sha256':sha.hexdigest(),
        'count':a.count,'batch_size':a.batch_size,'resolution':512,'steps':a.steps,'cfg':a.cfg,
        'cfg_channels':'first3','precision':'BF16 transformer operands, FP32 latent/VAE',
        'schedule':schedule,'diffusers':diffusers_version,'torch':torch.__version__,
        'noise_protocol':'independent per-image CUDA Generator; seed_base+index; differs from legacy global RNG after initialization',
        'pixel_protocol':'uint8 clamp(decoded*127.5+128,0,255), RGB PNG; official DiT sample convention',
        'paper_exact_protocol_recovered':False,'quality_metrics_evaluated':False,'completed':0,'samples':[],
        'source_sha256':{str(path.relative_to(root)).replace('\\','/'):hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in [pathlib.Path(__file__),root/'src/dit-original/models.py',root/'harp/adapters/dit.py']}}
    if a.paired:
        import os
        from harp.core import attention as attention_module
        from harp.protocol import schedule_digest
        from quality.dit_budget import public_schedule_record
        from quality.dit_stage4 import file_sha

        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("Paired evaluation requires the fixed CuBLAS workspace")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        selector = attention_module.select_region_mask
        attention_module.select_region_mask = lambda mass, threshold: selector(mass.cpu(), threshold).to(mass.device)

        table_path = root / "src/dit-original/harp/lambda.json"
        table = json.loads(table_path.read_text())
        identity = table["identity"]
        actual = {
            "checkpoint_sha256": sha.hexdigest(), "steps": a.steps, "cfg_scale": a.cfg,
            "noise_schedule_sha256": schedule_digest(public_schedule_record(scheduler)),
            "resolution": 512, "cfg_channels": "first3",
        }
        if any(identity.get(key) != value for key, value in actual.items()) or a.seed_base != 100000:
            raise ValueError("Paired evaluation settings differ from the fixed DiT protocol")
        source_files = sorted((root / "harp").rglob("*.py"))
        source_files += [
            root / path for path in ("src/dit-original/models.py", "quality/dit.py",
                                     "quality/dit_paired.py", "quality/dit_budget.py")
        ]
        report.update(mode=a.paired, paired_contract={
            **actual, "seed_base": a.seed_base, "batch_size": a.batch_size,
            "torch": torch.__version__, "diffusers": diffusers_version, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "source_sha256": {str(path.relative_to(root)): file_sha(path) for path in source_files},
            "budget_sha256": file_sha(table_path),
            "vae_sha256": {p.name: file_sha(p) for p in sorted((root / "assets/weights/dit-vae").iterdir())
                           if p.is_file()},
        })
        if a.balanced_cohort:
            report["paired_contract"]["cohort"] = "class-balanced-v1"
        if (out / "generation.json").is_file():
            report = resume_samples(out, report, rows, file_sha)
            if report["status"] == "complete":
                print(f"Verified completed DiT {a.paired} generation")
                return
        elif out.exists() and any(
            not ((path.name == "images" and path.is_dir() and not any(path.iterdir()))
                 or (path.name == "generation.tmp" and path.is_file()))
            for path in out.iterdir()
        ):
            raise ValueError("Existing DiT output has no generation receipt")
    elif out.exists():
        raise FileExistsError(out)
    model=DiT_models['DiT-XL/2'](input_size=64,num_classes=1000).cuda().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    model.to(torch.bfloat16)
    vae=AutoencoderKL.from_pretrained(root/'assets/weights/dit-vae',local_files_only=True).cuda().eval()
    out.mkdir(parents=True, exist_ok=bool(a.paired))
    (out / "images").mkdir(exist_ok=bool(a.paired))
    def save():
        temporary=out/'generation.tmp';temporary.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8');temporary.replace(out/'generation.json')
    save();started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    previous_seconds = report.get("elapsed_seconds", 0)
    runtime=paired_runtime(a.paired, table) if a.paired else HarpRuntime('dense',32,32)
    with torch.inference_mode(),DiTAdapter(model,runtime):
        for start in range(report['completed'],len(rows),a.batch_size):
            batch=rows[start:start+a.batch_size]
            if a.paired:
                runtime.reset()
            noise=initial_noise([r['seed'] for r in batch],device='cuda')
            latent=denoise(model,make_scheduler(),noise,[r['class_id'] for r in batch],steps=a.steps,runtime=runtime,cfg=a.cfg)
            images=[]
            for offset in range(0,len(batch),4):
                decoded=vae.decode(latent[offset:offset+4].float()/.18215).sample
                if not bool(torch.isfinite(decoded).all()):raise RuntimeError('Nonfinite VAE output')
                images.extend((decoded*127.5+128).clamp(0,255).permute(0,2,3,1).to(torch.uint8).cpu().numpy())
            for index,(item,image) in enumerate(zip(batch,images)):
                filename=f"images/{item['index']:06d}-class{item['class_id']:04d}-seed{item['seed']}.png"
                Image.fromarray(image).save(out/filename)
                report['samples'].append(dict(item,class_name=_IMAGENET_CATEGORIES[item['class_id']],image=filename,
                    noise_sha256=digest(noise[index]),latent_sha256=digest(latent[index]),image_sha256=hashlib.sha256((out/filename).read_bytes()).hexdigest()))
            report['completed']=len(report['samples']);report['elapsed_seconds']=previous_seconds+time.perf_counter()-started
            if a.paired:
                report["peak_mib"] = max(report.get("peak_mib", 0), torch.cuda.max_memory_allocated()/2**20)
            save()
            print(f"DiT {a.paired or 'dense'}: {report['completed']}/{a.count} images in {report['elapsed_seconds']:.1f}s",flush=True)
    report.update(status='complete',peak_mib=max(report.get("peak_mib", 0),
                                                 torch.cuda.max_memory_allocated()/2**20));save()


if __name__=='__main__':main()
