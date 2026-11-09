"""Official Wan generation with an optional, explicitly scoped HARP adapter."""
import argparse
import hashlib
import json
import math
import pathlib
import time
from contextlib import nullcontext
import torch
from .adapters.wan import WanAdapter
from .calibration import BudgetTable
from .protocol import schedule_digest


def arguments(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['official', 'dense', 'residual', 'planeclear', 'combined'], default='official')
    p.add_argument('--frames', type=int, default=81)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--shift', type=float, default=5.)
    p.add_argument('--cfg', type=float, default=5.)
    p.add_argument('--prompt', default='A cat walks across a sunlit room.')
    p.add_argument('--negative-prompt', default='')
    p.add_argument('--checkpoint', default='/opt/harp/assets/weights/wan2.1-t2v-14b')
    p.add_argument('--experimental-uniform', action='store_true')
    p.add_argument('--allowance', type=float)
    p.add_argument('--budget-table')
    p.add_argument('--mass-threshold', type=float, default=.95)
    p.add_argument('--refresh-interval', type=int, default=5)
    p.add_argument('--attention-backend',choices=['selected','masked_chunked'],default='selected',
                   help='masked_chunked is a dense masked FP32 quality oracle, not sparse execution')
    p.add_argument('--attention-query-chunk',type=int,default=64)
    p.add_argument('--output', required=True)
    a = p.parse_args(argv)
    if a.frames <= 0 or a.frames % 4 != 1 or a.steps <= 0 or a.seed < 0:
        raise ValueError('Use positive steps, 4n+1 frames and an explicit nonnegative seed')
    if not math.isfinite(a.shift) or a.shift <= 0 or not math.isfinite(a.cfg) or a.cfg < 0:
        raise ValueError('Invalid shift/CFG')
    if not 0 < a.mass_threshold <= 1 or a.refresh_interval <= 0:
        raise ValueError('Invalid attention schedule')
    if a.attention_query_chunk<=0:
        raise ValueError('Attention query chunk must be positive')
    if a.attention_backend!='selected' and a.mode not in {'residual','combined'}:
        raise ValueError('Masked quality backend applies only to residual/combined')
    if a.mode in {'planeclear', 'combined'}:
        if bool(a.budget_table) == bool(a.experimental_uniform):
            raise ValueError('Choose a table or explicit experimental uniform allowance')
        if a.experimental_uniform and (a.allowance is None or not math.isfinite(a.allowance) or not 0 <= a.allowance <= 1):
            raise ValueError('Invalid explicit allowance')
    elif a.budget_table or a.experimental_uniform or a.allowance is not None:
        raise ValueError('Budget options require a PlaneClear mode')
    return a


def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def verify_checkpoint(directory, plan):
    directory = pathlib.Path(directory).resolve()
    records = {}
    expected_paths = {row['path'] for row in plan['files']}
    if {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()} != expected_paths:
        raise ValueError('Checkpoint file inventory differs from pinned manifest')
    for row in plan['files']:
        path = (directory/row['path']).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or path.stat().st_size != row['bytes']:
            raise ValueError('Checkpoint inventory mismatch: '+row['path'])
        sha = hashlib.sha256()
        git = hashlib.sha1(f"blob {row['bytes']}\0".encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                sha.update(block)
                if not row.get('sha256'):
                    git.update(block)
        if (sha.hexdigest() if row.get('sha256') else git.hexdigest()) != (row.get('sha256') or row.get('git_blob_sha1')):
            raise ValueError('Checkpoint content mismatch: '+row['path'])
        records[row['path']] = sha.hexdigest()
    return hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()


def run(a, pipeline=None):
    table = BudgetTable.load(a.budget_table) if a.budget_table else None
    out = pathlib.Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    from wan import WanT2V
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import cache_video
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    import diffusers
    root = pathlib.Path(__file__).resolve().parents[1]
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = WAN_CONFIGS['t2v-14B']
    schedule = FlowUniPCMultistepScheduler(num_train_timesteps=cfg.num_train_timesteps, shift=1, use_dynamic_shifting=False)
    schedule.set_timesteps(a.steps, device='cpu', shift=a.shift)
    schedule_record = {'config': dict(schedule.config), 'timesteps': schedule.timesteps.tolist(), 'sigmas': schedule.sigmas.tolist()}
    source = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for folder in (root/'harp', root/'src/Wan2.1/wan') for p in folder.rglob('*.py')}
    identity = {'checkpoint_sha256': 'see_verified_asset_manifest',
                'adapter': 'wan-self-ffn-reference-v1' if a.mode != 'official' else 'official-unmodified',
                'sampler': 'official-FlowUniPCMultistepScheduler',
                'noise_schedule_sha256': schedule_digest(schedule_record),
                'steps': a.steps, 'precision': 'official BF16 autocast; FP32 latent',
                'resolution': [832, 480], 'frames': a.frames,
                'layout': 'frame-local-4x4-real-edge-regions-channel32-headlocal',
                'mode': a.mode, 'cfg_scale': a.cfg, 'cfg_channels': 'all',
                'mass_threshold': a.mass_threshold, 'refresh_interval': a.refresh_interval,
                'negative_prompt': a.negative_prompt or cfg.sample_neg_prompt,
                'cross_attention': 'dense'}
    identity['attention_execution_backend']=a.attention_backend
    identity['attention_query_chunk']=a.attention_query_chunk
    # Bind budget identity to the verified asset metadata, not a model name.
    asset_plan = root/'manifests/wan-download-plan.json'
    if pipeline is None:
        print('Verifying actual Wan checkpoint contents against pinned manifest',flush=True)
        asset_identity = verify_checkpoint(a.checkpoint, json.loads(asset_plan.read_text()))
    else:
        if str(pathlib.Path(a.checkpoint).resolve()) != pipeline.harp_checkpoint_path:
            raise ValueError('Cannot reuse pipeline with a different checkpoint')
        asset_identity = pipeline.harp_checkpoint_identity
    identity['checkpoint_sha256'] = asset_identity
    identity['implementation_sha256'] = hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
    if table is not None:
        table.validate(identity)
    result = {'status': 'running', 'identity': identity, 'seed': a.seed, 'prompt': a.prompt,
              'schedule': schedule_record, 'source_sha256': source,
              'asset_identity_note': 'checkpoint_sha256 hashes verified actual-file SHA256 inventory; full verification on pipeline load',
              'torch': torch.__version__, 'diffusers': diffusers.__version__,
              'paper_quality_verified': False, 'paper_exact_reproduction': False,
              'quality_metrics_evaluated': False, 'experimental_uniform': a.experimental_uniform,
              'allowance': a.allowance, 'budget_table': a.budget_table,
              'scope': 'single-device t2v-14B; self-attention/FFN approximation; cross-attention dense'}
    (out/'run.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    if pipeline is None:
        pipeline = WanT2V(cfg, a.checkpoint, t5_cpu=True)
        pipeline.harp_checkpoint_identity = asset_identity
        pipeline.harp_checkpoint_path = str(pathlib.Path(a.checkpoint).resolve())
    result['model_parameter_dtype'] = str(next(pipeline.model.parameters()).dtype)
    calls = [0]

    def capture_input(module, args, kwargs):
        if calls[0] == 0:
            initial = args[0][0].detach().cpu()
            torch.save(initial, out/'initial-noise.pt')
            result['initial_noise_sha256'] = tensor_hash(initial)
        calls[0] += 1
        if calls[0] % 2 == 0:
            print(f'Wan {a.mode}: denoising step {calls[0]//2}/{a.steps}', flush=True)

    handle = pipeline.model.register_forward_pre_hook(capture_input, with_kwargs=True)
    decode = pipeline.vae.decode

    def capture_final(z):
        final = z[0].detach().cpu()
        torch.save(final, out/'latent.pt')
        result['latent_sha256'] = tensor_hash(final)
        result['finite_latent'] = bool(torch.isfinite(final).all())
        return decode(z)

    pipeline.vae.decode = capture_final
    options = dict(experimental_uniform=a.experimental_uniform, allowance=a.allowance,
                   budget_table=table,
                   identity=identity, mass_threshold=a.mass_threshold, refresh_interval=a.refresh_interval,
                   residual_cache_device='cpu',plane_chunk_size=4096,
                   attention_backend=a.attention_backend,attention_query_chunk=a.attention_query_chunk)
    adapter = WanAdapter(pipeline.model, a.mode, **options) if a.mode != 'official' else None
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        with adapter if adapter is not None else nullcontext():
            video = pipeline.generate(a.prompt, size=(832,480), frame_num=a.frames,
                shift=a.shift, sample_solver='unipc', sampling_steps=a.steps,
                guide_scale=a.cfg, n_prompt=a.negative_prompt, seed=a.seed, offload_model=True)
            if adapter is not None:
                result['runtime'] = adapter.summary()
        result['finite_video'] = bool(torch.isfinite(video).all())
        result['video_tensor_sha256'] = tensor_hash(video)
        if calls[0] != 2*a.steps or not result['finite_latent'] or not result['finite_video']:
            raise RuntimeError('Unexpected CFG call count or nonfinite generation')
        cache_video(tensor=video[None], save_file=str(out/'sample.mp4'), fps=cfg.sample_fps,
                    nrow=1, normalize=True, value_range=(-1,1))
        import cv2
        capture = cv2.VideoCapture(str(out/'sample.mp4'))
        decoded_frames = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (480, 832):
                raise RuntimeError('Encoded video dimensions changed')
            decoded_frames += 1
        capture.release()
        if decoded_frames != a.frames:
            raise RuntimeError('Encoded video frame count mismatch')
        result['decoded_video_frames'] = decoded_frames
        result['video_file_sha256'] = hashlib.sha256((out/'sample.mp4').read_bytes()).hexdigest()
        result.update(status='complete', forward_calls=calls[0], functional_wall_seconds=time.perf_counter()-started,
                      peak_mib=torch.cuda.max_memory_allocated()/2**20,
                      timing_note='Functional Python reference including statistics and decode; not performance evidence')
    except Exception as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        handle.remove()
        pipeline.vae.decode = decode
        (out/'run.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status','initial_noise_sha256','latent_sha256','functional_wall_seconds')}, indent=2))
    return pipeline


def main():
    run(arguments())


if __name__ == '__main__':
    main()
