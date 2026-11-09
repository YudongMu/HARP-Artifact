"""Heldout Wan Stage4 scorer. Official point scores; explicit, auditable NA.

This module imports no tensor libraries until actual artifact validation/scoring.
The protocol is a fixed eleven-prompt, two-seed development experiment, not the
946x5 VBench benchmark and not a calibration of the missing hardware gain G.
"""
import argparse
import copy
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
import pathlib
import random
import shutil
import sys
import time
from contextlib import nullcontext
from .wan_semantic import ALL_DIMENSIONS, PROMPTS

MODES = ('dense', 'residual', 'planeclear', 'combined')
HELDOUT_PROMPTS = tuple('a red car' if p == 'a red bicycle' else p for p in PROMPTS)
VBENCH_COMMIT = 'fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490'
VBENCH_ARCHIVE_SHA = '571c1b5d6cf5d77c48b124d8f3070f625c3b7215f07b6a5d45edef70247ccaf4'
# Frozen extracted official vbench/**/*.py plus the two official aggregate files.
VBENCH_PY_TREE_SHA = 'cc4aa0b6133ff56e39a333488e037ac341247a53f79f4ab15f3f74889658af33'
# All233 original archive members in this scope were compared byte-for-byte.
# The former226-file local backup omitted7 non-Python members; no scorer changed.
# Evidence: results/stage4-vbench-source-archive-check.json (archive SHA above).
VBENCH_TREE_SHA = 'f8ec235f5763ab6057c8aa8c118d8cbb6a758e8193a6ca3922631fc1f2b06bb8'
VBENCH_TREE_FILES = 233


def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8*1024*1024), b''): digest.update(data)
    return digest.hexdigest()


def read(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = pathlib.Path(path)
    temp = path.with_suffix(path.suffix+'.incoming')
    temp.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def schedule_sha(record):
    public = dict(record, config={k:v for k,v in record['config'].items() if not k.startswith('_')})
    return hashlib.sha256(json.dumps(public, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_manifest(manifest, official):
    """Require the predeclared split and untouched, uniquely eligible metadata."""
    if manifest.get('frames') not in (9,17) or manifest.get('steps') != 50 or manifest.get('resolution') != [832,480]:
        raise ValueError('Independent short-video 9/17-frame fifty-step protocol required')
    inputs = manifest['inputs']; held = inputs['heldout']; cal = inputs['calibration']
    if len(held) != 22 or len(cal) != 1: raise ValueError('Expected 22 heldout and one independent calibration input')
    if (cal[0]['id'],cal[0]['prompt'],cal[0]['seed']) != ('cal-0','a dog and a horse',510000):
        raise ValueError('Calibration input changed or overlaps heldout')
    for i,prompt in enumerate(HELDOUT_PROMPTS):
        for j in range(2):
            row = held[2*i+j]
            if (row['id'],row['prompt_index'],row['seed_index'],row['prompt'],row['seed']) != (f'held-{i:02d}-s{j}',i,j,prompt,520000+2*i+j):
                raise ValueError('Heldout identity/order/seed differs from predeclared protocol')
    for row in cal+held:
        matches = [i for i,r in enumerate(official) if r['prompt_en'] == row['prompt']]
        if len(matches) != 1 or row['source_index'] != matches[0] or row['metadata'] != official[matches[0]]:
            raise ValueError('Original official metadata/source membership differs')
    if set().union(*(set(r['metadata']['dimension']) for r in held)) != set(ALL_DIMENSIONS):
        raise ValueError('Legal heldout membership does not cover all sixteen dimensions')
    if set(r['seed'] for r in cal) & set(r['seed'] for r in held): raise ValueError('Calibration/heldout seeds overlap')
    return held


def validate_index(index, manifest):
    """No mixing modes, candidate/control runs, duplicates, or incomplete pools."""
    if index.get('status') != 'complete' or index.get('frames') != manifest['frames']:
        raise ValueError('Heldout generation index is incomplete or has another frame protocol')
    expected = {r['id']:r for r in manifest['inputs']['heldout']}
    modes = {m:{} for m in MODES}; paths = set()
    for case in index['cases']:
        if case['split'] != 'heldout': continue
        key,mode = case['input_id'],case['mode']
        if key not in expected or mode not in modes or key in modes[mode] or case.get('label',mode) != mode:
            raise ValueError('Mixed, duplicate, or unrecognized heldout mode/input')
        if case['metadata'] != expected[key]['metadata']: raise ValueError('Index metadata differs from manifest')
        if case['run_directory'] in paths: raise ValueError('A video directory was pooled across modes')
        paths.add(case['run_directory']); modes[mode][key] = case
    if any(set(rows) != set(expected) for rows in modes.values()): raise ValueError('Each mode requires its own complete 22-video heldout pool')
    return modes


def validate_run_receipt(run, mode, item, frames, observer):
    identity = run['identity']
    if run.get('status') != 'complete' or run.get('forward_calls') != 100 or run.get('decoded_video_frames') != frames or not run.get('finite_latent') or not run.get('finite_video'):
        raise ValueError('Incomplete/nonfinite generation receipt')
    if (identity.get('mode'),identity.get('frames'),identity.get('wan_stage4_frames'),identity.get('steps'),identity.get('resolution')) != (mode,frames,frames,50,[832,480]):
        raise ValueError('Generation mode/frame/step deployment differs')
    if identity.get('adapter') != 'wan-self-ffn-reference-v1' or identity.get('cfg_scale') != 5 or identity.get('cfg_channels') != 'all' or identity.get('mass_threshold') != .95 or identity.get('refresh_interval') != 5:
        raise ValueError('Unexpected adapter/CFG/Residual protocol')
    if not identity.get('wan_stage4_sidecar_sha256') or not identity.get('checkpoint_sha256') or not run.get('source_sha256'):
        raise ValueError('Missing model/base-code/sidecar source identity')
    if run['prompt'] != item['prompt'] or run['seed'] != item['seed']: raise ValueError('Generated prompt/seed differs from heldout declaration')
    if identity.get('noise_schedule_sha256') != schedule_sha(run['schedule']): raise ValueError('Actual sampler schedule differs from recorded identity')
    if observer.get('status') != 'complete' or observer.get('completed_model_calls') != 100 or observer.get('completed_transformer_blocks') != 4000:
        raise ValueError('Incomplete actual observed model/block trajectory')


def validate_paired_receipts(reference, other):
    exclusions = {'mode','attention_execution_backend','attention_query_chunk'}
    if {k:v for k,v in reference['identity'].items() if k not in exclusions} != {k:v for k,v in other['identity'].items() if k not in exclusions}:
        raise ValueError('Paired deployment identity differs')
    for field in ('prompt','seed','source_sha256','initial_noise_sha256'):
        if reference[field] != other[field]: raise ValueError('Unpaired '+field)
    if schedule_sha(reference['schedule']) != schedule_sha(other['schedule']): raise ValueError('Public sampler schedules differ')


def validate_selection(selection, index, root):
    """Verify a real nonzero policy came only from the independent calibration."""
    if selection.get('heldout_used_for_selection') is not False or selection.get('paper_policy') is not False or selection.get('G') is not None:
        raise ValueError('Invalid calibration-only selection scope')
    selected = selection.get('selected'); exploratory = selection.get('exploratory')
    if bool(selected) == bool(exploratory): raise ValueError('Declare exactly one selected or exploratory development policy')
    policy = selected or exploratory
    values = list(policy.get('allowances',{}).values())
    if not values or not all(isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1 for v in values) or not any(v>0 for v in values):
        raise ValueError('A real nonzero finite development policy is required')
    if policy.get('G') is not None or policy.get('cycle_optimal') is not False: raise ValueError('Cannot assert gain/cycle-optimality without G')
    if index.get('selected_development_policy') is not bool(selected) or index.get('exploratory_only') is not bool(exploratory):
        raise ValueError('Generation index and calibration selection disagree')
    root = pathlib.Path(root).resolve(); candidate = pathlib.Path(policy['calibration_run_directory']).resolve()
    if not candidate.is_relative_to(root/'runs/cal-0'): raise ValueError('Selection used a heldout/noncalibration candidate')
    matches = [r for r in index['cases'] if r.get('split')=='calibration' and r.get('input_id')=='cal-0' and r.get('kind')=='candidate' and pathlib.Path(r['run_directory']).resolve()==candidate]
    if len(matches) != 1: raise ValueError('Missing unique actual calibration candidate receipt')
    if set(policy.get('tables',{})) != {'planeclear','combined'}: raise ValueError('Separate mode-matched tables required')
    for path in policy['tables'].values():
        if not pathlib.Path(path).resolve().is_relative_to(root/'tables'): raise ValueError('Policy table path is outside independent stage4 root')
    return policy


def validate_execution_receipt(execution, mode, input_id, sidecar, table_sha, artifact_shas):
    if execution.get('mode') != mode or execution.get('input_id') != input_id: raise ValueError('Observer mode/input identity differs')
    if execution.get('sidecar_sha256') != sidecar or execution.get('sidecar_sha256_after') != sidecar: raise ValueError('Executed sidecar changed before/after generation')
    if execution.get('budget_table_sha256_before') != table_sha or execution.get('budget_table_sha256_after') != table_sha: raise ValueError('Executed budget table changed before/after generation or differs from frozen selection')
    if any(execution.get('artifacts',{}).get(name) != digest for name,digest in artifact_shas.items()): raise ValueError('Executed artifact bytes differ from completion receipt')


def load_official_aggregate(source):
    """Load exactly the pinned pure aggregate functions, never submission's zero fill."""
    source = pathlib.Path(source)
    constant_spec = importlib.util.spec_from_file_location('constant', source/'scripts/constant.py')
    constant = importlib.util.module_from_spec(constant_spec); constant_spec.loader.exec_module(constant)
    previous = sys.modules.get('constant'); sys.modules['constant'] = constant
    try:
        spec = importlib.util.spec_from_file_location('_wan_stage4_official_aggregate', source/'scripts/cal_final_score.py')
        aggregate = importlib.util.module_from_spec(spec); spec.loader.exec_module(aggregate)
    finally:
        if previous is None: sys.modules.pop('constant',None)
        else: sys.modules['constant'] = previous
    if set(aggregate.TASK_INFO) != {d.replace('_',' ') for d in ALL_DIMENSIONS}: raise ValueError('Unexpected official aggregate dimensions')
    return aggregate


def normalize_scores(dimensions, aggregate):
    invalid = [d for d in ALL_DIMENSIONS if d not in dimensions or dimensions[d].get('status') != 'complete' or not isinstance(dimensions[d].get('score'),(int,float)) or not math.isfinite(dimensions[d]['score'])]
    empty = dict(quality=None,semantic=None,total=None)
    if invalid or set(dimensions) != set(ALL_DIMENSIONS):
        return dict(all_dimensions_finite=False,invalid_dimensions=invalid,display_percent=empty,quality_0_1=None,semantic_0_1=None,total_0_1=None)
    raw = {d.replace('_',' '):dimensions[d]['score'] for d in ALL_DIMENSIONS}
    normalized = aggregate.get_nomalized_score(raw)
    quality = aggregate.get_quality_score(normalized); semantic = aggregate.get_semantic_score(normalized)
    total = aggregate.get_final_score(quality,semantic)
    if not all(math.isfinite(v) for v in (quality,semantic,total)): raise ValueError('Official aggregate produced nonfinite output')
    return dict(all_dimensions_finite=True,invalid_dimensions=[],raw_dimension_scores=raw,weighted_normalized_dimensions=normalized,
        quality_0_1=quality,semantic_0_1=semantic,total_0_1=total,display_percent=dict(quality=100*quality,semantic=100*semantic,total=100*total),
        normalization='Official get_nomalized_score; no clipping; quality/semantic weights4:1; displayed times100')


def contribution_stats(dimension, detail, frames):
    """Single-rank official sufficient statistics, without replacing point scores."""
    if dimension in {'object_class','multiple_objects','scene'}:
        numerator,denominator = detail['success_frame_count'],detail['frame_count']; rule = 'sum_success_frames/sum_sampled_frames'
    elif dimension == 'background_consistency':
        numerator,denominator = detail['video_sim'],detail['cnt_per_video']; rule = 'sum_video_sim/sum_frame_transitions'
    elif dimension == 'subject_consistency':
        # Official load_video without num_frames decodes all actual verified frames.
        denominator = frames-1; numerator = float(detail['video_results'])*denominator; rule = 'sum_similarity/sum_verified_frame_transitions'
    elif dimension in {'appearance_style','spatial_relationship'}:
        values = detail['frame_results']; numerator,denominator = sum(values),len(values); rule = 'sum_frame_results/count_frame_results_single_rank'
    else:
        numerator,denominator = float(detail['video_results']),1; rule = 'mean_per_video'
        if dimension == 'imaging_quality': numerator /= 100.; rule = 'mean_per_video_divided_by100'
    numerator,denominator = float(numerator),float(denominator)
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0: raise ValueError('Invalid sufficient statistics')
    return dict(numerator=numerator,denominator=denominator,rule=rule)


def bind_details(dimension, official_score, details, eligible, frames):
    value = float(official_score)
    if not math.isfinite(value) or not isinstance(details,list) or not details or not eligible: raise ValueError('Nonfinite or empty official dimension result')
    by_path = {str(pathlib.Path(row['video_path']).resolve()):row for row in eligible}
    if len(by_path) != len(eligible): raise ValueError('Duplicate eligible video paths')
    seen = set(); contributions = []; bound = []
    for detail in details:
        path = str(pathlib.Path(detail['video_path']).resolve())
        if path not in by_path or path in seen: raise ValueError('Official details contain duplicated or another mode video')
        seen.add(path); item = by_path[path]
        bound.append(dict(detail,input_id=item['input_id'],prompt_index=item['prompt_index'],seed_index=item['seed_index'],seed=item['seed']))
        try: stats = contribution_stats(dimension,detail,frames)
        except (KeyError,TypeError,ValueError) as error:
            stats = dict(numerator=None,denominator=None,rule='unavailable_from_official_details',reason=type(error).__name__+': '+str(error))
        contributions.append(dict(input_id=item['input_id'],prompt_index=item['prompt_index'],seed_index=item['seed_index'],seed=item['seed'],**stats))
    omitted = [row for path,row in by_path.items() if path not in seen]
    if omitted and dimension != 'color': raise ValueError('Official dimension omitted eligible videos')
    # Official color intentionally excludes videos with no qualifying denominator.
    # Resampling an excluded video contributes0/0, not an invented zero score.
    for item in omitted:
        contributions.append(dict(input_id=item['input_id'],prompt_index=item['prompt_index'],seed_index=item['seed_index'],seed=item['seed'],
            numerator=0.,denominator=0.,rule='official_color_conditional_exclusion'))
    stats_complete = all(row['denominator'] is not None for row in contributions)
    denominator = sum(row['denominator'] for row in contributions) if stats_complete else 0
    reconstructed = sum(row['numerator'] for row in contributions)/denominator if denominator > 0 else None
    exact = reconstructed is not None and math.isclose(reconstructed,value,rel_tol=1e-6,abs_tol=1e-8)
    return dict(status='complete',score=value,eligible_videos=len(eligible),official_result_videos=len(details),per_video=bound,
        omitted_input_ids=[r['input_id'] for r in omitted],contributions=contributions,reconstructed_score=reconstructed,
        exact_reconstruction=exact,bootstrap_available=exact,
        bootstrap_reason='Verified official single-rank aggregate reconstruction' if exact else 'Official point retained; raw details did not accurately reconstruct aggregate; CI unavailable')


def resampled_dimension(result, input_ids):
    """Reconstruct a bootstrap dimension, using the same repeated IDs across modes."""
    if not result.get('exact_reconstruction'): return None
    stats = {r['input_id']:r for r in result['contributions']}
    selected = [stats[key] for key in input_ids if key in stats]
    denominator = sum(r['denominator'] for r in selected)
    return sum(r['numerator'] for r in selected)/denominator if denominator > 0 else None


def verify_sources(assets):
    assets = pathlib.Path(assets).resolve()
    source = assets/'source/VBench'; receipt = read(assets/'source-manifest.json')['repositories']['VBench']
    if receipt['commit'] != VBENCH_COMMIT or receipt['archive_sha256'] != VBENCH_ARCHIVE_SHA: raise ValueError('Official VBench source pin differs')
    files = sorted([p for p in (source/'vbench').rglob('*.py')]+[source/'scripts/cal_final_score.py',source/'scripts/constant.py'],key=lambda p:p.relative_to(source).as_posix())
    tree = {p.relative_to(source).as_posix():sha(p) for p in files}
    digest = hashlib.sha256(json.dumps(tree,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest != VBENCH_PY_TREE_SHA or len(tree) != 177: raise ValueError('Actual official scorer source tree differs from frozen extracted source')
    files = [p for p in (source/'vbench').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')]+[source/'scripts/cal_final_score.py',source/'scripts/constant.py']
    all_tree = {p.relative_to(source).as_posix():sha(p) for p in sorted(files)}
    full_digest = hashlib.sha256(json.dumps(all_tree,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if full_digest != VBENCH_TREE_SHA or len(all_tree) != VBENCH_TREE_FILES: raise ValueError('Official scorer code/config/metadata tree differs from frozen source')
    dependencies = {}
    for name,directory,count,expected in (
        ('DINO',assets/'vbench-cache/dino_model/facebookresearch_dino_main',12,'63768e6d91a29228ae29e125a6cbdb6f8b80b5fb9f6dfa66973e1b577903d84d'),
        ('PyIQA',assets/'source/PyIQA/pyiqa',108,'64a054670e80f751397dd9233903e4bccf2d4a7353a82e55e014ae82f6e106a1')):
        inventory = {p.relative_to(directory).as_posix():sha(p) for p in sorted(directory.rglob('*.py')) if '__pycache__' not in p.parts}
        digest_dep = hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if len(inventory) != count or digest_dep != expected: raise ValueError(name+' actual source differs from frozen dependency')
        dependencies[name] = dict(files_sha256=inventory,tree_sha256=digest_dep)
    metadata = source/'vbench/VBench_full_info.json'
    if sha(metadata) != read(assets/'source-manifest.json')['vbench_prompts']['sha256']: raise ValueError('Official metadata bytes changed')
    return dict(source=receipt,official_python_files_sha256=tree,official_python_tree_sha256=digest,
        official_tree_files=len(all_tree),official_all_files_sha256=all_tree,official_tree_sha256=full_digest,dependencies=dependencies,metadata_sha256=sha(metadata))


def verify_assets(assets):
    inventory_path = assets/'wan-semantic-deployed-inventory.json'; inventory = read(inventory_path)
    rows = inventory['checkpoint_files']
    if len(rows) != 12 or sum(r['bytes'] for r in rows) != 9425878528: raise ValueError('Expected complete frozen12-checkpoint inventory')
    for row in rows:
        file = (assets/row['path']).resolve()
        if not file.is_relative_to(assets) or file.stat().st_size != row['bytes'] or sha(file) != row['sha256']: raise ValueError('Deployed scorer checkpoint changed: '+row['path'])
    for name,digest in inventory['receipt_sha256'].items():
        if sha(assets/name) != digest: raise ValueError('Scorer asset/source receipt changed: '+name)
    return dict(inventory_sha256=sha(inventory_path),inventory=inventory,all_checkpoint_sha256_recomputed=True,**verify_sources(assets))


def validate_artifacts(root, manifest, pools, policy):
    """Decode actual videos and hash tensor bytes; recorded JSON alone is insufficient."""
    import torch
    import cv2
    items = {r['id']:r for r in manifest['inputs']['heldout']}; result = {m:[] for m in MODES}; reference_deployment = None
    records = {p.name:sha(p) for p in (pathlib.Path(__file__).with_name('wan_budget.py'),pathlib.Path(__file__).with_name('wan_stage4.py'))}
    sidecar = hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
    tables = {}
    for mode,path in policy['tables'].items():
        path = pathlib.Path(path).resolve(); table = read(path)
        if table.get('allowances') != policy['allowances']: raise ValueError('Selected table allowances differ from calibration policy')
        tables[mode] = dict(path=path,sha256=sha(path),content=table)
    for key,item in items.items():
        reference = None; reference_noise = None
        for mode in MODES:
            case = pools[mode][key]; folder = (root/'runs'/key/mode).resolve()
            if pathlib.Path(case['run_directory']).resolve() != folder: raise ValueError('Index points outside canonical heldout mode directory')
            run = read(folder/'run.json'); observer = read(folder/'stage4-execution.json')
            if sha(folder/'run.json') != case['run_sha256']: raise ValueError('Generation receipt changed since index')
            validate_run_receipt(run,mode,item,manifest['frames'],observer)
            if run['identity']['wan_stage4_sidecar_sha256'] != sidecar: raise ValueError('Executed stage4 source differs from current frozen sidecar')
            artifact_shas = {name:sha(folder/name) for name in ('run.json','initial-noise.pt','latent.pt','sample.mp4')}
            table = tables.get(mode)
            validate_execution_receipt(observer,mode,key,sidecar,table['sha256'] if table else None,artifact_shas)
            if table:
                if pathlib.Path(run['budget_table']).resolve() != table['path']: raise ValueError('Generation used another table path')
                if any(run['identity'].get(k) != v for k,v in table['content']['identity'].items()): raise ValueError('Selected budget table deployment identity differs')
            elif run.get('budget_table') is not None: raise ValueError('Dense/Residual must not use PlaneClear tables')
            deployment = {k:v for k,v in run['identity'].items() if k not in {'mode','attention_execution_backend','attention_query_chunk'}}
            actual = dict(identity=deployment,source=run['source_sha256'],schedule=schedule_sha(run['schedule']))
            if reference_deployment is None: reference_deployment = actual
            elif reference_deployment != actual: raise ValueError('Mixed source/deployment/sampler among heldout videos')
            noise = torch.load(folder/'initial-noise.pt',map_location='cpu',weights_only=True)
            if tuple(noise.shape) != (16,(manifest['frames']-1)//4+1,60,104) or noise.dtype != torch.float32 or not bool(torch.isfinite(noise).all()): raise ValueError('Actual initial noise shape/type/finite check failed')
            noise_sha = hashlib.sha256(noise.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            if noise_sha != run['initial_noise_sha256']: raise ValueError('Actual initial noise bytes differ from receipt')
            latent = torch.load(folder/'latent.pt',map_location='cpu',weights_only=True)
            if tuple(latent.shape) != tuple(noise.shape) or not bool(torch.isfinite(latent).all()): raise ValueError('Actual final latent shape/finite differs')
            if hashlib.sha256(latent.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() != run['latent_sha256']: raise ValueError('Actual final latent bytes differ from receipt')
            if reference is None: reference,reference_noise = run,noise_sha
            else:
                validate_paired_receipts(reference,run)
                if reference_noise != noise_sha: raise ValueError('Actual four-mode initial noise bytes differ')
            video = folder/'sample.mp4'
            if sha(video) != run['video_file_sha256']: raise ValueError('Video bytes differ from receipt')
            capture = cv2.VideoCapture(str(video)); count = 0
            try:
                while True:
                    ok,frame = capture.read()
                    if not ok: break
                    if frame.shape != (480,832,3): raise ValueError('Actual decoded video resolution differs')
                    count += 1
            finally: capture.release()
            if count != manifest['frames']: raise ValueError('Actual decoded frame count differs')
            result[mode].append(dict(input_id=key,prompt_index=item['prompt_index'],seed_index=item['seed_index'],seed=item['seed'],prompt=item['prompt'],
                metadata=item['metadata'],video_path=str(video),video_sha256=run['video_file_sha256'],run_sha256=case['run_sha256'],
                initial_noise_sha256=noise_sha,observer_sha256=sha(folder/'stage4-execution.json'),identity=run['identity'],source_sha256=run['source_sha256']))
    return result


def score_dimension(dimension, info, metadata, assets, device):
    import torch
    from .vbench import load_musiq_namespace
    from .wan_semantic_score import inference_guard,umt_factory_compat
    from vbench import utils
    if dimension == 'imaging_quality': load_musiq_namespace(assets/'source/PyIQA')
    module = importlib.import_module('vbench.'+dimension)
    if dimension == 'subject_consistency':
        model = torch.hub.load(str(assets/'vbench-cache/dino_model/facebookresearch_dino_main'),'dino_vitb16',source='local',pretrained=False)
        model.load_state_dict(torch.load(assets/'vbench-cache/dino_model/dino_vitbase16_pretrain.pth',map_location='cpu',weights_only=True),strict=True)
        model.eval().to(device)
        try:
            with torch.no_grad(): return module.subject_consistency(model,[row['video_list'][0] for row in metadata if dimension in row['dimension']],device,False)
        finally: del model
    paths = utils.init_submodules([dimension],local=True,read_frame=False)[dimension]
    guard = nullcontext()
    if dimension in {'object_class','multiple_objects','color','spatial_relationship'}:
        from vbench.third_party.grit_model import DenseCaptioning
        guard = inference_guard(DenseCaptioning)
    elif dimension == 'human_action':
        from vbench.third_party.umt.models.modeling_finetune import vit_large_patch16_224
        guard = umt_factory_compat(module,vit_large_patch16_224)
    with guard,torch.no_grad(): return getattr(module,'compute_'+dimension)(str(info),device,paths)


def fingerprint(value):
    return hashlib.sha256(json.dumps(json_safe(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def prepare_cache(directory, contract, resume):
    """Bind a cache to exact inputs/sources before any existing result is touched."""
    directory=pathlib.Path(directory); path=directory/'cache-contract.json'
    if directory.exists():
        if not resume: raise ValueError('Existing score output requires explicit --resume')
        if path.exists():
            if read(path) != contract: raise ValueError('Cached scoring contract/source/mode differs')
        elif any(directory.iterdir()): raise ValueError('Existing nonempty score output has no trusted cache contract')
    else: directory.mkdir(parents=True)
    if not path.exists(): write(path,contract)
    return fingerprint(contract)


def preserve_history(directory):
    """Snapshot JSON and unfinished atomic writes; never remove failure evidence."""
    directory=pathlib.Path(directory); history=directory/'history'/str(time.time_ns()); history.mkdir(parents=True)
    for path in directory.iterdir():
        if path.is_file(): shutil.copyfile(path,history/path.name)
    return history


def cached_dimension(out, dimension, binding, eligible, frames, prior=None):
    receipt_path=out/(dimension+'-state.json')
    if not receipt_path.exists():
        if prior is not None: raise ValueError('Labelled cached dimension is missing its source-bound receipt')
        return None
    receipt=read(receipt_path)
    if receipt.get('binding') != binding: raise ValueError('Cached dimension came from another mode/input/source')
    if prior is not None and (prior.get('sha256') != sha(receipt_path) or prior.get('result') != receipt.get('result')):
        raise ValueError('Cached dimension result/receipt SHA differs from mode label')
    result=receipt['result']; raw=out/(dimension+'-official-raw.json')
    if receipt.get('raw_sha256') is not None:
        if not raw.exists() or sha(raw) != receipt['raw_sha256']: raise ValueError('Cached official raw result SHA differs')
    elif raw.exists(): raise ValueError('Unbound official raw result present')
    if result.get('status')=='complete':
        if receipt.get('raw_sha256') is None: raise ValueError('Complete cached dimension lacks official raw output')
        original=read(raw); reconstructed=bind_details(dimension,original['score'],original['details'],eligible,frames)
        if reconstructed != {k:v for k,v in result.items() if k!='elapsed_seconds'}: raise ValueError('Cached result does not match revalidated official raw output')
    elif result.get('status') not in {'failed','undefined'} or result.get('score') is not None:
        raise ValueError('Invalid cached dimension status/score')
    return receipt


def score_mode(mode, rows, assets, out, frames, aggregate, *, resume=False, cache_contract=None, require_complete=False):
    import torch
    import numpy as np
    from .wan_semantic_score import canonical_video_name
    out=pathlib.Path(out); aliases = out/'video-inputs'
    if require_complete and not (out/'mode-scores.json').exists(): raise ValueError('Complete summary is missing a completed mode')
    metadata = []; bound_inputs = []
    for row in rows:
        alias = aliases/canonical_video_name(row['prompt'],row['seed'])
        metadata.append(dict(copy.deepcopy(row['metadata']),video_list=[str(alias)]))
        bound_inputs.append(dict(row,video_path=str(alias)))
    contract=dict(parent_contract_sha256=cache_contract,mode=mode,frames=frames,metadata=metadata,inputs=rows,scorer_seed=0,dimensions=list(ALL_DIMENSIONS))
    contract_sha=prepare_cache(out,contract,resume)
    info=out/'dimension-inputs.json'; path=out/'mode-scores.json'; previous=read(path) if path.exists() else None
    complete=previous is not None and previous.get('status') in {'complete','complete_with_unaccepted_dimensions'}
    if require_complete and not complete: raise ValueError('Complete summary points to an incomplete mode; refusing to rerun over completion evidence')
    if info.exists():
        if read(info)!=metadata: raise ValueError('Cached official metadata changed')
    elif complete: raise ValueError('Complete mode cache is missing metadata')
    else: write(info,metadata)
    aliases.mkdir(exist_ok=True)
    for row,item in zip(rows,bound_inputs):
        alias=pathlib.Path(item['video_path'])
        if alias.exists():
            if sha(alias)!=row['video_sha256']: raise ValueError('Cached canonical score alias changed bytes')
        elif complete: raise ValueError('Complete mode cache is missing a video')
        else:
            shutil.copyfile(row['video_path'],alias)
            if sha(alias)!=row['video_sha256']: raise ValueError('Canonical score alias changed bytes')
    if previous is not None and (previous.get('mode')!=mode or previous.get('cache_contract_sha256')!=contract_sha or previous.get('metadata_sha256')!=sha(info) or previous.get('inputs')!=bound_inputs):
        raise ValueError('Mode cache source/input/metadata identity differs')
    reusable={}
    for dimension in ALL_DIMENSIONS:
        eligible=[row for row in bound_inputs if dimension in row['metadata']['dimension']]
        binding=dict(contract_sha256=contract_sha,mode=mode,dimension=dimension,metadata_sha256=sha(info))
        prior=None
        if previous is not None and dimension in previous.get('dimension_scores',{}):
            prior=dict(sha256=previous.get('dimension_receipts',{}).get(dimension),result=previous['dimension_scores'][dimension])
        receipt=cached_dimension(out,dimension,binding,eligible,frames,prior)
        if receipt is not None: reusable[dimension]=receipt
    if complete:
        if set(previous.get('dimension_scores',{}))!=set(ALL_DIMENSIONS) or set(reusable)!=set(ALL_DIMENSIONS) or previous.get('all_16_dimensions_attempted') is not True:
            raise ValueError('Complete mode label lacks all sixteen verified dimensions')
        if normalize_scores(previous['dimension_scores'],aggregate)!=previous.get('normalized'): raise ValueError('Cached official aggregate changed')
        return previous
    if previous is not None: preserve_history(out)
    report = dict(status='running',mode=mode,frames=frames,steps=50,videos=22,dimension_scores={},
        cache_contract_sha256=contract_sha,dimension_receipts={},metadata_sha256=sha(info),inputs=bound_inputs,scorer_seed_per_dimension=0,
        primary_point='Official aggregate over all eligible heldout videos, separately for this mode',
        confidence_interval=None,confidence_interval_scope='Only fixed11-prompt within-prompt seed resampling; never946-prompt population',
        official_single_process_world_size=1,standard_suite_reproduced=False,paper_total_comparison_valid=False)
    write(path,report); started = time.perf_counter()
    try:
      for dimension in ALL_DIMENSIONS:
        cached=reusable.get(dimension)
        if cached is not None and cached['result'].get('status')=='complete':
            report['dimension_scores'][dimension]=cached['result'];report['dimension_receipts'][dimension]=sha(out/(dimension+'-state.json'))
            report['normalized']=normalize_scores(report['dimension_scores'],aggregate);write(path,report)
            print('REUSE SCORE',mode,dimension,flush=True);continue
        print('START SCORE',mode,dimension,flush=True)
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        eligible = [row for row in bound_inputs if dimension in row['metadata']['dimension']]
        dimension_start = time.perf_counter()
        raw_path=out/(dimension+'-official-raw.json')
        # Failed attempts were snapshotted above, including their raw output.
        if raw_path.exists():
            if previous is None: preserve_history(out)
            raw_path.unlink()
        try:
            value,details = score_dimension(dimension,info,metadata,assets,torch.device('cuda'))
            # Preserve the original details before any validation, even on failure.
            write(out/(dimension+'-official-raw.json'),json_safe(dict(score=value,details=details)))
            result = bind_details(dimension,value,details,eligible,frames)
        except Exception as error:
            result = dict(status='undefined' if isinstance(error,ZeroDivisionError) else 'failed',score=None,eligible_videos=len(eligible),
                error_type=type(error).__name__,error=str(error),bootstrap_available=False,bootstrap_reason='Official dimension evaluation did not produce an accepted finite point')
        result['elapsed_seconds'] = time.perf_counter()-dimension_start
        receipt=dict(binding=dict(contract_sha256=contract_sha,mode=mode,dimension=dimension,metadata_sha256=sha(info)),
            result=result,raw_sha256=sha(raw_path) if raw_path.exists() else None)
        receipt_path=out/(dimension+'-state.json');write(receipt_path,receipt)
        report['dimension_scores'][dimension] = result
        report['dimension_receipts'][dimension]=sha(receipt_path)
        report['normalized'] = normalize_scores(report['dimension_scores'],aggregate)
        report['elapsed_seconds'] = time.perf_counter()-started
        write(path,report); print('DONE SCORE',mode,dimension,result['status'],result['score'],flush=True)
        gc.collect(); torch.cuda.empty_cache()
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));write(path,report);raise
    report['status'] = 'complete' if report['normalized']['all_dimensions_finite'] else 'complete_with_unaccepted_dimensions'
    report['all_16_dimensions_attempted'] = set(report['dimension_scores']) == set(ALL_DIMENSIONS)
    report['quality_accepted'] = False  # Requires the separate paired heldout criterion, never mere scorer completion.
    report['bootstrap_all_dimensions_available'] = all(r.get('bootstrap_available',False) for r in report['dimension_scores'].values())
    write(path,report); return report


def json_safe(value):
    """Keep official raw outputs JSON-safe without promoting a NaN to a score."""
    if isinstance(value,dict): return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_safe(v) for v in value]
    if hasattr(value,'item'): value = value.item()
    if isinstance(value,float) and not math.isfinite(value): return dict(nonfinite=repr(value))
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True); parser.add_argument('--output',required=True); parser.add_argument('--assets',default='/quality')
    parser.add_argument('--resume',action='store_true',help='Reuse only source-bound, revalidated complete results; preserve partial failure history')
    args = parser.parse_args()
    root,assets,out = pathlib.Path(args.root).resolve(),pathlib.Path(args.assets).resolve(),pathlib.Path(args.output).resolve()
    if out.exists() and not args.resume: raise ValueError('Existing score output requires explicit --resume')
    report = dict(status='running',scope='Independent short-video Stage4 development heldout:11 fixed official prompts x2 fresh seeds x4 modes',
        G=None,formal_paper_policy=False,standard_suite_reproduced=False,paper_total_comparison_valid=False,quality_accepted=False,
        confidence_interval_scope='Only fixed11-prompt within-prompt seed resampling, same draws across all modes/dimensions. Does not represent946prompt population.',
        sample_duration_limitation='Short9/17-frame inputs; official16-frame sampling can repeat input indices. Input videos are never padded or changed.',
        paired_backend_limitation='Dense/PlaneClear use selected backend; Residual/combined use FP32 masked reference. Quality differences include backend arithmetic.',
        modes={},scorer_source_sha256={name:sha(pathlib.Path(__file__).with_name(name)) for name in ('wan_stage4_score.py','wan_semantic_score.py','wan_semantic.py','vbench.py')})
    target = out/'summary.json'; started = time.perf_counter(); write_allowed=False
    try:
        manifest = read(root/'manifest.json'); official = read(assets/'source/VBench/vbench/VBench_full_info.json')
        validate_manifest(manifest,official); index = read(root/'generation-index.json'); pools = validate_index(index,manifest)
        if index.get('heldout_never_used_for_selection') is not True: raise ValueError('Missing explicit heldout isolation receipt')
        selection = read(root/'selection.json')
        policy = validate_selection(selection,index,root)
        report.update(frames=manifest['frames'],manifest_sha256=sha(root/'manifest.json'),generation_index_sha256=sha(root/'generation-index.json'),
            selection_sha256=sha(root/'selection.json'),selection=selection,asset_validation=verify_assets(assets))
        rows = validate_artifacts(root,manifest,pools,policy); report['artifact_validation'] = rows
        import torch
        torch.set_num_threads(4); report['torch'] = torch.__version__
        sys.path.insert(0,str(assets/'source/VBench'))
        from vbench import utils
        from vbench.distributed import get_world_size
        if get_world_size() != 1: raise ValueError('Sufficient-statistics rules require official single-process world_size1')
        utils.CACHE_DIR = str(assets/'vbench-cache'); os.environ['VBENCH_CACHE_DIR'] = str(assets/'vbench-cache')
        aggregate = load_official_aggregate(assets/'source/VBench')
        contract=dict(version='wan-stage4-score-cache-v1',frames=manifest['frames'],manifest_sha256=report['manifest_sha256'],
            generation_index_sha256=report['generation_index_sha256'],selection_sha256=report['selection_sha256'],
            asset_validation_sha256=fingerprint(report['asset_validation']),artifact_validation_sha256=fingerprint(rows),
            scorer_source_sha256=report['scorer_source_sha256'],torch=report['torch'],modes=list(MODES),dimensions=list(ALL_DIMENSIONS))
        contract_sha=prepare_cache(out,contract,args.resume); report['cache_contract_sha256']=contract_sha
        previous=read(target) if target.exists() else None
        if previous is not None and previous.get('cache_contract_sha256')!=contract_sha: raise ValueError('Summary contract/source identity differs')
        if previous is not None and previous.get('status') in {'complete','complete_with_unaccepted_dimensions'}:
            if set(previous.get('modes',{}))!=set(MODES) or previous.get('all_modes_all_16_dimensions_attempted') is not True:
                raise ValueError('Complete summary label lacks four completed mode receipts')
            for mode in MODES:
                result=score_mode(mode,rows[mode],assets,out/mode,manifest['frames'],aggregate,resume=True,cache_contract=contract_sha,require_complete=True)
                cached=previous['modes'][mode]
                if result.get('status') not in {'complete','complete_with_unaccepted_dimensions'} or cached.get('mode_scores_sha256')!=sha(out/mode/'mode-scores.json') or cached.get('normalized')!=result['normalized']:
                    raise ValueError('Complete summary mode result/SHA differs')
            print('REUSE COMPLETE SCORE SUMMARY',str(target),flush=True);return
        if previous is not None: preserve_history(out)
        write_allowed=True;write(target,report)
        for mode in MODES:
            result = score_mode(mode,rows[mode],assets,out/mode,manifest['frames'],aggregate,resume=args.resume,cache_contract=contract_sha)
            report['modes'][mode] = dict(status=result['status'],normalized=result['normalized'],
                all_16_dimensions_attempted=result['all_16_dimensions_attempted'],bootstrap_all_dimensions_available=result['bootstrap_all_dimensions_available'],
                mode_scores_path=str(out/mode/'mode-scores.json'),mode_scores_sha256=sha(out/mode/'mode-scores.json'))
            write(target,report)
        all_finite = all(row['normalized']['all_dimensions_finite'] for row in report['modes'].values())
        report.update(status='complete' if all_finite else 'complete_with_unaccepted_dimensions',all_modes_all_16_dimensions_attempted=True,
            all_modes_all_dimensions_finite=all_finite,elapsed_seconds=time.perf_counter()-started,
            quality_acceptance_reason='Separate paired heldout loss/uncertainty assessment required' if all_finite else 'At least one official dimension is NA/error; total remains null and quality is unaccepted')
        write(target,report)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.perf_counter()-started)
        if write_allowed: write(target,report)
        elif out.exists():
            rejected=out/'resume-rejections';rejected.mkdir(exist_ok=True)
            write(rejected/(str(time.time_ns())+'.json'),dict(status='rejected',error_type=type(error).__name__,error=str(error)))
        raise


if __name__ == '__main__': main()
