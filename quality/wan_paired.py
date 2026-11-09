"""Independent Wan50 dense/Residual scoring sidecar; one-prompt diagnostics only."""
import copy,hashlib,json,math,pathlib
from .vbench import QUALITY_DIMS

def sha(path):
    result=hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()

def tensor_sha(tensor):
    import torch
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def public_schedule(record):
    return dict(record,config={k:v for k,v in record['config'].items() if not k.startswith('_')})

def dependency_ready(control_path,run_path):
    control_path,run_path=pathlib.Path(control_path),pathlib.Path(run_path)
    if not control_path.exists():return False
    control=json.loads(control_path.read_text(encoding='utf-8'))
    if control.get('status')=='running':return False
    if control.get('status')!='complete' or control.get('exit_code')!=0 or not control.get('own_container_absent') or control.get('run_status')!='complete':raise ValueError('Owned Residual generation dependency failed: '+str(control.get('status')))
    observer=control.get('observer') or {}
    if observer.get('status')!='complete' or observer.get('completed_model_calls')!=100 or observer.get('completed_transformer_blocks')!=4000:raise ValueError('Owned Residual controller has incomplete observed execution')
    if not run_path.exists():raise ValueError('Complete controller is missing generation receipt')
    run=json.loads(run_path.read_text(encoding='utf-8'))
    if run.get('status')!='complete' or run.get('forward_calls')!=100 or run.get('decoded_video_frames')!=81 or not run.get('finite_latent') or not run.get('finite_video') or run['identity']['frames']!=81 or run['identity']['steps']!=50:raise ValueError('Owned Residual generation receipt is incomplete')
    return True

def validate_pair(dense,residual,dense_noise,residual_noise,control):
    import torch
    for run,mode in ((dense,'dense'),(residual,'residual')):
        identity=run['identity']
        if run['status']!='complete' or identity['mode']!=mode or identity['adapter']!='wan-self-ffn-reference-v1':raise ValueError('Expected complete dense/Residual runs')
        if identity['resolution']!=[832,480] or identity['frames']!=81 or identity['steps']!=50 or run['decoded_video_frames']!=81 or run['forward_calls']!=100:raise ValueError('Requires complete full-size fifty-step generation')
        if not run['finite_latent'] or not run['finite_video'] or identity['cfg_scale']!=5 or identity['cfg_channels']!='all':raise ValueError('Invalid finite/CFG generation contract')
        effective=public_schedule(run['schedule'])
        if hashlib.sha256(json.dumps(effective,sort_keys=True,allow_nan=False).encode()).hexdigest()!=identity['noise_schedule_sha256']:raise ValueError('Recorded schedule identity differs from actual schedule')
    excluded={'mode','attention_execution_backend','attention_query_chunk'}
    if {k:v for k,v in dense['identity'].items() if k not in excluded}!={k:v for k,v in residual['identity'].items() if k not in excluded}:raise ValueError('Deployment identity mismatch')
    for field in ('prompt','seed','initial_noise_sha256','source_sha256'):
        if dense[field]!=residual[field]:raise ValueError('Unpaired '+field)
    if public_schedule(dense['schedule'])!=public_schedule(residual['schedule']):raise ValueError('Public schedule/timesteps/sigmas differ')
    for noise,run in ((dense_noise,dense),(residual_noise,residual)):
        if tuple(noise.shape)!=(16,21,60,104) or noise.dtype!=torch.float32 or not bool(torch.isfinite(noise).all()):raise ValueError('Unexpected actual initial noise')
        if tensor_sha(noise)!=run['initial_noise_sha256']:raise ValueError('Actual initial noise differs from recorded hash')
    if not torch.equal(dense_noise,residual_noise):raise ValueError('Actual initial noise tensors differ')
    if control.get('status')!='complete' or control.get('exit_code')!=0 or not control.get('own_container_absent') or control.get('run_status')!='complete':raise ValueError('Residual controller did not complete')
    observer=control.get('observer') or {}
    if observer.get('status')!='complete' or observer.get('completed_model_calls')!=100 or observer.get('completed_transformer_blocks')!=4000:raise ValueError('Incomplete observed fifty-step trajectory')

def mode_metadata(video,run,official):
    rows=[row for row in official if row['prompt_en']==run['prompt']]
    if len(rows)!=1 or run['prompt']!='a cat and a dog' or rows[0]['dimension']!=['multiple_objects'] or rows[0].get('auxiliary_info')!={'multiple_objects':{'object':'cat and dog'}}:raise ValueError('Expected exact official multiple_objects metadata')
    semantic=[dict(copy.deepcopy(rows[0]),video_list=[str(video)])]
    generic=[{'prompt_en':run['prompt'],'dimension':list(QUALITY_DIMS),'video_list':[str(video)]}]
    return generic,semantic

def dimension_change(dense,residual):
    for value in (dense,residual):
        if value is not None and not math.isfinite(value):raise ValueError('Nonfinite score')
    delta=None if dense is None or residual is None else residual-dense
    return {'dense':dense,'residual':residual,'delta_residual_minus_dense':delta,
            'absolute_delta':None if delta is None else abs(delta),
            'relative_loss':None if dense is None or residual is None or dense<=0 else (dense-residual)/dense,
            'relative_loss_definition':'(dense-residual)/dense; negative means increase in higher-is-better raw score; null for absent/nonpositive baseline'}

DIMENSIONS=QUALITY_DIMS+('multiple_objects',)

def combine_mode_scores(dense,residual):
    result={}
    for report,mode in ((dense,'dense'),(residual,'residual')):
        if report.get('status')!='complete' or report.get('mode')!=mode or set(report.get('dimension_scores',{}))!=set(DIMENSIONS):raise ValueError('Incomplete or incorrectly pooled mode scoring')
        for row in report['dimension_scores'].values():
            if row.get('eligible_prompts')!=1 or len(row.get('per_video',[]))!=1 or not math.isfinite(row['score']):raise ValueError('Each mode/dimension must contain exactly one finite video result')
    for name in DIMENSIONS:
        result[name]=dict(dimension_change(dense['dimension_scores'][name]['score'],residual['dimension_scores'][name]['score']),
                          evaluation_scope='official_semantic_membership' if name=='multiple_objects' else 'custom_input_quality_diagnostic')
    return result

def write_json(path,data):
    temporary=path.with_suffix('.incoming');temporary.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');temporary.replace(path)

def verify_video(folder,run):
    import cv2
    path=folder/'sample.mp4'
    if sha(path)!=run['video_file_sha256']:raise ValueError('Video bytes differ from generation receipt')
    cap=cv2.VideoCapture(str(path));count=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            if frame.shape!=(480,832,3):raise ValueError('Decoded video shape differs')
            count+=1
    finally:cap.release()
    if count!=81:raise ValueError('Expected eighty-one actual decoded frames')
    return count

def score_mode(mode,folder,run,official,assets,out):
    import importlib,random,shutil,time
    import numpy as np
    import torch
    from .vbench import load_musiq_namespace
    from .wan_semantic_score import inference_guard,checked_dimension_result,canonical_video_name
    from vbench import utils
    utils.CACHE_DIR=str(assets/'vbench-cache')
    out.mkdir(parents=True,exist_ok=False)
    video=out/canonical_video_name(run['prompt'],run['seed']);shutil.copyfile(folder/'sample.mp4',video)
    if sha(video)!=run['video_file_sha256']:raise ValueError('Scoring alias bytes differ')
    generic,semantic=mode_metadata(video,run,official)
    infos={'custom_input_quality_diagnostic':out/'quality-inputs.json','official_semantic_membership':out/'semantic-inputs.json'}
    write_json(infos['custom_input_quality_diagnostic'],generic);write_json(infos['official_semantic_membership'],semantic)
    report={'status':'running','mode':mode,'videos':1,'dimension_scores':{},'total_score':None,
            'metadata_scope':'Seven generic quality dimensions explicitly custom-input; multiple_objects retains official original membership/aux',
            'metadata_sha256':{key:sha(value) for key,value in infos.items()},'video_sha256':run['video_file_sha256'],
            'run_sha256':sha(folder/'run.json'),'official_scorer_sha256':{},'scorer_seed_per_dimension':0}
    path=out/'mode-scores.json';write_json(path,report);start=time.perf_counter();device=torch.device('cuda')
    try:
        for dimension in DIMENSIONS:
            random.seed(0);np.random.seed(0);torch.manual_seed(0)
            print('START',mode,dimension,flush=True)
            if dimension=='imaging_quality':load_musiq_namespace(assets/'source/PyIQA')
            module=importlib.import_module('vbench.'+dimension)
            report['official_scorer_sha256'][dimension]=sha(module.__file__)
            scope='official_semantic_membership' if dimension=='multiple_objects' else 'custom_input_quality_diagnostic'
            model=None
            if dimension=='subject_consistency':
                repository=assets/'vbench-cache/dino_model/facebookresearch_dino_main'
                model=torch.hub.load(str(repository),'dino_vitb16',source='local',pretrained=False)
                model.load_state_dict(torch.load(assets/'vbench-cache/dino_model/dino_vitbase16_pretrain.pth',map_location='cpu',weights_only=True),strict=True)
                model.eval().to(device)
                with torch.no_grad():score,details=module.subject_consistency(model,[str(video)],device,False)
            else:
                paths=utils.init_submodules([dimension],local=True,read_frame=False)[dimension]
                if dimension=='multiple_objects':
                    from vbench.third_party.grit_model import DenseCaptioning
                    with inference_guard(DenseCaptioning),torch.no_grad():score,details=module.compute_multiple_objects(str(infos[scope]),device,paths)
                else:
                    with torch.no_grad():score,details=getattr(module,'compute_'+dimension)(str(infos[scope]),device,paths)
            row=checked_dimension_result(score,details,1)
            if len(row['per_video'])!=1:raise ValueError('Mode scorer returned a pooled or empty result')
            row['evaluation_scope']=scope;report['dimension_scores'][dimension]=row
            report['elapsed_seconds']=time.perf_counter()-start;write_json(path,report)
            print('DONE',mode,dimension,float(score),flush=True)
            del model;torch.cuda.empty_cache()
        report['status']='complete';write_json(path,report)
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));write_json(path,report);raise
    return report

def main():
    import argparse,sys,time,shutil
    import torch
    p=argparse.ArgumentParser();p.add_argument('--dense',required=True);p.add_argument('--residual',required=True)
    p.add_argument('--controller',required=True);p.add_argument('--output',required=True);p.add_argument('--assets',default='/quality');a=p.parse_args()
    folders=[pathlib.Path(a.dense).resolve(),pathlib.Path(a.residual).resolve()];assets=pathlib.Path(a.assets).resolve();out=pathlib.Path(a.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    summary=out/'paired-vbench.json'
    if summary.exists() or (out/'dense').exists() or (out/'residual').exists():raise ValueError('Refusing to overwrite an existing paired score run')
    torch.set_num_threads(4);start=time.perf_counter()
    report={'status':'running','scope':'One fixed prompt/seed paired Wan81/50 development evaluation',
            'total_score':None,'standard_suite_reproduced':False,'paper_total_comparison_valid':False,
            'primary_quality_loss_measured':False,'confidence_interval':None,'confidence_interval_reason':'One paired prompt/seed has no representative bootstrap CI',
            'backend_note':'dense uses official FlashAttention; Residual uses FP32 masked quality reference; score differences include backend arithmetic',
            'source_sha256':{name:sha(pathlib.Path(__file__).with_name(name)) for name in ('wan_paired.py','vbench.py','wan_semantic_score.py','wan_semantic.py')},
            'torch':torch.__version__,'scoring_image':'harp-ae-gpu:quality-v0.4'}
    snapshot=out/'source-snapshot';snapshot.mkdir(exist_ok=False)
    for name,expected in report['source_sha256'].items():
        shutil.copyfile(pathlib.Path(__file__).with_name(name),snapshot/name)
        if sha(snapshot/name)!=expected:raise ValueError('Scoring source changed while snapshotting')
    write_json(summary,report)
    try:
        runs=[json.loads((folder/'run.json').read_text(encoding='utf-8')) for folder in folders]
        control=json.loads(pathlib.Path(a.controller).read_text(encoding='utf-8'))
        noises=[torch.load(folder/'initial-noise.pt',map_location='cpu',weights_only=True) for folder in folders]
        validate_pair(*runs,*noises,control)
        decoded=[verify_video(folder,run) for folder,run in zip(folders,runs)]
        inventory=json.loads((assets/'wan-semantic-deployed-inventory.json').read_text(encoding='utf-8'))
        for row in inventory['checkpoint_files']:
            file=assets/row['path']
            if file.stat().st_size!=row['bytes'] or sha(file)!=row['sha256']:raise ValueError('Deployed scoring asset differs from verified inventory: '+row['path'])
        sys.path.insert(0,str(assets/'source/VBench'))
        fullpath=assets/'source/VBench/vbench/VBench_full_info.json';official=json.loads(fullpath.read_text())
        report.update(pair_validation={'same_whole_source':True,'actual_initial_noise_equal':True,'public_schedule_and_actual_timesteps_sigmas_equal':True,
            'decoded_frames':decoded,'controller_sha256':sha(a.controller),'run_sha256':{mode:sha(folder/'run.json') for mode,folder in zip(('dense','residual'),folders)},
            'initial_noise_sha256':runs[0]['initial_noise_sha256'],'identities':{mode:run['identity'] for mode,run in zip(('dense','residual'),runs)}},
            original_official_metadata_sha256=sha(fullpath),asset_inventory_sha256=sha(assets/'wan-semantic-deployed-inventory.json'),
            asset_inventory=inventory,vbench_source=json.loads((assets/'source-manifest.json').read_text())['repositories']['VBench'])
        write_json(summary,report)
        scores=[score_mode(mode,folder,run,official,assets,out/mode) for mode,folder,run in zip(('dense','residual'),folders,runs)]
        report['dimensions']=combine_mode_scores(*scores)
        report['mode_score_receipts']={mode:{'path':str(out/mode/'mode-scores.json'),'sha256':sha(out/mode/'mode-scores.json')} for mode in ('dense','residual')}
        report.update(status='complete',elapsed_seconds=time.perf_counter()-start,paired_development_scores_measured=True)
        write_json(summary,report);print(json.dumps(report['dimensions'],indent=2),flush=True)
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));write_json(summary,report);raise

if __name__=='__main__':main()
