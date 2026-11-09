"""All sixteen official VBench dimensions on frozen dimension-specific inputs.

Reports diagnostic dimension scores only, never a standard-suite total.
"""
import argparse,hashlib,importlib,json,pathlib,sys,time,random,shutil,math
from contextlib import contextmanager,nullcontext
from .wan_semantic import ALL_DIMENSIONS,validate_selection
from .vbench import load_musiq_namespace

def checked_dimension_result(score,details,eligible):
    value=float(score)
    if not math.isfinite(value) or eligible<=0 or not isinstance(details,list) or not details:raise ValueError('Nonfinite or empty dimension evaluation')
    return {'score':value,'per_video':details,'eligible_prompts':eligible}

@contextmanager
def umt_factory_compat(module,entrypoint):
    """Use official UMT factory directly; new timm adds unsupported cache_dir."""
    original=module.create_model
    def create(name,pretrained=False,**kwargs):
        if name!='vit_large_patch16_224' or pretrained:raise ValueError('Unexpected UMT model request')
        return entrypoint(pretrained=pretrained,**{k:v for k,v in kwargs.items() if v is not None})
    module.create_model=create
    try:yield
    finally:module.create_model=original

@contextmanager
def inference_guard(model_class):
    """Observe swallowed GRiT per-frame exceptions without changing predictions."""
    originals={name:getattr(model_class,name) for name in ('run_caption_tensor','run_det_tensor')}
    errors=[]
    for name,original in originals.items():
        def forward(this,*args,original=original,**kwargs):
            try:return original(this,*args,**kwargs)
            except Exception as error:
                errors.append(type(error).__name__+': '+str(error));raise
        setattr(model_class,name,forward)
    try:
        yield errors
    finally:
        for name,original in originals.items():setattr(model_class,name,original)
        if errors:raise RuntimeError('GRiT inference failed: '+errors[0])

def canonical_video_name(prompt,seed):
    if '/' in prompt or '\\' in prompt:raise ValueError('Prompt cannot be used as a portable video basename')
    return f'{prompt}-{seed}.mp4'

def validate_dense_protocol(run,reference):
    identity=run['identity']
    if identity.get('mode')!='official' or identity.get('adapter')!='official-unmodified' or identity.get('cfg_scale')!=5 or identity.get('cfg_channels')!='all':raise ValueError('Pilot requires official dense CFG5 generation')
    for field in ('checkpoint_sha256','negative_prompt'):
        if identity.get(field)!=reference['identity'].get(field):raise ValueError('Mixed deployment: '+field)
    def effective(record):
        return dict(record,config={k:v for k,v in record['config'].items() if not k.startswith('_')})
    if effective(run['schedule'])!=effective(reference['schedule']):raise ValueError('Mixed sampler/noise schedule')

def metadata_from_completed(completed,alias_directory=None):
    result=[];reference=None
    for row in completed:
        folder=pathlib.Path(row['run_directory']);run=json.loads((folder/'run.json').read_text())
        if reference is None:reference=run
        validate_dense_protocol(run,reference)
        if run['status']!='complete' or run['identity']['resolution']!=[832,480] or run['identity']['frames']!=81 or run['identity']['steps']!=50 or run['decoded_video_frames']!=81:raise ValueError('Full-size generation incomplete')
        if run['prompt']!=row['metadata']['prompt_en'] or run['seed']!=row['seed']:raise ValueError('Generation/metadata identity mismatch')
        path=folder/'sample.mp4'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=run['video_file_sha256']:raise ValueError('Video content changed')
        if alias_directory is not None:
            alias_directory.mkdir(parents=True,exist_ok=True)
            alias=alias_directory/canonical_video_name(run['prompt'],run['seed'])
            if not alias.exists():shutil.copyfile(path,alias)
            if hashlib.sha256(alias.read_bytes()).hexdigest()!=run['video_file_sha256']:raise ValueError('Canonical video copy differs')
            path=alias
        result.append(dict(row['metadata'],video_list=[str(path)]))
    if set().union(*(set(row['dimension']) for row in result))!=set(ALL_DIMENSIONS):raise ValueError('Not all official dimensions covered')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--completed',required=True);p.add_argument('--output',required=True);p.add_argument('--assets',default='/quality');p.add_argument('--dimensions',nargs='+',choices=ALL_DIMENSIONS,default=list(ALL_DIMENSIONS));a=p.parse_args()
    import torch
    import numpy as np
    assets=pathlib.Path(a.assets).resolve();out=pathlib.Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(assets/'source/VBench'))
    from vbench import utils
    utils.CACHE_DIR=str(assets/'vbench-cache')
    completed=json.loads(pathlib.Path(a.completed).read_text())
    if completed['status']!='complete':raise ValueError('Generation plan incomplete')
    full=json.loads((assets/'source/VBench/vbench/VBench_full_info.json').read_text())
    validate_selection(full,completed['completed'])
    metadata=metadata_from_completed(completed['completed'],out/'video-inputs');info=out/'dimension-inputs.json';info.write_text(json.dumps(metadata,indent=2))
    report={'status':'running','scope':'11 official prompts; one seed per prompt; original dimension labels and auxiliary_info retained',
            'standard_suite_reproduced':False,'total_score':None,'paper_total_comparison_valid':False,
            'dimension_scores':{},'videos':len(metadata),'metadata_sha256':hashlib.sha256(info.read_bytes()).hexdigest(),
            'scorer_seed_per_dimension':0,'torch':torch.__version__,
            'video_naming':'Byte-identical copies named official prompt-seed.mp4; human_action extracts target label from basename',
            'compatibility':['Official UMT entrypoint used directly; omit None options as timm factory does, bypass unsupported new cache_dir metadata',
                             'MUSIQ official modules loaded through minimal package namespace',
                             'GRiT swallowed per-frame inference exceptions observed and cause score rejection'],
            'scorer_source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (pathlib.Path(__file__),pathlib.Path(__file__).with_name('vbench.py'),pathlib.Path(__file__).with_name('wan_semantic.py'))},
            'asset_receipts':{name:{'sha256':hashlib.sha256((assets/name).read_bytes()).hexdigest(),'content':json.loads((assets/name).read_text())} for name in ('vbench-assets.json','wan-semantic-assets.json','wan-semantic-bert.json','wan-semantic-bpe.json','wan-semantic-official-lfs.json')},
            'source':json.loads((assets/'source-manifest.json').read_text())['repositories']['VBench']}
    def save():
        temp=out/'summary.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False));temp.replace(out/'vbench-semantic.json')
    save();torch.set_num_threads(4);device=torch.device('cuda');start=time.perf_counter()
    for dim in a.dimensions:
        random.seed(0);np.random.seed(0);torch.manual_seed(0)
        print('START SCORE',dim,flush=True)
        if dim=='imaging_quality':load_musiq_namespace(assets/'source/PyIQA')
        module=importlib.import_module('vbench.'+dim)
        paths=utils.init_submodules([dim],local=True,read_frame=False)[dim]
        if dim=='subject_consistency':
            repository=assets/'vbench-cache/dino_model/facebookresearch_dino_main'
            model=torch.hub.load(str(repository),'dino_vitb16',source='local',pretrained=False)
            model.load_state_dict(torch.load(assets/'vbench-cache/dino_model/dino_vitbase16_pretrain.pth',map_location='cpu',weights_only=True),strict=True)
            model.eval().to(device)
            with torch.no_grad():score,details=module.subject_consistency(model,[r['video_list'][0] for r in metadata if dim in r['dimension']],device,False)
            del model
        else:
            guard=nullcontext()
            if dim in {'object_class','multiple_objects','color','spatial_relationship'}:
                from vbench.third_party.grit_model import DenseCaptioning
                guard=inference_guard(DenseCaptioning)
            elif dim=='human_action':
                from vbench.third_party.umt.models.modeling_finetune import vit_large_patch16_224
                guard=umt_factory_compat(module,vit_large_patch16_224)
            with guard,torch.no_grad():score,details=getattr(module,'compute_'+dim)(str(info),device,paths)
        report['dimension_scores'][dim]=checked_dimension_result(score,details,sum(dim in row['dimension'] for row in metadata))
        report['elapsed_seconds']=time.perf_counter()-start;save();print('DONE SCORE',dim,float(score),flush=True);torch.cuda.empty_cache()
    report['status']='complete';save()

if __name__=='__main__':main()
