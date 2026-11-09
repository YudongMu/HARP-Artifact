"""Official VBench-v1 dimension scorers on a small full-size dense pilot.

Diagnostic dimensions on custom inputs are never labeled a full VBench total.
"""
import argparse,hashlib,importlib,json,pathlib,sys,types,time,zipfile

QUALITY_DIMS=('subject_consistency','background_consistency','temporal_flickering',
              'motion_smoothness','dynamic_degree','aesthetic_quality','imaging_quality')


def build_metadata(runs,dimensions):
    result=[]
    for folder,run in runs:
        identity=run['identity']
        if run['status']!='complete' or identity['resolution']!=[832,480] or identity['frames']!=81 or identity['steps']!=50 or run['decoded_video_frames']!=81:
            raise ValueError('Full-size pilot requires complete 832x480 / 81 frames / 50 steps')
        result.append({'prompt_en':run['prompt'],'dimension':list(dimensions),'video_list':[(folder/'sample.mp4').as_posix()]})
    return result


def load_musiq_namespace(source):
    # Load official MUSIQ dependencies without importing unrelated PyIQA scorers,
    # training helpers or their incompatible Transformers pins. No model file is edited.
    for name,relative in [('pyiqa','pyiqa'),('pyiqa.archs','pyiqa/archs'),('pyiqa.utils','pyiqa/utils'),('pyiqa.data','pyiqa/data')]:
        module=types.ModuleType(name);module.__path__=[str(source/relative)];sys.modules[name]=module


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--dimensions',nargs='+',choices=QUALITY_DIMS,default=list(QUALITY_DIMS))
    p.add_argument('--assets',default='/quality');a=p.parse_args()
    import torch
    assets=pathlib.Path(a.assets).resolve();out=pathlib.Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(assets/'source/VBench'))
    runs=[(pathlib.Path(folder).resolve(),json.loads((pathlib.Path(folder)/'run.json').read_text(encoding='utf-8'))) for folder in a.runs]
    metadata=build_metadata(runs,a.dimensions)
    for folder,run in runs:
        if hashlib.sha256((folder/'sample.mp4').read_bytes()).hexdigest()!=run['video_file_sha256']:raise ValueError('Video hash changed')
    info=out/'dimension-inputs.json';info.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    cache=assets/'vbench-cache'
    raft=cache/'raft_model/models/raft-things.pth'
    if not raft.exists() and (cache/'raft_model/models.zip').exists():
        with zipfile.ZipFile(cache/'raft_model/models.zip') as archive:
            member=next(name for name in archive.namelist() if name.endswith('raft-things.pth'))
            raft.parent.mkdir(parents=True,exist_ok=True);raft.write_bytes(archive.read(member))
    paths={
        'background_consistency':[str(cache/'clip_model/ViT-B-32.pt'),False],
        'temporal_flickering':[],
        'motion_smoothness':{'config':str(assets/'source/VBench/vbench/third_party/amt/cfgs/AMT-S.yaml'),'ckpt':str(cache/'amt_model/amt-s.pth')},
        'dynamic_degree':{'model':str(raft)},
        'aesthetic_quality':[str(cache/'clip_model/ViT-L-14.pt'),str(cache/'aesthetic_model/emb_reader')],
        'imaging_quality':{'model_path':str(cache/'pyiqa_model/musiq_spaq_ckpt-358bb6af.pth')}}
    report={'status':'running','benchmark':'VBench-v1 official dimension implementations',
        'mode':'custom-input diagnostic; two or few standard-prompt videos, not the standard prompt suite',
        'videos':len(runs),'frames':81,'steps':50,'dimension_scores':{},'total_score':None,
        'paper_total_score':86.220,'paper_total_comparison_valid':False,
        'source':json.loads((assets/'source-manifest.json').read_text())['repositories']['VBench']}
    def save():
        temporary=out/'vbench.tmp';temporary.write_text(json.dumps(report,indent=2),encoding='utf-8');temporary.replace(out/'vbench-pilot.json')
    save();torch.set_num_threads(4);device=torch.device('cuda');started=time.perf_counter()
    for dimension in a.dimensions:
        print('Evaluating VBench dimension:',dimension,flush=True)
        module=None;model=None
        if dimension=='imaging_quality':load_musiq_namespace(assets/'source/PyIQA')
        module=importlib.import_module('vbench.'+dimension)
        if dimension=='subject_consistency':
            repository=cache/'dino_model/facebookresearch_dino_main'
            model=torch.hub.load(str(repository),'dino_vitb16',source='local',pretrained=False)
            model.load_state_dict(torch.load(cache/'dino_model/dino_vitbase16_pretrain.pth',map_location='cpu',weights_only=True),strict=True)
            model.eval().to(device)
            with torch.no_grad():score,details=module.subject_consistency(model,[entry['video_list'][0] for entry in metadata],device,False)
        else:
            with torch.no_grad():score,details=getattr(module,'compute_'+dimension)(str(info),device,paths[dimension])
        report['dimension_scores'][dimension]={'score':float(score),'per_video':details}
        report['elapsed_seconds']=time.perf_counter()-started;save()
        print(dimension,float(score),flush=True)
        del model;torch.cuda.empty_cache()
    report['status']='complete';save();print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
