"""CPU-only postprocessing of four-mode official held-out VBench results."""
import argparse,json,pathlib,time,sys
from quality.wan_stage4_score import read,sha,write,load_official_aggregate,MODES,validate_manifest
from quality.wan_stage4_uncertainty import evaluate_uncertainty

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--assets',default='/quality');p.add_argument('--replicates',type=int,default=400);a=p.parse_args()
root=pathlib.Path(a.root);out=root/'heldout-scores';assets=pathlib.Path(a.assets);started=time.monotonic()
summary=read(out/'summary.json');manifest=read(root/'manifest.json')
if summary['status'] not in {'complete','complete_with_unaccepted_dimensions'} or not summary['all_modes_all_16_dimensions_attempted']:raise ValueError('All four modes must finish all16 dimension attempts')
if summary['manifest_sha256']!=sha(root/'manifest.json') or summary['selection_sha256']!=sha(root/'selection.json') or summary['generation_index_sha256']!=sha(root/'generation-index.json'):raise ValueError('Scoring inputs/selection changed')
validate_manifest(manifest,read(assets/'source/VBench/vbench/VBench_full_info.json'))
helper=pathlib.Path(__file__).parents[1]/'quality/wan_stage4_score.py'
if summary['scorer_source_sha256']['wan_stage4_score.py']!=sha(helper):raise ValueError('Sufficient-statistics scorer changed since official points')
reports={}
for mode in MODES:
 path=out/mode/'mode-scores.json'
 if sha(path)!=summary['modes'][mode]['mode_scores_sha256']:raise ValueError('Independent mode score changed')
 reports[mode]=read(path)
binding={'summary_sha256':sha(out/'summary.json'),'selection_sha256':sha(root/'selection.json'),'manifest_sha256':sha(root/'manifest.json'),
         'uncertainty_source_sha256':sha(pathlib.Path(__file__).parents[1]/'quality/wan_stage4_uncertainty.py'),
         'point_source_sha256':sha(helper),'driver_source_sha256':sha(__file__)}
target=out/'paired-uncertainty.json'
if target.exists():
 cached=read(target)
 if cached.get('status')!='complete' or cached.get('source')!=binding or cached.get('replicates')!=a.replicates:raise ValueError('Completed Wan uncertainty binding changed; never overwrite it')
 print('Completed paired Wan uncertainty verified; no overwrite');sys.exit(0)
result=evaluate_uncertainty(reports,manifest,load_official_aggregate(assets/'source/VBench'),replicates=a.replicates)
result.update(frames=manifest['frames'],steps=manifest['steps'],selection=read(root/'selection.json'),
 source=binding,
 elapsed_seconds=time.monotonic()-started,paper_full_suite_reproduced=False)
write(target,result)
print(json.dumps({'status':result['status'],'frames':result['frames'],'total':result['total'],'elapsed_seconds':result['elapsed_seconds']},indent=2))
