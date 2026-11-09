"""Official ADM FID, one distinct feature pool per phase/mode; CPU only."""
import argparse,hashlib,json,os,pathlib,sys,time
import numpy as np
from PIL import Image
from quality.dit_stage4 import (group_names,calibration_items,heldout_items,file_sha,canonical_sha,write_json,
 validate_generation,validate_smokes,choose_candidate,freeze_selection,load_freeze,CANDIDATES)
from quality.paired_uncertainty import low_rank_fid
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['calibration','heldout'],required=True)
p.add_argument('--output',required=True);p.add_argument('--candidates',required=True);p.add_argument('--selection',required=True)
p.add_argument('--assets',default='/quality');a=p.parse_args()
out=pathlib.Path(a.output).resolve();assets=pathlib.Path(a.assets).resolve()
items=calibration_items() if a.phase=='calibration' else heldout_items();ids=[x['id'] for x in items]
progress=json.loads((out/'progress.json').read_text())
if progress['status']!='complete' or progress['phase']!=a.phase:raise ValueError('Generation phase incomplete')
execution_sha=file_sha(out/'execution.json');execution=json.loads((out/'execution.json').read_text());runs={}
validate_smokes(out,a.phase,canonical_sha(execution),ids[0])
for group in group_names(a.phase):
 runs[group]=json.loads((out/group/'generation.json').read_text());validate_generation(out/group,runs[group],a.phase,group,ids)
 if runs[group]['execution_sha256']!=canonical_sha(json.loads((out/'execution.json').read_text())):raise ValueError('Generation/source binding differs')
 if [r['noise_sha256'] for r in runs[group]['samples']]!=[r['noise_sha256'] for r in runs['dense']['samples']]:raise ValueError('Different paired noise')
if a.phase=='heldout':
 frozen=load_freeze(a.selection)
 if json.loads((out/'execution.json').read_text())['freeze_sha256']!=frozen['freeze_sha256']:raise ValueError('Heldout selection changed')
sources={str(path):file_sha(path) for path in [pathlib.Path(__file__),pathlib.Path(sys.modules['quality.dit_stage4'].__file__),pathlib.Path(sys.modules['quality.paired_uncertainty'].__file__),assets/'source/guided-diffusion/evaluations/evaluator.py',assets/'fid/imagenet512-moments.npz',assets/'fid/classify_image_graph_def.pb']}
existing=out/'quality.json'
if existing.exists():
 report=json.loads(existing.read_text())
 if report['status']!='complete' or report['execution_sha256']!=execution_sha or report['metric_source_sha256']!=sources:raise ValueError('Completed score identity changed')
 for g,row in report['groups'].items():
  if file_sha(out/g/'inception-features.npy')!=row['feature_sha256']:raise ValueError('Cached features changed')
else:
 sys.path.insert(0,str(assets/'source/guided-diffusion/evaluations'));os.chdir(assets/'fid')
 import tensorflow.compat.v1 as tf
 tf.disable_eager_execution();import evaluator
 config=tf.ConfigProto(device_count={'GPU':0},intra_op_parallelism_threads=8,inter_op_parallelism_threads=2)
 report=dict(status='running',phase=a.phase,input_ids=ids,execution_sha256=execution_sha,metric='Official ADM TensorFlow Inception FID',metric_source_sha256=sources,groups={},formal_paper_quality=False,small_sample_warning='Calibration N10 and heldout N256 are development estimates, not FID50K or a general quality guarantee.')
 started=time.monotonic()
 with tf.Session(config=config) as session:
  metric=evaluator.Evaluator(session,batch_size=32)
  with np.load(assets/'fid/imagenet512-moments.npz') as ref:reference=evaluator.FIDStatistics(ref['mu'],ref['sigma'])
  if abs(reference.frechet_distance(reference))>1e-5:raise ValueError('Reference self FID failed')
  for group,run in runs.items():
   cache_key=canonical_sha(dict(generation=file_sha(out/group/'generation.json'),sources=sources))
   cache=out/group/'feature-receipt.json';feature_path=out/group/'inception-features.npy'
   if cache.exists():
    saved=json.loads(cache.read_text())
    if saved['key']!=cache_key or file_sha(feature_path)!=saved['sha256']:raise ValueError('Feature cache identity changed')
    features=np.load(feature_path,allow_pickle=False)
   else:
    def batches():
     for start in range(0,len(run['samples']),32):
      images=[]
      for row in run['samples'][start:start+32]:
       with Image.open(out/group/row['image']) as image:images.append(np.array(image.convert('RGB')))
      yield np.stack(images)
    features,_=metric.compute_activations(batches());np.save(feature_path,features)
    write_json(cache,dict(key=cache_key,sha256=file_sha(feature_path)))
   if features.shape!=(len(ids),2048):raise ValueError('Wrong mode feature shape')
   equivalent=float(low_rank_fid(features,reference.mu,reference.sigma))
   official=None;failure=None
   try:
    official=float(metric.compute_statistics(features).frechet_distance(reference))
    if not np.isfinite(official):raise ValueError('Nonfinite official ADM FID')
   except (ValueError,np.linalg.LinAlgError) as error:
    failure=dict(type=type(error).__name__,message=str(error));official=None
   difference=abs(official-equivalent) if official is not None else None
   if difference is not None and difference>max(1e-4,1e-6*abs(official)):raise ValueError('Official and low-rank FID point estimates disagree')
   report['groups'][group]=dict(official_adm_fid=official,official_adm_status='PASS' if official is not None else 'FAILED',official_adm_failure=failure,
    low_rank_fid=equivalent,arithmetic_absolute_difference=difference,count=len(ids),input_ids=ids,feature_sha256=file_sha(feature_path))
   write_json(out/'score-progress.json',report);print('FID',group,'official',official,'low_rank',equivalent,'failure',failure,flush=True)
 report.update(status='complete',elapsed_seconds=time.monotonic()-started)
 use_official=all(row['official_adm_status']=='PASS' for row in report['groups'].values())
 report['arithmetic_used']='official_adm_scipy_sqrtm' if use_official else 'official_inception_features_low_rank_fid_equivalent'
 report['official_adm_all_groups_passed']=use_official
 for row in report['groups'].values():row['fid']=row['official_adm_fid'] if use_official else row['low_rank_fid']
 dense=report['groups']['dense']['fid']
 if dense<=0:raise ValueError('Invalid relative FID denominator')
 for row in report['groups'].values():row['loss_percent_dense']=100*(row['fid']-dense)/dense
 write_json(existing,report)
if a.phase=='calibration':
 selection=choose_candidate(report);selection.update(calibration_report=str(existing),calibration_report_sha256=file_sha(existing),
  primary_metric='FID: fixed ImageNet real moments, relative to paired dense',arithmetic_used=report['arithmetic_used'],official_adm_all_groups_passed=report['official_adm_all_groups_passed'],
  table_sha256={c:file_sha(pathlib.Path(a.candidates)/(c+'-table.json')) for c in CANDIDATES})
 freeze_selection(a.selection,selection)
 print('FROZEN',selection['candidate'],'passed',selection['calibration_passed'],'exploratory',selection['exploratory'],flush=True)
else:print('Heldout complete; frozen selection untouched',flush=True)
