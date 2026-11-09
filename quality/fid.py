"""ADM's official TensorFlow Inception features and FID, with explicit pilot sizes."""
import argparse,hashlib,json,os,pathlib,sys,time
import numpy as np
from PIL import Image


def pilot_subsets(count):
    if count<2:raise ValueError('FID requires at least two samples')
    sizes=sorted(set([min(250,count),min(500,count),count]))
    return {str(n):np.linspace(0,count,n,endpoint=False,dtype=int).tolist() for n in sizes}


def class_balanced_subsets(samples):
    groups={c:[] for c in range(1000)}
    for i,row in enumerate(samples):
        if row['class_id'] not in groups:raise ValueError('Invalid ImageNet class')
        groups[row['class_id']].append(i)
    counts={len(g) for g in groups.values()}
    if len(counts)!=1 or 0 in counts:raise ValueError('All 1000 classes require equal positive counts')
    repeats=counts.pop()
    return {str(k*1000):sorted(i for group in groups.values() for i in group[:k])
            for k in sorted({1,min(2,repeats),repeats})}


def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',required=True);p.add_argument('--assets',default='/quality')
    p.add_argument('--balanced-classes',action='store_true')
    p.add_argument('--full-only',action='store_true');a=p.parse_args()
    assets=pathlib.Path(a.assets).resolve();out=pathlib.Path(a.samples).resolve()
    generation=(out/'generation.json').read_bytes()
    run=json.loads(generation)
    if run['status']!='complete' or run['completed']!=run['count']:raise ValueError('Generation incomplete')
    subsets=class_balanced_subsets(run['samples']) if a.balanced_classes else pilot_subsets(run['count'])
    if a.full_only:subsets={str(run['count']):list(range(run['count']))}
    sys.path.insert(0,str(assets/'source/guided-diffusion/evaluations'))
    os.chdir(assets/'fid')
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    import evaluator
    config=tf.ConfigProto(device_count={'GPU':0},intra_op_parallelism_threads=8,inter_op_parallelism_threads=2)
    started=time.perf_counter()
    with tf.Session(config=config) as session:
        metric=evaluator.Evaluator(session,batch_size=32)
        def batches():
            for start in range(0,len(run['samples']),32):
                images=[]
                for item in run['samples'][start:start+32]:
                    path=out/item['image']
                    if hashlib.sha256(path.read_bytes()).hexdigest()!=item['image_sha256']:raise ValueError('Image hash changed')
                    with Image.open(path) as image:images.append(np.array(image.convert('RGB')))
                yield np.stack(images)
        print('Extracting official ADM Inception features',flush=True)
        features,_=metric.compute_activations(batches())
        np.save(out/'inception-features.npy',features)
        with np.load(assets/'fid/imagenet512-moments.npz') as reference:
            ref=evaluator.FIDStatistics(reference['mu'],reference['sigma'])
        print('Checking reference self-distance',flush=True)
        self_distance=float(ref.frechet_distance(ref))
        if abs(self_distance)>1e-5:raise ValueError('Reference FID self-test failed')
        results={}
        for name,indices in subsets.items():
            print('Computing FID for',name,'generated images',flush=True)
            value=float(metric.compute_statistics(features[indices]).frechet_distance(ref))
            if not np.isfinite(value):raise ValueError('Nonfinite FID')
            results[name]=value
    report={'metric':'ADM TensorFlow Inception FID','generated_samples':len(features),'fid_by_sample_count':results,
        'generation_manifest_sha256':hashlib.sha256(generation).hexdigest(),
        'class_coverage_by_subset':{name:len({run['samples'][i]['class_id'] for i in indices}) for name,indices in subsets.items()},
        'balanced_classes':a.balanced_classes,
        'reference_self_fid':self_distance,'feature_sha256':hashlib.sha256((out/'inception-features.npy').read_bytes()).hexdigest(),
        'reference_sha256':hashlib.sha256((assets/'fid/imagenet512-moments.npz').read_bytes()).hexdigest(),
        'inception_graph_sha256':hashlib.sha256((assets/'fid/classify_image_graph_def.pb').read_bytes()).hexdigest(),
        'source':json.loads((assets/'source-manifest.json').read_text())['repositories']['guided-diffusion'],
        'tensorflow':tf.__version__,'elapsed_seconds':time.perf_counter()-started}
    if not a.full_only:
        report.update(paper_dense_fid_50k=14.250,paper_harp_fid_50k=14.414,formal_paper_quality_reproduced=False,
            interpretation=('All reported subsets cover 1000 classes equally. Finite-sample uncertainty remains; do not extrapolate FID-50K or claim paper reproduction.' if a.balanced_classes else 'Pilot FID is sample-count biased. Legacy small subsets also cover different class sets; this is not a pure sample-size convergence curve.'))
    pending=out/'fid-pilot.json.tmp'
    pending.write_text(json.dumps(report,indent=2),encoding='utf-8')
    pending.replace(out/'fid-pilot.json')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
