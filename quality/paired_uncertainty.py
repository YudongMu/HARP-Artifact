"""Paired exploratory uncertainty; fixed real FID moments, never frame bootstrap.

The low-rank identity avoids a 2048-square matrix square root in every bootstrap
replicate. Point estimates must also be checked against the official scorer.
Intervals describe input-pair resampling at the observed development scale;
they do not remove small-sample FID bias or establish paper-suite performance.
"""
import numpy as np


def _reference(mu,cov):
 mu=np.asarray(mu,dtype=np.float64);cov=np.asarray(cov,dtype=np.float64)
 if mu.ndim!=1 or cov.shape!=(len(mu),len(mu)) or not np.isfinite(mu).all() or not np.isfinite(cov).all():
  raise ValueError('Incompatible or nonfinite real reference moments')
 if not np.allclose(cov,cov.T,atol=1e-10,rtol=1e-10):raise ValueError('Reference covariance is not symmetric')
 eig=np.linalg.eigvalsh(cov)
 tolerance=64*np.finfo(np.float64).eps*len(mu)*float(np.abs(eig).max())
 if eig.min() < -tolerance:raise ValueError('Reference covariance is not positive semidefinite')
 return mu,cov


def _features(x,dimension):
 x=np.asarray(x,dtype=np.float64)
 if x.ndim!=2 or len(x)<2 or x.shape[1]!=dimension or not np.isfinite(x).all():
  raise ValueError('FID requires at least two finite compatible feature vectors')
 return x


def _fid_from_gram(x,reference_mean,reference_trace,gram,indices):
 sample=x[indices];mean=sample.mean(axis=0);n=len(sample)
 centered=sample-mean
 generated_trace=float(np.einsum('ij,ij->',centered,centered)/(n-1))
 projected=gram[np.ix_(indices,indices)]
 row_mean=projected.mean(axis=1,keepdims=True)
 centered_gram=(projected-row_mean-row_mean.T+projected.mean())/(n-1)
 centered_gram=(centered_gram+centered_gram.T)*.5
 eigenvalues=np.linalg.eigvalsh(centered_gram)
 scale=max(float(np.abs(eigenvalues).max()),float(np.abs(projected).max())*n/(n-1))
 tolerance=64*np.finfo(np.float64).eps*n*scale
 if eigenvalues.min() < -tolerance:
  raise ValueError('Projected feature covariance is not positive semidefinite')
 # Centering leaves an exact nullspace. Roundoff-sized positive eigenvalues
 # must also be zeroed before sqrt, which otherwise amplifies them to1e-7.
 eigenvalues=np.where(np.abs(eigenvalues)<=tolerance,0.,eigenvalues)
 cross=float(np.sqrt(np.maximum(eigenvalues,0)).sum())
 value=float(np.sum((mean-reference_mean)**2)+generated_trace+reference_trace-2*cross)
 if value < -1e-8*(reference_trace+generated_trace):raise ValueError('Unexpected negative FID')
 return max(value,0.)


def low_rank_fid(features,reference_mean,reference_covariance):
 mu,cov=_reference(reference_mean,reference_covariance);x=_features(features,len(mu))
 centered=x-x.mean(axis=0)
 return _fid_from_gram(x,mu,float(np.trace(cov)),centered@cov@centered.T,np.arange(len(x)))


def _pairs(values,input_ids,minimum):
 if 'dense' not in values or len(values)<2 or set(input_ids)!=set(values):raise ValueError('Dense and comparison modes with explicit pair IDs are required')
 ids=list(input_ids['dense']);n=len(ids)
 if n<minimum or len(set(ids))!=n:raise ValueError('Insufficient or duplicate input pairs')
 for mode,v in values.items():
  if list(input_ids[mode])!=ids or len(v)!=n:raise ValueError('Input pair order/count mismatch')
 return ids,n


def _loss(reference,comparison,direction):
 if reference<=0:return None
 return float((comparison-reference)/reference if direction=='lower' else (reference-comparison)/reference)


def _summary(point,draws,ids,replicates,seed,direction):
 baseline=point['dense'];rows={}
 for mode,value in point.items():
  relatives=[_loss(draw['dense'],draw[mode],direction) for draw in draws]
  valid=[x for x in relatives if x is not None]
  valid_interval=baseline>0 and len(ids)>1 and len(valid)==len(draws) and bool(valid)
  interval=np.percentile(valid,[2.5,97.5]).tolist() if valid_interval else None
  score_interval=np.percentile([draw[mode] for draw in draws],[2.5,97.5]).tolist() if len(ids)>1 and draws else None
  rows[mode]={'score':float(value),'signed_delta':float(value-baseline),
              'relative_loss':_loss(baseline,value,direction),'relative_loss_interval':interval,
              'score_interval':score_interval,'valid_relative_replicates':len(valid),
              'invalid_relative_replicates':len(draws)-len(valid),
              'relative_interval_status':'complete' if valid_interval else 'point_denominator_nonpositive' if baseline<=0 else 'single_input' if len(ids)==1 else 'invalid_bootstrap_denominator'}
 return {'sample_count':len(ids),'input_ids':ids,'modes':rows,'replicates':replicates,'seed':seed,
         'direction':direction,'interval_level':.95,'resampling_unit':'paired input prompt/class and seed; identical indices across modes',
         'interpretation':'Exploratory percentile intervals at the observed sample size; no paper-suite/general-population claim'}


def paired_fid_bootstrap(feature_sets,reference_mean,reference_covariance,*,input_ids,replicates=400,seed=20261105):
 if type(replicates) is not int or replicates<2:raise ValueError('At least two bootstrap replicates required')
 ids,n=_pairs(feature_sets,input_ids,2);mu,cov=_reference(reference_mean,reference_covariance)
 data={mode:_features(value,len(mu)) for mode,value in feature_sets.items()}
 # A single fixed centering before projection is mathematically immaterial to
 # every resampled covariance, and avoids cancelling huge common offsets.
 centered={mode:x-x.mean(axis=0) for mode,x in data.items()}
 gram={mode:x@cov@x.T for mode,x in centered.items()};trace=float(np.trace(cov))
 def evaluate(indices):return {mode:_fid_from_gram(x,mu,trace,gram[mode],indices) for mode,x in data.items()}
 point=evaluate(np.arange(n));rng=np.random.default_rng(seed)
 draws=[evaluate(rng.integers(0,n,n)) for _ in range(replicates)]
 result=_summary(point,draws,ids,replicates,seed,'lower')
 result.update(metric='FID on fixed real reference moments',small_sample_bias_removed=False,
               algorithm='trace-sqrt of centered n-by-n projected feature Gram matrix; unbiased sample covariance ddof1')
 return result


def paired_metric_bootstrap(values_by_mode,*,input_ids,replicates=400,seed=20261105,direction='higher'):
 if direction not in {'higher','lower'} or type(replicates) is not int or replicates<2:raise ValueError('Invalid metric direction/replicates')
 ids,n=_pairs(values_by_mode,input_ids,1)
 data={mode:np.asarray(value,dtype=np.float64) for mode,value in values_by_mode.items()}
 if any(v.shape!=(n,) or not np.isfinite(v).all() for v in data.values()):raise ValueError('Expected one finite score per eligible paired input')
 point={mode:float(v.mean()) for mode,v in data.items()};rng=np.random.default_rng(seed);draws=[]
 if n>1:
  for _ in range(replicates):
   indices=rng.integers(0,n,n);draws.append({mode:float(v[indices].mean()) for mode,v in data.items()})
 return _summary(point,draws,ids,replicates,seed,direction)
