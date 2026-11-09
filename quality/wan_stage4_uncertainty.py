"""Conditional paired seed uncertainty for the frozen eleven-prompt Wan subset."""
import math
import numpy as np
from .wan_semantic import ALL_DIMENSIONS
from .wan_stage4_score import MODES,resampled_dimension,normalize_scores


def _rows(points,draws,replicates):
 result={};base=points['dense']
 for mode,value in points.items():
  scores=[r[mode] for r in draws if r[mode] is not None]
  relative=[(r['dense']-r[mode])/r['dense'] for r in draws if r['dense'] is not None and r['dense']>0 and r[mode] is not None]
  point_relative=(base-value)/base if base is not None and base>0 and value is not None else None
  complete_scores=len(scores)==replicates
  complete_relative=point_relative is not None and len(relative)==replicates
  interval=np.percentile(relative,[2.5,97.5]).tolist() if complete_relative else None
  result[mode]={'score':value,'signed_delta':value-base if value is not None and base is not None else None,
   'relative_loss':point_relative,'relative_loss_interval':interval,
   'score_interval':np.percentile(scores,[2.5,97.5]).tolist() if complete_scores else None,
   'invalid_score_replicates':replicates-len(scores),'invalid_relative_replicates':replicates-len(relative),
   'point_within_2percent':point_relative<=.02 if point_relative is not None else None,
   'interval_supports_2percent':interval[1]<=.02 if interval is not None else None,
   'interval_status':'complete' if complete_relative else 'unavailable_due_to_NA_point_or_unreconstructable_or_undefined_draw; no conditioning_away_NA'}
 return result


def evaluate_uncertainty(mode_reports,manifest,aggregate,*,replicates=400,seed=20261106):
 if set(mode_reports)!=set(MODES) or type(replicates) is not int or replicates<2:raise ValueError('Four independent modes and >=2 replicates required')
 items=manifest['inputs']['heldout'];ids=[r['id'] for r in items]
 if len(ids)!=22 or len(set(ids))!=22:raise ValueError('Expected22 unique prompt/seed inputs')
 strata={p:[r for r in items if r['prompt_index']==p] for p in range(11)}
 if any(len(rows)!=2 or {r['seed_index'] for r in rows}!={0,1} for rows in strata.values()):raise ValueError('Expected eleven fixed prompt strata with two seeds each')
 dimensions={};normalized_points={};checks={}
 for mode,report in mode_reports.items():
  if report.get('mode')!=mode or report.get('status') not in {'complete','complete_with_unaccepted_dimensions'} or set(report['dimension_scores'])!=set(ALL_DIMENSIONS):raise ValueError('Incomplete or mixed mode reports')
  dimensions[mode]=report['dimension_scores'];normalized_points[mode]=report['normalized']['display_percent'];checks[mode]={}
  for dimension,row in dimensions[mode].items():
   point=row.get('score') if row.get('status')=='complete' else None
   if point is not None and not math.isfinite(point):raise ValueError('Nonfinite official score must be explicit NA')
   reconstructed=resampled_dimension(row,ids)
   if reconstructed is not None and (point is None or not math.isclose(reconstructed,point,abs_tol=1e-8,rel_tol=1e-6)):raise ValueError('Contribution reaggregation changed the official point')
   checks[mode][dimension]={'official_point':point,'reconstructed_point':reconstructed,'CI_reconstruction_available':reconstructed is not None}
  normalized=normalize_scores(dimensions[mode],aggregate)['display_percent']
  for key in ('quality','semantic','total'):
   a,b=normalized[key],normalized_points[mode][key]
   if (a is None)!=(b is None) or (a is not None and not math.isclose(a,b,abs_tol=1e-8,rel_tol=1e-6)):raise ValueError('Normalized official point reconstruction differs')
 rng=np.random.default_rng(seed);dimension_draws={d:[] for d in ALL_DIMENSIONS};aggregate_draws={k:[] for k in ('quality','semantic','total')}
 stratum_ids=[[r['id'] for r in rows] for rows in strata.values()]
 for _ in range(replicates):
  sampled=[group[index] for group in stratum_ids for index in rng.integers(0,2,2)]
  current={}
  for mode in MODES:
   raw={d:resampled_dimension(dimensions[mode][d],sampled) for d in ALL_DIMENSIONS}
   results={d:dict(status='complete' if v is not None else 'undefined',score=v) for d,v in raw.items()}
   current[mode]={'raw':raw,'aggregate':normalize_scores(results,aggregate)['display_percent']}
  for d in ALL_DIMENSIONS:dimension_draws[d].append({m:current[m]['raw'][d] for m in MODES})
  for k in aggregate_draws:aggregate_draws[k].append({m:current[m]['aggregate'][k] for m in MODES})
 report={'status':'complete','videos_per_mode':22,'prompt_strata':11,'seeds_per_prompt':2,'replicates':replicates,'seed':seed,
  'resampling_unit':'Paired seed within each fixed prompt; two draws per prompt; same22 drawn IDs across modes/dimensions',
  'interval_level':.95,'scope':'Conditional fixed11-prompt seed variability, not the946-prompt population; two seeds give limited uncertainty',
  'NA_handling':'Any undefined/unreconstructable replicate suppresses the associated interval; invalid counts retained, never silently dropped',
  'primary_target':'Development relative normalized total loss <=2%; distinct from lambda/C/E/R and not a recovered paper guarantee',
  'point_reaggregation_checks':checks,'dimensions':{}}
 for k in aggregate_draws:report[k]=_rows({m:normalized_points[m][k] for m in MODES},aggregate_draws[k],replicates)
 for d in ALL_DIMENSIONS:
  points={m:dimensions[m][d].get('score') if dimensions[m][d].get('status')=='complete' else None for m in MODES}
  report['dimensions'][d]=_rows(points,dimension_draws[d],replicates)
 return report
