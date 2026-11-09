"""Phased short-video Wan calibration and generation; GPU scheduling is external."""
import argparse,copy,hashlib,importlib,json,math,pathlib,random,shutil,sys,time
import torch
from .wan_budget import (Protocol,input_split,PRIMARY_TARGETS,SURVEY_TARGETS,LAMBDAS,key_of,sidecar_sha,tensor_sha,
    WanSurvey,WanProbe,RuntimeFactory,patched_runtime,risk_summary,proposals,select_policy,table_identity)
from .wan_paired import sha,write_json,public_schedule
from .vbench import QUALITY_DIMS,load_musiq_namespace
from .wan_semantic_score import checked_dimension_result,inference_guard,canonical_video_name
from harp.calibration import BudgetTable
from harp.protocol import schedule_digest

def make_manifest(official,protocol):
    split=copy.deepcopy(input_split())
    for rows in split.values():
        for row in rows:
            matches=[i for i,r in enumerate(official) if r['prompt_en']==row['prompt']]
            if len(matches)!=1:raise ValueError('Prompt must uniquely match original official metadata')
            index=matches[0];row.update(source_index=index,metadata=copy.deepcopy(official[index]))
    coverage=sorted(set().union(*(set(r['metadata']['dimension']) for r in split['heldout'])))
    return {'version':'wan-stage4-v1','frames':protocol.frames,'steps':50,'resolution':[832,480],'inputs':split,'heldout_official_dimension_coverage':coverage,
        'heldout_videos_per_mode':22,'unique_heldout_prompts':11,'seeds_per_heldout_prompt':2,
        'replacement':'Predeclared red bicycle -> official red car only; old undefined result retained elsewhere',
        'expected_modes':['dense','residual','planeclear','combined'],
        'artifact_pattern':'runs/<input_id>/<mode>/{run.json,initial-noise.pt,latent.pt,sample.mp4,stage4-execution.json}',
        'scorer_sampling_note':'Official uniform16 frame sampling may repeat short9/17 input frames; temporal diagnostic is not standard81 quality. Never pad or forge a new video.'}

def validate_run(run,protocol,noise=None):
    identity=run['identity']
    if run['status']!='complete' or run['forward_calls']!=100 or run['decoded_video_frames']!=protocol.frames or not run['finite_latent'] or not run['finite_video']:raise ValueError('Incomplete short-protocol trajectory')
    if identity['frames']!=protocol.frames or identity['wan_stage4_frames']!=protocol.frames or identity['steps']!=50 or identity['resolution']!=[832,480] or identity['mode'] not in {'dense','residual','planeclear','combined'}:raise ValueError('Short protocol/deployment mode mismatch; not81 transfer')
    if identity['cfg_scale']!=5 or identity['cfg_channels']!='all' or identity['mass_threshold']!=.95 or identity['refresh_interval']!=5 or identity['wan_stage4_sidecar_sha256']!=sidecar_sha():raise ValueError('Changed stage4 arithmetic/control deployment')
    implementation=hashlib.sha256(json.dumps(run['source_sha256'],sort_keys=True).encode()).hexdigest()
    if identity.get('noise_schedule_sha256')!=schedule_digest(run['schedule']) or identity.get('implementation_sha256')!=implementation:raise ValueError('Actual schedule/source identity digest differs')
    if not isinstance(identity.get('checkpoint_sha256'),str) or len(identity['checkpoint_sha256'])!=64 or identity.get('adapter')!='wan-self-ffn-reference-v1' or identity.get('cross_attention')!='dense' or not identity.get('negative_prompt'):raise ValueError('Missing checkpoint/adapter/cross-attention/negative-prompt identity')
    if identity['attention_execution_backend']!=('masked_chunked' if identity['mode'] in {'residual','combined'} else 'selected') or identity['attention_query_chunk']!=128:raise ValueError('Changed short protocol attention backend')
    if noise is not None:
        if tuple(noise.shape)!=(16,protocol.latent_frames,60,104) or noise.dtype!=torch.float32 or not bool(torch.isfinite(noise).all()) or tensor_sha(noise)!=run['initial_noise_sha256']:raise ValueError('Actual short-protocol noise differs')

def validate_comparison(reference,other,protocol,noise,other_noise):
    validate_run(reference,protocol,noise);validate_run(other,protocol,other_noise)
    if not torch.equal(noise,other_noise):raise ValueError('Actual noise tensors differ')
    for key in ('prompt','seed','source_sha256','initial_noise_sha256'):
        if reference[key]!=other[key]:raise ValueError('Unpaired '+key)
    if public_schedule(reference['schedule'])!=public_schedule(other['schedule']):raise ValueError('Actual public schedule differs')
    excluded={'mode','attention_execution_backend','attention_query_chunk'}
    if {k:v for k,v in reference['identity'].items() if k not in excluded}!={k:v for k,v in other['identity'].items() if k not in excluded}:raise ValueError('Unpaired deployment identity')

def generic_metadata(video,metadata):
    return ([dict(prompt_en=metadata['prompt_en'],dimension=list(QUALITY_DIMS),video_list=[str(video)])],
            [dict(copy.deepcopy(metadata),video_list=[str(video)])])

def _load(path):return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
def _latent(path):return torch.load(path/'latent.pt',map_location='cpu',weights_only=True)
def _noise(path):return torch.load(path/'initial-noise.pt',map_location='cpu',weights_only=True)

def immutable_json(path,data):
    if path.exists():
        if _load(path)!=data:raise ValueError('Immutable artifact differs: '+str(path))
    else:write_json(path,data)

def current_generator_source(root):
    return {p.relative_to(root).as_posix():sha(p) for directory in (root/'harp',root/'src/Wan2.1/wan') for p in directory.rglob('*.py')}

def prepare_manifest(out,protocol,metadata_path,root,resume,gpu_deterministic):
    from .wan_semantic import ALL_DIMENSIONS
    manifest=make_manifest(_load(metadata_path),protocol)
    if set(manifest['heldout_official_dimension_coverage'])!=set(ALL_DIMENSIONS):raise ValueError('Predeclared heldout does not cover all16 official dimensions')
    sources=current_generator_source(root)
    if not any(k.startswith('harp/') for k in sources) or not any(k.startswith('src/Wan2.1/wan/') for k in sources):raise ValueError('Missing mounted actual generator sources')
    manifest.update(official_metadata_sha256=sha(metadata_path),generator_source_sha256=sources,sidecar_sha256=sidecar_sha(),
        gpu_deterministic_algorithms=bool(gpu_deterministic))
    if out.exists():
        if not resume:raise ValueError('Existing phase root requires explicit --resume')
        if not (out/'manifest.json').exists() or _load(out/'manifest.json')!=manifest:raise ValueError('Frozen manifest/source/inputs/protocol changed; use a new independent root')
    else:out.mkdir(parents=True)
    immutable_json(out/'manifest.json',manifest)
    snapshot=out/'source-snapshot';snapshot.mkdir(exist_ok=True)
    for name in ('wan_budget.py','wan_stage4.py'):
        path=snapshot/name;original=pathlib.Path(__file__).with_name(name)
        if path.exists():
            if sha(path)!=sha(original):raise ValueError('Frozen sidecar snapshot differs')
        else:shutil.copyfile(original,path)
    return manifest

def _verify_decoded(folder,protocol):
    import cv2
    capture=cv2.VideoCapture(str(folder/'sample.mp4'));count=0
    try:
        while True:
            ok,frame=capture.read()
            if not ok:break
            if frame.shape!=(480,832,3):raise ValueError('Cached decoded shape differs')
            count+=1
    finally:capture.release()
    if count!=protocol.frames:raise ValueError('Cached decoded frame count differs')

def verified_case(folder,item,mode,protocol,table,expected_source):
    """Verify complete originals; never treat a partial folder as completed work."""
    folder=pathlib.Path(folder);run=_load(folder/'run.json');execution=_load(folder/'stage4-execution.json')
    validate_run(run,protocol,_noise(folder))
    if (run['prompt'],run['seed'],run['identity']['mode'])!=(item['prompt'],item['seed'],mode) or run['source_sha256']!=expected_source:raise ValueError('Cached input/source/mode differs')
    if execution.get('status')!='complete' or execution.get('mode')!=mode or execution.get('input_id')!=item['id'] or execution.get('completed_model_calls')!=100 or execution.get('completed_transformer_blocks')!=4000:raise ValueError('Cached observed execution incomplete')
    expected_table=sha(table) if table else None
    if execution.get('budget_table_sha256_before')!=expected_table or execution.get('budget_table_sha256_after')!=expected_table:raise ValueError('Cached actual executed table SHA differs')
    if table is not None:BudgetTable.load(table).validate(run['identity'])
    if execution.get('sidecar_sha256')!=sidecar_sha() or execution.get('sidecar_sha256_after')!=sidecar_sha():raise ValueError('Cached sidecar source differs')
    for name in ('run.json','initial-noise.pt','latent.pt','sample.mp4'):
        if execution.get('artifacts',{}).get(name)!=sha(folder/name):raise ValueError('Cached artifact bytes differ: '+name)
    latent=_latent(folder)
    if tuple(latent.shape)!=(16,protocol.latent_frames,60,104) or not bool(torch.isfinite(latent).all()) or tensor_sha(latent)!=run['latent_sha256'] or sha(folder/'sample.mp4')!=run['video_file_sha256']:raise ValueError('Cached actual latent/video differs')
    _verify_decoded(folder,protocol);return run

def restore_probe(probe,run):
    states=run['runtime']['runtimes'];row=copy.deepcopy(states['branch0:sample0'].get('intervention',{}))
    if 'intervention' in states['branch1:sample0']:raise ValueError('Cached probe was not branch0-only')
    expected=dict(probe.target,allowance=probe.allowance,actual_slice=probe.index)
    if row.get('hits')!=1 or any(row.get(k)!=v for k,v in expected.items()) or row.get('input_slice_sha256')!=probe.expected_input or row.get('committed_slice_sha256')!=probe.expected_committed:raise ValueError('Cached probe measurement/hit/bits differ')
    candidate=next(r for r in __import__('quality.wan_budget',fromlist=['candidates']).candidates(probe.expected) if r['allowance']==probe.allowance)
    if abs(row.get('cleared_mass',-1)-candidate['cleared_mass'])>1e-12:raise ValueError('Cached actual C differs from committed bits')
    probe.measurement=row;probe.measurement['cache_reused']=True

def restore_survey(survey,folder):
    sites=_load(folder/'survey-sites.json');selected=torch.load(folder/'survey-selected.pt',map_location='cpu',weights_only=True)
    if set(sites)!=survey.targets:raise ValueError('Cached survey target plan differs')
    from .wan_budget import candidates
    for key,row in sites.items():
        if row['status']=='sampled':
            if key not in selected or tensor_sha(selected[key])!=row['input_slice_sha256'] or candidates(selected[key])!=row['candidates']:raise ValueError('Cached survey sample/committed mask differs')
    survey.sites.update(sites);survey.selected.update(selected)

def preserve_incomplete(out,folder):
    target=out/'failed-attempts'/folder.parent.name/(folder.name+'-'+str(time.time_ns()))
    if not folder.resolve().is_relative_to(out.resolve()) or not target.resolve().is_relative_to(out.resolve()):raise ValueError('Retry archive must stay within owned stage output')
    target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(folder),str(target))
    with (out/'retry-ledger.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(dict(original=str(folder),preserved=str(target)))+'\n')

def verify_scoring_assets(assets):
    """Read every deployed weight in full, not just its receipt or filename."""
    inventory_path=assets/'wan-semantic-deployed-inventory.json';inventory=_load(inventory_path);seen=set();total=0
    if not inventory.get('checkpoint_files'):raise ValueError('Empty scorer inventory')
    for row in inventory['checkpoint_files']:
        path=assets/row['path']
        if not path.resolve().is_relative_to(assets.resolve()) or row['path'] in seen:raise ValueError('Invalid/duplicate scoring asset path')
        seen.add(row['path'])
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('Actual deployed scoring asset differs: '+row['path'])
        total+=row['bytes']
    return dict(inventory_sha256=sha(inventory_path),inventory=inventory,total_bytes=total,all_files_fully_sha256_verified=True)

def scoring_context(assets,manifest):
    verified=verify_scoring_assets(assets)
    metadata=assets/'source/VBench/vbench/VBench_full_info.json'
    if sha(metadata)!=manifest['official_metadata_sha256']:raise ValueError('Original official metadata differs from frozen generation inputs')
    files={}
    for directory in (assets/'source/VBench',assets/'source/PyIQA'):
        for path in directory.rglob('*.py'):files[str(path.relative_to(assets))]=sha(path)
    for name in ('wan_stage4.py','wan_budget.py','wan_semantic.py','wan_semantic_score.py','wan_paired.py','vbench.py'):
        files['quality/'+name]=sha(pathlib.Path(__file__).with_name(name))
    return dict(assets=verified,scorer_source_sha256=files,official_metadata_sha256=sha(metadata),
        source_manifest_sha256=sha(assets/'source-manifest.json'),sidecar_sha256=sidecar_sha(),
        torch_version=torch.__version__,scoring_image='harp-ae-gpu:quality-v0.4')

def verified_score_cache(out,binding):
    receipt=_load(out/'score-execution.json');report=_load(out/'scores.json')
    if receipt.get('status')!='complete' or report.get('status')!='complete' or receipt.get('binding')!=binding or report.get('binding')!=binding:raise ValueError('Cached calibration score source/assets/input binding differs')
    if 'scores.json' not in receipt.get('artifacts',{}):raise ValueError('Missing raw scores artifact SHA')
    for name,digest in receipt['artifacts'].items():
        file=out/name
        if not file.resolve().is_relative_to(out.resolve()) or sha(file)!=digest:raise ValueError('Cached calibration scorer artifact bytes differ')
    rows=report['dimension_scores']
    if set(rows)!=set(QUALITY_DIMS)|{'multiple_objects'}:raise ValueError('Incomplete eight-dimension calibration scores')
    for row in rows.values():
        checked_dimension_result(row['score'],row['per_video'],row['eligible_prompts'])
        if row['eligible_prompts']!=1 or len(row['per_video'])!=1:raise ValueError('Calibration score is not one matched video')
    return {d:row['score'] for d,row in rows.items()}

class Generator:
    def __init__(self,out,protocol,manifest,resume=True):
        self.out=out;self.protocol=protocol;self.manifest=manifest;self.pipeline=None
        self.resume=resume;self.expected_source=manifest.get('generator_source_sha256',current_generator_source(pathlib.Path('/opt/harp')))
        self.index=_load(out/'generation-index.json') if (out/'generation-index.json').exists() else dict(status='running',frames=protocol.frames,steps=50,
            expected_heldout_videos_per_mode=22,unique_heldout_prompts=11,cases=[])
        self.ledger=out/'run-ledger.jsonl'
    def register(self,record):
        previous=[r for r in self.index['cases'] if (r['input_id'],r['label'])==(record['input_id'],record['label'])]
        if len(previous)>1:raise ValueError('Duplicate generation-index key')
        if previous:
            if previous[0]['run_sha256']!=record['run_sha256'] or previous[0]['run_directory']!=record['run_directory']:raise ValueError('Existing index points at different bytes')
        else:self.index['cases'].append(record)
        write_json(self.out/'generation-index.json',self.index)
    def execute(self,item,label,mode,*,survey=None,probe=None,table=None,kind='control',split='calibration'):
        from harp import wan_cli
        import wan
        folder=self.out/'runs'/item['id']/label;folder.parent.mkdir(parents=True,exist_ok=True)
        if folder.exists():
            if not self.resume:raise ValueError('Existing output requires verified resume')
            complete=(folder/'run.json').exists() and (folder/'stage4-execution.json').exists() and _load(folder/'run.json').get('status')=='complete' and _load(folder/'stage4-execution.json').get('status')=='complete'
            if complete:
                run=verified_case(folder,item,mode,self.protocol,table,self.expected_source)
                if survey is not None:
                    execution=_load(folder/'stage4-execution.json')
                    for name in ('survey-sites.json','survey-selected.pt'):
                        if execution['artifacts'].get(name)!=sha(folder/name):raise ValueError('Cached survey artifact differs')
                    restore_survey(survey,folder)
                if probe is not None:restore_probe(probe,run)
                self.register(dict(input_id=item['id'],split=split,mode=mode,label=label,kind=kind,run_directory=str(folder),run_sha256=sha(folder/'run.json'),
                    metadata=item['metadata'],initial_noise_sha256=run['initial_noise_sha256'],budget_table_sha256=sha(table) if table else None,verified_cache_reuse=True))
                with (self.out/'resume-ledger.jsonl').open('a') as stream:stream.write(json.dumps(dict(input_id=item['id'],label=label,run_sha256=sha(folder/'run.json'),event='verified_complete_reuse'))+'\n')
                print('REUSED VERIFIED',item['id'],label,flush=True);return folder,run
            preserve_incomplete(self.out,folder)
        factory=RuntimeFactory(self.protocol,survey,probe);counter=dict(status='running',completed_model_calls=0,completed_transformer_blocks=0,
            mode=mode,input_id=item['id'],sidecar_sha256=sidecar_sha(),budget_table_sha256_before=sha(table) if table else None)
        handles=[];original_ctor=wan.WanT2V
        def attach(pipeline):
            if len(pipeline.model.blocks)!=40:raise ValueError('Requires Wan14B40 blocks')
            def after_model(*args):
                counter['completed_model_calls']+=1
                write_json(folder/'stage4-execution.json',counter)
            handles.append(pipeline.model.register_forward_hook(after_model))
            def after_block(*args):counter['completed_transformer_blocks']+=1
            for block in pipeline.model.blocks:handles.append(block.register_forward_hook(after_block))
            return pipeline
        def create(*args,**kwargs):return attach(original_ctor(*args,**kwargs))
        argv=['--mode',mode,'--frames',str(self.protocol.frames),'--steps','50','--seed',str(item['seed']),'--prompt',item['prompt'],
              '--attention-backend','masked_chunked' if mode in {'residual','combined'} else 'selected','--attention-query-chunk','128','--output',str(folder)]
        if table is not None:argv+=['--budget-table',str(table)]
        started=time.perf_counter();wan.WanT2V=create
        try:
            if self.pipeline is not None:attach(self.pipeline)
            with patched_runtime(factory):self.pipeline=wan_cli.run(wan_cli.arguments(argv),self.pipeline)
            if len(factory.created)!=2:raise ValueError('Trajectory did not create two isolated CFG runtimes')
            if counter['completed_model_calls']!=100 or counter['completed_transformer_blocks']!=4000:raise ValueError('Incomplete observed execution')
            run=_load(folder/'run.json');validate_run(run,self.protocol,_noise(folder))
            if run['source_sha256']!=self.expected_source:raise ValueError('Loaded generator source differs from frozen snapshot')
            if probe is not None:
                if probe.measurement['hits']!=1 or 'intervention' in run['runtime']['runtimes']['branch1:sample0']:raise ValueError('Probe hit/branch isolation failed')
            if survey is not None:
                write_json(folder/'survey-sites.json',survey.sites);torch.save(survey.selected,folder/'survey-selected.pt')
            counter.update(budget_table_sha256_after=sha(table) if table else None,sidecar_sha256_after=sidecar_sha())
            if counter['budget_table_sha256_before']!=counter['budget_table_sha256_after'] or counter['sidecar_sha256']!=counter['sidecar_sha256_after']:raise ValueError('Table/sidecar changed during actual generation')
            names=['run.json','initial-noise.pt','latent.pt','sample.mp4']+(['survey-sites.json','survey-selected.pt'] if survey is not None else [])
            counter.update(status='complete',artifacts={name:sha(folder/name) for name in names});write_json(folder/'stage4-execution.json',counter)
            record=dict(input_id=item['id'],split=split,mode=mode,label=label,kind=kind,run_directory=str(folder),run_sha256=sha(folder/'run.json'),
                metadata=item['metadata'],initial_noise_sha256=run['initial_noise_sha256'],budget_table_sha256=sha(table) if table else None,elapsed_seconds=time.perf_counter()-started)
            self.register(record)
            with self.ledger.open('a',encoding='utf-8') as stream:stream.write(json.dumps(record)+'\n')
            print('COMPLETE',item['id'],label,record['elapsed_seconds'],flush=True)
            return folder,run
        except BaseException as error:
            counter.update(status='failed',error_type=type(error).__name__,error=str(error))
            if folder.exists():write_json(folder/'stage4-execution.json',counter)
            raise
        finally:
            wan.WanT2V=original_ctor
            for handle in handles:handle.remove()

def _equal_reference(reference,other,protocol):
    a,b=_load(reference/'run.json'),_load(other/'run.json')
    validate_comparison(a,b,protocol,_noise(reference),_noise(other))
    if not torch.equal(_latent(reference),_latent(other)) or a['video_tensor_sha256']!=b['video_tensor_sha256']:raise ValueError('End-to-end repeat/zero control not exact')

def _save_tables(out,identity,proposal):
    paths={}
    for mode in ('planeclear','combined'):
        path=out/'tables'/(proposal['id']+'-'+mode+'.json');path.parent.mkdir(exist_ok=True)
        table=BudgetTable(table_identity(identity,mode),proposal['allowances'],dict(policy=proposal['policy'],G=None,cycle_optimal=False,frozen=False,
            measurement_branch=0,deployment_branch_multiplier=2,short_frames=identity['frames']))
        immutable_json(path,dict(identity=table.identity,allowances=table.allowances,provenance=table.provenance));paths[mode]=str(path)
    return paths

def calibrate(out,protocol,manifest):
    generator=Generator(out,protocol,manifest);item=manifest['inputs']['calibration'][0]
    survey=WanSurvey(item['id']);ref,ref_run=generator.execute(item,'residual','residual',survey=survey,kind='reference')
    immutable_json(out/'survey.json',survey.sites)
    repeat,_=generator.execute(item,'reference-repeat','residual',kind='repeat');_equal_reference(ref,repeat,protocol)
    targets=[]
    for primary in PRIMARY_TARGETS:
        options=[primary] if primary['site']!='v' else [t for t in SURVEY_TARGETS if t['site']=='v']
        picked=next((t for t in options if survey.sites.get(key_of(t),{}).get('status')=='sampled'),None)
        if picked is None:raise ValueError('Required real-mask target missing: '+key_of(primary)+'; retain noC zero and extend a new predeclared target plan')
        targets.append(picked)
    immutable_json(out/'selected-targets.json',targets)
    dense,_=generator.execute(item,'dense','dense',kind='dense')
    zero=proposals({})[0];zero_paths=_save_tables(out,ref_run['identity'],zero)
    pc0,_=generator.execute(item,'planeclear-zero','planeclear',table=zero_paths['planeclear']);_equal_reference(dense,pc0,protocol)
    cb0,_=generator.execute(item,'combined-zero','combined',table=zero_paths['combined']);_equal_reference(ref,cb0,protocol)
    observations=[];zref=_latent(ref).double()
    for target in targets:
        key=key_of(target);row=survey.sites[key];sample=survey.selected[key];index=row['actual_slice']
        zero_probe=WanProbe(target,0,index,sample);zp,_=generator.execute(item,'probe-'+key.replace(':','-')+'-zero','residual',probe=zero_probe,kind='zero_probe');_equal_reference(ref,zp,protocol)
        known={row['input_slice_sha256']:(0.,str(zp))}
        for candidate in row['candidates']:
            committed=candidate['committed_slice_sha256'];lam=candidate['allowance']
            if committed in known:error,path=known[committed];method='same_committed_bits_reuse_from_actual_zero_or_nonzero_replay'
            else:
                probe=WanProbe(target,lam,index,sample);path,_=generator.execute(item,'probe-'+key.replace(':','-')+f'-lambda-{lam:g}','residual',probe=probe,kind='nonzero_probe')
                validate_comparison(ref_run,_load(path/'run.json'),protocol,_noise(ref),_noise(path))
                if abs(probe.measurement['cleared_mass']-candidate['cleared_mass'])>1e-12:raise ValueError('Actual C differs from CPU committed mask')
                error=float((_latent(path).double()-zref).norm()/zref.norm());path=str(path);known[committed]=(error,path);method='actual_full_suffix_replay'
            observations.append(dict(key=key,input_id=item['id'],allowance=lam,cleared_mass=candidate['cleared_mass'],final_error=error,
                slices_at_site=row['slices_at_site'],actual_slice=index,input_slice_sha256=row['input_slice_sha256'],committed_slice_sha256=committed,
                method=method,evaluation_source=path,G=None,measurement_branch=0,deployment_branch_multiplier=2))
            write_json(out/'observations-progress.json',observations)
    immutable_json(out/'observations.json',observations)
    sites=risk_summary(observations);policies=proposals(sites);immutable_json(out/'risk-curves.json',sites)
    for policy in policies:
        policy['tables']=_save_tables(out,ref_run['identity'],policy)
        if any(policy['allowances'].values()):
            path,_=generator.execute(item,'candidate-'+policy['id'],'combined',table=policy['tables']['combined'],kind='candidate')
            policy['calibration_run_directory']=str(path)
    immutable_json(out/'candidates.json',policies)
    immutable_json(out/'calibration-complete.json',dict(status='complete',awaiting='calibration eight-dimension scoring/selection',real_mask_sites=len(targets),G=None,
        zero_controls_exact=True,repeat_exact=True,unmeasured_sites_zero=True,formal100_20=False))

def score_calibration_case(folder,item,assets,out,protocol,context):
    """Single custom7 + original legal semantic metadata; no edited official files."""
    import numpy as np
    from vbench import utils
    utils.CACHE_DIR=str(assets/'vbench-cache');run=_load(folder/'run.json');validate_run(run,protocol,_noise(folder))
    binding=dict(context=context,input_id=item['id'],metadata=item['metadata'],run_sha256=sha(folder/'run.json'),
        generation_receipt_sha256=sha(folder/'stage4-execution.json'),video_sha256=sha(folder/'sample.mp4'),frames=protocol.frames)
    if binding['video_sha256']!=run['video_file_sha256']:raise ValueError('Actual scoring input video changed')
    if out.exists():
        if (out/'score-execution.json').exists() and _load(out/'score-execution.json').get('status')=='complete':
            print('REUSED VERIFIED CALIBRATION SCORES',out,flush=True);return verified_score_cache(out,binding)
        preserve_incomplete(out.parent.parent,out)
    out.mkdir(parents=True,exist_ok=False);alias=out/canonical_video_name(item['prompt'],item['seed']);shutil.copyfile(folder/'sample.mp4',alias)
    if sha(alias)!=run['video_file_sha256']:raise ValueError('Scoring video bytes changed')
    import cv2
    capture=cv2.VideoCapture(str(alias));decoded=0
    while True:
        ok,frame=capture.read()
        if not ok:break
        if frame.shape!=(480,832,3):raise ValueError('Video dimensions changed')
        decoded+=1
    capture.release()
    if decoded!=protocol.frames:raise ValueError('Official scorer input frame count differs')
    generic,semantic=generic_metadata(alias,item['metadata']);write_json(out/'generic.json',generic);write_json(out/'semantic.json',semantic)
    dimensions=list(QUALITY_DIMS)+[d for d in item['metadata']['dimension'] if d not in QUALITY_DIMS]
    if dimensions!=list(QUALITY_DIMS)+['multiple_objects']:raise ValueError('Calibration protocol expects original multiple_objects plus custom7')
    report=dict(status='running',dimension_scores={},total_score=None,frames=protocol.frames,official_scorer_sha256={},binding=binding,
        scorer_sampling_note='Official uniform16 may repeat original short-video indices; input video is unmodified')
    device=torch.device('cuda')
    for dimension in dimensions:
        random.seed(0);np.random.seed(0);torch.manual_seed(0)
        if dimension=='imaging_quality':load_musiq_namespace(assets/'source/PyIQA')
        module=importlib.import_module('vbench.'+dimension)
        report['official_scorer_sha256'][dimension]=sha(module.__file__)
        if dimension=='subject_consistency':
            model=torch.hub.load(str(assets/'vbench-cache/dino_model/facebookresearch_dino_main'),'dino_vitb16',source='local',pretrained=False)
            model.load_state_dict(torch.load(assets/'vbench-cache/dino_model/dino_vitbase16_pretrain.pth',map_location='cpu',weights_only=True),strict=True);model.eval().to(device)
            with torch.no_grad():score,details=module.subject_consistency(model,[str(alias)],device,False)
            del model
        elif dimension=='multiple_objects':
            from vbench.third_party.grit_model import DenseCaptioning
            with inference_guard(DenseCaptioning),torch.no_grad():score,details=module.compute_multiple_objects(str(out/'semantic.json'),device,utils.init_submodules([dimension],local=True,read_frame=False)[dimension])
        else:
            with torch.no_grad():score,details=getattr(module,'compute_'+dimension)(str(out/'generic.json'),device,utils.init_submodules([dimension],local=True,read_frame=False)[dimension])
        row=checked_dimension_result(score,details,1)
        if len(row['per_video'])!=1:raise ValueError('Calibration scoring must be exactly one video')
        report['dimension_scores'][dimension]=row;write_json(out/'scores.json',report);torch.cuda.empty_cache()
    report.update(status='complete',run_sha256=sha(folder/'run.json'),video_sha256=sha(alias),source_sha256=sidecar_sha());write_json(out/'scores.json',report)
    write_json(out/'score-execution.json',dict(status='complete',binding=binding,
        artifacts={name:sha(out/name) for name in ('scores.json','generic.json','semantic.json',alias.name)}))
    return verified_score_cache(out,binding)

def score_calibration(out,protocol,manifest,assets):
    if _load(out/'calibration-complete.json')['status']!='complete':raise ValueError('Calibration replay guards did not complete')
    context=scoring_context(assets,manifest);immutable_json(out/'calibration-scoring-context.json',context)
    sys.path.insert(0,str(assets/'source/VBench'));item=manifest['inputs']['calibration'][0];base=out/'runs'/item['id'];scores=out/'calibration-scores'
    generator=Generator(out,protocol,manifest)
    for mode in ('dense','residual'):verified_case(base/mode,item,mode,protocol,None,generator.expected_source)
    dense=score_calibration_case(base/'dense',item,assets,scores/'dense',protocol,context)
    score_calibration_case(base/'residual',item,assets,scores/'residual',protocol,context)
    candidates=_load(out/'candidates.json');tested=[]
    for candidate in candidates:
        if not any(candidate['allowances'].values()):continue
        folder=pathlib.Path(candidate['calibration_run_directory'])
        verified_case(folder,item,'combined',protocol,candidate['tables']['combined'],generator.expected_source)
        candidate['scores']=score_calibration_case(folder,item,assets,scores/candidate['id'],protocol,context);tested.append(candidate)
    selection=select_policy(tested,dense,item['id'])
    policy=selection['selected'] or selection['exploratory']
    if policy is not None:
        original=dict(policy['tables']);updated={}
        for mode,path in original.items():
            table=BudgetTable.load(path)
            table.provenance.update(frozen=selection['selected'] is not None,frozen_development_policy=selection['selected'] is not None,paper_policy=False,G=None,
                calibration_id=item['id'],heldout_used_for_selection=False,rule='worst original raw-dimension relative loss against matched dense <=.02',
                calibration_scores_sha256=sha(scores/policy['id']/'scores.json'))
            target=out/'tables'/(("selected-" if selection['selected'] else 'exploratory-')+policy['id']+'-'+mode+'.json')
            immutable_json(target,dict(identity=table.identity,allowances=table.allowances,provenance=table.provenance));updated[mode]=str(target)
        policy['candidate_tables']=original;policy['tables']=updated
        policy['frozen']=selection['selected'] is not None
    selection['binding']=dict(manifest_sha256=sha(out/'manifest.json'),candidates_sha256=sha(out/'candidates.json'),
        scoring_context_sha256=sha(out/'calibration-scoring-context.json'),
        scores={p.parent.name:sha(p) for p in scores.glob('*/scores.json')},
        score_execution={p.parent.name:sha(p) for p in scores.glob('*/score-execution.json')},
        tables={mode:dict(path=path,sha256=sha(path)) for mode,path in policy['tables'].items()} if policy else {})
    immutable_json(out/'selection.json',selection)

def verify_selection(out,decision):
    binding=decision['binding']
    for name,key in [('manifest.json','manifest_sha256'),('candidates.json','candidates_sha256'),('calibration-scoring-context.json','scoring_context_sha256')]:
        if sha(out/name)!=binding[key]:raise ValueError('Frozen selection input changed: '+name)
    for label,digest in binding['scores'].items():
        if sha(out/'calibration-scores'/label/'scores.json')!=digest or sha(out/'calibration-scores'/label/'score-execution.json')!=binding['score_execution'][label]:raise ValueError('Frozen calibration scores/receipts changed')
    for mode,row in binding['tables'].items():
        if sha(row['path'])!=row['sha256'] or BudgetTable.load(row['path']).identity['mode']!=mode:raise ValueError('Frozen deployment table changed')
    if decision.get('heldout_used_for_selection') is not False:raise ValueError('Heldout selection leakage')

def heldout(out,protocol,manifest):
    decision=_load(out/'selection.json');policy=decision['selected'] or decision['exploratory']
    if policy is None:raise ValueError('No real nonzero policy; cannot claim all-zero optimization')
    verify_selection(out,decision)
    generator=Generator(out,protocol,manifest);cal=manifest['inputs']['calibration'][0]
    # Preserve candidate run provenance; copy metadata index entry rather than regenerate it.
    cal_combo=pathlib.Path(policy['calibration_run_directory'])
    cal_run=verified_case(cal_combo,cal,'combined',protocol,policy['candidate_tables']['combined'],generator.expected_source)
    if BudgetTable.load(policy['candidate_tables']['combined']).allowances!=BudgetTable.load(policy['tables']['combined']).allowances:raise ValueError('Selected policy allowances differ from actual calibration candidate')
    generator.register(dict(input_id=cal['id'],split='calibration',mode='combined',label='combined',kind='selected_candidate_reuse',
        run_directory=str(cal_combo),run_sha256=sha(cal_combo/'run.json'),metadata=cal['metadata'],
        budget_table_sha256=sha(policy['candidate_tables']['combined']),policy_table_sha256=sha(policy['tables']['combined']),initial_noise_sha256=cal_run['initial_noise_sha256']))
    generator.execute(cal,'planeclear','planeclear',table=policy['tables']['planeclear'],kind='selected_policy')
    for item in manifest['inputs']['heldout']:
        runs={}
        for mode in ('dense','residual','planeclear','combined'):
            path,run=generator.execute(item,mode,mode,table=policy['tables'].get(mode),kind='heldout',split='heldout');runs[mode]=(path,run)
            if mode!='dense':validate_comparison(runs['dense'][1],run,protocol,_noise(runs['dense'][0]),_noise(path))
    generator.index.update(status='complete',selected_development_policy=decision['selected'] is not None,exploratory_only=decision['selected'] is None,
        heldout_never_used_for_selection=True);write_json(out/'generation-index.json',generator.index)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['calibration','score-calibration','heldout'],required=True);parser.add_argument('--root',required=True)
    parser.add_argument('--frames',type=int,default=9);parser.add_argument('--gpu-deterministic',type=int,choices=[0,1],default=1);parser.add_argument('--resume',action='store_true');a=parser.parse_args()
    out=pathlib.Path(a.root);protocol=Protocol(a.frames);root=pathlib.Path('/opt/harp');assets=pathlib.Path('/quality') if pathlib.Path('/quality').exists() else root/'assets/quality'
    torch.set_num_threads(4)
    if a.phase!='calibration' and not (out/'manifest.json').exists():raise ValueError('Calibration frozen manifest must exist first')
    declared_det=bool(a.gpu_deterministic) if a.phase=='calibration' else _load(out/'manifest.json')['gpu_deterministic_algorithms']
    if a.phase=='heldout' and declared_det!=bool(a.gpu_deterministic):raise ValueError('Heldout deterministic mode differs from calibration')
    manifest=prepare_manifest(out,protocol,assets/'source/VBench/vbench/VBench_full_info.json',root,a.resume,declared_det)
    state=dict(status='running',phase=a.phase,frames=a.frames,sidecar_sha256=sidecar_sha());progress=out/(a.phase+'-phase.json')
    if progress.exists():
        history=out/'phase-history';history.mkdir(exist_ok=True);shutil.copyfile(progress,history/(a.phase+'-'+str(time.time_ns())+'.json'))
    write_json(progress,state)
    try:
        if a.phase!='score-calibration':
            import os
            if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('Set fixed CuBLAS workspace before model use')
            torch.use_deterministic_algorithms(bool(a.gpu_deterministic));torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
            state['actual_gpu_deterministic_algorithms']=torch.are_deterministic_algorithms_enabled();write_json(progress,state)
        {'calibration':lambda:calibrate(out,protocol,manifest),'score-calibration':lambda:score_calibration(out,protocol,manifest,assets),'heldout':lambda:heldout(out,protocol,manifest)}[a.phase]()
        state['status']='complete';write_json(progress,state)
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__,error=str(error));write_json(progress,state);raise

if __name__=='__main__':main()
