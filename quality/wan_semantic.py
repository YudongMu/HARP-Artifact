"""Frozen official-membership pilot covering all VBench-v1 dimensions."""
import copy,hashlib,json,pathlib

ALL_DIMENSIONS=('subject_consistency','background_consistency','temporal_flickering','motion_smoothness','dynamic_degree','aesthetic_quality','imaging_quality','object_class','multiple_objects','human_action','color','spatial_relationship','scene','appearance_style','temporal_style','overall_consistency')
PROMPTS=('a cat and a dog','a person swimming in ocean','A person is riding a bike','a person','a red bicycle','A beautiful coastal beach in spring, waves lapping on sand, Van Gogh style','A beautiful coastal beach in spring, waves lapping on sand, in super slow motion','Close up of grapes on a rotating table.','alley','a bicycle on the left of a car, front view','In a still frame, a stop sign')

def select_pilot(rows):
    selected=[]
    for index,prompt in enumerate(PROMPTS):
        source_index=next(i for i,r in enumerate(rows) if r['prompt_en']==prompt)
        selected.append({'id':f'wan-semantic-{index:02d}','source_index':source_index,'metadata':copy.deepcopy(rows[source_index]),
                         'seed':index if index<2 else 200000+index,
                         'reuse':('wan-paper-dense-v1' if index==0 else 'wan-motion-dense-v1' if index==1 else None)})
    return selected

def validate_selection(rows,selected):
    covered=set()
    for row in selected:
        if row['metadata']!=rows[row['source_index']]:raise ValueError('Official metadata changed')
        covered.update(row['metadata']['dimension'])
    if covered!=set(ALL_DIMENSIONS):raise ValueError('Incomplete official dimension coverage')
    if len({r['metadata']['prompt_en'] for r in selected})!=len(selected):raise ValueError('Duplicate prompt')

def main():
    import argparse
    from harp.wan_cli import arguments,run
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output-root',required=True);a=p.parse_args()
    plan=json.loads(pathlib.Path(a.plan).read_text());root=pathlib.Path(a.output_root);root.mkdir(exist_ok=False,parents=True)
    pipeline=None;completed=[]
    for item in plan['selection']:
        if item['reuse']:
            folder=root.parent/item['reuse']
            prior=json.loads((folder/'run.json').read_text())
            if prior['status']!='complete' or prior['prompt']!=item['metadata']['prompt_en'] or prior['seed']!=item['seed']:raise ValueError('Reuse identity mismatch')
        else:
            folder=root/item['id']
            args=arguments(['--mode','official','--frames','81','--steps','50','--seed',str(item['seed']),'--prompt',item['metadata']['prompt_en'],'--output',str(folder)])
            print('START',item['id'],item['metadata']['prompt_en'],flush=True)
            pipeline=run(args,pipeline)
        completed.append(dict(item,run_directory=str(folder)))
        (root/'completed.json').write_text(json.dumps({'status':'complete' if len(completed)==len(plan['selection']) else 'running','completed':completed,'total_score':None},indent=2))
    print('COMPLETE official-membership pilot generation',len(completed),flush=True)

if __name__=='__main__':main()
