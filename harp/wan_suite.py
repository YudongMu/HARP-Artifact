"""Run paired development checks with one verified model load and one code snapshot."""
import argparse
import json
import pathlib
import torch
from .wan_cli import arguments, run
from .protocol import schedule_digest


def cases(output_root, *, frames=5, steps=6, seed=0, prompt='A cat walks across a sunlit room.'):
    result={}
    for label,mode,allowance in [('official','official',None),('dense','dense',None),
          ('residual','residual',None),('combined-zero','combined',0),
          ('planeclear-zero','planeclear',0),('combined-nonzero','combined',.005)]:
        argv=['--mode',mode,'--frames',str(frames),'--steps',str(steps),'--seed',str(seed),
              '--prompt',prompt,'--output',str(pathlib.Path(output_root)/label)]
        if allowance is not None:
            argv+=['--experimental-uniform','--allowance',str(allowance)]
        result[label]=arguments(argv)
    return result


def main():
    p=argparse.ArgumentParser(description='Development parity suite, not FID/VBench')
    p.add_argument('--output-root',required=True)
    p.add_argument('--frames',type=int,default=5)
    p.add_argument('--steps',type=int,default=6)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--prompt',default='A cat walks across a sunlit room.')
    a=p.parse_args()
    rows=cases(a.output_root,frames=a.frames,steps=a.steps,seed=a.seed,prompt=a.prompt)
    root=pathlib.Path(a.output_root)
    root.mkdir(parents=True,exist_ok=False)
    pipeline=None;runs={};latents={};comparisons={}
    for label,args in rows.items():
        pipeline=run(args,pipeline)
        runs[label]=json.loads((pathlib.Path(args.output)/'run.json').read_text(encoding='utf-8'))
        latents[label]=torch.load(pathlib.Path(args.output)/'latent.pt',weights_only=True)
        reference=runs['official']
        for field in ('initial_noise_sha256','seed','prompt','source_sha256'):
            if runs[label][field]!=reference[field]:raise RuntimeError('Unpaired suite field: '+field)
        if schedule_digest(runs[label]['schedule'])!=schedule_digest(reference['schedule']):
            raise RuntimeError('Unpaired effective schedule')
        if runs[label]['identity']['checkpoint_sha256']!=reference['identity']['checkpoint_sha256']:
            raise RuntimeError('Unpaired actual checkpoint')
        base='residual' if label.startswith('combined') else 'official'
        z,b=latents[label].double(),latents[base].double()
        comparisons[label]={'reference':base,'exact_equal':torch.equal(z,b),'latent_relative_l2':float((z-b).norm()/b.norm())}
        if label in ('dense','planeclear-zero','combined-zero') and not comparisons[label]['exact_equal']:
            raise RuntimeError('Zero-approximation parity failed: '+label)
        if label=='combined-nonzero' and not sum(rt['plane_totals']['cleared_planes'] for rt in runs[label]['runtime']['runtimes'].values()):
            raise RuntimeError('Nonzero suite case did not clear planes')
        (root/'summary.json').write_text(json.dumps({'scope':'development functional checks only; no quality metrics',
            'completed_cases':list(runs),'comparisons':comparisons,'quality_metrics_evaluated':False},indent=2),encoding='utf-8')
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':
    main()
