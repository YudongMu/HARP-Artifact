"""Generate paired examples from the maintained DiT and Wan source-tree budgets."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path('/opt/harp')


def dit(out, count):
    from quality import dit_stage4 as stage

    source = ROOT / 'src/dit-original/harp/lambda.json'
    table = json.loads(source.read_text())
    items = stage.heldout_items()[::16][:count]
    names = ('risk-cap-0', 'source-budget')
    stage.CANDIDATES, stage.NONZERO = names, names[1:]
    stage.heldout_items = lambda: items
    tables = out / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    zero, chosen = (tables / (name+'-table.json') for name in names)
    if chosen.exists() and stage.file_sha(chosen) != stage.file_sha(source):
        raise ValueError('Budget changed for existing output root')
    if not chosen.exists():
        with chosen.open('xb') as stream:
            stream.write(source.read_bytes())
    control = dict(table, allowances=dict.fromkeys(table['allowances'], 0.0))
    if zero.exists():
        if json.loads(zero.read_text()) != control:
            raise ValueError('Zero control changed')
    else:
        stage.write_json(zero, control)
    protocol = out / 'source-budget.json'
    record = {'source': str(source), 'sha256': stage.file_sha(source), 'inputs': items}
    if protocol.exists() and json.loads(protocol.read_text()) != record:
        raise ValueError('Existing source-budget experiment differs')
    if not protocol.exists():
        stage.write_json(protocol, record)
    selection = out / 'selection.json'
    stage.freeze_selection(selection, {'candidate': names[1], 'exploratory': True,
        'calibration_passed': None, 'calibration_report': str(protocol),
        'calibration_report_sha256': stage.file_sha(protocol),
        'table_sha256': {n: stage.file_sha(p) for n, p in zip(names, (zero, chosen))}})
    stage.generate(SimpleNamespace(output=out/'heldout', phase='heldout', candidates=tables, selection=selection))


def wan(out, count):
    import torch
    from quality import wan_stage4 as stage

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    old = ROOT / 'results/wan-stage4-9f-v1'
    manifest = stage._load(old/'manifest.json')
    selection = stage._load(old/'selection.json')['selected']
    source = ROOT/'src/Wan2.1/harp/lambda.json'
    source_bytes = source.read_bytes()
    budget = json.loads(source_bytes)
    if budget['allowances'] != selection['allowances']:
        raise ValueError('Wan budget allowances differ from the frozen mixed policy')
    if stage.current_generator_source(ROOT) != manifest['generator_source_sha256']:
        raise ValueError('Use the frozen DiT/Wan reference view configured by the shell launcher')
    items = manifest['inputs']['heldout'][:count]
    out.mkdir(parents=True, exist_ok=True)
    stage.immutable_json(out/'source-budget.json', {
        'source': str(source), 'sha256': hashlib.sha256(source_bytes).hexdigest(), 'inputs': items,
    })
    # The runner records a validated deployment identity for each execution.
    tables = {}
    for mode in ('planeclear', 'combined'):
        path = out/'.runtime'/mode/'lambda.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        identity = stage.table_identity(budget['identity'], mode)
        stage.immutable_json(path, dict(budget, identity=identity))
        tables[mode] = path
    protocol = stage.Protocol(9)
    runner = stage.Generator(out, protocol, manifest)
    for item in items:
        ref_dir, ref = runner.execute(item, 'dense', 'dense', split='smoke')
        for mode in ('residual', 'planeclear', 'combined'):
            folder, run = runner.execute(item, mode, mode, table=tables.get(mode), split='smoke')
            stage.validate_comparison(ref, run, protocol, stage._noise(ref_dir), stage._noise(folder))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=('dit', 'wan'))
    parser.add_argument('output_name')
    parser.add_argument('--count', type=int, default=1, choices=range(1, 17))
    args = parser.parse_args()
    if not args.output_name.startswith('source-budget-') or not args.output_name.replace('-', '').isalnum():
        parser.error('use a fresh source-budget-<name> output directory')
    {'dit': dit, 'wan': wan}[args.model](ROOT/'results'/args.output_name, args.count)
