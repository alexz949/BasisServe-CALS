"""Section 4, experiment 2: Base / residual component ablation table from saved RULER-128K records and the routing diagnostic.
Every arm is a `<run>/ours/evaluate` (or `full/evaluate`) directory of eval_llama_cal128_p1.py records on the same frozen prompts;
the prompts are matched by index and input_sha256 and restricted to ordinal < --max-ordinal of every task (the same subset for
every arm, including arms that were evaluated on all 1100 prompts). All arms must share the identity (compressed model) hash.
The routing diagnostic is a page_granularity_sweep.py JSON (--diag) whose `router:<name>` scorers are the same arms.
usage: section4_base_residual_summary.py --prompts <run>/prompts.json --arms b16r16=DIR r16=DIR r32=DIR b16=DIR [full=DIR]
       --dims b16r16=16:16 r16=0:16 r32=0:32 b16=16:0 --diag sweep.json --page-size 4 --budget 2048 --max-ordinal 30 --output DIR"""
import argparse
import json
from pathlib import Path
import random
import shlex
import subprocess
import sys

from evaluation.v96kl_common import read_json, sha256, write_json

ORDER = ('b16', 'b16sink', 'r16', 'b16r16', 'r32')
LABEL = {'b16': 'B16', 'b16sink': 'B16 + pinned first page (4-token sink)', 'r16': 'R16', 'b16r16': 'B16R16', 'r32': 'R32', 'full': 'Full-K (exact routing)'}


def load_records(directory, frozen, max_ordinal):
    out, protocols = {}, set()
    for path in sorted(Path(directory).glob('sample_*.json')):
        r = json.loads(path.read_text())
        s = r['sample']
        if s['ordinal'] >= max_ordinal:
            continue
        f = frozen[s['index']]
        assert s['task'] == f['task'] and s['ordinal'] == f['ordinal'] and s['input_sha256'] == f['input_sha256'], path
        assert r['status'] == 'complete'
        out[s['index']] = dict(score=float(r['result']['score']), task=s['task'], routing=r['result'].get('routing'),
                               seconds=r['result']['seconds'], peak_gib=r['result']['peak_gib'])
        protocols.add((r['protocol']['identity_sha256'], r['protocol']['format'], json.dumps(r['protocol'].get('ours'), sort_keys=True),
                       r['protocol'].get('prompts_sha256')))
    return out, protocols


def task_mean(scores, ids, tasks):
    by = {t: [] for t in tasks}
    for i in ids:
        by[scores[i]['task']].append(scores[i]['score'])
    means = {t: 100 * sum(v) / len(v) for t, v in by.items() if v}
    return sum(means.values()) / len(means), means


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--prompts', type=Path, required=True)
    p.add_argument('--arms', nargs='+', required=True, help='name=<dir of sample_*.json>')
    p.add_argument('--dims', nargs='+', required=True, help='name=<V-derived dims>:<K-derived dims>')
    p.add_argument('--diag', type=Path, required=True)
    p.add_argument('--extra-diag', nargs='*', default=[], help='name=<sweep json>:<scorer name>: diagnostic of an arm taken from another sweep (e.g. a pinned-sink control)')
    p.add_argument('--page-size', type=int, default=4)
    p.add_argument('--budget', type=int, default=2048)
    p.add_argument('--max-ordinal', type=int, default=30)
    p.add_argument('--bootstrap', type=int, default=4000)
    p.add_argument('--reference', default='b16r16', help='arm the paired differences are taken against')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    random.seed(0)
    prompts = read_json(args.prompts)
    frozen = {r['index']: r for r in prompts['rows']}
    tasks = list(dict.fromkeys(r['task'] for r in prompts['rows']))
    arms = dict(a.split('=', 1) for a in args.arms)
    dims = {k: tuple(int(x) for x in v.split(':')) for k, v in (a.split('=', 1) for a in args.dims)}
    scores, protocols = {}, {}
    for name, directory in arms.items():
        scores[name], protocols[name] = load_records(directory, frozen, args.max_ordinal)
    ids = sorted(set.intersection(*(set(s) for s in scores.values())))
    expected = len(tasks) * args.max_ordinal
    assert len(ids) == expected, f'{len(ids)} common prompts, expected {expected}: ' + str({n: len(s) for n, s in scores.items()})
    identities = {proto[0] for protos in protocols.values() for proto in protos}
    assert len(identities) == 1, identities                       # same compressed model in every arm
    diag = read_json(args.diag)
    assert diag['status'] == 'complete' and diag['identity_sha256'] == next(iter(identities))
    P, B = args.page_size, args.budget
    assert P in diag['page_sizes'] and B in diag['budgets'] and diag['sink'] == 0
    extra = {}
    for item in args.extra_diag:
        name, rest = item.split('=', 1)
        path, scorer = rest.rsplit(':', 1)
        extra[name] = (read_json(path), scorer, path)
        assert extra[name][0]['status'] == 'complete' and extra[name][0]['identity_sha256'] == next(iter(identities))
    rows = {}
    for name in arms:
        mean, per_task = task_mean(scores[name], ids, tasks)
        source_diag, key = (extra[name][0], f'router:{extra[name][1]}') if name in extra else (diag, f'router:{name}')
        diagnostic = None if name == 'full' else dict(retained_mass=source_diag['overall'][f'mass|{key}|P{P}|B{B}'], page_recall=source_diag['overall'][f'recall|{key}|P{P}|B{B}'],
                                                     dispersion=source_diag['overall'][f'dispersion|{key}|P{P}|B{B}'], sink=source_diag['sink'],
                                                     oracle_mass=source_diag['overall'][f'mass|oracle|P{P}|B{B}'], source=extra[name][2] if name in extra else str(args.diag))
        rows[name] = dict(label=LABEL.get(name, name), v_derived_dims=dims.get(name, (None, None))[0], k_derived_dims=dims.get(name, (None, None))[1],
                          ruler_average=mean, per_task=per_task, samples=len(ids), diagnostic=diagnostic, source=arms[name],
                          protocols=[dict(identity_sha256=a, format=b, ours=json.loads(c), prompts_sha256=d) for a, b, c, d in sorted(protocols[name])])
    oracle = dict(retained_mass=diag['overall'][f'mass|oracle|P{P}|B{B}'], dispersion=diag['overall'][f'dispersion|oracle|P{P}|B{B}'])
    paired = {}
    reference = args.reference
    assert reference in arms, reference
    for name in arms:
        if name == reference:
            continue
        def delta(sel):
            return task_mean(scores[name], sel, tasks)[0] - task_mean(scores[reference], sel, tasks)[0]
        d0 = delta(ids)
        boots = sorted(delta([random.choice(ids) for _ in ids]) for _ in range(args.bootstrap))
        wins = sum(scores[name][i]['score'] > scores[reference][i]['score'] for i in ids)
        losses = sum(scores[name][i]['score'] < scores[reference][i]['score'] for i in ids)
        paired[f'{name} - {reference}'] = dict(delta=d0, ci95=[boots[int(0.025 * args.bootstrap)], boots[int(0.975 * args.bootstrap) - 1]], wins=wins, losses=losses, ties=len(ids) - wins - losses)
    result = dict(status='complete', format='basisserve.section4.base_residual.v1', identity_sha256=next(iter(identities)),
                  prompts=str(args.prompts), prompts_sha256=sha256(args.prompts), data_sha256=prompts.get('data_sha256'),
                  subset=dict(rule=f'ordinal < {args.max_ordinal} of every task', tasks=tasks, samples=len(ids), indices=ids),
                  routing=dict(page_size=P, routed_budget=B, recent=64, sink=0, protocol_note='all arms: page-4 Page-Fisher fit on the same calibration windows, 2048 routed + recent 64, exact selected-K attention'),
                  diagnostic=dict(source=str(args.diag), windows=diag['windows'], windows_sha256=diag['windows_sha256'], queries_per_window=diag['queries_per_window'],
                                  sequence_length=diag['sequence_length'], layers=len(diag['layers']), oracle=oracle, banks=diag['banks']),
                  arms=rows, paired=paired, bootstrap=args.bootstrap, seed=0,
                  git_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]).stdout.strip(),
                  command=shlex.join(sys.argv))
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'result.json', result)
    lines = ['# Section 4 / Experiment 2: Base / residual components (Llama-3.1-8B-Instruct, V96, page 4, 2048 routed + recent 64, no sink)', '',
             f"RULER-128K on the first {args.max_ordinal} prompts of each of the {len(tasks)} tasks ({len(ids)} prompts, identical across arms; matched by index and input hash). "
             f"Retained mass = exact attention mass captured by the selected support (sink 0 + recent 64 + {B} routed tokens in pages of {P}) on the {len(diag['windows'])} held-out "
             f"calibration windows ({diag['sequence_length']} tokens, {diag['queries_per_window']} queries per window, all layers); oracle (exact-QK page selection) = {100 * oracle['retained_mass']:.2f}%.", '',
             '| Method | V-derived dims | K-derived dims | Retained mass | Page recall | RULER Avg. |', '|---|---:|---:|---:|---:|---:|']
    for name in [n for n in ORDER if n in rows] + [n for n in rows if n not in ORDER]:
        r = rows[name]
        d = r['diagnostic']
        lines.append(f"| {r['label']} | {r['v_derived_dims'] if r['v_derived_dims'] is not None else '-'} | {r['k_derived_dims'] if r['k_derived_dims'] is not None else '-'} | "
                     f"{100 * d['retained_mass']:.2f}%{' (sink ' + str(d['sink']) + ', oracle ' + format(100 * d['oracle_mass'], '.2f') + '%)' if d['sink'] else ''} | {100 * d['page_recall']:.1f}% | {r['ruler_average']:.2f} |" if d else
                     f"| {r['label']} | - | - | {100 * oracle['retained_mass']:.2f}% (oracle) | 100% | {r['ruler_average']:.2f} |")
    lines += ['', f'Paired differences vs {LABEL.get(reference, reference)} (task-balanced mean, bootstrap 95% CI on the common prompts):', '']
    for k, v in paired.items():
        lines.append(f"- {k}: {v['delta']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}], win/loss/tie {v['wins']}/{v['losses']}/{v['ties']}")
    lines += ['', '## Per task', '', '| task | ' + ' | '.join(rows[n]['label'] for n in rows) + ' |', '|---|' + '---:|' * len(rows)]
    for t in tasks:
        lines.append(f'| {t} | ' + ' | '.join(f"{rows[n]['per_task'][t]:.1f}" for n in rows) + ' |')
    lines += ['', 'Exact values, protocols and prompt indices in `result.json`.', '']
    (args.output / 'summary.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:12]))


if __name__ == '__main__':
    main()
