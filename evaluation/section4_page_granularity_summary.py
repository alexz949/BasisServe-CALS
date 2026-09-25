"""Section 4, experiment 3: page-granularity tables and plots from the no-sink page_granularity_sweep.py JSON (3A) and the
saved RULER-128K records of the page-size refits (3B).
3A (--diag): oracle and router retained mass, router page recall and support dispersion for every page size and budget;
    `--fixed <bank name>` is the router scored at every page size with fixed factors, and a bank named `p<P>` (if present) gives
    the router refit at page size P.
3B (--arms p1=DIR p4=DIR ... [full=DIR], --tasks): per-task RULER scores on the frozen prompts (matched by index + input hash,
    all prompts of the listed tasks); --averages name=<summary.txt or run dir> records the 1100-prompt RULER averages when available.
usage: section4_page_granularity_summary.py --diag sweep.json --fixed p4 --arms p1=DIR p4=DIR p8=DIR p32=DIR full=DIR
       --prompts <run>/prompts.json --tasks niah_multikey_2,fwe --output DIR [--notes p32="..."]"""
import argparse
import json
from pathlib import Path
import random
import shlex
import subprocess
import sys

from evaluation.v96kl_common import read_json, sha256, write_json


def load_records(directory, frozen, tasks):
    out, protocols = {}, set()
    for path in sorted(Path(directory).glob('sample_*.json')):
        r = json.loads(path.read_text())
        s = r['sample']
        if s['task'] not in tasks:
            continue
        f = frozen[s['index']]
        assert s['task'] == f['task'] and s['ordinal'] == f['ordinal'] and s['input_sha256'] == f['input_sha256'], path
        assert r['status'] == 'complete'
        out[s['index']] = dict(score=float(r['result']['score']), task=s['task'])
        protocols.add((r['protocol']['identity_sha256'], r['protocol']['format'], json.dumps(r['protocol'].get('ours'), sort_keys=True)))
    return out, protocols


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--diag', type=Path, required=True)
    p.add_argument('--fixed', default='p4', help='bank name in the sweep JSON scored at every page size with fixed factors')
    p.add_argument('--arms', nargs='+', required=True, help='name=<dir of sample_*.json>; names p1, p4, p8, p32, full')
    p.add_argument('--prompts', type=Path, required=True)
    p.add_argument('--tasks', default='niah_multikey_2,fwe')
    p.add_argument('--notes', nargs='*', default=[], help='name="protocol note" shown in the downstream table')
    p.add_argument('--bootstrap', type=int, default=4000)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    random.seed(0)
    diag = read_json(args.diag)
    assert diag['status'] == 'complete' and diag['sink'] == 0
    pages, budgets = diag['page_sizes'], diag['budgets']
    fixed = f'router:{args.fixed}'
    table_a = {}
    for P in pages:
        row = {}
        for B in budgets:
            row[f'B{B}'] = dict(oracle_mass=diag['overall'][f'mass|oracle|P{P}|B{B}'], router_mass=diag['overall'][f'mass|{fixed}|P{P}|B{B}'],
                                router_recall=diag['overall'][f'recall|{fixed}|P{P}|B{B}'], oracle_dispersion=diag['overall'][f'dispersion|oracle|P{P}|B{B}'],
                                router_dispersion=diag['overall'][f'dispersion|{fixed}|P{P}|B{B}'])
            refit = f'router:p{P}'
            if f'mass|{refit}|P{P}|B{B}' in diag['overall']:
                row[f'B{B}']['refit_router_mass'] = diag['overall'][f'mass|{refit}|P{P}|B{B}']
                row[f'B{B}']['refit_router_recall'] = diag['overall'][f'recall|{refit}|P{P}|B{B}']
        table_a[P] = row
    diagnostic = dict(status='complete', format='basisserve.section4.page_granularity_diagnostic.v1', source=str(args.diag), source_sha256=sha256(args.diag),
                      identity_sha256=diag['identity_sha256'], model_config_sha256=diag.get('model_config_sha256'), checkpoint_manifest_sha256=diag.get('checkpoint_manifest_sha256'),
                      windows=diag['windows'], windows_sha256=diag['windows_sha256'], sequence_length=diag['sequence_length'], queries_per_window=diag['queries_per_window'],
                      query_positions=diag.get('query_positions'), layers=diag['layers'], sink=0, recent=diag['recent'], segment=diag.get('segment'),
                      fixed_bank=args.fixed, banks=diag['banks'], page_sizes=pages, budgets=budgets, table=table_a, per_layer=diag['per_layer'],
                      gpu=diag.get('gpu'), dtype=diag.get('dtype'), sweep_command=diag.get('command'))
    prompts = read_json(args.prompts)
    frozen = {r['index']: r for r in prompts['rows']}
    tasks = args.tasks.split(',')
    arms = dict(a.split('=', 1) for a in args.arms)
    notes = dict(a.split('=', 1) for a in args.notes)
    scores, protocols = {}, {}
    for name, directory in arms.items():
        scores[name], protocols[name] = load_records(directory, frozen, tasks)
    ids = sorted(set.intersection(*(set(s) for s in scores.values())))
    assert ids and all(len(s) == len(ids) for s in scores.values()), {n: len(s) for n, s in scores.items()}
    identities = {proto[0] for protos in protocols.values() for proto in protos}
    assert len(identities) == 1 and next(iter(identities)) == diag['identity_sha256'], identities
    per_task = {name: {t: 100 * sum(s[i]['score'] for i in ids if s[i]['task'] == t) / sum(1 for i in ids if s[i]['task'] == t) for t in tasks} for name, s in scores.items()}
    paired = {}
    for name in arms:
        for other in arms:
            if name >= other or 'full' in (name, other):
                continue
            for t in tasks:
                sel = [i for i in ids if scores[name][i]['task'] == t]
                d = [scores[name][i]['score'] - scores[other][i]['score'] for i in sel]
                boots = sorted(100 * sum(random.choice(d) for _ in d) / len(d) for _ in range(args.bootstrap))
                paired[f'{name} - {other} | {t}'] = dict(delta=100 * sum(d) / len(d), ci95=[boots[int(0.025 * args.bootstrap)], boots[int(0.975 * args.bootstrap) - 1]],
                                                          wins=sum(x > 0 for x in d), losses=sum(x < 0 for x in d))
    downstream = dict(status='complete', format='basisserve.section4.page_granularity_downstream.v1', identity_sha256=next(iter(identities)),
                      prompts=str(args.prompts), prompts_sha256=sha256(args.prompts), data_sha256=prompts.get('data_sha256'), tasks=tasks, samples=len(ids), indices=ids,
                      arms={name: dict(source=arms[name], per_task=per_task[name], mean=sum(per_task[name].values()) / len(tasks), note=notes.get(name, ''),
                                       protocols=[dict(identity_sha256=a, format=b, ours=json.loads(c)) for a, b, c in sorted(protocols[name])]) for name in arms},
                      paired=paired, bootstrap=args.bootstrap, seed=0,
                      git_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]).stdout.strip(),
                      command=shlex.join(sys.argv))
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'diagnostic.json', diagnostic)
    write_json(args.output / 'downstream.json', downstream)
    order = [n for n in ('p1', 'p4', 'p8', 'p32') if n in arms] + [n for n in arms if n not in ('p1', 'p4', 'p8', 'p32')]
    lines = ['# Section 4 / Experiment 3: page granularity (Llama-3.1-8B-Instruct, V96, B16R16, no sink, recent 64)', '',
             f"3A: exact attention mass captured by sink 0 + recent 64 + B routed tokens in pages of P on the {len(diag['windows'])} held-out calibration windows "
             f"({diag['sequence_length']} tokens, {diag['queries_per_window']} queries per window, {len(diag['layers'])} layers). Oracle = exact-QK page selection; "
             f"router = the page-{args.fixed[1:]} B16R16 factors scored at every page size (fixed, no refit); refit = the B16R16 bank fitted at that page size. "
             "Page recall = fraction of the oracle's routed pages the router selects; dispersion = fraction of 256-token segments of the routable region touched by the routed support.", '',
             '## Table A: retained attention mass', '',
             '| Page | ' + ' | '.join(f'Router mass B{B} | Oracle mass B{B}' for B in budgets) + ' | ' + ' | '.join(f'Refit router mass B{B}' for B in budgets) + ' |',
             '|---:|' + '---:|' * (3 * len(budgets))]
    for P in pages:
        cells = [f"{100 * table_a[P][f'B{B}']['router_mass']:.2f}% | {100 * table_a[P][f'B{B}']['oracle_mass']:.2f}%" for B in budgets]
        cells += [f"{100 * table_a[P][f'B{B}']['refit_router_mass']:.2f}%" if 'refit_router_mass' in table_a[P][f'B{B}'] else '-' for B in budgets]
        lines.append(f'| {P} | ' + ' | '.join(cells) + ' |')
    lines += ['', '| Page | ' + ' | '.join(f'Router page recall B{B} | Router dispersion B{B} | Oracle dispersion B{B}' for B in budgets) + ' |', '|---:|' + '---:|' * (3 * len(budgets))]
    for P in pages:
        lines.append(f'| {P} | ' + ' | '.join(f"{100 * table_a[P][f'B{B}']['router_recall']:.1f}% | {100 * table_a[P][f'B{B}']['router_dispersion']:.1f}% | {100 * table_a[P][f'B{B}']['oracle_dispersion']:.1f}%" for B in budgets) + ' |')
    lines += ['', f"## Table B: RULER-128K on {', '.join(tasks)} ({len(ids)} prompts, identical across arms)", '',
              '| Page | ' + ' | '.join(tasks) + ' | mean | protocol |', '|---|' + '---:|' * (len(tasks) + 1) + '---|']
    for name in order:
        a = downstream['arms'][name]
        lines.append(f"| {name} | " + ' | '.join(f"{a['per_task'][t]:.1f}" for t in tasks) + f" | {a['mean']:.2f} | {a['note']} |")
    lines += ['', 'Paired per-task differences (bootstrap 95% CI, 100 prompts per task):', '']
    for k, v in paired.items():
        lines.append(f"- {k}: {v['delta']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}], win/loss {v['wins']}/{v['losses']}")
    lines += ['', 'Plots: `page_granularity_mass.pdf` (3A), `page_granularity_ruler.pdf` (3B). Exact values in `diagnostic.json` and `downstream.json`.', '']
    (args.output / 'summary.md').write_text('\n'.join(lines))
    plots(args.output, pages, budgets, table_a, order, downstream['arms'], tasks)
    print('\n'.join(lines[:24]))


def plots(output, pages, budgets, table_a, order, arms, tasks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(budgets), figsize=(4.2 * len(budgets), 3.4), sharey=False)
    axes = list(axes) if len(budgets) > 1 else [axes]
    for ax, B in zip(axes, budgets):
        ax.plot(pages, [100 * table_a[P][f'B{B}']['oracle_mass'] for P in pages], color='black', linestyle='--', marker='o', markersize=4, label='oracle (exact QK)')
        ax.plot(pages, [100 * table_a[P][f'B{B}']['router_mass'] for P in pages], color='tab:blue', marker='s', markersize=4, label='B16R16 router (fixed)')
        refit = [(P, 100 * table_a[P][f'B{B}']['refit_router_mass']) for P in pages if 'refit_router_mass' in table_a[P][f'B{B}']]
        if refit:
            ax.plot([x for x, _ in refit], [y for _, y in refit], color='tab:blue', linestyle=':', marker='^', markersize=4, label='B16R16 router (refit at P)')
        ax.set_xscale('log', base=2); ax.set_xticks(pages); ax.set_xticklabels(pages); ax.set_xlabel('page size P'); ax.set_title(f'B = {B} routed tokens', fontsize=10); ax.grid(alpha=0.3)
    axes[0].set_ylabel('retained attention mass (%)'); axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(output / 'page_granularity_mass.pdf'); plt.close(fig)
    page_arms = [n for n in order if n.startswith('p')]
    xs = [int(n[1:]) for n in page_arms]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for t, color, marker in zip(tasks, ('tab:red', 'tab:green', 'tab:purple', 'tab:orange'), ('o', 's', '^', 'D')):
        ax.plot(xs, [arms[n]['per_task'][t] for n in page_arms], color=color, marker=marker, markersize=4, label=t)
        if 'full' in arms:
            ax.axhline(arms['full']['per_task'][t], color=color, linestyle=':', linewidth=1)
    ax.set_xscale('log', base=2); ax.set_xticks(xs); ax.set_xticklabels(xs); ax.set_xlabel('page size P'); ax.set_ylabel('RULER-128K accuracy (%)'); ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, title='dotted: Full-K' if 'full' in arms else None, title_fontsize=8)
    fig.tight_layout(); fig.savefig(output / 'page_granularity_ruler.pdf'); plt.close(fig)


if __name__ == '__main__':
    main()
