"""Rebuild the E2E table and separate diagnostic profile inventory."""
import argparse
import csv
import json
from pathlib import Path
import statistics
from evaluation.prepare_kcuu_llama import OUTPUT


def summarize(root):
    plan = json.loads((root / 'experiment_plan.json').read_text())
    rows = []
    raw = {}
    for job in plan['formal']:
        path = Path(job['output'])
        result = json.loads(path.read_text()) if path.exists() else {}
        row = dict(model=job['model'], batch=job['batch'], arm=job['arm'],
                   status=result.get('status', 'not_run'), median_e2e_seconds=None,
                   speedup_vs_dense=None, hybrid_change_vs_ag_pct=None,
                   output_tokens_per_second=None, ttft_p50_ms=None, ttft_p95_ms=None,
                   peak_allocated_gib=None, max_decode_batch=None, preemptions=None,
                   kv_token_capacity_min_rank=None, raw_json=str(path))
        if result.get('status') == 'complete':
            assert not result['profile']
            assert result['source_sha256'] == plan['source_sha256'], path
            assert (result['prefill_tokens'], result['output_tokens']) == (8192, 128)
            runs = result['runs']; assert len(runs) == 3
            assert all(r['successful_requests'] == job['batch'] and r['total_output_tokens'] == job['batch']*128 for r in runs)
            row.update(median_e2e_seconds=statistics.median(r['wall_seconds'] for r in runs),
                output_tokens_per_second=statistics.median(r['output_tokens_per_second'] for r in runs),
                ttft_p50_ms=statistics.median(r['ttft_ms']['median'] for r in runs),
                ttft_p95_ms=statistics.median(r['ttft_ms']['p95'] for r in runs),
                peak_allocated_gib=max(r['max_rank_peak_allocated_bytes'] for r in runs)/2**30,
                max_decode_batch=max(r['max_pure_decode_batch'] for r in runs),
                preemptions=sum(r['preemptions'] for r in runs),
                kv_token_capacity_min_rank=min(g['token_capacity'] for r in runs for w in r['workers'] for g in w['kv_groups']))
            if row['kv_token_capacity_min_rank'] < job['batch'] * 8192:
                row['status'] = 'complete_capacity_limited'
            raw[(job['model'], job['batch'], job['arm'])] = result
        rows.append(row)
    for row in rows:
        if row['median_e2e_seconds'] is None:
            continue
        base = next(r for r in rows if (r['model'],r['batch'],r['arm']) == (row['model'],row['batch'],'dense'))
        if base['median_e2e_seconds'] is not None:
            row['speedup_vs_dense'] = base['median_e2e_seconds']/row['median_e2e_seconds']
        ag = next(r for r in rows if (r['model'],r['batch'],r['arm']) == (row['model'],row['batch'],'v64_ag'))
        if row['arm'] == 'v64_hybrid' and ag['median_e2e_seconds'] is not None:
            row['hybrid_change_vs_ag_pct'] = 100*(row['median_e2e_seconds']/ag['median_e2e_seconds']-1)
        peers = [raw[(row['model'],row['batch'],arm)] for arm in ('dense','v64_ag','v64_hybrid')
                 if (row['model'],row['batch'],arm) in raw]
        assert len({p['prompt_sha256'] for p in peers}) == 1
    smoke = []
    for job in plan['smoke']:
        path = Path(job['output'])
        result = json.loads(path.read_text()) if path.exists() else {}
        smoke.append(dict(name=job['name'], status=result.get('status','not_run'),
                          current_source=result.get('source_sha256') == plan['source_sha256']))
    breakdown = []
    for job in plan['breakdown']:
        path = Path(job['output'])
        result = json.loads(path.read_text()) if path.exists() else {}
        breakdown.append(dict(name=job['name'], status=result.get('status', 'not_run'),
            current_source=result.get('source_sha256') == plan['source_sha256'],
            raw_json=str(path), ranks=result.get('breakdown', [])))
    report = dict(expected_configurations=len(rows), completed=sum(r['median_e2e_seconds'] is not None for r in rows),
                  breakdown=breakdown, expected_profiles=len(plan['breakdown']),
                  cells=rows, smoke=smoke, deferred=plan['deferred'],
                  metric_notes='Medians of three independent repeats. Speedup = Dense / arm. B is submitted requests; actual concurrency is measured. Memory includes persistent model/cache allocations; it is not live KV usage.')
    (root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    with (root/'summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"Completed {report['completed']}/{len(rows)}; smoke",[(s['name'],s['status']) for s in smoke])
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    args=parser.parse_args()
    summarize(args.output_dir)


if __name__=='__main__':
    main()
