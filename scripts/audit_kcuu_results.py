"""Independent final audit of the 30-cell KCuu matrix; runs on CPU only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root
    plan = json.loads((root / 'experiment_plan.json').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    assert len(plan['formal']) == len(summary['cells']) == 30
    with (root / 'summary.csv').open(newline='') as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 30
    for row, csv_row in zip(summary['cells'], csv_rows):
        assert csv_row == {key: '' if value is None else str(value) for key, value in row.items()}
    current = {p: sha(Path(p)) for p in plan['source_sha256']}
    assert current == plan['source_sha256']
    expected = dict(tensor_parallel_size=4, dtype='bfloat16', max_model_len=8320,
                    max_num_batched_tokens=8192, enable_chunked_prefill=True,
                    enable_prefix_caching=False, async_scheduling=False,
                    gpu_memory_utilization=0.8, worker_cls='basisserve.vllm.kcuu_worker.KCuuWorker')
    prompts, cells, smokes, compressed_configs = {}, [], [], {}
    for job in plan['smoke']:
        path = Path(job['output'])
        value = json.loads(path.read_text()) if path.exists() else {}
        smokes.append(dict(name=job['name'], status=value.get('status', 'not_run'),
                           current_source=value.get('source_sha256') == current,
                           relative_l2=value.get('relative_l2')))
    for job, row in zip(plan['formal'], summary['cells']):
        assert (job['model'], job['arm'], job['batch']) == (row['model'], row['arm'], row['batch'])
        path = Path(job['output'])
        value = json.loads(path.read_text()) if path.exists() else {}
        cell = dict(name=job['name'], status=value.get('status', 'not_run'),
                    reason=value.get('failure_category'), log=value.get('log'),
                    raw_json=str(path), raw_json_sha256=sha(path) if path.exists() else None)
        if cell['status'] == 'complete':
            assert value['source_sha256'] == current
            assert value['model_revision'] == plan['model_revisions'][job['model']]
            assert value['factors_sha256'] == plan['factor_manifest_sha256'][job['model']]
            assert (value['batch'], value['arm'], value['prefill_tokens'], value['output_tokens'], value['repeats'], value['warmups']) == (job['batch'], job['arm'], 8192, 128, 3, 1)
            config = value['configuration']
            assert all(config[k] == v for k, v in expected.items())
            assert config['max_num_seqs'] == job['batch']
            assert config['compilation_config']['cudagraph_mode'] == 'FULL_DECODE_ONLY'
            inputs = json.loads(Path(value['prompt_token_ids_file']).read_text())
            assert len(inputs) == job['batch'] and all(len(p['prompt_token_ids']) == 8192 for p in inputs)
            digest = hashlib.sha256(json.dumps(inputs).encode()).hexdigest()
            assert digest == value['prompt_sha256']
            prompts.setdefault((job['model'], job['batch']), set()).add(digest)
            assert len(value['runs']) == 3
            cell['measurements'] = []
            for run in value['runs']:
                assert run['successful_requests'] == job['batch']
                assert run['total_output_tokens'] == job['batch'] * 128
                assert run['admission_included'] and len(run['workers']) == 4
                assert sorted(w['rank'] for w in run['workers']) == list(range(4))
                assert all(w['bf16_reduced_precision_reduction'] is False for w in run['workers'])
                assert all(len(r['output_token_ids']) == 128 for r in run['requests'])
                assert all(w['actual_max_num_seqs'] == job['batch'] and w['actual_max_num_batched_tokens'] == 8192 and w['actual_max_model_len'] == 8320 for w in run['workers'])
                assert all(w['steps'] == run['workers'][0]['steps'] for w in run['workers'])
                assert all('NONE' in s['graph_mode'] for s in run['workers'][0]['steps'] if s['kind'] in ('prefill', 'mixed'))
                steps = run['workers'][0]['steps']
                assert all(s['prefill_tokens'] + s['decode_tokens'] <= 8192 for s in steps)
                active = [s['decode_requests'] for s in steps if s['kind'] == 'decode']
                assert run['pure_decode_active_batch_histogram'] == {str(b): active.count(b) for b in sorted(set(active))}
                assert run['max_pure_decode_batch'] == max(active, default=0) <= job['batch']
                assert run['output_tokens_per_second'] == job['batch'] * 128 / run['wall_seconds']
                cell['measurements'].append(dict(e2e_seconds=run['wall_seconds'],
                    output_tokens_per_second=run['output_tokens_per_second'], ttft_ms=run['ttft_ms'],
                    rank_peak_allocated_bytes=[w['peak_allocated_bytes'] for w in run['workers']],
                    decode_concurrency=run['pure_decode_active_batch_histogram'],
                    max_decode_concurrency=run['max_pure_decode_batch'], preemptions=run['preemptions'],
                    max_effective_step_requests=max(s['prefill_requests'] + s['decode_requests'] for s in steps),
                    rank_kv_capacity=[w['kv_groups'] for w in run['workers']]))
            assert row['median_e2e_seconds'] == statistics.median(r['wall_seconds'] for r in value['runs'])
            assert row['output_tokens_per_second'] == statistics.median(r['output_tokens_per_second'] for r in value['runs'])
            assert row['ttft_p50_ms'] == statistics.median(r['ttft_ms']['median'] for r in value['runs'])
            assert row['ttft_p95_ms'] == statistics.median(r['ttft_ms']['p95'] for r in value['runs'])
            assert row['preemptions'] == sum(r['preemptions'] for r in value['runs'])
            assert row['max_decode_batch'] == max(r['max_pure_decode_batch'] for r in value['runs'])
            assert row['peak_allocated_gib'] == max(w['peak_allocated_bytes'] for r in value['runs'] for w in r['workers']) / 2**30
            assert row['kv_token_capacity_min_rank'] == min(g['token_capacity'] for r in value['runs'] for w in r['workers'] for g in w['kv_groups'])
            if job['arm'] != 'dense':
                assert all(w['decode_gather'] == 'nccl_symmetric_inplace' for w in value['c1_workers'])
                keys = ('attention_backend', 'attention_impl', 'prefill_specialization',
                        'decode_specialization', 'decode_output_layout', 'key_head_size',
                        'value_head_size', 'loaded_value_layers', 'decode_gather')
                signature = [tuple(w[k] for k in keys) for w in value['c1_workers']]
                pair = (job['model'], job['batch'])
                if pair in compressed_configs:
                    assert compressed_configs[pair] == signature
                compressed_configs[pair] = signature
        cells.append(cell)
    rows = {(r['model'], r['batch'], r['arm']): r for r in summary['cells']}
    for row in summary['cells']:
        if row['median_e2e_seconds'] is None:
            continue
        dense = rows[(row['model'], row['batch'], 'dense')]
        if dense['median_e2e_seconds'] is not None:
            assert row['speedup_vs_dense'] == dense['median_e2e_seconds'] / row['median_e2e_seconds']
        if row['arm'] == 'v64_hybrid':
            ag = rows[(row['model'], row['batch'], 'v64_ag')]
            if ag['median_e2e_seconds'] is not None:
                assert row['hybrid_change_vs_ag_pct'] == 100 * (row['median_e2e_seconds'] / ag['median_e2e_seconds'] - 1)
    assert all(len(values) == 1 for values in prompts.values())
    ready = all(s['status'] == 'passed' and s['current_source'] for s in smokes)
    complete = sum(c['status'] == 'complete' for c in cells)
    report = dict(audit_script_sha256=sha(Path(__file__)), smoke_gates_passed=ready,
                  source_hashes_sha256=hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest(),
                  completed=complete, total=30, all_successful=ready and complete == 30,
                  fairness_checks_passed_for_completed_cells=True, smoke=smokes, cells=cells,
                  metric_notes=dict(
                      memory='Per-rank torch.cuda.max_memory_allocated, reset after warmup before each repeat; external CUDA/NCCL allocations are not included.',
                      capacity_status='complete_capacity_limited means KV capacity is below B*8192; it does not by itself prove that KV capacity limited measured throughput. Consult preemptions and actual concurrency.',
                      concurrency='B counts submitted requests. Decode histogram excludes prefill/mixed steps and CUDA Graph padding.',
                      repetitions='Three complete formal measurements after one warmup per fresh configuration process.'))
    (root / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Audit:', complete, '/30 successful; smoke gates:', ready)


if __name__ == '__main__':
    main()
