"""Merge available measurements without inventing absent methods or stages."""
import argparse
import csv
import json
import itertools
from pathlib import Path


def csv_file(path,rows):
    assert rows
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def router_row(r):
    position_bytes=0;factor_bytes=None;coordinate_bytes=None;fixed_id_bytes=0
    if 'state' in r:
        state=r['state'];audit=r['audit']
        budget=state['nominal_budget'];width=state['routing_read_dimensions_per_token']
        state_bytes=state['routing_state_bytes'];physical=audit['unique_tokens_per_kv'];pages=audit['pages32_per_kv']
        per_query=[n for batch in audit['tokens_per_query_head'] for head in batch for n in head]
        if r['method']=='shadow':
            fixed_id_bytes=r['batch']*r['kv_heads']*(state['outlier_tokens']+state['local_tokens'])*8
            state_bytes+=fixed_id_bytes
            scan_bytes=r['batch']*r['kv_heads']*((r['length']-state['outlier_tokens']-state['local_tokens'])//state['chunk_size'])*r['head_dim']*2
        elif r['method']=='lrqk':
            fixed_id_bytes=state['lite']*8
            state_bytes+=fixed_id_bytes
            scan_bytes=state['scan_state_bytes']
        else:
            scan_bytes=r['batch']*r['kv_heads']*r['length']*state['rank']*2
    else:
        budget=r['budget'];width=r['logical_routing_width'];state_bytes=r['routing_cache_bytes']
        physical=[n for batch in r['audit']['physical_tokens_per_kv'] for n in batch]
        pages=r['audit']['physical_pages_per_kv'];per_query=physical
        coordinate_bytes=state_bytes
        position_bytes=(r['length']-r['recent'])*64*2*2
        # Retained B16R16 factor tensors: bias, left, right, residual encoder,
        # residual query. Validation-only full cos/sin tensors are excluded.
        factor_bytes=(8*128+8*r['v_rank']*16+8*16*128+8*128*16+32*128*16)*2
        state_bytes+=position_bytes+factor_bytes
        scan_bytes=r['estimated_coordinate_scan_bytes']
    pre=r['query_preprocessing']
    return dict(method=r['method'],layer=r['layer'],length=r['length'],batch=r['batch'],
        logical_budget=budget,logical_width=width,routing_state_bytes=state_bytes,
        coordinate_cache_bytes=coordinate_bytes,position_table_bytes=position_bytes,
        retained_factor_bytes=factor_bytes,fixed_id_storage_bytes=fixed_id_bytes,
        logical_coordinate_scan_bytes=scan_bytes,logical_scan_bytes_with_position_tables=scan_bytes+position_bytes,
        logical_effective_gbps=(scan_bytes+position_bytes)/r['route_to_ids']['p50_us']/1000,
        query_p50_us=pre['p50_us'] if pre is not None else 0,
        route_mean_us=r['route_to_ids']['mean_us'],route_p50_us=r['route_to_ids']['p50_us'],
        route_p95_us=r['route_to_ids']['p95_us'],route_stddev_us=r['route_to_ids']['stddev_us'],
        total_p50_us=r['total_with_preprocessing']['p50_us'],
        min_support_per_query=min(per_query),max_support_per_query=max(per_query),
        minimum_unique_fetch_tokens_per_kv=min(physical),maximum_unique_fetch_tokens_per_kv=max(physical),
        max_physical_pages_per_kv=max(pages))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('results/system_benchmarks/l40s'))
    a=p.parse_args();root=a.output
    report_paths=[*root.glob('router_basis_rank*.json'),*root.glob('router_loki_t*.json'),*root.glob('router_shadow_t*.json'),*root.glob('router_lrqk_t*.json')]
    reports=[json.loads(p.read_text()) for p in sorted(report_paths)]
    rows=[r for d in reports for r in d['records']]
    if rows:
        csv_file(root/'router.csv',[router_row(r) for r in rows])
        outcomes={(r['method'],r['length'],r['batch']):dict(method=r['method'],length=r['length'],batch=r['batch'],status='complete') for r in rows}
        capacity_path=root/'lrqk_capacity_review.json'
        if capacity_path.exists():
            for r in json.loads(capacity_path.read_text())['records']:
                key=('lrqk',r['length'],r['batch'])
                if key not in outcomes:outcomes[key]=dict(method='lrqk',**r)
        methods=['BasisKV B16R16','loki','shadow','lrqk']
        expected=set(itertools.product(methods,[16384,32768,65536,131072],[1,4,8]))
        finished={key for key,r in outcomes.items() if r['status'] in ['complete','gpu_oom']}
        status='partial' if finished!=expected else 'complete_with_capacity_limits' if len(rows)<48 else 'complete'
        (root/'router_raw.json').write_text(json.dumps(dict(status=status,
            pending_methods=[m for m in methods if any(key[0]==m for key in expected-finished)],
            outcomes=list(outcomes.values()),source_reports=reports),indent=2)+'\n')
    if (root/'nccl_raw.json').exists():
        nccl=json.loads((root/'nccl_raw.json').read_text())
        csv_file(root/'nccl.csv',[dict(operation=r['operation'],input_bytes_per_rank=r['input_bytes_per_rank'],
            output_bytes_per_rank=r['output_bytes_per_rank'],algorithm_gbps=r['algorithm_gbps'],bus_gbps=r['bus_gbps'],
            **{k:v for k,v in r['timing'].items() if k.endswith('_ms') and k!='raw_ms'}) for r in nccl['records']])
    h2d=[r for p in sorted(root.glob('h2d_rank*.json')) for r in json.loads(p.read_text())['records']]
    if h2d:csv_file(root/'h2d.csv',[dict(gpu=r['gpu'],numa_node=r['numa_node'],bytes=r['bytes'],gbps=r['one_direction_h2d_gbps'],
        **{k:v for k,v in r['timing'].items() if k.endswith('_ms') and k!='raw_ms'}) for r in h2d])
    fetch_reports=[json.loads(p.read_text()) for p in sorted(root.glob('k_offload_budget*.json'))]
    if fetch_reports:
        (root/'k_offload_raw.json').write_text(json.dumps(dict(source_reports=fetch_reports),indent=2)+'\n')
        csv_file(root/'k_offload.csv',[dict(budget=r['physical_budget'],pages=r['actual_pages'],length=r['length'],
            logical_k_bytes=r['logical_unique_k_bytes'],staged_dma_payload_bytes=r['staged_dma_payload_bytes'],
            fetch_us=r['fetch']['p50_us'],qk_us=r['exact_qk']['p50_us'],pv_us=r['softmax_pv_with_selected_v_gather']['p50_us'],
            staged_effective_payload_gbps=r['staged_dma_payload_bytes']/r['fetch']['p50_us']/1000,
            staged_total_us=r['staged_total']['p50_us'],mapped_fused_total_us=r['mapped_fused_total']['p50_us'])
            for d in fetch_reports for r in [d['record']]])
    operator_reports=[json.loads(p.read_text()) for p in sorted(root.glob('sparse_operator_t*.json'))]
    if operator_reports:
        corrected_staged={r['length']:r for p in sorted(root.glob('sparse_staged_total_t*.json')) for r in [json.loads(p.read_text())]}
        (root/'sparse_operator_raw.json').write_text(json.dumps(dict(source_reports=operator_reports,
            allocator_excluded_staged_totals=list(corrected_staged.values()),
            note='Original staged total includes eager router allocation. Use separately captured replacement totals; original fused and component timings remain valid.'),indent=2)+'\n')
        csv_file(root/'sparse_operator.csv',[dict(length=r['length'],batch=r['batch'],budget=r['budget'],
            dense_local_us=r['dense_local']['p50_us'],dense_offload_us=r['dense_k_offload']['p50_us'],
            sparse_local_us=r['sparse_local']['p50_us'],sparse_offload_us=r['sparse_k_offload']['p50_us'],
            dense_offload_over_sparse_offload=r['primary_speedup'],
            staged_query_us=r['staged_breakdown']['query_preprocessing']['p50_us'],
            staged_route_us=r['staged_breakdown']['route_to_ids']['p50_us'],
            staged_fetch_us=r['staged_breakdown']['staged_fetch']['p50_us'],
            staged_qk_us=r['staged_breakdown']['staged_exact_qk']['p50_us'],
            staged_softmax_pv_us=r['staged_breakdown']['staged_softmax_pv']['p50_us'],
            staged_total_us=corrected_staged[r['length']]['staged_sparse_total']['p50_us'] if r['length'] in corrected_staged else None,
            dense_local_over_sparse_local=r['dense_local_over_sparse_local'])
            for d in operator_reports for r in [d['record']]])
    tp_reports=[json.loads(p.read_text()) for p in sorted(root.glob('tp_collective_layer*.json'))]
    if tp_reports:
        records=[r for d in tp_reports for r in d['records']]
        (root/'tp_collective_raw.json').write_text(json.dumps(dict(status='complete' if len(records)==32*6*3 else 'partial',source_reports=tp_reports),indent=2)+'\n')
        csv_file(root/'tp_collective.csv',[dict(layer=r['layer'],rank=r['rank_per_kv_head'],batch=r['batch'],method=r['method'],
            projection_ms=r['projection']['p50_ms'],collective_ms=r['collective']['p50_ms'],
            decoder_ms=0 if r['decoder'] is None else r['decoder']['p50_ms'],total_ms=r['total']['p50_ms'],
            collective_input_bytes_per_rank=r['collective_input_bytes_per_rank'],analytical_bus_bytes_per_rank=r['analytical_bus_bytes_per_rank']) for r in records])
    common=[json.loads(p.read_text()) for p in [root/f'common_{method}.json' for method in ['basis','loki','shadow','lrqk']] if p.exists()]
    if common:
        (root/'common_offload_raw.json').write_text(json.dumps(dict(status='complete' if len(common)==4 else 'partial',source_reports=common),indent=2)+'\n')
        csv_file(root/'common_offload.csv',[dict(method=r['method'],length=r['length'],batch=r['batch'],
            query_p50_us=0 if r['query_preprocessing'] is None else r['query_preprocessing']['p50_us'],
            route_p50_us=r['route_to_ids']['p50_us'],fetch_p50_us=r['fetch']['p50_us'],
            exact_qk_softmax_pv_fused_p50_us=r['exact_qk_softmax_pv_fused']['p50_us'],
            total_p50_us=r['total_with_query_preprocessing']['p50_us'],
            unique_tokens=r['traffic']['unique_key_tokens'],h2d_dma_payload_bytes=r['traffic']['h2d_dma_payload_bytes']) for r in common])
    trials=[json.loads(path.read_text()) for path in sorted((root/'e2e').glob('*/trial.json'))]
    if trials:
        table=[]
        for trial in trials:
            row=dict(mode=trial['mode'],length=trial['length'],batch=trial['batch'],status=trial['status'],
                ttft_seconds=None,steady_decode_mean_ms=None,steady_aggregate_tokens_per_second=None,
                max_prefill_gpu_gib=None,max_decode_gpu_gib=None,total_host_key_gib=None,
                nccl_input_bytes_per_rank=None,logical_host_k_read_bytes_per_rank=None)
            if trial['status']=='complete':
                ranks=trial['ranks'];r=ranks[0]
                row.update(ttft_seconds=max(x['ttft_seconds'] for x in ranks),
                    steady_decode_mean_ms=r['steady_decode_mean_ms'],
                    steady_aggregate_tokens_per_second=r['steady_aggregate_tokens_per_second'],
                    max_prefill_gpu_gib=max(x['prefill_peak_allocated_bytes'] for x in ranks)/2**30,
                    max_decode_gpu_gib=max(x['decode_peak_allocated_bytes'] for x in ranks)/2**30,
                    total_host_key_gib=sum(x['host_key_bytes'] for x in ranks)/2**30,
                    nccl_input_bytes_per_rank=r['total_nccl_input_bytes_per_rank_per_decode'],
                    logical_host_k_read_bytes_per_rank=r['logical_unique_host_k_read_bytes_per_rank_per_decode'])
            table.append(row)
        csv_file(root/'e2e_tp4.csv',table)
    print(dict(router_cases=len(rows),h2d_cases=len(h2d)),flush=True)

if __name__=='__main__':main()
