"""Check recorded evidence and grid coverage, without certifying missing work."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('results/system_benchmarks/l40s'))
    args=parser.parse_args();root=args.output
    counters=dict(timing_distributions=0,numa_buffers=0,metadata_records=0,e2e_complete=0,e2e_oom=0)
    files=[]
    def inspect(value):
        if isinstance(value,list):
            for item in value:inspect(item)
        elif isinstance(value,dict):
            if 'raw_us' in value or 'raw_ms' in value:
                unit='us' if 'raw_us' in value else 'ms';raw=value['raw_'+unit]
                assert len(raw)==value['iterations'] and value['iterations']>=500
                assert value['warmup']>=100
                assert all(math.isfinite(v) and v>0 for v in raw)
                assert math.isclose(statistics.mean(raw),value['mean_'+unit],rel_tol=1e-6)
                counters['timing_distributions']+=1
            if 'page_nodes' in value:
                assert value['pinned'] and value['query_returncode']==0
                assert len(value['page_nodes'])==value['sampled_pages'] and value['sampled_pages']>0
                assert all(node==value['expected_numa_node'] for node in value['page_nodes'])
                counters['numa_buffers']+=1
            if 'git_commit' in value:
                assert value['git_commit'] and value['source_sha256'] and value['hostname']
                assert value['command_line'] and value['pytorch'] and value['cuda']
                assert 'external_gpu_processes' in value
                counters['metadata_records']+=1
            for item in value.values():inspect(item)
    def load(path):
        data=json.loads(path.read_text());inspect(data);files.append(str(path));return data
    routers=[]
    for pattern in ['router_basis_rank*.json','router_loki_t*.json','router_shadow_t*.json','router_lrqk_t*.json']:
        for path in sorted(root.glob(pattern)):
            for row in load(path)['records']:
                assert row['audit']['reference_ids_equal']
                if 'logical_routing_width' in row:
                    assert row['logical_routing_width']==32
                    assert row['audit']['nonrouting_payload_poisoned_without_effect']
                    assert row['audit']['split_query_preprocessing_bitwise_equal']
                routers.append((row['method'],row['length'],row['batch']))
    assert len(routers)==len(set(routers))
    frozen_updates=[]
    for path in sorted(root.glob('lrqk_frozen_update_cusolver*.json')):
        if 'smoke' in path.name:continue
        row=load(path)
        assert row['native_output_bitwise_equal']
        assert row['audit']['reference_ids_equal']
        assert 'Cusolver' in row['linalg_preference']
        frozen_updates.append((row['length'],row['batch']))
    assert len(frozen_updates)==len(set(frozen_updates))
    frozen_pipelines=[]
    for path in sorted(root.glob('lrqk_frozen_pipeline_t*.json')):
        if 'smoke' in path.name:continue
        row=load(path)
        assert row['native_output_bitwise_equal'] and row['native_ids_equal']
        assert row['native_audit']['reference_ids_equal']
        if row['common_backend'] is not None:
            assert row['common_backend']['audit']['independent_query_support_reference']
            assert row['common_backend']['audit']['exact_key_equal']
        frozen_pipelines.append((row['length'],row['batch']))
    staged_lengths=[]
    for path in sorted(root.glob('sparse_staged_total_t*.json')):
        row=load(path)
        assert row['reference_output_bitwise_equal'] and row['route_audit']['reference_ids_equal']
        staged_lengths.append(row['length'])
    tp=[]
    for path in sorted(root.glob('tp_collective_layer*.json')):
        for row in load(path)['records']:
            assert all(row['audit'].values())
            tp.append((row['layer'],row['batch'],row['method']))
    assert len(tp)==len(set(tp))
    assert len(tp)==576
    nccl=load(root/'nccl_raw.json')['records']
    assert {(r['operation'],r['input_bytes_per_rank']) for r in nccl}==set(itertools.product(['all_reduce','all_gather'],[2**i for i in range(8,25)]))
    for pattern in ['h2d_rank*.json','numa_validation_rank*.json','basis_validation_rank*.json']:
        for path in sorted(root.glob(pattern)):load(path)
    for path in sorted(root.glob('k_offload_budget*.json')):
        row=load(path)['record'];audit=row['validation']
        assert all(audit[key] for key in ['staged_exact_keys_equal','staged_reference_equal','mapped_reference_equal','no_full_gpu_k_buffer'])
        assert audit['gpu_k_staging_bytes']<audit['full_cpu_k_bytes']
        assert row['logical_unique_k_bytes']==row['staged_dma_payload_bytes']
    for path in sorted(root.glob('sparse_operator_t*.json')):
        row=load(path)['record']
        assert row['route_audit']['reference_ids_equal'] and row['sparse_audit']['no_full_gpu_k_buffer']
    common=[]
    for method in ['basis','loki','shadow','lrqk']:
        path=root/f'common_{method}.json'
        if path.exists():
            row=load(path);common.append(method)
            assert row['audit']['exact_key_equal'] and row['audit']['independent_query_support_reference']
            assert row['audit']['persistent_gpu_key_staging_bytes']<row['audit']['full_host_key_bytes']
    basis_rows=[row for path in root.glob('basis_validation_rank*.json') for row in json.loads(path.read_text())['rows']]
    assert {r['layer'] for r in basis_rows}==set(range(32))
    assert all(r['separate_storage'] and r['logical_router_dimensions']==32 and r['fp64_output_rel_mse']<1e-16 for r in basis_rows)
    h2d_rows=[row for path in root.glob('h2d_rank*.json') for row in json.loads(path.read_text())['records']]
    assert {(r['gpu'],r['bytes']) for r in h2d_rows}==set(itertools.product(range(4),[i*2**20 for i in [1,4,16,64,256,1024]]))
    expected=set(itertools.product(['dense','c1','sparse_local','offload'],[16384,32768,65536],[1,4,8,16]))
    cases=[];source_hashes={};historical_sources={}
    for path in sorted((root/'e2e').glob('*/trial.json')):
        trial=load(path);case=(trial['mode'],trial['length'],trial['batch']);cases.append(case)
        assert case in expected and trial['status'] in ['complete','gpu_oom']
        for name,digest in trial['source_sha256'].items():
            if name not in source_hashes:source_hashes[name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
            if source_hashes[name]!=digest:
                snapshot=root/'source_snapshots'/f'{digest}.py'
                assert snapshot.is_file() and hashlib.sha256(snapshot.read_bytes()).hexdigest()==digest
                historical_sources[name]=dict(measured_sha256=digest,snapshot=str(snapshot),current_sha256=source_hashes[name])
        if trial['status']=='gpu_oom':counters['e2e_oom']+=1;continue
        counters['e2e_complete']+=1
        assert {r['rank'] for r in trial['ranks']}==set(range(4))
        for rank in trial['ranks']:
            assert rank['tp']==4 and rank['generated_tokens']==256 and rank['steady_forward_count']==224
            assert rank['discard_first_generated_tokens']==32 and len(rank['raw_decode_ms'])==255
            assert all(math.isfinite(t) and t>0 for t in rank['raw_decode_ms'])
            assert math.isclose(statistics.mean(rank['raw_decode_ms'][31:]),rank['steady_decode_mean_ms'],rel_tol=1e-6)
            assert math.isclose(rank['steady_aggregate_tokens_per_second'],trial['batch']*224/rank['steady_wall_seconds'],rel_tol=1e-6)
            if trial['mode']=='offload':
                assert len(rank['host_buffer_audit'])==32 and rank['host_key_bytes']>0
                assert rank['logical_unique_host_k_read_bytes_per_rank_per_decode']>0
            else:assert rank['host_key_bytes']==0
    assert len(cases)==len(set(cases))
    result=dict(status='available_records_passed',scope='Recorded invariants and grid coverage only; not a completion certificate or a replacement for GPU numerical tests and profiler inspection',
        counters=counters,router_cases=len(routers),tp_cases=len(tp),common_methods=common,
        lrqk_frozen_cusolver_update_cases=frozen_updates,
        lrqk_frozen_cusolver_pipeline_cases=frozen_pipelines,allocator_excluded_staged_lengths=staged_lengths,
        e2e_missing=sorted(expected-set(cases)),source_sha256=source_hashes,inspected_files=files,
        historical_measured_sources=historical_sources,
        remaining_manual_checks=['LRQK native eager timing protocol exception','Profiler metric and timeline interpretation','Figures and report inspection','All required grids and actual OOM evidence'])
    if (root/'router_raw.json').exists():
        outcomes=json.loads((root/'router_raw.json').read_text()).get('outcomes',[])
        result['router_terminal_outcomes']=len(outcomes)
        for record in outcomes:
            if record['status']=='gpu_oom':
                assert 'out of memory' in Path(record['log']).read_text().lower()
    (root/'record_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(dict(**counters,router_cases=len(routers),tp_cases=len(tp),e2e_missing=len(expected-set(cases))),flush=True)


if __name__=='__main__':main()
