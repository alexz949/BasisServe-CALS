"""Exclusive device accounting must retain graph attribution gaps."""
from evaluation.kcuu_profile import summarize_events


def test_rank_qualified_trace(tmp_path):
    import gzip
    import json
    from evaluation.kcuu_profile import summarize_trace
    path = tmp_path / 'llama_dense_b128_dp0_pp0_tp2_dcp0_ep0_rank2.123.pt.trace.json.gz'
    with gzip.open(path, 'wt') as handle:
        json.dump({'traceEvents': []}, handle)
    assert summarize_trace(path)['rank'] == 2


def test_gpu_events_count_once_and_graph_gemm_stays_unattributed():
    def cpu(name, ts, dur, eid=None, cat='cpu_op'):
        return dict(ph='X', cat=cat, pid=1, tid=1, name=name, ts=ts, dur=dur,
                    args={} if eid is None else {'External id': eid})
    def gpu(name, dur, eid):
        return dict(ph='X', cat='kernel', pid=2, tid=7, name=name, ts=100, dur=dur,
                    args={'External id': eid})
    events = [cpu('kcuu::output_boundary', 0, 30, cat='user_annotation'),
              cpu('aten::copy_', 1, 2, 1),
              cpu('kcuu::decoder', 4, 20, cat='user_annotation'),
              cpu('aten::mm', 5, 2, 2),
              cpu('kcuu::allreduce', 8, 8, cat='user_annotation'),
              cpu('vllm::all_reduce', 9, 2, 3),
              gpu('vectorized_copy', 3, 1), gpu('gemm', 7, 2),
              gpu('custom_collective', 5, 3), gpu('graph_gemm', 11, 999),
              gpu('ncclDevKernel_AllGather', 2, 999)]
    result = summarize_events(events)
    assert result['gpu_events'] == 5
    assert result['gpu_sum_us'] == 28
    assert result['categories_gpu_sum_us'] == dict(layout_conversion=3, decoder=7,
        allreduce=5, unattributed=11, allgather=2)


def test_matrix_replaces_b16_with_separate_b128_profiles(tmp_path):
    from evaluation.run_kcuu_experiments import prepare
    plan = prepare(tmp_path)
    assert len(plan['formal']) == 24 and len(plan['breakdown']) == 6
    assert {job['batch'] for job in plan['formal']} == {32, 64, 128, 256}
    assert all('--profile' not in job['command'] for job in plan['formal'])
    assert all(job['batch'] == 128 and job['command'][-1] == '--profile' for job in plan['breakdown'])
    assert len({job['output'] for job in plan['formal'] + plan['breakdown']}) == 30
