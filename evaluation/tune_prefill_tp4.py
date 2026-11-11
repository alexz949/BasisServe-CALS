"""Single-GH200 prefill tuning at TP4-local geometry, without collectives.

Measure attention plus its local pack separately from the replicated decoder
GEMM. Synthetic full-width decoder inputs are NOT an AllGather measurement.
Serving defaults remain fixed while candidate layouts and pipelines are tested.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import statistics
import sys

import torch
import triton

from basisserve.kernels.diffkv_prefill import diffkv_prefill
from evaluation.benchmark_hopper_warps import make_case, reference


def capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
        for _ in range(4):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def time_graph(graph):
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    start.record()
    for _ in range(10):
        graph.replay()
    end.record()
    end.synchronize()
    return 1000 * start.elapsed_time(end) / 40


def check(actual, expected):
    error = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
    finite = bool(torch.isfinite(actual).all())
    assert finite and error < 0.005, (finite, error)
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (9, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    rng = random.Random(20261110)
    source_paths = [Path(__file__), Path('basisserve/kernels/diffkv_prefill.py'),
                    Path('evaluation/benchmark_hopper_warps.py')]
    result = dict(status='running', command=sys.argv, environment=os.environ.get('CONDA_DEFAULT_ENV'),
                  job=os.environ.get('SLURM_JOB_ID'), host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__, triton=triton.__version__,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                  scope='One GPU, TP4-local two KV heads. Synthetic full-width decoder input. NO communication or full-model timing.',
                  metric='Median of 5 shuffled rounds, warm CUDA graph microseconds per call',
                  config=dict(query_head_rows=64, kv_tile=128, warps=4, compiler_stages=2), records=[])

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')

    def measure(functions):
        graphs = {name: capture(fn) for name, fn in functions.items()}
        samples = {name: [] for name in functions}
        for _ in range(5):
            names = list(functions)
            rng.shuffle(names)
            for name in names:
                samples[name].append(time_graph(graphs[name]))
        return {name: dict(samples_us=values, median_us=statistics.median(values))
                for name, values in samples.items()}

    save()
    with torch.cuda.stream(stream):
        for model, heads_per_kv, hidden in (('llama', 4, 4096), ('qwen', 8, 5120)):
            for value_rank in (64, 96):
                local_heads = heads_per_kv * 2
                local_width = local_heads * value_rank
                total_width = local_width * 4
                torch.manual_seed(20261110)
                decoder = torch.randn(total_width, hidden, device=device, dtype=torch.bfloat16) / total_width ** 0.5
                decoder_col = decoder.T.contiguous().T
                for batch, qlen, length in ((1, 129, 146), (1, 4096, 4096), (2, 4096, 4096)):
                    torch.manual_seed(20261110)
                    tokens = batch * qlen
                    label = f'{model}-v{value_rank}-b{batch}-q{qlen}'
                    print(f'START {label}', flush=True)
                    data = make_case(batch, qlen, length, heads_per_kv, value_rank)
                    # Guard the destination's physical padding as well as its strides.
                    storage = torch.full((local_width + 3, tokens + 3), float('nan'),
                                         device=device, dtype=torch.bfloat16)
                    slot = storage[:local_width, :tokens]
                    direct_data = dict(data, out=slot.T.view(tokens, local_heads, value_rank))

                    def attention():
                        return diffkv_prefill(**data)

                    def pack():
                        return slot.copy_(data['out'].flatten(1).T)

                    def token_pack(stages=None):
                        diffkv_prefill(**data, loop_stages=stages)
                        pack()
                        return direct_data['out']

                    def direct(stages=None):
                        diffkv_prefill(**direct_data, loop_stages=stages)
                        return direct_data['out']

                    attention()
                    local_expected = data['out'].clone()
                    fp32_error = check(local_expected, reference(data, qlen)) if qlen == 129 else None
                    record = dict(case=label, model=model, value_rank=value_rank, tokens=tokens,
                                  independent_attention_relative_l2=fp32_error,
                                  correctness={}, components={}, attention_pack={}, gemm={}, resources={})
                    runners = dict(baseline_token_pack=token_pack, direct_feature=direct,
                                   direct_pipeline2=lambda: direct(2), direct_pipeline3=lambda: direct(3),
                                   token_pipeline2_pack=lambda: token_pack(2),
                                   token_pipeline3_pack=lambda: token_pack(3))
                    result['records'].append(record)
                    save()
                    for name, run in runners.items():
                        error = check(run(), local_expected)
                        assert torch.isnan(storage[local_width:]).all()
                        assert torch.isnan(storage[:, tokens:]).all()
                        record['correctness'][name] = error
                        # Check dynamic inputs for every candidate inside CUDA graph replay.
                        graph = capture(run)
                        data['q'].normal_()
                        data['v'].normal_()
                        attention()
                        local_expected = data['out'].clone()
                        graph.replay()
                        torch.cuda.synchronize()
                        check(direct_data['out'], local_expected)
                        assert torch.isnan(storage[local_width:]).all()
                        assert torch.isnan(storage[:, tokens:]).all()
                        if qlen == 129:
                            check(direct_data['out'], reference(data, qlen))
                        del graph
                        print(f'CHECK {label} {name} relative_l2={error:.6g} graph_replay=passed', flush=True)
                        save()
                    record['changed_input_replay_passed'] = True
                    components = dict(attention_token=attention,
                                      attention_direct=lambda: diffkv_prefill(**direct_data),
                                      attention_pipeline2=lambda: diffkv_prefill(**direct_data, loop_stages=2),
                                      attention_pipeline3=lambda: diffkv_prefill(**direct_data, loop_stages=3),
                                      pack=pack)
                    # Real communication slots are contiguous [features, tokens]; use that
                    # layout for timings after testing the stricter padded-stride version.
                    slot = torch.empty(local_width, tokens, device=device, dtype=torch.bfloat16)
                    direct_data = dict(data, out=slot.T.view(tokens, local_heads, value_rank))
                    for run in runners.values():
                        check(run(), local_expected)
                    for name, fn in components.items():
                        if name != 'pack':
                            kernel = fn()
                            record['resources'][name] = dict(registers=kernel.n_regs, spills=kernel.n_spills,
                                                             shared_bytes=kernel.metadata.shared)
                    record['components'] = measure(components)
                    record['attention_pack'] = measure(runners)
                    baseline_us = record['attention_pack']['baseline_token_pack']['median_us']
                    for values in record['attention_pack'].values():
                        values['speedup'] = baseline_us / values['median_us']
                    save()
                    # Synthetic gathered input: isolates cuBLAS layout selection without
                    # pretending a single GPU can measure TP4 communication.
                    arena = torch.randn(total_width, tokens, device=device, dtype=torch.bfloat16)
                    packed = arena.T.contiguous()
                    expected = torch.mm(packed, decoder)
                    gemms = dict(feature_row_decoder=lambda: torch.mm(arena.T, decoder),
                                 feature_column_decoder=lambda: torch.mm(arena.T, decoder_col),
                                 copy_then_token_gemm=lambda: torch.mm(arena.T.contiguous(), decoder),
                                 token_gemm_only=lambda: torch.mm(packed, decoder))
                    for name, fn in gemms.items():
                        record['correctness'][name] = check(fn(), expected)
                    record['gemm'] = measure(gemms)
                    record['status'] = 'passed'
                    save()
                    print(json.dumps(record), flush=True)
        result['status'] = 'passed'
        save()
    print('PASS prefill tuning', flush=True)


if __name__ == '__main__':
    main()
