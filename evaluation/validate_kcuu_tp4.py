"""Four-rank C1 attention/output validation; run only in a one-node GPU allocation."""
import argparse
import json
import os
from pathlib import Path
import socket
import shlex
import sys
from types import SimpleNamespace
import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import GraphCaptureContext
from basisserve.core.qwen3_8b_vllm_c1 import file_sha256
from basisserve.core.tp4_c1_factors import load_layer, load_manifest
from basisserve.core.tp4_step import classify_step
from basisserve.vllm.diffkv_attention import C1DiffKVImpl
from basisserve.vllm.prepared_c1_output import PreparedC1Output
from evaluation.validate_gh200_tp4 import tp_group
from evaluation.prepare_kcuu_llama import source_hashes


def error(actual, expected, tolerance):
    delta = actual.float() - expected.float()
    result = dict(relative_l2=(delta.norm() / expected.float().norm().clamp_min(1e-12)).item(),
                  max_absolute_error=delta.abs().max().item(), finite=bool(torch.isfinite(actual).all()))
    assert result['finite'] and result['relative_l2'] < tolerance, result
    return result


def attention_reference(q, cache, table, lengths, contexts, heads):
    outputs = []
    start = 0
    for seq, length in enumerate(lengths):
        if not length:
            continue
        full = cache[table[seq].long()].flatten(0, 1)[:contexts[seq]].float()
        k = full[..., :128].repeat_interleave(heads // 2, dim=1).transpose(0, 1)
        v = full[..., 128:].repeat_interleave(heads // 2, dim=1).transpose(0, 1)
        query = q[start:start + length].float().transpose(0, 1)
        scores = query @ k.transpose(1, 2) * 128 ** -0.5
        pos = torch.arange(length, device=q.device) + contexts[seq] - length
        mask = torch.arange(contexts[seq], device=q.device)[None, :] > pos[:, None]
        scores.masked_fill_(mask, float('-inf'))
        outputs.append((scores.softmax(-1) @ v).transpose(0, 1))
        start += length
    return torch.cat(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--factor-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ['SLURM_NNODES'] == '1'
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    assert torch.cuda.get_device_capability() == (9, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda', torch.cuda.current_device())
    group = tp_group(4, rank)
    hosts = [None] * 4
    dist.all_gather_object(hosts, socket.gethostname())
    assert len(set(hosts)) == 1
    manifest = load_manifest(args.factor_dir, file_sha256(args.factor_dir / 'results.json'))
    fit = manifest['fit_config']
    assert fit['cache_rank_per_head'] == 64
    heads = fit['num_query_heads'] // 4
    boundaries = {mode: PreparedC1Output(group.device_group, device, heads * 64, 64,
                                         output_mode=mode, audit_routes=True)
                  for mode in ('ag', 'hybrid', 'ar')}
    impl = C1DiffKVImpl(num_heads=heads, head_size=128, scale=128 ** -0.5,
                       num_kv_heads=2, alibi_slopes=None, sliding_window=None,
                       kv_cache_dtype='auto', value_rank=64)
    layer = SimpleNamespace(impl=impl, head_size_v=64)
    cases = [('prefill', [17], [True]), ('mixed', [17, 1], [True, False]),
             ('prefill_tail', [1], [True]), ('mixed_tail', [1, 1], [True, False]),
             ('decode', [1, 1, 1], [False] * 3)]
    records = []
    stream = torch.cuda.Stream()
    for layer_id in (0, (fit['num_hidden_layers'] - 1) // 2, fit['num_hidden_layers'] - 1):
        encoder, decoder = load_layer(args.factor_dir, manifest, layer_id, rank, device=device, dtype=torch.bfloat16)
        for name, lengths, flags in cases:
            torch.manual_seed(919 + rank + layer_id)
            n = sum(lengths)
            padded = 4 if name == 'decode' else n
            contexts = [33] * len(lengths)
            cache = torch.randn(len(lengths) * 3, 16, 2, 192, device=device, dtype=torch.bfloat16)
            raw_v = torch.randn(*cache.shape[:-1], 128, device=device, dtype=torch.bfloat16)
            cache[..., 128:].copy_(torch.einsum('pthd,hdr->pthr', raw_v.float(), encoder.float()).bfloat16())
            table = torch.randperm(len(lengths) * 3, device=device, dtype=torch.int32).view(len(lengths), 3)
            q = torch.randn(padded, heads, 128, device=device, dtype=torch.bfloat16)
            k_before = cache[..., :128].clone()
            starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device=device, dtype=torch.int32)
            metadata = SimpleNamespace(num_actual_tokens=n, max_query_len=max(lengths),
                max_seq_len=33, query_start_loc=starts, block_table=table,
                seq_lens=torch.tensor(contexts, device=device, dtype=torch.int32),
                use_cascade=False, seq_threshold_3D=128,
                softmax_segm_output=torch.empty(len(lengths), heads, 4, 64, device=device),
                softmax_segm_max=torch.empty(len(lengths), heads, 4, device=device),
                softmax_segm_expsum=torch.empty(len(lengths), heads, 4, device=device),
                basisserve_step=classify_step(lengths, flags))
            choices = [None] * 4
            dist.all_gather_object(choices, vars(metadata.basisserve_step))
            assert all(choice == choices[0] for choice in choices)
            local = torch.empty(padded, heads, 64, device=device, dtype=torch.bfloat16)
            impl.forward(layer, q, None, None, cache.transpose(1, 2), metadata, local)
            ref_local = attention_reference(q, cache, table, lengths, contexts, heads)
            attention_error = error(local[:n], ref_local, 0.01)
            expected = ref_local.flatten(1) @ decoder[rank * heads * 64:(rank + 1) * heads * 64].float()
            dist.all_reduce(expected)  # Independent FP32 sum reference.
            outputs = {}
            for mode, boundary in boundaries.items():
                before_routes = boundary.route_counts.clone()
                def run():
                    return boundary.attention(layer, q, None, None, cache.transpose(1, 2), metadata, decoder)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        actual = run()
                torch.cuda.current_stream().wait_stream(stream)
                graph = None
                if name == 'decode':
                    graph = torch.cuda.CUDAGraph()
                    with group.graph_capture(GraphCaptureContext(stream)):
                        with torch.cuda.graph(graph, stream=stream):
                            actual = run()
                    graph.replay()
                torch.cuda.synchronize()
                outputs[mode] = actual[:n].clone()
                record = dict(layer=layer_id, case=name, mode=mode, rank=rank,
                              expected_collective=metadata.basisserve_step.collective(mode),
                              attention_error=attention_error, output_error=error(actual[:n], expected, 0.02),
                              tokens=n, padded_tokens=padded, graph_replay=graph is not None)
                if graph is not None:
                    original_q = q.clone()
                    for replay in range(2):
                        q.normal_()
                        ref = attention_reference(q, cache, table, lengths, contexts, heads).flatten(1)
                        ref = ref @ decoder[rank * heads * 64:(rank + 1) * heads * 64].float()
                        dist.all_reduce(ref)
                        graph.replay()
                        torch.cuda.synchronize()
                        record[f'changed_input_replay_{replay}'] = error(actual[:n], ref, 0.02)
                    q.copy_(original_q)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    if graph is not None:
                        graph.replay()
                    else:
                        run()
                    torch.cuda.synchronize()
                record['gpu_kernel_names'] = sorted({e.name for e in prof.events()
                    if e.device_type == torch.autograd.DeviceType.CUDA})
                assert record['gpu_kernel_names'], 'GPU trace is required to inspect real collective dispatch'
                record['routes'] = boundary.statistics()['route_counts']
                delta_routes = (boundary.route_counts - before_routes).tolist()
                route_index = ('prefill', 'mixed', 'decode').index(metadata.basisserve_step.kind) * 2
                route_index += int(record['expected_collective'] == 'ar')
                assert delta_routes[route_index] > 0
                assert sum(delta_routes) == delta_routes[route_index]
                record['route_increments'] = delta_routes
                records.append(record)
            assert torch.equal(cache[..., :128], k_before)
            for mode in ('hybrid', 'ar'):
                records.append(dict(layer=layer_id, case=name, comparison=f'{mode}_vs_ag',
                                    rank=rank, error=error(outputs[mode], outputs['ag'], 0.02)))
    workers = [None] * 4
    dist.all_gather_object(workers, records)
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(status='passed', job=os.environ['SLURM_JOB_ID'],
            command=shlex.join([sys.executable, *sys.argv]), environment=os.environ.get('CONDA_DEFAULT_ENV'),
            hosts=hosts, workers=workers, source_sha256=source_hashes(),
            factors_sha256=file_sha256(args.factor_dir / 'results.json')), indent=2) + '\n')
        print('PASS KCuu four-rank boundary validation', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
