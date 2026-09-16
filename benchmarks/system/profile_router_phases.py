"""Real-query phase attribution and transformed-query feasibility experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import statistics
import subprocess
import sys
import torch
from torch.utils.cpp_extension import load
import benchmarks.system.native_basis_cache as basis
import benchmarks.system.transformed_query_router as transformed
from basisserve.kernels.mapped_host_paged_attention import _load_extension, conditional_router_query_code, select_fixed_group_max_pages_cuda


PHASES = ['load_base_right', 'B16_to_K128', 'bias_BF16_round', 'RoPE_BF16_round',
          'load_query_codes', 'QK', 'residual_dot_score_round', 'Page_LSE']


def build_sources(root):
    source = Path('basisserve/kernels/csrc')
    folder = root / 'source'; folder.mkdir(parents=True, exist_ok=True)
    text = (source / 'conditional_router_page32.cu').read_text()
    left = text.index('__global__ void conditional_router_page_lse_kernel(')
    right = text.index('__global__ void conditional_router_append_decode_kernel(', left)
    kernel = text[left:right]
    loop = '  for (int index = thread; index < kPageSize * kHalfHeadDim;'
    start = kernel.index('    const float first_gemm =')
    end = kernel.index('    float cosine =', start)
    kernel = kernel[:start] + '''    const float first_pre = shared_accumulator[first_index];
    const float second_pre = shared_accumulator[second_index];
''' + kernel[end:]
    index = kernel.index(loop)
    kernel = kernel[:index] + '''  for (int index = thread; index < kPageSize * kQueryKeyDim; index += kThreads) {
    const int feature = index % kQueryKeyDim;
    shared_accumulator[index] = round_bfloat16(round_bfloat16(shared_accumulator[index]) +
        static_cast<float>(base_bias[kv_head * kQueryKeyDim + feature]));
  }
  __syncthreads();

''' + kernel[index:]
    begin = kernel.index('  for (int head = warp;')
    cut = kernel.index('    const float maximum = warp_max(local_max);', begin)
    stop = kernel.index('\n#endif', cut)
    residual = kernel[begin:cut].replace('    float scores[(kPageSize + 31) / 32];\n', '').replace('    float local_max = -CUDART_INF_F;\n', '')
    residual = residual.replace('      scores[slot] = score;\n      local_max = fmaxf(local_max, score);', '      shared_scores[head * kPageSize + page_token] = score;')
    lse = '''  }
  __syncthreads();
  for (int head = warp; head < kQueriesPerKv; head += kThreads / kWarpSize) {
    const float score = shared_scores[head * kPageSize + lane];
    const float maximum = warp_max(score);
    const float sum = warp_sum(__expf(score - maximum));
    if (lane == 0) output[batch * output_stride_batch + kv_head * output_stride_head + head * output_stride_query + page] = maximum + __logf(sum);
  }
'''
    kernel = kernel[:begin] + residual + lse + kernel[stop:]
    kernel = kernel.replace('    float scale) {', '    float scale, int64_t* cycles) {', 1)
    needle = '  const int64_t page_start = page * kPageSize;'
    kernel = kernel.replace(needle, needle + '\n  __shared__ unsigned long long stamps[9];\n  if (thread == 0) stamps[0] = clock64();', 1)
    parts = kernel.split('  __syncthreads();')
    assert len(parts) == 8
    kernel = parts[0] + ''.join('  __syncthreads();\n  if (thread == 0) stamps[' + str(i) + '] = clock64();' + p for i, p in enumerate(parts[1:], 1))
    kernel = kernel.replace('\n#endif\n}', '''
  __syncthreads();
  if (thread == 0) {
    stamps[8] = clock64();
    for (int stage = 0; stage < 8; ++stage) cycles[row * 8 + stage] = stamps[stage + 1] - stamps[stage];
  }
#endif
}''', 1)
    text = text[:left] + kernel + text[right:]
    text = text.replace('}  // namespace', '}  // namespace\nstatic at::Tensor router_cycle_buffer;\nat::Tensor get_router_cycles() { return router_cycle_buffer; }', 1)
    text = text.replace('  conditional_router_page_lse_kernel\n      <<<', '  router_cycle_buffer = at::empty({base_code.size(0) * base_code.size(1) * pages, 8}, base_code.options().dtype(at::kLong));\n  conditional_router_page_lse_kernel\n      <<<', 1)
    text = text.replace('          static_cast<float>(scale));', '          static_cast<float>(scale), router_cycle_buffer.mutable_data_ptr<int64_t>());', 1)
    (folder / 'conditional_router_page32.cu').write_text(text)
    cpp = (source / 'mapped_host_paged_attention.cpp').read_text().replace('PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {', 'at::Tensor get_router_cycles();\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {\n  module.def("get_router_cycles", &get_router_cycles);', 1)
    (folder / 'mapped_host_paged_attention.cpp').write_text(cpp)
    (folder / 'mapped_host_paged_attention.cu').write_bytes((source / 'mapped_host_paged_attention.cu').read_bytes())
    return [str(folder / name) for name in ['mapped_host_paged_attention.cpp', 'mapped_host_paged_attention.cu', 'conditional_router_page32.cu']]


def timing(fn, repeats):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(20): fn()
    raw = []
    for _ in range(repeats):
        first = torch.cuda.Event(enable_timing=True); last = torch.cuda.Event(enable_timing=True)
        first.record(); graph.replay(); last.record(); last.synchronize()
        raw.append(first.elapsed_time(last) / 20)
    return {'median_ms': statistics.median(raw), 'raw_ms': raw}


def eager_timing(fn, repeats):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    raw = []
    for _ in range(repeats):
        first = torch.cuda.Event(enable_timing=True); last = torch.cuda.Event(enable_timing=True)
        first.record(); fn(); last.record(); last.synchronize()
        raw.append(first.elapsed_time(last))
    return {'median_ms': statistics.median(raw), 'raw_ms': raw}


def linear_reference(q, b, res, kw, code):
    pages = (b.shape[2] + 31) // 32
    chosen = torch.tensor(sorted({0, 1, pages // 2, pages - 2, pages - 1}), device=q.device)
    tokens = (chosen[:, None] * 32 + torch.arange(32, device=q.device)).flatten()
    valid = tokens < b.shape[2]; tokens = tokens.clamp_max(b.shape[2] - 1)
    pre = b[:, :, tokens].float() @ kw['base_right'].float() + kw['base_bias'][None, :, None].float()
    c = kw['rope_cos'][tokens].float(); s = kw['rope_sin'][tokens].float()
    post = torch.cat((pre[..., :64] * c - pre[..., 64:] * s, pre[..., 64:] * c + pre[..., :64] * s), -1)
    base_score = torch.einsum('bhqd,bhtd->bhqt', q.reshape(q.shape[0], 8, 4, 128).float(), post).bfloat16().float()
    residual_score = torch.einsum('bhqr,bhtr->bhqt', code.float(), res[:, :, tokens].float()).bfloat16().float()
    score = ((base_score + residual_score).bfloat16().float() * kw['scale']).bfloat16().float()
    score[..., ~valid] = -torch.inf
    return chosen, score.reshape(q.shape[0], 8, 4, -1, 32).logsumexp(-1)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--smoke', action='store_true'); args = parser.parse_args()
    root = Path('results/system_benchmarks/router_phases') / ('smoke' if args.smoke else '64k')
    root.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    sources = build_sources(root)
    ext = load(name='basis_router_eight_phases', sources=sources, extra_cflags=['-O3', '-std=c++17'],
        extra_cuda_cflags=['-O3', '-std=c++17', '--use_fast_math', '-DBASIS_VALUE_DIM=80', '-DBASIS_GQA=4',
                          '-DBASIS_PAGE_SIZE=32', '-DBASIS_BASE_RANK=16', '-DBASIS_RESIDUAL_RANK=16'])
    normal = _load_extension(80, 4, 32, 16, 16)
    rows = []; seen = set(); current = {}
    original_attention = basis.BasisCache.attention
    def attention(self, q, k, v, layer, positions):
        current.update(v=v, factors=self.factors[layer])
        return original_attention(self, q, k, v, layer, positions)
    basis.BasisCache.attention = attention
    original = basis.conditional_router_page_lse
    def measured(q, b, res, **kw):
        result = original(q, b, res, **kw)
        identity = kw['base_right'].data_ptr()
        if identity in seen: return result
        seen.add(identity)
        code = torch.empty(q.shape[0], 8, 4, 16, device=q.device, dtype=torch.bfloat16)
        conditional_router_query_code(q, kw['residual_query'], code)
        out = torch.empty_like(result)
        call = (q, b, res, kw['base_right'], kw['base_bias'], kw['residual_query'], kw['rope_cos'], kw['rope_sin'], code, out, kw['scale'], True)
        ext.conditional_router_page_lse(*call)
        torch.testing.assert_close(out, result, rtol=0, atol=0)
        cycles = []
        for _ in range(3):
            ext.conditional_router_page_lse(*call)
            cycles.append(ext.get_router_cycles().double().mean(0).cpu())
        mean = torch.stack(cycles).mean(0); fractions = mean / mean.sum()
        coefficients, candidate = transformed.allocate(q, b)
        def candidate_run():
            return transformed.run(q, b, res, kw['base_right'], kw['base_bias'], kw['rope_cos'], kw['rope_sin'], code, coefficients, candidate)
        candidate_run()
        assert bool(torch.isfinite(candidate).all())
        chosen, reference = linear_reference(q, b, res, kw, code)
        delta_linear = candidate[..., chosen] - reference
        assert float(delta_linear.square().mean().sqrt()) < .05
        delta = candidate - result
        selected = select_fixed_group_max_pages_cuda(result, pages_per_kv_head=62, pinned_prefix_pages=1, force_current_page=False)
        selected_candidate = select_fixed_group_max_pages_cuda(candidate, pages_per_kv_head=62, pinned_prefix_pages=1, force_current_page=False)
        overlap = (selected[..., :, None] == selected_candidate[..., None, :]).any(-1).float().mean()
        repeats = 2 if args.smoke else 7
        times = {
            'original_page_kernel': timing(lambda: normal.conditional_router_page_lse(*call), repeats),
            'original_wrapper_eager': eager_timing(lambda: original(q, b, res, **kw), repeats),
            'instrumented_split_kernel': timing(lambda: ext.conditional_router_page_lse(*call), repeats),
            'transformed_query_total': timing(candidate_run, repeats),
            'transformed_coefficients_once': timing(lambda: transformed._coefficients[(q.shape[0] * 32,)](
                q, kw['base_right'], kw['base_bias'], *coefficients, q.stride(0), q.stride(1), num_warps=4), repeats),
            'residual_query_projection': timing(lambda: conditional_router_query_code(q, kw['residual_query'], code), repeats),
        }
        v = current['v']; left = current['factors']['base_left_b16']
        newbase = torch.empty(*v.shape[:-1], 16, device=v.device, dtype=v.dtype)
        times['append_V128_to_B16'] = timing(lambda: torch.matmul(v, left, out=newbase), repeats)
        times['append_V80_to_B16_dimension_control'] = timing(lambda: torch.matmul(v[..., :80], left[:, :80], out=newbase), repeats)
        row = dict(layer=len(rows),tokens=b.shape[2],phase_fraction=dict(zip(PHASES, fractions.tolist())),
            mean_block_cycles=dict(zip(PHASES, mean.tolist())),timing=times,
            original_split_bitwise_equal=True,linear_reference_rmse=float(delta_linear.square().mean().sqrt()),
            linear_reference_max_abs=float(delta_linear.abs().max()),
            original_score_rmse=float(delta.square().mean().sqrt()),original_score_max_abs=float(delta.abs().max()),
            selected_page_overlap=float(overlap))
        rows.append(row)
        (root / 'layers.json').write_text(json.dumps(rows, indent=2) + '\n')
        print('LAYER', row['layer'], {k:round(v['median_ms'],4) for k,v in times.items()}, 'overlap',row['selected_page_overlap'],flush=True)
        if len(rows) == 1:
            (root / 'transformed_pages.ptx').write_text(transformed.last_page_kernel.asm['ptx'])
            (root / 'transformed_coefficients.ptx').write_text(transformed.last_coefficient_kernel.asm['ptx'])
            (root / 'triton_resources.json').write_text(json.dumps(dict(
                page_registers=transformed.last_page_kernel.n_regs,
                page_spills=transformed.last_page_kernel.n_spills,
                page_shared_bytes=transformed.last_page_kernel.metadata.shared), indent=2) + '\n')
            with (root / 'original.sass').open('w') as f:
                subprocess.run(['cuobjdump', '--dump-sass', normal.__file__], stdout=f, check=True)
        return result
    basis.conditional_router_page_lse = measured
    sys.argv = ['bench_shadow_native', '--method', 'basis16', '--length', '65536', '--batch', '1', '--output', str(root / 'runtime')]
    if args.smoke: sys.argv.append('--smoke')
    runpy.run_module('benchmarks.system.bench_shadow_native', run_name='__main__')
    assert len(rows) == 32
    totals = {name:sum(row['timing'][name]['median_ms'] for row in rows) for name in rows[0]['timing']}
    proxy = {name:sum(row['phase_fraction'][name] * row['timing']['original_page_kernel']['median_ms'] for row in rows) for name in PHASES}
    report = dict(status='complete',environment='basis',command='python -m benchmarks.system.profile_router_phases' + (' --smoke' if args.smoke else ''),
        model='Llama-3.1-8B base',gpu=torch.cuda.get_device_name(),batch=1,length=8192 if args.smoke else 65536,
        summed_median_kernel_ms=totals,phase_wall_proxy_ms=proxy,
        average_page_overlap=statistics.mean(row['selected_page_overlap'] for row in rows),
        max_score_difference=max(row['original_score_max_abs'] for row in rows),
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources + [__file__, 'benchmarks/system/transformed_query_router.py']},
        scope='First real decode query in each of 32 layers. Graph-replayed kernels exclude Python overhead. Phase proportions are clock64 block cycles from a split, instrumented layout, including barriers/stalls; scaled wall proxies are NOT independent measured phase latencies. Transformed query omits original intermediate BF16 K/RoPE rounding and is a diagnostic prototype, not a serving replacement. V80 is a dimension-only control; actual native values are Dense V128.')
    (root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Router phase experiment', '', report['scope'], '', f"Command: `{report['command']}`; environment: `basis`.", '', '| Timed component | Sum over 32 layers (ms) |', '|---|---:|']
    lines += [f'| {name} | {value:.4f} |' for name,value in totals.items()]
    lines += ['', '| Phase | Approximate wall proxy (ms) |', '|---|---:|']
    lines += [f'| {name} | {value:.4f} |' for name,value in proxy.items()]
    lines += ['', f"Mean selected-page overlap with original: {report['average_page_overlap']:.6%}.", f"Maximum page-score difference: {report['max_score_difference']:.6f}."]
    (root / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    print('SUMMARY',json.dumps(report),flush=True)


if __name__ == '__main__': main()
