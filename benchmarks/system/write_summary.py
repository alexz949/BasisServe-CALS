"""Render available measurements; missing evidence stays explicitly pending."""
import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


def table(rows, columns):
    if not rows:
        return 'Pending: no measurements available.\n'
    def cell(value):
        if value is None or value == '':
            return '—'
        if isinstance(value, float):
            return f'{value:.3f}'
        if isinstance(value, str) and re.fullmatch(r'-?\d+\.\d+(?:e[+-]?\d+)?', value):
            return f'{float(value):.3f}'
        return str(value).replace('|', '\\|')
    return '\n'.join(['| ' + ' | '.join(title for key, title in columns) + ' |',
        '| ' + ' | '.join('---' for _ in columns) + ' |',
        *['| ' + ' | '.join(cell(row.get(key)) for key, title in columns) + ' |' for row in rows]]) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/system_benchmarks/l40s'))
    args = parser.parse_args()
    root = args.output
    def read_csv(name):
        path = root / name
        return list(csv.DictReader(path.open())) if path.exists() else []
    def read_json(name):
        path = root / name
        return json.loads(path.read_text()) if path.exists() else {}
    router = read_csv('router.csv')
    tp = read_csv('tp_collective.csv')
    e2e = read_csv('e2e_tp4.csv')
    common = read_csv('common_offload.csv')
    lines = ['# L40S systems benchmark results', '',
        'Generated UTC: ' + datetime.now(timezone.utc).isoformat(), '',
        '**Status: completed with the capacity and measurement limitations below.**' if read_json('completion_review.json').get('status')=='complete' else '**Status: partial until all required grids, profiling and correctness evidence are reviewed.**', '',
        'Model: Llama-3.1-8B **base**, BF16, TP4 for model/collective tests. Existing two-sided KL allocation averages V96; actual layer ranks are retained. B16R16, Page32, hard 2048 including sink32 and recent64. The earlier Instruct/Dense-V capacity scan is a separate experiment.', '',
        'Environment: `basis`. Rebuild tables with `python -m benchmarks.system.summarize`, then run `python -m benchmarks.system.write_summary` and `python -m benchmarks.system.plot_figures`. Per-run JSON metadata and E2E trial records contain exact measurement commands, versions, source hashes and occupancy snapshots. [input_identity.json](input_identity.json) records model, factor and calibration-input content hashes at its stated observation time.', '',
        '## 1. Hardware and topology', '',
        'Four L40S GPUs on lovelace. Existing external GPU0 occupancy is allowed and recorded per run; these are not empty-machine measurements. See [hardware.txt](hardware.txt) and [hardware.json](hardware.json).', '',
        table(read_json('hardware.json').get('gpu_numa', []), [('gpu','GPU'),('pci_bus','PCI bus'),('numa_node','NUMA'),('local_cpu_list','Local CPUs')]),
        'Pinned allocation status and physical NUMA placement are checked separately with sampled host pages. Detailed reports: `numa_validation_rank*.json`; per-run offload audits verify the actual buffers.', '',
        '## 2. NCCL and host-to-device bandwidth', '',
        'NCCL uses default algorithm selection. Bus bandwidth below is the analytical collective convention, not a PCIe hardware counter. Full sweep: [nccl.csv](nccl.csv).', '']
    nccl = read_csv('nccl.csv')
    lines += [table([r for r in nccl if int(r['input_bytes_per_rank']) in (256, 16384, 1048576, 16777216)],
        [('operation','Collective'),('input_bytes_per_rank','Input B/rank'),('p50_ms','p50 ms'),('algorithm_gbps','Algorithm GB/s'),('bus_gbps','Analytical bus GB/s')]),
        'Pinned H2D 1 GiB transfer per GPU; all six transfer sizes are in [h2d.csv](h2d.csv).', '',
        table([r for r in read_csv('h2d.csv') if int(r['bytes']) == 2**30], [('gpu','GPU'),('numa_node','NUMA'),('gbps','H2D GB/s'),('p50_ms','p50 ms')]),
        '## 3. TP output block', '',
        f'{len(tp)}/576 layer/batch/method records. Values below average the 32 layer p50 values; they are not a full-model latency. [Detailed CSV](tp_collective.csv).', '',
        'LR-AR embeds the same C1 per-head coordinates in a zero-initialized global coordinate buffer and AllReduces it. LR-AG gathers those coordinates. Both use identical real C1 decoders. This is not an independently fitted global low-rank AR model.', '']
    groups = defaultdict(list)
    for row in tp:
        groups[(int(row['batch']), row['method'])].append(row)
    means = [dict(batch=b, method=m, layers=len(rs), collective_us=1000*mean(float(r['collective_ms']) for r in rs),
        total_us=1000*mean(float(r['total_ms']) for r in rs)) for (b,m),rs in sorted(groups.items())]
    lines += [table(means, [('batch','Batch'),('method','Method'),('layers','Layers'),('collective_us','Collective µs'),('total_us','Block µs')]),
        'Figure A: [PNG](figure_A.png). Synthetic attention-output inputs, actual weights/factors; AR/AG coordinate and output equality is checked per case.', '',
        '## 4. Router only', '',
        f'{len(router)} numeric records out of48 requested cases; three LRQK configurations run out of memory during native prefill. Representative64K, batch1 below. [All lengths and batches](router.csv). Route timing ends at selected IDs; query preprocessing is separately measured and included in joint total.', '',
        table([r for r in router if r['length']=='65536' and r['batch']=='1'], [('method','Router'),('query_p50_us','Query µs'),('route_p50_us','Route µs'),('total_p50_us','Joint µs'),('routing_state_bytes','State bytes'),('min_support_per_query','Min/query'),('max_support_per_query','Max/query')]),
        'State accounting includes the Basis Base16/Residual16 coordinate cache, compact RoPE tables and retained factor tensors. At64K/batch1 the coordinate cache alone is32 MiB; the RoPE tables add about16 MiB. The V tail is not scanned. LRQK includes active exact K needed for online query updates and allocated code capacity; its JSON also reports scan state separately. ShadowKV reconstruction factors are excluded from routing state and reported separately in raw JSON. Output workspaces and validation-only copies are excluded. CSV columns separate the added position/factor/fixed-ID accounting. Logical coordinate bytes are not total DRAM traffic.', '',
        table([r for r in router if r['length']=='65536' and r['batch']=='1'], [('method','Router'),('logical_coordinate_scan_bytes','Logical coordinate/landmark B'),('position_table_bytes','RoPE table B'),('logical_effective_gbps','Logical effective GB/s')]),
        'Logical effective GB/s divides unique coordinate/landmark bytes plus position tables by route p50. It excludes small query/parameter reads, output writes and repeated/cache-served accesses; it is not measured DRAM bandwidth.', '',
        'ShadowKV uses upstream CUTLASS landmark routing, rank160/chunk8. Native CPU-cache alignment gives routed2048 + outlier384 + local64 =2496 at aligned lengths. Loki uses PCA rank32 with independent top2048/query. LRQK uses official rank32/top2048 plus native lite64, after 64 continuous teacher-forced online updates.', '',
        'LRQK native import permits TF32 for FP32 matrix multiplications, recorded as `torch_matmul_tf32=true` in formal metadata. BF16 stored routing codes and FP32 solve tensors do not imply that every internal multiplication is full IEEE FP32. The upstream numerical settings are preserved.', '',
        'LRQK formal setup outcomes (a missing timing row must not be treated as zero latency):', '',
        table([r for r in read_json('router_raw.json').get('outcomes', []) if r['method']=='lrqk'], [('length','Context'),('batch','Batch'),('status','Outcome'),('log','Log')]),
        '## 5. Exact-K offload', '',
        '64K, batch1, real router pages, V resident. Staged rows are independently measured components. The mapped kernel fuses host reads, QK, softmax and PV; its total cannot be decomposed by summing the staged measurements.', '',
        table(read_csv('k_offload.csv'), [('budget','Budget'),('logical_k_bytes','Unique K B'),('fetch_us','Staged fetch µs'),('qk_us','QK µs'),('pv_us','Softmax/PV µs'),('staged_total_us','Staged total µs'),('mapped_fused_total_us','Mapped fused µs')]),
        'H2D DMA payload and logical mapped-host reads are reported separately. Actual PCIe bus traffic is unavailable unless explicitly recorded by a profiler.', '',
        table(read_csv('k_offload.csv'), [('budget','Budget'),('staged_effective_payload_gbps','Effective staged payload GB/s')]),
        'Effective staged payload throughput includes the whole fetch interval, including host gather and dispatch; the separate H2D sanity test measures copy bandwidth.', '',
        '## 6. Single-layer attention operator', '',
        'Layer3, full32Q/8KV heads, batch1. Sparse totals include query preprocessing and routing. These are not TP4 full-model speeds.', '',
        table(read_csv('sparse_operator.csv'), [('length','Context'),('dense_local_us','Dense local µs'),('dense_offload_us','Dense offload µs'),('sparse_local_us','Sparse local µs'),('sparse_offload_us','Sparse offload µs'),('dense_offload_over_sparse_offload','Offload speedup')]),
        'Figure B and full raw timing distributions accompany [sparse_operator.csv](sparse_operator.csv).', '',
        'Staged breakdown below is a separate implementation. Its total uses pre-captured routing/attention around the host-dependent fetch and supersedes the earlier eager-routing staged total. These component medians are not the internal costs of the mapped fused kernel.', '',
        table(read_csv('sparse_operator.csv'), [('length','Context'),('staged_query_us','Query µs'),('staged_route_us','Route µs'),('staged_fetch_us','Fetch µs'),('staged_qk_us','QK µs'),('staged_softmax_pv_us','Softmax/PV µs'),('staged_total_us','Measured staged total µs')]),
        '## 7. Common exact-K backend', '',
        f'{len(common)}/4 methods. Native routing + common exact-K fetch/attention; **ShadowKV here is not a full official serving-system reproduction**. Fetch deduplicates the GQA union but attention retains each query head\'s original support.', '',
        table(common, [('method','Router'),('query_p50_us','Query µs'),('route_p50_us','Route µs'),('fetch_p50_us','Fetch µs'),('exact_qk_softmax_pv_fused_p50_us','Attention µs'),('total_p50_us','Total µs'),('unique_tokens','Unique K tokens'),('h2d_dma_payload_bytes','DMA payload B')]),
        '## 8. TP4 full-model decode', '',
        f'{len(e2e)}/48 terminal configurations currently available. Generate256 tokens; discard the first32 generated tokens, retaining224 decode forwards. TTFT includes prefill and first-token selection. Peak columns are the maximum per-rank PyTorch allocation; host K is summed over four ranks. CUDA-event steady latency and synchronized wall throughput have distinct timing boundaries.', '',
        table(e2e, [('length','Context'),('batch','Batch'),('mode','Mode'),('status','Status'),('ttft_seconds','TTFT s'),('steady_decode_mean_ms','Decode ms'),('steady_aggregate_tokens_per_second','Tokens/s'),('max_prefill_gpu_gib','Prefill GiB'),('max_decode_gpu_gib','Decode GiB'),('total_host_key_gib','Host K GiB')]),
        'Decode backend mapping: Dense uses `flash_attn_with_kvcache` with16 splits; C1 full attention uses the general CUDA GPU paged kernel with32 splits and split transformed V; sparse-local uses that GPU kernel on selected tokens; offload uses the mapped-host fused CUDA kernel on the same selected support. All modes use native FlashAttention prefill. Dense versus C1 therefore measures the implemented systems, not an isolated effect of V compression with a matched attention kernel.', '',
        'See [e2e_tp4.csv](e2e_tp4.csv) and individual `e2e/*/trial.json` files for communication bytes, support counts, external occupancy and commands. OOM is a capacity outcome, not a numeric timing result.', '',
        '## 9. Profiling and observed bottlenecks', '',
        'The current Basis router is slower than Loki and ShadowKV in the measured 64K/batch1 route-only cases. Offload can benefit from fetching fewer keys even when routing itself is not the fastest. Communication-only gains do not guarantee a faster total output block; projection, decoder and launch costs remain included.', '',
        'Nsight Systems records four steady TP4 decode steps at64K/batch1. Mapped K read/QK/softmax/PV is one fused range; separate internal durations would be misleading. Nsight Compute targets only the warmed Basis router. Counter data must distinguish L2/cache traffic from DRAM and logical bytes.', '']
    ncu_status=read_json('ncu_status.json')
    if ncu_status:
        lines += ['', 'NCU availability: **'+ncu_status['status']+'**. '+ncu_status['message']+
            ' DRAM bytes, measured bandwidth, occupancy, arithmetic intensity and stalls are unavailable, not zero. See [ncu_status.json](ncu_status.json).']
    review=read_json('profile_review.json')
    if review:
        lines += ['', 'Accepted trace: ['+review['trace']+']('+review['trace']+'). Four GPUs each contain128 router calls and128 mapped attention calls across four decode steps. All four processes completed256 generated tokens. The earlier2023 trace was rejected after SIGSEGV and incomplete coverage; it is retained as failure evidence.', '',
            f'Profiler window averages {review["profiled_mean_step_ms"]:.2f} ms/step, versus {review["unprofiled_mean_step_ms"]:.2f} ms/step in the formal run. Use the formal run for performance claims. [Coverage review](profile_review.json).', '']
        profile=read_json('e2e_trace_nsys2024_analysis.json')
        stages=[]
        for row in profile.get('nsys',{}).get('nvtx_gpu_proj_sum',{}).get('rows',[]):
            if row['Range'].startswith(':') and not row['Range'].endswith('basis_full_decode_window'):
                stages.append(dict(stage=row['Range'][1:],calls=row['Range Instances'],gpu_us=float(row['Proj Avg (ns)'])/1000,cpu_us=float(row['Total Range Time (ns)'])/int(row['Range Instances'])/1000))
        lines += [table(stages,[('stage','NVTX stage'),('calls','Calls over4 GPUs/4 steps'),('gpu_us','GPU projected mean µs'),('cpu_us','CPU range mean µs')]),
            'The trace shows substantial collective/synchronization time and host launch overhead, especially in page selection. NCCL kernel duration can include waiting for another rank; this does not establish a PCIe bandwidth bottleneck. Ranges overlap and their means must not be added to reconstruct the model step.', '']
    lines += ['', '## 10. Correctness and limits on interpretation', '',
        'Basis transform uses T=[A16,Q-perp], E′=ET and D′=T⁻¹D; Base16 and the remaining coordinates have separate storage. Per-layer FP64/BF16 checks are in `basis_validation_rank*.json`. Real-capture router IDs are checked against references; poisoning the unused tail checks that routing does not read it.', '',
        'The 4K batch1/4 full-model smoke compares Dense prefill against native HF, verifies C1 prefill equality across three cache modes, and verifies sparse-local/offload generated-token equality. See [e2e_smoke_validation.json](e2e_smoke_validation.json). This establishes small-case implementation consistency, not language-model quality at long contexts.', '',
        'The common backend test checks independent query supports, duplicates, invalid IDs, changing requests, buffer reuse and exact-K attention against reference computation. Microbenchmark selected-K validation does not imply equality to dense full-support attention.', '',
        'Run `python -m benchmarks.system.audit_results` to check raw timing distributions, sampled NUMA pages, recorded correctness flags, E2E timing arithmetic, source hashes and missing grid points. [record_audit.json](record_audit.json) documents the inspected files and coverage; it is not a completion certificate.', '',
        '**LRQK timing qualification:** the native online update retains eager temporary allocations and convergence-related host synchronization. Its measured preprocessing/joint totals therefore do not satisfy the allocator-excluded microbenchmark protocol and must not be presented as isolated kernel latency. The route-only scan has a separate timing boundary. Native precision and update semantics have not been changed to make timing look better.', '',
        'Nsight Compute hardware counters are unavailable due to permissions. Hardware PCIe counters are not inferred from tensor size. Existing external GPU activity can affect latency as well as capacity.', '',
        'LRQK32K/batch8 initially failed an over-strict FP32-rounded top-k check. On one query head, a native BF16 score of−104.0 corresponded to FP32−103.7499847, which rounds to−103.5 and changes one boundary token. Native official IDs remained exact. Validation now checks native BF16 IDs exactly and bounds independent FP32 score error separately; smoke and the original device-index regression passed. The original failed report and diagnostic are retained. The formal configuration also passed an unchanged-gate rerun on GPU0.', '',
        'All results are systems measurements, with synthetic inputs in the TP-block test and real captured activations in router/operator tests. They do not establish RULER accuracy or prove that a V-coordinate change has no long-context quality impact.', '']
    frozen=[]
    for path in sorted(root.glob('lrqk_frozen_update_cusolver*.json')):
        if 'smoke' in path.name:continue
        row=read_json(path.name);timing=row['query_preprocessing']
        frozen.append(dict(length=row['length'],batch=row['batch'],p50_us=timing['p50_us'],p95_us=timing['p95_us'],
            bitwise=row['native_output_bitwise_equal'],source=path.name))
    frozen.sort(key=lambda row:(row['length'],row['batch']))
    if frozen:
        lines += ['### LRQK allocator-excluded supplementary measurement', '',
            'Native default solver dispatch could not be captured. This separate experiment prefers cuSOLVER, records the native scalar convergence decisions at a fixed final online state, and captures the unchanged tensor algebra. Each case requires bitwise equality of all update outputs against the native reference. CUDA graph replay excludes allocation and CPU branch synchronization; solver dispatch differs and this is not a general continuous-online implementation. The main tables retain native eager timings. Do not add supplementary update p50 to route p50 and call that a measured joint total.', '',
            table(frozen,[('length','Context'),('batch','Batch'),('p50_us','Update p50 µs'),('p95_us','Update p95 µs'),('bitwise','Native outputs bitwise equal'),('source','Raw record')]),
            'Command: `python -m benchmarks.system.bench_lrqk_frozen_update --cusolver --length T --batch B` in `basis`;100 warmups and500 measured replays. The64K/batch1 run used the equivalent default length/batch arguments.', '']
    pipelines=[]
    for path in sorted(root.glob('lrqk_frozen_pipeline_t*.json')):
        if 'smoke' in path.name:continue
        row=read_json(path.name);common=row['common_backend']
        pipelines.append(dict(length=row['length'],batch=row['batch'],query_us=row['query_preprocessing']['p50_us'],
            route_us=row['route_to_ids']['p50_us'],joint_us=row['total_with_preprocessing']['p50_us'],
            common_us=None if common is None else common['total_with_query_preprocessing']['p50_us']))
    pipelines.sort(key=lambda row:(row['length'],row['batch']))
    if pipelines:
        lines += ['Joint fixed-state timings, measured directly rather than summed from stages. The same cuSOLVER and frozen-branch qualification applies. The common-backend measurement checks the native IDs and exact selected attention, with unchanged physical support.', '',
            table(pipelines,[('length','Context'),('batch','Batch'),('query_us','Query µs'),('route_us','Route µs'),('joint_us','Measured joint µs'),('common_us','Measured common backend µs')]),
            'Command: `python -m benchmarks.system.bench_lrqk_frozen_pipeline --length T --batch B`, with `--common` at64K/batch1; `basis`,100 warmups/500 samples. Raw records are `lrqk_frozen_pipeline_t*_b*.json`.', '']
    (root/'SUMMARY.md').write_text('\n'.join(lines))
    print(dict(summary=str(root/'SUMMARY.md'), router_cases=len(router), e2e_terminal_cases=len(e2e)))


if __name__ == '__main__':
    main()
