"""Summarize coverage, lost mass, output fidelity, and matched timing scopes."""
import json
from pathlib import Path
import statistics
import numpy as np


def flat(value):
    if isinstance(value,list):return [y for x in value for y in flat(x)]
    return [value]


def main():
    root=Path('results/system_benchmarks/two_stage')
    traces=[json.loads(p.read_text()) for p in sorted(root.glob('trace_w*/diagnostic.json'))]
    rows=[r for t in traces for r in t['records']]
    lines=['# Exact-K coarse screening plus B16R16 refinement','',
        'Conda environment: basis. Slurm on lovelace, one L40S, TP1, Llama-3.1-8B base, 64K context, batch 1, Dense V128, original Wo. Full reference uses the new 8-warp register-consumed B16R16 router, not the older slower implementation. Final support is the existing 62 historical Page32 pages (including sink page 0), masked at the historical boundary, plus exactly recent64; no budget expansion.', '',
        '## Implementation and interpretation','',
        'Exact BF16 post-RoPE K supplies coordinatewise page minima/maxima. FP32 coarse dot products add log(valid historical token count). Four heads normalize independently over non-forced eligible pages, then merge by max in log space. Coarse top-k is 512 total per physical KV head, sorted by original page ID. Page0 and every page intersecting recent64 are forced into these 512. Recent-only pages have fine score -infinity. Candidate-only MMA maps page IDs to original positions; it does not gather full routing codes into a new sequence. Fine normalization is over candidates and is not equivalent to full normalization.', '',
        'Metadata construction occurs before CPU K offload. A 64-token GPU ring delays historical-summary updates until each token exits recent64. The coarse scoring and normalization kernels are Triton; selection uses torch.topk and sort, and final selection reuses the existing CUDA selector. This is a first implementation, not a claim of fully optimized selection. No CPU historical K is read during routing.', '',
        'Min/max page scoring is borrowed from [Quest](https://arxiv.org/html/2406.10774v2); the normalized GQA candidate generator and B16R16 cascade here are the tested adaptation, not an official Quest reproduction.', '',
        '## Validation','',
        'Synthetic checks cover 30 incremental metadata checkpoints, page-boundary lengths 127/128/129 and 8K/64K, multiple batches/heads, forced recent pages, unique sorted candidates, and candidate/full score equivalence. 8K two-stage model smoke checks actual attention and finite logits. Real trace additionally asserts bitwise-equal candidate scores versus corresponding full-scan scores (before renormalization).','',
        f'Completed diagnostic windows: {[t["window"] for t in traces]}; total layer/query samples: {len(rows)}. Each 64K trace uses full-router generation, collecting step 0 and steps 10–19 across all 32 layers. All three methods receive identical queries, K/V and positions. Dense output uses FP32 QK/softmax/PV with TF32 disabled. These are local fixed-input diagnostics, not RULER accuracy.', '',
        '## Candidate quality','',
        '| Window | Adaptive-page candidate recall mean / p1 / minimum | Lost reference head mass mean / p95 / maximum | Final two-stage page overlap |','|---|---|---|---|']
    for name,rr in [(str(t['window']),t['records']) for t in traces]+([('all',rows)] if rows else []):
        c=np.array([x for r in rr for x in flat(r['candidate_recall'])]);e=np.array([x for r in rr for x in flat(r['lost_reference_mass'])]);o=np.array([x for r in rr for x in flat(r['two_overlap'])])
        lines.append(f'| {name} | {c.mean():.6%} / {np.quantile(c,.01):.6%} / {c.min():.6%} | {e.mean():.6%} / {np.quantile(e,.95):.6%} / {e.max():.6%} | {o.mean():.6%} |')
    lines+=['','Candidate recall excludes forced sink/recent-intersecting pages. Lost reference mass uses full B16R16 page probabilities over that same non-forced set. Final page overlap includes sink among 62 historical pages. A low mean does not rule out bad individual heads/queries.','',
        '## Attention output','',
        '| Window | Method | Dense-relative MSE (sum squared error / sum squared dense output) | Mean retained mass | Mean non-sink/non-recent retained mass | Mean output delta vs full router |','|---|---|---|---|---|---|']
    for name,rr in [(str(t['window']),t['records']) for t in traces]+([('all',rows)] if rows else []):
        for method in ['full','coarse','two']:
            ss=[r['output'][method] for r in rr];mse=sum(s['error_squared'] for s in ss)/sum(s['dense_squared'] for s in ss)
            delta=statistics.mean(s.get('delta_vs_full_mse',0) for s in ss)
            lines.append(f'| {name} | {method} | {mse:.8f} | {statistics.mean(s["mass"] for s in ss):.4%} | {statistics.mean(s["non_sink_recent_mass"] for s in ss):.4%} | {delta:.8f} |')
    lines+=['','## Complete routing latency','',
        'CUDA-graph microbench of full routing calls, including output allocations within capture, coarse scoring, candidate normalization/top512/sort, fine projection/scan, final selection and original-ID mapping. Summary construction/update, attention, fetch and other model computation are excluded here; normal decode below includes incremental updates. Timings are sums of layer medians averaged over steps 10 and 19. Diagnostic trace wall time is not serving performance.','',
        '| Window | Full B16R16 ms/step | Coarse-only ms/step | Two-stage ms/step |','|---|---|---|---|']
    for t in traces:
        rr=[r for r in t['records'] if r['step'] in [10,19] and 'routing_ms' in r]
        totals={m:sum(r['routing_ms'][m] for r in rr)/2 for m in ['full','coarse','two']}
        lines.append(f'| {t["window"]} | {totals["full"]:.4f} | {totals["coarse"]:.4f} | {totals["two"]:.4f} |')
    lines+=['','## Normal continuous decode','',
        'One GPU allocation, order full/coarse/two/two/coarse/full, fresh processes, 100 generated tokens; steady CUDA median discards the first ten steps. No full-scan reference is evaluated in coarse/two serving. Prefill is excluded. Different algorithms can change generation and subsequent cache reuse; paired microbench above isolates routing cost on matched queries.','',
        '| Method | Mean of two steady CUDA medians ms/step | Native wall ms/step | Matching recorded decode-input positions vs full (first run) |','|---|---|---|---|']
    reports={}
    for method in ['full','coarse','two']:
        pp=[root/f'{method}_w64_r{r}/optimized_b16_w4_r0/summary.json' for r in [0,1]]
        if all(p.exists() for p in pp):reports[method]=[json.loads(p.read_text()) for p in pp]
    for method,rr in reports.items():
        same='pending'
        if 'full' in reports:
            x=rr[0]['decode_input_ids'];y=reports['full'][0]['decode_input_ids'];same=f'{sum(a==b for a,b in zip(x,y))}/{len(y)}'
        lines.append(f'| {method} | {statistics.mean(r["cuda_median_after10_ms"] for r in rr):.4f} | {statistics.mean(r["native_wall_ms"] for r in rr):.4f} | {same} |')
    lines+=['','Native whole-loop wall measurements can include cold Triton compilation at newly encountered page counts; they are retained for transparency and are not used as steady speedup estimates. The coarse-only control still retains the common Base/Residual code cache and append computation, while skipping its scan; it is not a fully optimized standalone Quest runtime. The inherited `w4` directory label is the old harness slot-attention parameter; the full and candidate-only router kernels both use 8 warps.','',
        'Per-run steady decode medians:','']
    for method,rr in reports.items():
        lines.append(f'- {method}: '+', '.join(f'{r["cuda_median_after10_ms"]:.6f} ms (finite logits={r["all_decode_logits_finite"]}, slot hit fraction={r["hit_fraction"]:.6f})' for r in rr))
    if traces:
        lines+=['',f'Metadata allocation (min/max plus recent ring): {traces[0]["metadata_bytes"]/2**20:.3f} MiB. Nominal 64K min/max alone is 256 MiB; capacity slack and ring account for the remainder. Prefill-summary event totals include cold compilation if present and are retained in diagnostic JSON, not interpreted as steady kernel cost.']
    lines+=['','## Commands','', '```bash','python -m benchmarks.system.validate_two_stage','python -m benchmarks.system.run_two_stage --phase trace','python -m benchmarks.system.run_two_stage --phase decode','python -m benchmarks.system.summarize_two_stage','```','',
        'Implementation: benchmarks/system/two_stage_router.py; real benchmark: bench_two_stage.py; runner: run_two_stage.py. The reusable native benchmark now accepts a diagnostic window index. Production router/cache defaults remain unchanged. No GitHub commit or push.','']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    if rows:
        worst=sorted(rows,key=lambda r:min(flat(r['candidate_recall'])))[:20]
        (root/'worst_cases.json').write_text(json.dumps(worst,indent=2)+'\n')
    print('\n'.join(lines[:50]))

if __name__=='__main__':main()
