"""Record measured router and continuous-decode results without mixing scopes."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


def main():
    root=Path('results/system_benchmarks/register_router')
    validation=json.loads((root/'validation.json').read_text())
    lines=['# Register-consumed MMA router','',
        'Environment: `basis`; Slurm, one L40S per job on lovelace. Model: meta-llama/Llama-3.1-8B (base, not Instruct), snapshot d04e592bb4f6aa9cfee91e2e20afa771667e1d4b, TP1. K128, GQA4, Page32; Dense V128 and original Wo in the model benchmark. Nominal 2048 budget, existing sink/recent and slot-cache policy unchanged. B8R8 truncates existing B16R16 factors; no refit.', '',
        '## Implementation','',
        'A warp owns 16 tokens. The factor is transposed and RoPE-paired while loading shared memory; no new persistent model-side factor. BF16 m16n8k8 / m16n8k16 reconstructs four pairs per output tile. Each lane consumes two token pairs in registers, preserves explicit BF16 rounding, and accumulates four query scores. Four-lane reductions write only token scores. The residual and Page-LSE formulas are unchanged. The full FP32 reconstructed-K shared buffer is removed. The 4-warp/8-warp CTA covers two/four independent pages. This is a benchmark candidate; production dispatch is unchanged.', '',
        '## Correctness','',
        f'Synthetic status: {validation["status"]}; {len(validation["cases"])} cases. Ranks 8/16, warps 4/8, lengths 1/31/32/33/63/64/65/127/128/129/257/8192/65536. Short cases also use an independent FP32 matrix product with explicit BF16 boundary reference; multiple batches and KV heads are covered. Tolerance: atol=rtol=0.003. Maximum candidate/current page-LSE difference: {max(r["max_abs"] for r in validation["cases"]):.9g}.', '',
        'Changing MMA tiling and QK reduction order is not bitwise-equivalent by construction. Real selected-page comparisons are reported below; no new RULER quality evaluation is claimed.','',
        '## Synthetic 64K router','',
        '| Rank | Warps | Current ms/layer | Candidate ms/layer | Latency reduction |','|---|---|---|---|---|']
    for r in validation['cases']:
        if r['tokens']==65536:
            x=r['baseline']['median_ms'];y=r['candidate']['median_ms']
            lines.append(f'| {r["rank"]} | {r["warps"]} | {x:.6f} | {y:.6f} | {100*(1-y/x):.2f}% |')
    lines+=['','## Real-query trace','',
        'Matched inputs from continuous decode; 352 samples per trace (step 0 and steps 10–19 across 32 layers). Timings sum per-layer CUDA-graph microbench medians at steps 10 and 19, including residual-query projection, excluding selector. Both versions use preallocated outputs and the same C++ API. Probed decode wall time is not a serving metric.','',
        '| Rank | Warps | Current ms/step | Candidate ms/step | Reduction | Minimum / mean page overlap | Max score difference |','|---|---|---|---|---|---|---|']
    for p in sorted(root.glob('trace_b*/summary.json')):
        r=json.loads(p.read_text());s=r['router_samples'];x=r['router_ms_per_step']['original'];y=r['router_ms_per_step']['candidate']
        lines.append(f'| {r["rank"]} | {r["warps"]} | {x:.4f} | {y:.4f} | {100*(1-y/x):.2f}% | {min(t["page_overlap"] for t in s):.8f} / {statistics.mean(t["page_overlap"] for t in s):.8f} | {max(t["max_abs"] for t in s):.6g} |')
    lines+=['','## Continuous decode','',
        'Each rank uses same-GPU ABBA order: current, candidate, candidate, current; 64K input, batch 1, 100 generated steps, fresh process each run, vector fetch and existing slot attention in both paths. Report the mean of two per-run steady CUDA medians, discarding the first ten steps. Prefill and placement are excluded.','',
        '| Rank | Current ms/step | Candidate ms/step | Reduction | Speedup | All recorded token sequences equal |','|---|---|---|---|---|---|']
    for rank in [8,16]:
        paths=[root/m/f'optimized_b{rank}_w8_r{i}'/'summary.json' for m in ['baseline','candidate'] for i in [0,1]]
        if all(p.exists() for p in paths):
            runs=[json.loads(p.read_text()) for p in paths]
            x=statistics.mean(r['cuda_median_after10_ms'] for r in runs[:2]);y=statistics.mean(r['cuda_median_after10_ms'] for r in runs[2:])
            equal=all(r['decode_input_ids']==runs[0]['decode_input_ids'] for r in runs)
            lines.append(f'| {rank} | {x:.4f} | {y:.4f} | {100*(1-y/x):.2f}% | {x/y:.4f}x | {equal} |')
    lines+=['','Per-run steady CUDA median and native decode wall latency (the latter includes startup within the decode loop):','',
        '| Rank | Path | Repeat | CUDA ms/step | Native wall ms/step | Finite logits |','|---|---|---|---|---|---|']
    for p in sorted(root.glob('*/optimized_b*_w8_r[01]/summary.json')):
        r=json.loads(p.read_text())
        lines.append(f'| {r["rank"]} | {r["register_router_mode"]} | {p.parent.name[-1]} | {r["cuda_median_after10_ms"]:.4f} | {r["native_wall_ms"]:.4f} | {r["all_decode_logits_finite"]} |')
    lines+=['','B16 4-warp trace additionally ran the previous slot/fetch microbench probes; subsequent traces only probe the router. Neither trace wall time is used for speedup. Continuous-decode jobs have no such probes.','',
        'Files: `register_router_body.cuh` contains the candidate kernel body; `register_router.py` builds isolated current/candidate extensions; `validate_register_router.py` checks mathematics and boundaries; `bench_register_router.py` and `run_register_router.py` run matched real workloads. `bench_local_kernels.py` now permits the experiment output root and skipping unrelated slot probes, and uses symmetric preallocated C++ calls for router A/B timing.']
    resources=[]
    for p in Path('/home/zhangal/.cache/torch_extensions/py311_cu124').glob('register_router*/*.so'):
        text=subprocess.check_output(['/deac/opt/rocky9-noarch/nvidia/cuda/12.3.2/bin/cuobjdump','--dump-resource-usage',str(p)],text=True).splitlines()
        for i,line in enumerate(text):
            if 'Function ' in line and 'conditional_router_page_lse_kernel' in line:resources.append(dict(module=p.stem,resources=text[i+1].strip()))
    (root/'resources.json').write_text(json.dumps(resources,indent=2)+'\n')
    lines+=['','## Compiled resources','', 'Current kernels use dynamic shared memory (23,680 bytes for B16); cuobjdump reports only their static shared portion. Candidates use static shared.','', '```']+[f'{r["module"]}: {r["resources"]}' for r in resources]+['```','',
        '## Commands','', 'Conda environment: `basis`. CUDA 12.3.2, TORCH_CUDA_ARCH_LIST=8.9, MAX_JOBS=2.','', '```bash',
        'python -m benchmarks.system.validate_register_router',
        'python -m benchmarks.system.run_register_router --phase trace',
        'python -m benchmarks.system.run_register_router --phase decode --rank 16 --warps 8',
        'python -m benchmarks.system.run_register_router --phase decode --rank 8 --warps 8','```','',
        'Commands for decode are applicable when the corresponding completed runs appear above. Exact child commands and progress are in trace.log / decode_b*.log and the per-run summary JSON. No commit or push performed.','']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    sources=list(Path('benchmarks/system').glob('*register_router*'))+[Path('benchmarks/system/bench_local_kernels.py')]
    (root/'sources.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.is_file()},indent=2)+'\n')
    print('\n'.join(lines[:50]))

if __name__=='__main__':main()
