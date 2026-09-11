"""Write the paired uniform96 comparison report after both audits complete."""
from pathlib import Path
import sys
import json

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.v96kl_common import read_json,sha256


def main():
    arms=['full','recent_extra','recent_fixed','lrqk','shadowkv']
    lines=['# Uniform V96: paired RULER comparison','',
        'Qwen3-8B-Base and Llama-3.1-8B Base. BF16, 11 tasks x8 examples/model; primary87 excludes global index86.',
        'Every new arm passed protocol, prompt identity, EOS/cap, decoding and scoring checks, and its first generated token matches the uniform ShadowKV reference.',
        'Within-model prompts are paired. Cross-model prompts differ because their tokenizers differ.','',
        '| Model | Full-K | Base16/R16 extra64 | Base16/R16 fixed2048 | LRQK | ShadowKV |',
        '|---|---:|---:|---:|---:|---:|']
    results={}
    for family in ['qwen3','llama31']:
        path=ROOT/f'results/evaluation/{family}_uniform96_compare/result.json'
        result=read_json(path)
        assert result['status']=='complete' and result['verified']==440 and result['first_token_agreement']
        results[family]=result
        lines.append('| '+family+' | '+' | '.join(f"{result['means87'][a]:.6f}" for a in arms)+' |')
    for family,result in results.items():
        lines += ['',f'## {family}', '', '| Scope | '+' | '.join(arms)+' |', '|---|'+'---:|'*len(arms),
            '| All88 | '+' | '.join(f"{result['means88'][a]:.6f}" for a in arms)+' |',
            '| Delta versus Full-K (87) | '+' | '.join(f"{result['means87'][a]-result['means87']['full']:+.6f}" for a in arms)+' |',
            '', 'Per-task scores use all8 examples, including sample86 in QA2.', '',
            '| Task | '+' | '.join(arms)+' |','|---|'+'---:|'*len(arms)]
        for task,means in result['tasks'].items():
            lines.append('| '+task+' | '+' | '.join(f'{means[a]:.6f}' for a in arms)+' |')
        bank=ROOT/f'results/checkpoints/{family}_uniform96_b16r16'
        records=[read_json(bank/f'layer_{i:03d}.json') for i in range(36 if family=='qwen3' else 32)]
        assert all(d['sha256']==sha256(bank/f"layer_{d['layer']:03d}.safetensors") for d in records)
        pcg=[d['losses']['b16_r16']['final_query_maximum_relative_residual'] for d in records]
        fit=[d['losses']['b16_r16']['fit_page_fisher_nmse'] for d in records]
        validation=[d['losses']['b16_r16']['validation_page_fisher_nmse'] for d in records]
        lines += ['',f'Bank: {len(records)} authenticated layers. Mean layer fit/diagnostic Page-Fisher NMSE: {sum(fit)/len(fit):.6f} / {sum(validation)/len(validation):.6f}.',
            f'Maximum recorded final query-PCG relative residual: {max(pcg):.6g}; {sum(v>1e-5 for v in pcg)} layers exceed the requested1e-5 tolerance. Fixed40 sweeps/PCG100 were retained; no factor selection used validation.']
    lines += ['', '## Protocol and execution','',
        'HF revision `f1a6253b5d5c747a2475cbf9e704a67d97930b31`, uniform C1 V96 at every layer/head. Each model has a separately fitted Base16+Page-Fisher R16 bank:64x32768 fit,16x32768 diagnostics, Query-Gram Q32 in four8K bins.',
        'Base16/R16 extra64 selects2048 page tokens including sink32 then unions sliding recent64 (max2112); fixed2048 reserves sink32 and recent64 within2048. LRQK is R32/k1152/recent64 with FP32 routing and2/2 iterations; its GQA union is not capped at2048. ShadowKV rank160/chunk8/routed2048 has additional outlier/local/generated tokens.',
        '', 'Environment `lowrank`, direct shell execution without Slurm, two OMP/MKL threads per worker. Initial fitting GPUs Qwen3/6, Llama4/7; baseline queues Qwen0, Llama2. Qwen even-layer fitting migrated from externally busy GPU3 to GPU6 after layer10 saved, preserving factors and replaying dense hidden states. Ours runs after the banks complete: Qwen2/6 (recent_extra redirected from queued GPU0 to idle GPU2), Llama4/7.',
        '', '```bash',
        'python -u evaluation/calibrate_uniform96_router.py --model-family FAMILY --num-shards 2 --shard-index SHARD',
        'python -u evaluation/eval_uniform96_compare.py --model-family FAMILY --arm ARM',
        'python -u evaluation/eval_uniform96_compare.py --model-family FAMILY --stage summarize',
        'python -u evaluation/summarize_uniform96_compare.py','```','',
        '`FAMILY`: qwen3/llama31; `SHARD`:0/1; `ARM`:full/lrqk/recent_extra/recent_fixed. ShadowKV completed earlier and is reused with hashes.',
        'Logs: `results/logs/uniform96_compare/`. Detailed configuration: `docs/uniform96_compare_protocol.md`.',
        'Initial baseline smoke had a cache-constructor argument error before writing results; fixed and rerun with appended logs. Six native-adapter/recent-budget tests passed. The earlier Qwen ShadowKV run had an SVD warning and automatic solver fallback, then completed successfully.',
        '', 'These are resident accuracy measurements. They do not establish throughput or offload memory cost. The uploaded uniform V factor banks were fitted on256x2048 with64x2048 validation; the old Qwen L40S protocol records32x32768 fit plus4x32768 diagnostics. Thus even their V-factor fitting settings differ. The new64x32K router fit is separate from the frozen HF V-factor fit.']
    path=ROOT/'results/evaluation/uniform96_compare_summary.md'
    content='\n'.join(lines)+'\n'
    if path.exists(): assert path.read_text()==content
    else: path.write_text(content)
    print('VERIFIED both models, 880 records including reused ShadowKV; report',path,flush=True)
    print(json.dumps({f:r['means87'] for f,r in results.items()}),flush=True)


if __name__=='__main__': main()
