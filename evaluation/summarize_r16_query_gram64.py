"""Compare verified FP16 offline R16 Query-Gram count runs."""
import json
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    runs={}
    for name in ('q32','q64'):
        root=ROOT/'results/evaluation'/('longbench_r16_s40p100_fp16' if name=='q32' else 'longbench_r16_q64_s40p100_fp16')
        path=root/'result.json'
        result=json.loads(path.read_text()); audit=json.loads((root/'audit.json').read_text())
        assert result['status']==audit['status']=='complete'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==audit['result_sha256']
        assert len(result['records'])==192 and result['protocol']['dtype']=='float16'
        runs[name]=result
    a,b=runs.values()
    assert [r['sample'] for r in a['records']]==[r['sample'] for r in b['records']]
    for key in ('page_size','physical_token_budget','pinned_prefix_pages','full_protocol','fp16_full_reference_sha256'):
        assert a['protocol'][key]==b['protocol'][key]
    for key in ('fit_indices','diagnostic_indices','initialization','c1_layer_sha256','captures'):
        assert a['protocol']['bank_protocol'][key]==b['protocol']['bank_protocol'][key]
    assert a['protocol']['bank_protocol']['bcd_sweeps']==40 and a['protocol']['bank_protocol']['pcg_iterations']==100
    assert a['protocol']['bank_protocol']['query_count']==32 and b['protocol']['bank_protocol']['query_count']==64
    assert b['protocol']['bank_protocol']['diagnostic_query_count']==32
    assert b['protocol']['bank_protocol']['diagnostic_position_manifest_sha256']==a['protocol']['bank_protocol']['query_position_manifest_sha256']
    assert b['protocol']['bank_protocol']['diagnostic_query_capture_manifest_sha256']==a['protocol']['bank_protocol']['query_capture_manifest_sha256']
    assert b['protocol']['bank_protocol']['bcd_sweeps']==40 and b['protocol']['bank_protocol']['pcg_iterations']==100
    delta=[y['score']-x['score'] for x,y in zip(a['records'],b['records'],strict=True)]
    summary=dict(means={n:r['means']['v96_sparse'] for n,r in runs.items()},
        delta_pp=100*sum(delta)/192,improved=sum(x>0 for x in delta),regressed=sum(x<0 for x in delta),tied=sum(x==0 for x in delta))
    target=ROOT/'results/evaluation/r16_query_gram64'
    target.mkdir(parents=True,exist_ok=True)
    (target/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# R16 Q32 vs Q64: matched V100 FP16 LongBench','',
        'Same192 prompts, fixed C1-V96/Base16, Page32/B2048,40 BCD sweeps,PCG100. Q32 factors were fitted on L40S and Q64 on V100 using FP32; generation is matched V100 FP16. Diagnostic Q32 stays fixed.','',
        '| Task | Q32 | Q64 |','|---|---:|---:|']
    for task in a['tasks']:
        lines.append(f"| {task} | {a['tasks'][task]['v96_sparse']:.4f} | {b['tasks'][task]['v96_sparse']:.4f} |")
    lines+=['','```json',json.dumps(summary,indent=2),'```','']
    (target/'summary.md').write_text('\n'.join(lines))
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
