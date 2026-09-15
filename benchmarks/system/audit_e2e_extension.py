"""Audit and summarize a separately recorded extension of the TP4 model grid."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output
    protocol=json.loads((root/'protocol.json').read_text())
    expected=set(itertools.product(protocol['lengths'],protocol['batches'],protocol['modes']))
    records=[];seen=set();sources={}
    for path in sorted((root/'e2e').glob('*/trial.json')):
        trial=json.loads(path.read_text());case=(trial['length'],trial['batch'],trial['mode'])
        assert case in expected and case not in seen;seen.add(case)
        for name,digest in trial['source_sha256'].items():
            if name not in sources:sources[name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
            assert sources[name]==digest
        row=dict(length=trial['length'],batch=trial['batch'],mode=trial['mode'],status=trial['status'],
            ttft_s=None,decode_ms=None,tokens_s=None,prefill_gib=None,decode_gib=None,host_key_gib=None)
        if trial['status']=='complete':
            ranks=trial['ranks'];assert {r['rank'] for r in ranks}==set(range(4))
            for r in ranks:
                assert r['length']==trial['length'] and r['batch']==trial['batch'] and r['tp']==4
                assert r['generated_tokens']==256 and r['steady_forward_count']==224 and r['discard_first_generated_tokens']==32
                raw=r['raw_decode_ms'];assert len(raw)==255 and all(math.isfinite(x) and x>0 for x in raw)
                assert math.isclose(statistics.mean(raw[31:]),r['steady_decode_mean_ms'],rel_tol=1e-6)
                assert math.isclose(trial['batch']*224/r['steady_wall_seconds'],r['steady_aggregate_tokens_per_second'],rel_tol=1e-6)
                assert r['input_window_ids']==[[64+i,64+(i+8)%16] for i in range(trial['batch'])]
                assert 'external_gpu_processes' in r['metadata']
                if trial['mode']=='offload':
                    assert len(r['host_buffer_audit'])==32 and r['host_key_bytes']>0
                    for audit in r['host_buffer_audit']:
                        assert audit['pinned'] and audit['query_returncode']==0
                        assert audit['sampled_pages']>0 and all(n==audit['expected_numa_node'] for n in audit['page_nodes'])
            row.update(ttft_s=max(r['ttft_seconds'] for r in ranks),decode_ms=ranks[0]['steady_decode_mean_ms'],
                tokens_s=ranks[0]['steady_aggregate_tokens_per_second'],
                prefill_gib=max(r['prefill_peak_allocated_bytes'] for r in ranks)/2**30,
                decode_gib=max(r['decode_peak_allocated_bytes'] for r in ranks)/2**30,
                host_key_gib=sum(r['host_key_bytes'] for r in ranks)/2**30)
        else:
            assert trial['status']=='gpu_oom' and 'out of memory' in Path(trial['log']).read_text().lower()
        records.append(row)
    assert seen==expected,dict(missing=sorted(expected-seen))
    with (root/'e2e_tp4.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    passed=sum(r['status']=='complete' for r in records)
    report=dict(status='complete',successful=passed,oom=len(records)-passed,records=records,source_sha256=sources,
        scope='128K input plus256 generated tokens. Concatenated diagnostic windows; unchanged64K-fitted factors. Systems performance, not language-model quality.')
    (root/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# L40S128K TP4 extension','',
        'Llama-3.1-8B base; existing two-sided KL average V96; B16R16; hard2048 including sink32/recent64. Four L40S, BF16. No refit.', '',
        'Input:131072 tokens from two distinct64K diagnostic windows;256 tokens generated, first32 excluded. Hardware and factors shared with the original suite. Existing GPU occupancy is recorded per run.', '',
        f'{passed} successful configurations; {len(records)-passed} GPU OOM outcomes. OOM is not a zero-latency measurement.', '',
        '| Batch | Mode | Status | TTFT s | Decode ms/step | Tokens/s | Prefill GiB | Decode GiB | Host K GiB |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in sorted(records,key=lambda r:(r['batch'],r['mode'])):
        cells=[row[k] for k in ['batch','mode','status','ttft_s','decode_ms','tokens_s','prefill_gib','decode_gib','host_key_gib']]
        lines.append('| '+' | '.join('—' if x is None else f'{x:.3f}' if isinstance(x,float) else str(x) for x in cells)+' |')
    lines += ['', 'GPU peaks are maximum per-rank PyTorch allocation; host K sums four ranks. Dense uses FlashAttention decode; C1 and sparse modes use existing CUDA kernels. Fused offload traffic is logical bytes, not measured PCIe counters.', '',
        'Environment: `basis`. Command: `python -m benchmarks.system.run_e2e_tp4 --root /home/zhangal/BasisServe-CALS-runs/llama31_8b_64k --output results/system_benchmarks/l40s_128k --lengths 131072`. Exact per-trial commands, timing distributions, communication/support counts and source hashes are in `e2e/*/trial.json`.', '']
    (root/'SUMMARY.md').write_text('\n'.join(lines))
    print(dict(status='complete',successful=passed,oom=len(records)-passed),flush=True)


if __name__=='__main__':main()
