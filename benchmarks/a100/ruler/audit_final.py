import json, hashlib, statistics
from pathlib import Path
from types import SimpleNamespace
from collections import Counter, defaultdict
from transformers import AutoTokenizer
from evaluation import eval_llama_exact_sparse_page4 as e
root=Path('/mnt/basiskv/ruler')
archive=root/'archive'
verified=0
for line in (archive/'files.sha256').read_text().splitlines():
    digest,name=line.split('  ',1)
    if not name.startswith('l31-exact-page4/'):
        continue
    p=archive/name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==digest, name
    verified+=1
config,key=e.load_configuration(SimpleNamespace(output=archive/'l31-exact-page4'),verify=True)
e.settings()
tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
rows=e.read(e.PROMPTS/'prompts.json')['rows']
old=[e.read(p) for p in (archive/'l31-exact-page4/exact_sparse_page4/evaluate').glob('sample_*.json')]
old=[r for r in old if r['status']=='complete']
new=[e.read(p) for p in (root/'a100/resume/exact_sparse_page4').glob('sample_*.json')]
plan=e.read(root/'plan.json')
assert len(old)==397 and len(new)==703
assert all(r['status']=='complete' for r in new)
assert {r['sample']['index'] for r in new}==set(plan['remaining'])
all_rows=old+new
assert len({r['sample']['index'] for r in all_rows})==1100
assert {r['sample']['index'] for r in all_rows}==set(range(1100))
hashes=old[0]['bank_sha256']
for i,digest in hashes.items():
    bank=e.CHECKPOINT/'router_page4_b16r16'/f'layer_{int(i):03d}.safetensors'
    assert e.file_hash(bank)==digest
scores=defaultdict(list)
for r in all_rows:
    result=e.audit_result(r,rows[r['sample']['index']],key,hashes,tokenizer,'exact_sparse_page4')
    scores[r['sample']['task']].append(result['score'])
assert len(scores)==11 and all(len(v)==100 for v in scores.values())
for i in range(8):
    lines=(root/f'logs/resume{i}.log').read_text().splitlines()
    n=len(plan['remaining'][i::8])
    assert lines[-1].startswith(f'DONE {n} {n} ')
    assert not any(any(k in s for k in ('Traceback','Error:','FAILED','out of memory')) for s in lines)
summary=dict(status='passed',configuration_hash=key,environment='lowrank',command='bash /mnt/basiskv/ruler/setup/resume.sh',audit_command='python /mnt/basiskv/ruler/setup/audit_final.py',archive_files_verified=verified,original_complete=397,new_complete=703,total=1100,per_task={k:dict(n=len(v),score_percent=100*statistics.mean(v)) for k,v in scores.items()},mean_11_percent=100*statistics.mean(statistics.mean(v) for v in scores.values()),paper_10_percent=100*statistics.mean(statistics.mean(scores[k]) for k in e.PAPER_TASKS),new_seconds_mean=statistics.mean(r['result']['seconds'] for r in new),hardware=dict(Counter(r['gpu'] for r in all_rows)),numerical_note='8/8 replay generated tokens and scores match; prefill logits hashes differ across A6000/A100. User accepted output equivalence.',comparison_note='No matched proxy rerun; no paired baseline comparison.')
(root/'a100/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
