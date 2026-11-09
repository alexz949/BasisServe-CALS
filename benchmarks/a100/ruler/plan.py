import hashlib
import json
from pathlib import Path

root = Path('/mnt/basiskv/ruler')
directory = root/'archive/l31-exact-page4/exact_sparse_page4/evaluate'
records = [json.loads(p.read_text()) for p in sorted(directory.glob('sample_*.json'))]
assert len(records) == 397 and all(r['status'] == 'complete' for r in records)
completed = [r['sample']['index'] for r in records]
assert len(set(completed)) == 397
tasks = list(dict.fromkeys(r['sample']['task'] for r in records))
chosen = [next(r['sample']['index'] for r in records if r['sample']['task'] == task) for task in tasks]
for task in tasks:
    failures = [r['sample']['index'] for r in records if r['sample']['task'] == task and r['result']['score'] < 1]
    if failures and failures[0] not in chosen:
        chosen.append(failures[0])
for index in completed:
    if len(chosen) == 8:
        break
    if index not in chosen:
        chosen.append(index)
assert len(chosen) == 8
remaining = sorted(set(range(1100)) - set(completed))
assert len(remaining) == 703
plan = dict(archive_revision='bb3e548586ecd47cddd8201eb3e1f33077600205',
            original_completed=completed, remaining=remaining, validation=chosen,
            per_worker=[len(remaining[i::8]) for i in range(8)],
            validation_policy='Require equal full generated token IDs, score, first argmax, and bitwise prefill logits hash on all eight replays.',
            wrapper_sha256=hashlib.sha256((root/'setup/worker.py').read_bytes()).hexdigest())
path = root/'plan.json'
if path.exists():
    assert json.loads(path.read_text()) == plan
else:
    path.write_text(json.dumps(plan, indent=2)+'\n')
print(json.dumps({k:v for k,v in plan.items() if k not in ('original_completed','remaining')}, indent=2))
