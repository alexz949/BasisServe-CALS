import hashlib
import json
import sys
from pathlib import Path

root = Path('/mnt/basiskv/ruler')
checks = [json.loads((root/f'a100/validate/worker{i}.json').read_text()) for i in range(8)]
fields = ('tokens_equal', 'score_equal', 'first_argmax_equal', 'prefill_logits_bitwise_equal')
plan = json.loads((root/'plan.json').read_text())
assert [row['index'] for row in checks] == plan['validation']
assert len({row['configuration_hash'] for row in checks}) == 1
status = 'passed' if all(row[field] for row in checks for field in fields) else 'failed'
gate = dict(status=status, configuration_hash=checks[0]['configuration_hash'],
            plan_sha256=hashlib.sha256((root/'plan.json').read_bytes()).hexdigest(),
            matched={field:sum(row[field] for row in checks) for field in fields},
            samples=checks, per_worker_remaining=plan['per_worker'])
if '--accept-output-equivalence' in sys.argv:
    assert all(row[field] for row in checks for field in fields[:3])
    gate.update(status='passed', strict_bitwise_status=status,
                accepted_criterion='Full generated token IDs, score, and first argmax agree in all eight replays; bitwise logits equality is waived by the user.',
                user_authorization=dict(output_equivalence='接受，继续剩余 703 道',
                                        local_execution='允许这次直接在本机运行'),
                driver_sha256=hashlib.sha256((root/'setup/worker.py').read_bytes()).hexdigest())
    destination = root/'a100/resume_gate.json'
else:
    destination = root/'a100/consistency.json'
if destination.exists():
    assert json.loads(destination.read_text()) == gate
else:
    destination.write_text(json.dumps(gate, indent=2)+'\n')
print(json.dumps(gate, indent=2))
