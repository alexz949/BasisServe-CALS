import json
import sys

import pytest

from evaluation.eval_qwen35_k_routing_ruler import ARMS, TASKS
from evaluation.summarize_qwen35_k_routing import main


def make_records(root, count=88):
    common = dict(v_bank_sha256='v', data_sha256='data', model_identity={'revision': 'fixed'},
        prompt_format='completion', thinking=False, sequence_length=65536, samples=88,
        code_sha256={'runtime': 'fixed'}, loki={'topk': 2048},
        ours_budget=2048, page_size=32, pinned_prefix_pages=0, recent_tokens=64,
        wo_compression=False, prefill='shared full', selection='greedy')
    for arm in ARMS:
        (root/arm).mkdir()
        for index in range(count):
            row = dict(status='complete', protocol=dict(common, arm=arm), index=index,
                task=TASKS[index//8], ordinal=index%8, input_tokens=100,
                generated_ids=[42], prediction='alpha', answers=['alpha'], score=1.0,
                input_ids_sha256=f'input{index}', first_logits_sha256=f'logits{index}',
                seconds=1.0, peak_gib=1.0)
            (root/arm/f'{index:03d}.json').write_text(json.dumps(row))


def test_complete_report_and_reject_mismatched_prefill(tmp_path, monkeypatch):
    make_records(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['summary', '--results', str(tmp_path)])
    main()
    report = json.loads((tmp_path/'summary.json').read_text())
    assert report['records'] == 616
    assert all(row['mean'] == 100 and row['delta_vs_full_pp'] == 0 for row in report['scores'].values())
    path = tmp_path/'lrqk'/'000.json'
    row = json.loads(path.read_text())
    row['first_logits_sha256'] = 'different'
    path.write_text(json.dumps(row))
    with pytest.raises(AssertionError):
        main()


def test_missing_sample_cannot_produce_summary(tmp_path, monkeypatch):
    make_records(tmp_path)
    (tmp_path/'shadowkv'/'087.json').unlink()
    monkeypatch.setattr(sys, 'argv', ['summary', '--results', str(tmp_path)])
    with pytest.raises(FileNotFoundError):
        main()
    assert not (tmp_path/'summary.json').exists()


def test_smoke_gate_audits_every_arm_and_marks_its_scope(tmp_path, monkeypatch):
    make_records(tmp_path, count=1)
    monkeypatch.setattr(sys, 'argv', ['summary', '--results', str(tmp_path), '--smoke'])
    main()
    report = json.loads((tmp_path/'summary.json').read_text())
    assert report['scope'] == 'smoke' and report['records'] == 7 and report['samples_per_arm'] == 1
    path = tmp_path/'b32r32'/'000.json'
    row = json.loads(path.read_text())
    row['input_ids_sha256'] = 'different'
    path.write_text(json.dumps(row))
    with pytest.raises(AssertionError):
        main()
