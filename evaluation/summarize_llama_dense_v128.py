"""Audit the V128 experiment, retaining evidence of TP2 prefill differences."""
import argparse
from pathlib import Path
from types import SimpleNamespace

from safetensors.torch import load_file
from transformers import AutoTokenizer

from evaluation.eval_llama_dense_v_routing128 import ARMS, TASKS, audit_saved, bank_for, inputs
from evaluation.v96kl_common import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    for name in ('evaluation', 'reference', 'prefill-check', 'identity', 'data', 'bank', 'loki'):
        parser.add_argument('--'+name, type=Path, required=True)
    options = parser.parse_args()
    selected = options.arms
    assert len(selected) == len(set(selected))
    args = SimpleNamespace(
        identity=options.identity, data=options.data,
        bank=options.bank, loki=options.loki, output=options.evaluation)
    identity = read_json(args.identity)
    manifest, rows, _, spec = inputs(args, identity)
    tokenizer = AutoTokenizer.from_pretrained(identity['model'], local_files_only=True)
    reference = options.reference
    check_root = options.prefill_check
    check = read_json(check_root/'summary.json')
    assert check['status'] == 'complete'
    assert check['protocol']['identity_sha256'] == spec['identity_sha256']
    assert check['protocol']['prompts_sha256'] == spec['prompts_sha256']
    assert check['protocol']['source_sha256'] == sha256(Path('evaluation/audit_llama_dense_prefill.py'))
    assert check['protocol']['generation_cap'] == 1
    _, ours_hashes = bank_for(args, identity, manifest, 'ours')
    first_tokens, differences, evidence = {}, [], {}
    for row in rows:
        name = f"sample_{row['index']:03d}.json"
        ours_path = args.output/'ours/evaluate'/name
        ours = read_json(ours_path)
        audit_saved(ours, row, spec, ours_hashes, 'ours', tokenizer)
        dense_path = reference/name
        dense = read_json(dense_path)
        assert dense['status'] == 'complete' and dense['sample'] == row
        assert dense['first_argmax'] == dense['result']['ids'][0]
        first_tokens[row['index']] = ours['first_argmax']
        if ours['first_argmax'] != dense['first_argmax']:
            differences.append(row['index'])
        evidence[str(ours_path)] = sha256(ours_path)
        evidence[str(dense_path)] = sha256(dense_path)
    expected = sorted(set(differences) | {0, 7*spec['samples_per_task']})
    assert check['protocol']['selected'] == expected
    assert check['checked'] == len(expected) and check['mismatching_tp2'] == len(differences)
    for index in expected:
        path = check_root/f'sample_{index:03d}.json'
        record = read_json(path)
        assert record['protocol'] == check['protocol'] and record['sample'] == rows[index]
        assert record['kernel_calls'] == {'prefill': 32}
        assert record['matches_ours']
        logits_path = path.with_suffix('.safetensors')
        assert sha256(logits_path) == record['logits_sha256']
        logits = load_file(str(logits_path))['first_logits']
        assert int(logits.argmax()) == record['first_argmax'] == first_tokens[index]
        dense = read_json(reference/path.name)
        assert record['tp2_first_argmax'] == dense['first_argmax']
        evidence[str(path)] = sha256(path)
    evidence[str(check_root/'summary.json')] = sha256(check_root/'summary.json')
    tasks = {task: {} for task in TASKS}
    for arm in selected:
        _, hashes = bank_for(args, identity, manifest, arm)
        results = []
        for row in rows:
            path = args.output/arm/'evaluate'/f"sample_{row['index']:03d}.json"
            saved = read_json(path)
            result = audit_saved(saved, row, spec, hashes, arm, tokenizer)
            assert saved['first_argmax'] == first_tokens[row['index']], (arm, row['index'])
            results.append(result)
            evidence[str(path)] = sha256(path)
        for task in TASKS:
            tasks[task][arm] = 100*sum(result['score'] for row, result in
                zip(rows, results, strict=True) if row['task'] == task)/spec['samples_per_task']
    means = {arm: sum(scores[arm] for scores in tasks.values())/11 for arm in selected}
    means10 = {arm: sum(scores[arm] for task, scores in tasks.items()
                       if task != 'niah_single_3')/10 for arm in selected}
    report = dict(status='complete', protocol=spec, arms=selected, tasks=tasks,
        means=means, means_10_excluding_single_3=means10,
        verified_predictions=len(rows)*len(selected), source_sha256=sha256(Path(__file__)),
        prefill_audit=dict(tp2_different_indices=differences,
            native_full_k_checked_indices=expected,
            interpretation='All routing first tokens match ours; all ours/TP2 differences '
                           'and two controls checked against native single-GPU Full-K. '
                           'Dense TP2 and single-GPU execution are not bitwise equivalent.'),
        evidence_sha256=evidence)
    name = ('audit_summary.json' if set(selected) == set(ARMS)
            else 'audit_'+'_'.join(selected)+'.json')
    write_json(args.output/name, report)
    print('VERIFIED', report['verified_predictions'], means, means10, flush=True)


if __name__ == '__main__':
    main()
