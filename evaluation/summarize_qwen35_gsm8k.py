"""Audit paired complete GSM8K results and their immutable inputs."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from datasets import Dataset

from evaluation.eval_qwen35_hybrid_gsm8k import validate_gsm8k_samples
from evaluation.qwen35_hybrid_common import atomic_save, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results/q35_hybrid/gsm8k_vllm'))
    p.add_argument('--test-arrow', type=Path, default=Path('/home/lz299/.cache/huggingface/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-test.arrow'))
    a = p.parse_args()
    data = Dataset.from_file(str(a.test_arrow))
    assert len(data) == 1319
    arms = ('dense', 'uniform64', 'twosided64', 'twosided64_wo')
    rows, per_arm, prompts, shared_provenance = [], {}, None, None
    for arm in arms:
        path = a.root / f'result_{arm}.json'
        result = json.loads(path.read_text())
        assert result['status'] == 'complete' and result['thinking'] is False
        settings = result['args']
        assert settings['limit'] is None and not settings['smoke'] and not settings['enforce_eager']
        assert settings['max_new_tokens'] == 1024 and settings['max_num_seqs'] == 32
        assert settings['max_model_len'] == 8192 and settings['seed'] == 20260909
        evaluation = result['evaluation']
        samples = evaluation['samples']['gsm8k']
        validate_gsm8k_samples(samples, 1319)
        by_doc_filter = {(r['doc_id'], r['filter']): r for r in samples}
        local_prompts = {}
        for i in range(1319):
            strict, flexible = (by_doc_filter[i, f] for f in ('strict-match', 'flexible-extract'))
            assert strict['doc'] == flexible['doc'] == data[i]
            assert strict['arguments'] == flexible['arguments'] and strict['resps'] == flexible['resps']
            context, kwargs = strict['arguments'][0]
            assert context.endswith('<think>\n\n</think>\n\n')
            assert kwargs['do_sample'] is False and kwargs['temperature'] == 0 and kwargs['max_gen_toks'] == 1024
            local_prompts[i] = context
        if prompts is None:
            prompts = local_prompts
        assert local_prompts == prompts
        generations = result['generation_records']
        assert len(generations) == 1319 and {r['doc_id'] for r in generations} == set(range(1319))
        for r in generations:
            assert r['original_prompt_tokens'] == r['retained_prompt_tokens']
            assert r['prompt_sha256'] == hashlib.sha256(prompts[r['doc_id']].encode()).hexdigest()
            assert 0 < r['generated_tokens'] <= 1024
        capped = sum(r['finish_reason'] == 'length' for r in generations)
        assert capped == result['length_capped']
        provenance = result['provenance']
        fixed = {k: provenance[k] for k in ('model_identity', 'tokenizer_sha256', 'chat_template_sha256')}
        if shared_provenance is None:
            shared_provenance = fixed
        assert fixed == shared_provenance
        if settings['bank']:
            assert sha256(settings['bank']) == provenance['v_bank_sha256']
        if settings['wo_bank']:
            assert sha256(settings['wo_bank']) == provenance['wo_bank_sha256']
        metrics = evaluation['results']['gsm8k']
        row = {'arm': arm, 'result_sha256': sha256(path), 'length_capped': capped,
               'generated_tokens': sum(r['generated_tokens'] for r in generations),
               'closing_think_in_response': sum('</think>' in by_doc_filter[i, 'strict-match']['resps'][0][0] for i in range(1319)),
               'repeated_closing_think': sum(by_doc_filter[i, 'strict-match']['resps'][0][0].count('</think>') >= 3 for i in range(1319)),
               'flexible_correct_strict_wrong': sum(by_doc_filter[i, 'flexible-extract']['exact_match'] == 1 and by_doc_filter[i, 'strict-match']['exact_match'] == 0 for i in range(1319)),
               'elapsed_seconds': result['elapsed_seconds'], 'command': result['command'],
               'settings': settings, 'provenance': provenance, 'versions': result['versions']}
        outcomes = {}
        for f in ('strict-match', 'flexible-extract'):
            values = [by_doc_filter[i, f]['exact_match'] for i in range(1319)]
            assert all(v in (0, 1) for v in values)
            correct = sum(values)
            assert math.isclose(correct / 1319, metrics['exact_match,' + f], abs_tol=1e-12)
            row[f] = {'correct': int(correct), 'accuracy': correct / 1319}
            outcomes[f] = values
        per_arm[arm] = outcomes
        rows.append(row)
        print(json.dumps({k: row[k] for k in ('arm', 'strict-match', 'flexible-extract', 'length_capped')}), flush=True)
    paired = {}
    for arm in arms[1:]:
        paired[arm] = {}
        for f in ('strict-match', 'flexible-extract'):
            dense, compressed = per_arm['dense'][f], per_arm[arm][f]
            differences = [b-a for a, b in zip(dense, compressed)]
            mean = sum(differences) / 1319
            se = math.sqrt(sum((d-mean)**2 for d in differences) / 1318 / 1319)
            paired[arm][f] = {'both_correct': sum(a == b == 1 for a, b in zip(dense, compressed)),
                             'dense_only_correct': sum(a == 1 and b == 0 for a, b in zip(dense, compressed)),
                             'compressed_only_correct': sum(a == 0 and b == 1 for a, b in zip(dense, compressed)),
                             'accuracy_difference': mean, 'paired_question_standard_error': se}
    evidence = {}
    for name in ('smoke_review.json', 'pilot_review.json', 'long_reference_review.json'):
        evidence[name] = sha256(a.root / name)
    atomic_save(a.root / 'summary.json', {'status': 'all_four_complete_and_audited', 'rows': rows,
        'test_questions': 1319, 'test_arrow_sha256': sha256(a.test_arrow),
        'identical_prompts': True, 'paired_vs_dense': paired, 'review_sha256': evidence,
        'physical_gpus_from_launch_records': {'dense': 6, 'uniform64': 5, 'twosided64': 0, 'twosided64_wo': 2},
        'environment': 'lowrankarena for vLLM; lowrank for HF reference and this audit',
        'scope': 'non-thinking, 5-shot, one calibration bank per arm; padded V cache, TP1 Wo equivalent'})


if __name__ == '__main__':
    main()
