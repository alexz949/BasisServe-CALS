"""Triage saved answers without inference or changing the official scores.

Numeric presence and repeated lines are review flags, not semantic grading.
"""

import argparse
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import re

from evaluation.qwen35_hybrid_common import atomic_save, sha256


NUMBER = re.compile(r'(?<![\w.])-?\$?\d+(?:,\d{3})*(?:\.\d+)?(?![\w.])')


def numbers(text):
    return {Decimal(m.group().replace('$', '').replace(',', '')) for m in NUMBER.finditer(text)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results/q35_hybrid/gsm8k_vllm'))
    p.add_argument('--output', required=True)
    a = p.parse_args()
    arms = {}
    for arm in ('dense', 'uniform64', 'twosided64', 'twosided64_wo'):
        path = a.root / f'result_{arm}.json'
        result = json.loads(path.read_text())
        samples = {(r['doc_id'], r['filter']): r for r in result['evaluation']['samples']['gsm8k']}
        generations = {r['doc_id']: r for r in result['generation_records']}
        counts, rows = Counter(), []
        for i in range(1319):
            strict = samples[i, 'strict-match']
            flex = samples[i, 'flexible-extract']
            response = strict['resps'][0][0]
            gold = strict['target'].rsplit('####', 1)[1].strip()
            gold_number = Decimal(gold.replace(',', ''))
            extracted = flex['filtered_resps'][0].strip().replace('$', '').replace(',', '').rstrip('.')
            equivalent = bool(re.fullmatch(r'-?\d+(?:\.\d+)?', extracted)) and Decimal(extracted) == gold_number
            lines = Counter(line.strip() for line in response.splitlines() if len(line.strip()) >= 8)
            repeated = [line for line, count in lines.items() if count >= 3]
            flags = {
                'flexible_correct_strict_wrong': bool(flex['exact_match'] and not strict['exact_match']),
                'flexible_wrong': not bool(flex['exact_match']),
                'length_capped': generations[i]['finish_reason'] == 'length',
                'repeated_line_or_think': bool(repeated) or response.count('</think>') >= 3,
                'flexible_wrong_gold_number_present': not flex['exact_match'] and gold_number in numbers(response),
                'extracted_number_equivalent': equivalent,
                'flexible_wrong_extracted_number_equivalent': not flex['exact_match'] and equivalent,
            }
            if flags['flexible_wrong']:
                flags['flexible_wrong_capped'] = flags['length_capped']
                flags['flexible_wrong_repeated'] = flags['repeated_line_or_think']
                flags['flexible_wrong_neither_capped_nor_repeated'] = not (flags['length_capped'] or flags['repeated_line_or_think'])
            counts.update(k for k, v in flags.items() if v)
            if not strict['exact_match'] or not flex['exact_match']:
                rows.append({'doc_id': i, 'question': strict['doc']['question'], 'gold': gold,
                             'strict_extracted': strict['filtered_resps'], 'flexible_extracted': flex['filtered_resps'],
                             'flags': flags, 'response': response, 'repeated_lines': repeated,
                             'semantic_review': 'unreviewed'})
        arms[arm] = {'source_sha256': sha256(path), 'counts': dict(counts), 'review_rows': rows}
        print(json.dumps({'arm': arm, 'counts': dict(counts)}), flush=True)
    atomic_save(a.output, {'scope': 'Saved responses only; overlapping triage flags, not corrected accuracy. Gold-number presence can be incidental; flexible extraction can also be accidentally correct.',
                          'arms': arms})


if __name__ == '__main__':
    main()
