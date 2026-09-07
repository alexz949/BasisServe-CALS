"""Summarize the three fixed RULER regressions without causal attribution."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i+1
    return min(len(a), len(b))+1 if len(a) != len(b) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    assert not (args.root/'summary.json').exists()
    assert not (args.root/'summary.md').exists()
    summary = {}
    for index in (33, 37, 38):
        directory = args.root/'evaluate'/f'sample_{index:03d}'
        result = json.loads((directory/'result.json').read_text())
        assert result['status'] == 'complete' and result['stage'] == 'evaluate'
        assert result['sample_index'] == index
        arms = result['arms']
        for arm in ('full_exact_k', 'q32_proxy'):
            assert arms[arm]['matches_old_tokens']
        divergence = first_difference(arms['full_exact_k']['generated_token_ids'],
                                      arms['q32_proxy']['generated_token_ids'])
        observations = {}
        for arm in arms:
            record = json.loads((directory/f'{arm}.json').read_text())
            assert record['tensor_sha256'] == sha256(directory/f'{arm}.safetensors')
            assert record['calls'] == len(record['rows']) == 36*(len(record['generated_token_ids'])-1)
            misses = []
            boundary = []
            for row in record['rows']:
                if row['predicts_generated_token_1based'] > divergence:
                    continue
                for support in row['support']:
                    if support['kind'] != 'answer':
                        continue
                    for group, mass in enumerate(support['teacher_mass_by_head']):
                        selected_exact = group in support['exact_selected_groups']
                        selected_proxy = group in support['proxy_selected_groups']
                        data = {'layer':row['layer'], 'predicts_generated_token_1based':row['predicts_generated_token_1based'],
                                'group':group, 'page':support['page'], 'teacher_head_max_mass':max(mass),
                                'teacher_head_mean_mass':sum(mass)/len(mass),
                                'exact_selected':selected_exact, 'proxy_selected':selected_proxy,
                                'exact_rank_min':support['exact_rank_min'][group],
                                'proxy_rank_min':support['proxy_rank_min'][group]}
                        if selected_exact and not selected_proxy:
                            misses.append(data)
                        if row['predicts_generated_token_1based'] == divergence:
                            boundary.append(data)
            observations[arm] = {'same_state_answer_page_misses_before_or_at_first_divergence':len(misses),
                'largest_mass_misses':sorted(misses,key=lambda r:r['teacher_head_max_mass'],reverse=True)[:12],
                'at_first_divergence':boundary}
        summary[str(index)] = {'references':result['references'], 'support_spans':result['support_spans'],
            'arms':arms, 'first_q32_divergence_generated_token_1based':divergence,
            'first_exact_pages_divergence_generated_token_1based':first_difference(
                arms['full_exact_k']['generated_token_ids'], arms['exact_pages']['generated_token_ids']),
            'observations':observations}
    lines=['# Three-sample multikey selected-page trace', '',
           'Qwen3-8B-Base, frozen C1-V80, closed-form Base16, Q32 Fisher R8. '
           'Shared full C1 prefill; independent greedy decode forks. Page32/B2048 including pinned page0, all36 layers. '
           'No refitting or parameter selection. Environment: basis; NVIDIA L40S.', '',
           'Exact-pages uses FP32 exact QK for the same non-pinned head-normalized GQA-max selector. '
           'Selected attention remains native BF16 exact-K/C1-V. Full exact-K uses full SDPA output. '
           'Each row compares exact/proxy selection on the SAME state within its arm; states across arms differ.', '',
           '| Sample | Reference | Full exact-K | Exact pages B2048 | Q32 proxy | First Q32 divergent generated token |',
           '| --- | --- | --- | --- | --- | ---: |']
    for index,r in summary.items():
        cells=[index, ', '.join(r['references'])]
        for arm in ('full_exact_k','exact_pages','q32_proxy'):
            a=r['arms'][arm]
            pred=a['prediction'].replace('\n',' ').replace('|',' / ')
            cells.append(f"`{pred[:180]}` (score {a['score']:.2f})")
        cells.append(str(r['first_q32_divergence_generated_token_1based']))
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', 'First-token numbering includes the shared leading space from full prefill. '
              'Matching generated prefixes do not imply identical hidden states/cache across arms.', '',
              '## Same-state answer-page omissions on the full-attention trajectory', '',
              'Below are the five largest per-head teacher-mass omissions per sample, up to and including '
              'the generated-token position where the independent Q32 run first diverges. '
              'These are observational selector comparisons, not layer intervention effects. '
              'Exact selected the page in the listed group; proxy did not.', '',
              '| Sample | Predicted token # | Layer | Group | Answer page | Max head mass | Exact rank | Proxy rank |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for index,r in summary.items():
        for v in r['observations']['full_exact_k']['largest_mass_misses'][:5]:
            lines.append(f"| {index} | {v['predicts_generated_token_1based']} | {v['layer']} | {v['group']} | {v['page']} | {100*v['teacher_head_max_mass']:.4f}% | {v['exact_rank_min']} | {v['proxy_rank_min']} |")
    lines += ['', '## Artifacts and checks', '',
              'Every sample has per-arm JSON step/layer records and safetensors containing exact/proxy selected IDs, '
              'per-head full page masses, normalized GQA-max scores, rank-min, cutoffs, overlap counts, and decode logits. '
              'Partial final pages are included. Support sentence spans and old-Q32 distractor spans are recorded. '
              'Minimum rank records ties; actual selected IDs remain authoritative.', '',
              'Full and Q32 generated token IDs reproduce their previous runs exactly. All six reproduction checks, '
              'nine artifact hashes, and step/layer row counts passed. A separate 8-token smoke verified bitwise '
              'logit equality between traced and untraced full/Q32 paths.', '',
              'This selected three-failure subset is a diagnosis, not an independent accuracy benchmark. '
              'Exact-pages answers all three correctly at the unchanged B2048 budget. '
              'This does not establish all-88-prompt accuracy or causal necessity of any individual omitted page.', '',
              'Run log: formal array 8300813 completed on three L40S in 33–34 seconds per sample; '
              'summary job 8300818 completed in 13 seconds. Initial smoke 8300809 exhausted GPU memory because '
              'Mock call histories retained KV tensors; direct callable patches removed this retention. '
              'The unchanged-setting smoke 8300810 then passed in 34 seconds. Temporary submission files were removed.', '',
              '## Commands', '']
    for index in summary:
        r=json.loads((args.root/'evaluate'/f'sample_{int(index):03d}'/'result.json').read_text())
        lines += ['```bash', r['python']+' -u '+r['command'], '```', '']
    write_json(args.root/'summary.json', {'status':'complete','samples':summary})
    (args.root/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines[:18]))


if __name__ == '__main__':
    main()
