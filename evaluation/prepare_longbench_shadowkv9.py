"""LongBench-v1 rows for the ShadowKV nine-task protocol.

For each task the official prompt template is rendered exactly as upstream
``pred.py`` does, tokenized with the evaluated model's own tokenizer, and only
rows whose prompt is longer than ``--min-input-tokens`` are kept, in source
order, without truncation. Chat wrapping is applied later by the evaluator.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.longbench_metrics import DATASET2METRIC, LONGBENCH_REPO_REVISION, OFFICIAL_MAXLEN, OFFICIAL_PROMPT
from evaluation.v96kl_common import read_json, sha256, write_json

SHADOWKV9 = ['narrativeqa', 'multifieldqa_en', 'hotpotqa', 'musique', 'dureader',
             'gov_report', 'samsum', 'passage_retrieval_en', 'lcc']
DATA_ZIP_SHA256 = 'cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64'
FORMAT = 'basisserve.longbench_v1.shadowkv9.v1'


def prepare_task(task, data_root, tokenizer, output, min_input_tokens, sequence_length):
    rows = [json.loads(line) for line in (data_root / f'{task}.jsonl').read_text(encoding='utf-8').splitlines()]
    prompts = [OFFICIAL_PROMPT[task].format(**row) for row in rows]
    counts = [len(ids) for ids in tokenizer(prompts, add_special_tokens=False)['input_ids']]
    max_gen = OFFICIAL_MAXLEN[task]
    kept = []
    for source_index, (row, prompt, count) in enumerate(zip(rows, prompts, counts)):
        if count <= min_input_tokens:
            continue
        assert count + max_gen <= sequence_length, f'{task}[{source_index}] {count} + {max_gen} > {sequence_length}'
        assert row['dataset'] == task, (task, source_index, row['dataset'])
        kept.append(dict(
            index=len(kept), source_index=source_index, _id=row['_id'], task=task, input=prompt,
            outputs=list(row['answers']), answer_prefix='', max_gen=max_gen, metric=DATASET2METRIC[task],
            all_classes=row['all_classes'], length=row['length'], input_tokens=count, language=row['language']))
    assert kept, f'{task}: no prompt longer than {min_input_tokens} tokens'
    path = output / task / 'validation.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in kept), encoding='utf-8')
    tokens = [record['input_tokens'] for record in kept]
    return dict(file=f'{task}/validation.jsonl', sha256=sha256(path), rows_total=len(rows), rows_kept=len(kept),
                min_tokens=min(tokens), max_tokens=max(tokens), mean_tokens=round(sum(tokens) / len(tokens), 2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('/home/Ubuntu/longbench_data/data'))
    p.add_argument('--model', type=Path, required=True, help='HF snapshot directory with the tokenizer files')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', default=','.join(SHADOWKV9))
    p.add_argument('--min-input-tokens', type=int, default=4096, help='keep rows with prompt tokens > this')
    p.add_argument('--sequence-length', type=int, default=131072)
    p.add_argument('--model-tag', required=True)
    args = p.parse_args()
    tasks = args.tasks.split(',')
    assert all(task in DATASET2METRIC for task in tasks), tasks
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        assert read_json(manifest_path)['status'] == 'complete', str(manifest_path)
        print(f'{manifest_path} already complete', flush=True)
        return
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    artifacts = {}
    for task in tasks:
        artifacts[task] = prepare_task(task, args.data_root, tokenizer, args.output,
                                       args.min_input_tokens, args.sequence_length)
        print(f'{task}: kept {artifacts[task]["rows_kept"]}/{artifacts[task]["rows_total"]}', flush=True)
    write_json(manifest_path, dict(
        status='complete', format=FORMAT, benchmark='longbench_v1', model_tag=args.model_tag,
        model=str(args.model.resolve()),
        tokenizer_config_sha256=sha256(args.model / 'tokenizer_config.json'),
        tokenizer_sha256=sha256(args.model / 'tokenizer.json'),
        protocol=dict(
            tasks=tasks, samples_per_task={task: artifacts[task]['rows_kept'] for task in tasks},
            filter=f'official prompt tokens > {args.min_input_tokens} under the model tokenizer '
                   '(add_special_tokens=False); no truncation',
            min_input_tokens=args.min_input_tokens, sequence_length=args.sequence_length,
            prompt_source='evaluation/longbench_official/dataset2prompt.json',
            maxlen_source='evaluation/longbench_official/dataset2maxlen.json',
            longbench_repo_revision=LONGBENCH_REPO_REVISION, data_zip_sha256=DATA_ZIP_SHA256,
            chat_wrapping='applied by the evaluator (official build_chat equivalent: chat template + generation prompt)'),
        artifacts=artifacts, command=sys.argv, created_at=datetime.now(timezone.utc).isoformat()))
    print(f'\n{args.model_tag}: prompts > {args.min_input_tokens} tokens')
    print(f'{"task":22s} {"total":>6s} {"kept":>6s} {"min":>8s} {"max":>8s} {"mean":>10s}')
    for task in tasks:
        a = artifacts[task]
        print(f'{task:22s} {a["rows_total"]:6d} {a["rows_kept"]:6d} {a["min_tokens"]:8d} {a["max_tokens"]:8d} {a["mean_tokens"]:10.1f}')
    print(f'{"total":22s} {sum(a["rows_total"] for a in artifacts.values()):6d} '
          f'{sum(a["rows_kept"] for a in artifacts.values()):6d}')
    print(f'wrote {manifest_path}', flush=True)


if __name__ == '__main__':
    main()
