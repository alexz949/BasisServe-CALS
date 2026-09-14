"""Freeze up to 100 untruncated 32K-64K LongBench-v2 CoT inputs."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

from evaluation.qwen35_hybrid_common import atomic_save, sha256


def render(template, item, reasoning=None):
    fields = {'$DOC$':'context', '$Q$':'question', '$C_A$':'choice_A',
        '$C_B$':'choice_B', '$C_C$':'choice_C', '$C_D$':'choice_D'}
    for marker, field in fields.items():
        if marker in template:
            template = template.replace(marker, item[field].strip())
    if reasoning is not None:
        template = template.replace('$COT$', reasoning.strip())
    return template


def chat_ids(tokenizer, text):
    text = tokenizer.apply_chat_template([dict(role='user', content=text)],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer(text, add_special_tokens=False)['input_ids']


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    inv = json.loads(args.inventory.read_text())
    assert inv['status'] == 'complete'
    assert inv['tokenizer_config_sha256'] == sha256(args.model/'tokenizer_config.json')
    source = Path(inv['source_path'])
    assert sha256(source) == inv['source_sha256']
    data = {r['_id']:r for r in json.loads(source.read_text())}
    eligible = [r for r in inv['rows'] if 32768 <= r['prompt_tokens']['cot'] <= 65536-1024]
    pools = defaultdict(list)
    for row in eligible:
        pools[row['domain']].append(row)
    for pool in pools.values():
        pool.sort(key=lambda r:hashlib.sha256(f"73:{r['source_id']}".encode()).hexdigest())
    chosen = []
    while len(chosen) < min(100, len(eligible)):
        for domain in sorted(pools):
            if pools[domain] and len(chosen) < min(100, len(eligible)):
                chosen.append(pools[domain].pop(0))
    assert chosen
    official = Path('external/LongBench/prompts')
    template = (official/'0shot_cot.txt').read_text()
    answer_template = (official/'0shot_cot_ans.txt').read_text()
    assert '$DOC$' not in answer_template
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows, tensors = [], {}
    for index, selected in enumerate(chosen):
        item = data[selected['source_id']]
        ids = chat_ids(tokenizer, render(template, item))
        assert len(ids) == selected['prompt_tokens']['cot']
        assert 32768 <= len(ids) and len(ids)+1024 <= 65536
        tensor = torch.tensor(ids, dtype=torch.int32)
        tensors[f'sample_{index:03d}'] = tensor
        rows.append(dict(index=index, **{k:item[k] for k in ('_id','domain','sub_domain','difficulty',
            'length','question','choice_A','choice_B','choice_C','choice_D','answer')},
            prompt_tokens=len(ids), input_ids_sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest()))
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output/'tokens.safetensors'))
    atomic_save(args.output/'samples.json', rows)
    protocol = dict(dataset_revision=inv['dataset_revision'], dataset_sha256=inv['source_sha256'],
        inventory_sha256=sha256(args.inventory), model=str(args.model),
        model_config_sha256=sha256(args.model/'config.json'), tokenizer_config_sha256=sha256(args.model/'tokenizer_config.json'),
        samples=len(rows), eligible=len(eligible), selection='seed73 hash order within domains; round-robin domains; labels unused',
        prompt_min_tokens=32768, total_limit=65536, reasoning_cap=1024, answer_cap=128,
        truncation=False, padding=False, native_thinking=False, sampling='greedy (official default is temperature0.1)',
        official_templates={name:sha256(official/name) for name in ('0shot_cot.txt','0shot_cot_ans.txt')},
        answer_template=answer_template, answer_stage='official second stage omits document, includes generated reasoning',
        source_code_sha256=sha256(Path(__file__)))
    atomic_save(args.output/'manifest.json', dict(status='complete', protocol=protocol,
        samples_sha256=sha256(args.output/'samples.json'), tokens_sha256=sha256(args.output/'tokens.safetensors')))
    print(json.dumps(dict(samples=len(rows), eligible=len(eligible),
        min_tokens=min(r['prompt_tokens'] for r in rows), max_tokens=max(r['prompt_tokens'] for r in rows),
        domains={d:sum(r['domain']==d for r in rows) for d in sorted(pools)})), flush=True)


if __name__ == '__main__':
    main()
