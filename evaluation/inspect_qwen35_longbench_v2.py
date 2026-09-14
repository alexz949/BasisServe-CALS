"""Inventory LongBench-v2 lengths before selecting a paired routing pilot."""

import argparse
from collections import Counter
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from evaluation.qwen35_hybrid_common import atomic_save, sha256


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    revision = '2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9'
    source = Path(hf_hub_download('THUDM/LongBench-v2', 'data.json', repo_type='dataset', revision=revision))
    data = json.loads(source.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = []
    templates = {mode:Path(f'external/LongBench/prompts/{name}.txt').read_text()
        for mode, name in (('direct', '0shot'), ('cot', '0shot_cot'))}
    for item in data:
        lengths = {}
        for mode, template in templates.items():
            prompt = template
            for marker, field in (('$DOC$', 'context'), ('$Q$', 'question'), ('$C_A$', 'choice_A'),
                ('$C_B$', 'choice_B'), ('$C_C$', 'choice_C'), ('$C_D$', 'choice_D')):
                prompt = prompt.replace(marker, item[field].strip())
            text = tokenizer.apply_chat_template([dict(role='user', content=prompt)],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            lengths[mode] = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        rows.append(dict(source_id=item['_id'], domain=item['domain'], sub_domain=item['sub_domain'],
            difficulty=item['difficulty'], official_length=item['length'], prompt_tokens=lengths))
        if len(rows)%50 == 0:
            print('inventoried', len(rows), flush=True)
    report = dict(status='complete', dataset_revision=revision, source_path=str(source), source_sha256=sha256(source),
        model=str(args.model), tokenizer_config_sha256=sha256(args.model/'tokenizer_config.json'),
        source_code_sha256=sha256(Path(__file__)), rows=rows,
        domains=dict(Counter(r['domain'] for r in rows)))
    atomic_save(args.output, report)
    for mode, reserve in (('direct',128), ('cot',2048)):
        eligible = [r for r in rows if r['prompt_tokens'][mode]+reserve <= 65536]
        print(json.dumps(dict(mode=mode, eligible64k=len(eligible),
            eligible32to64k=sum(r['prompt_tokens'][mode]>=32768 for r in eligible),
            domains=dict(Counter(r['domain'] for r in eligible)))), flush=True)


if __name__ == '__main__':
    main()
