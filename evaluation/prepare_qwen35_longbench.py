"""Freeze a 100-question, five-task LongBench-v1 Qwen3.5 pilot."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import save_file
from transformers import AutoTokenizer

from evaluation.qwen35_hybrid_common import atomic_save, sha256


TASKS = ('hotpotqa', '2wikimqa', 'musique', 'gov_report', 'qmsum')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--official-root', type=Path, default=Path('external/LongBench'))
    args = p.parse_args()
    torch.set_num_threads(2)
    assert not (args.output/'manifest.json').exists()
    official = args.official_root/'LongBench'
    prompts = json.loads((official/'config/dataset2prompt.json').read_text())
    caps = json.loads((official/'config/dataset2maxlen.json').read_text())
    revision = HfApi().dataset_info('THUDM/LongBench').sha
    archive = Path(hf_hub_download('THUDM/LongBench', 'data.zip', repo_type='dataset', revision=revision))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows, tensors, source_hashes, excluded = [], {}, {}, {}
    with zipfile.ZipFile(archive) as data:
        for task in TASKS:
            names = [n for n in data.namelist() if n == f'{task}.jsonl' or n.endswith(f'/{task}.jsonl')]
            assert len(names) == 1
            raw = data.read(names[0])
            source_hashes[task] = hashlib.sha256(raw).hexdigest()
            records = [json.loads(line) for line in raw.splitlines() if line.strip()]
            assert len({r['_id'] for r in records}) == len(records)
            records.sort(key=lambda r: (hashlib.sha256(f"73:{task}:{r['_id']}".encode()).hexdigest(), r['_id']))
            chosen, excluded[task] = 0, []
            for record in records:
                prompt = prompts[task].format(**record)
                rendered = tokenizer.apply_chat_template([dict(role='user', content=prompt)],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                ids = tokenizer(rendered, add_special_tokens=False)['input_ids']
                if len(ids)+caps[task] > 65536:
                    excluded[task].append(dict(source_id=record['_id'], prompt_tokens=len(ids)))
                    continue
                index = len(rows)
                tensor = torch.tensor(ids, dtype=torch.int32)
                tensors[f'sample_{index:03d}'] = tensor
                rows.append(dict(index=index, task=task, ordinal=chosen, source_id=record['_id'],
                    answers=record['answers'], all_classes=record['all_classes'],
                    official_length=record['length'], prompt_tokens=len(ids), maximum_tokens=caps[task],
                    prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                    input_ids_sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest()))
                chosen += 1
                if chosen == 20:
                    break
            assert chosen == 20
            print(json.dumps(dict(task=task, selected=chosen, excluded_too_long=len(excluded[task]))), flush=True)
    assert len(rows) == 100
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output/'tokens.safetensors'))
    atomic_save(args.output/'samples.json', rows)
    protocol = dict(tasks=list(TASKS), samples_per_task=20, sequence_length=65536, seed=73,
        sample_selection='SHA256(seed:task:source_id) order; first 20 fitting full prompt plus generation cap per task; no answer filtering',
        excluded_too_long=excluded, truncation=False, padding=False,
        dataset_repo='THUDM/LongBench', dataset_revision=revision, data_zip_sha256=sha256(archive),
        task_jsonl_sha256=source_hashes, official_root=str(args.official_root.resolve()),
        official_revision=subprocess.check_output(['git', '-C', str(args.official_root), 'rev-parse', 'HEAD'], text=True).strip(),
        official_sha256={name: sha256(official/name) for name in (
            'config/dataset2prompt.json', 'config/dataset2maxlen.json', 'eval.py', 'metrics.py')},
        model=str(args.model.resolve()), model_config_sha256=sha256(args.model/'config.json'),
        tokenizer_config_sha256=sha256(args.model/'tokenizer_config.json'),
        prompt='official task template inside native user chat template; thinking disabled',
        source_sha256=sha256(Path(__file__)))
    atomic_save(args.output/'manifest.json', dict(status='complete', protocol=protocol,
        tokens_sha256=sha256(args.output/'tokens.safetensors'), samples_sha256=sha256(args.output/'samples.json')))
    print(json.dumps(dict(samples=100, min_tokens=min(r['prompt_tokens'] for r in rows),
        max_tokens=max(r['prompt_tokens'] for r in rows),
        mean_tokens=sum(r['prompt_tokens'] for r in rows)/100)), flush=True)


if __name__ == '__main__':
    main()
