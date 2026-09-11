"""Freeze document-disjoint C4 calibration and validation PPL windows."""

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from evaluation.qwen35_hybrid_common import atomic_save, sha256


def sample(dataset, count, tokenizer, seed):
    generator = random.Random(seed)
    indices = list(range(len(dataset)))
    generator.shuffle(indices)
    windows, records, seen = [], [], set()
    for index in indices:
        row = dataset[index]
        document = hashlib.sha256(row['text'].encode()).hexdigest()
        if document in seen:
            continue
        ids = tokenizer(row['text'], add_special_tokens=False)['input_ids']
        if len(ids) < 2048:
            continue
        start = generator.randrange(len(ids) - 2048 + 1)
        windows.append(torch.tensor(ids[start:start + 2048], dtype=torch.int32))
        records.append({'row': index, 'document_sha256': document, 'token_start': start})
        seen.add(document)
        if len(windows) % 32 == 0:
            print(f'sampled {len(windows)}/{count}', flush=True)
        if len(windows) == count:
            break
    assert len(windows) == count
    return torch.stack(windows), records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--seed', type=int, default=20260909)
    args = p.parse_args()
    root = Path(args.output_dir)
    assert not root.exists()
    cache = Path.home() / '.cache/huggingface/datasets'
    c4 = cache / 'allenai___c4/en-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2'
    train_path, val_path = c4 / 'c4-train-00000-of-00002.arrow', c4 / 'c4-validation.arrow'
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    train, train_records = sample(Dataset.from_file(str(train_path)), 464, tokenizer, args.seed)
    val, val_records = sample(Dataset.from_file(str(val_path)), 128, tokenizer, args.seed + 1)
    assert not {r['document_sha256'] for r in train_records} & {r['document_sha256'] for r in val_records}
    wiki_paths = sorted(cache.glob('wikitext/wikitext-2-raw-v1*/0.0.0/*/wikitext-test.arrow'))
    assert wiki_paths
    wiki = Dataset.from_file(str(wiki_paths[0]))
    wiki_ids = tokenizer('\n\n'.join(wiki['text']), add_special_tokens=False)['input_ids']
    payload = {'fit': train[:256], 'heldout': train[256:320], 'profile': train[320:448],
        'confirm': train[448:464], 'c4_eval': val, 'wikitext': torch.tensor(wiki_ids, dtype=torch.int32)}
    atomic_save(root / 'windows.pt', payload)
    atomic_save(root / 'manifest.json', {'format': 'basisserve.qwen35.hybrid_windows.v1',
        'seed': args.seed, 'sequence_length': 2048, 'sampling': 'seeded shuffled local arrow rows; unique documents; uniform token start',
        'model_config_sha256': sha256(Path(args.model_path) / 'config.json'),
        'tokenizer_sha256': sha256(Path(args.model_path) / 'tokenizer.json'),
        'windows_sha256': sha256(root / 'windows.pt'), 'counts': {k: list(v.shape) for k, v in payload.items()},
        'sources': {'c4_train': str(train_path), 'c4_validation': str(val_path), 'wikitext_test': str(wiki_paths[0])},
        'train_records': train_records, 'c4_eval_records': val_records})
    print(json.dumps({k: list(v.shape) for k, v in payload.items()}), flush=True)


if __name__ == '__main__':
    main()
