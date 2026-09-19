"""Prepare the Qwen3-8B-matched C4-validation PPL bank for Nemotron-H."""
import argparse
import hashlib
from pathlib import Path
import random
import shlex
import sys

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.v96kl_common import configure, read_json, write_json, save_tensors, sha256, tensor_hash
from evaluation.build_llama31_8b_palu_m_checkpoint import _document_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    configure()
    audit = read_json(args.audit)
    assert audit['status'] == 'complete'
    model = Path(audit['model'])
    output = args.output / 'windows.safetensors'
    manifest_path = args.output / 'manifest.json'
    spec = dict(audit_sha256=sha256(args.audit), model=str(model),
        model_config_sha256=audit['config_sha256'], dataset='allenai/c4', config='en',
        split='validation', revision='1588ec454efa1a09f29cd18ddd04fe05fc8653a2',
        seed=20260824, shuffle_buffer=10000, sequence_length=2048, samples=128,
        source_sha256=sha256(__file__), document_identity_source_sha256=sha256(
            ROOT / 'evaluation/build_llama31_8b_palu_m_checkpoint.py'))
    if manifest_path.exists():
        record = read_json(manifest_path)
        assert record['status'] == 'complete' and record['protocol'] == spec
        assert record['sha256'] == sha256(output)
        print('QUALITY C4 WINDOWS VERIFIED', flush=True)
        return
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True,
        trust_remote_code=False, use_fast=True)
    dataset = load_dataset('allenai/c4', 'en', split='validation', streaming=True,
        revision=spec['revision']).shuffle(seed=spec['seed'], buffer_size=spec['shuffle_buffer'])
    generator = random.Random(spec['seed'])
    seen, seen_text, windows, records = set(), set(), [], []
    for stream_index, row in enumerate(dataset):
        document = _document_id(row, stream_index)
        text_hash = hashlib.sha256(row['text'].encode()).hexdigest()
        if document in seen or text_hash in seen_text:
            continue
        ids = tokenizer(row['text'], add_special_tokens=False)['input_ids']
        if len(ids) < spec['sequence_length']:
            continue
        start = generator.randint(0, len(ids) - spec['sequence_length'])
        window = torch.tensor(ids[start:start + spec['sequence_length']], dtype=torch.int32)
        windows.append(window)
        records.append(dict(index=len(windows)-1, document_id=document,
            document_sha256=text_hash, stream_index=stream_index, token_start=start,
            document_token_count=len(ids), input_ids_sha256=tensor_hash(window)))
        seen.add(document); seen_text.add(text_hash)
        if len(windows) % 16 == 0:
            print('QUALITY C4 WINDOWS', len(windows), '/128', flush=True)
        if len(windows) == spec['samples']:
            break
    assert len(windows) == len(seen) == len(seen_text) == spec['samples']
    save_tensors(output, dict(input_ids=torch.stack(windows)))
    write_json(manifest_path, dict(status='complete', protocol=spec, records=records,
        sha256=sha256(output), shape=[spec['samples'], spec['sequence_length']],
        command=shlex.join(sys.argv), python=sys.executable, document_disjoint=True,
        duplicate_text_excluded=True,
        sampling='one contiguous uniform-start excerpt per document; no special tokens'))
    print('QUALITY C4 WINDOWS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
