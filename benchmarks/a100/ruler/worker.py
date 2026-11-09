"""Replay archived samples, then resume only missing exact-score samples."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from evaluation import eval_llama_exact_sparse_page4 as experiment

ROOT = Path('/mnt/basiskv/ruler')
ARCHIVE = ROOT/'archive/l31-exact-page4'
OUTPUT = ROOT/'a100'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('validate', 'resume'))
    parser.add_argument('--worker', type=int, required=True)
    args = parser.parse_args()
    assert 0 <= args.worker < 8 and torch.cuda.device_count() == 1
    experiment.settings()
    experiment.configure_deterministic_evaluation()
    torch.set_num_threads(2)
    config, key = experiment.load_configuration(SimpleNamespace(output=ARCHIVE), verify=True)
    plan = experiment.read(ROOT/'plan.json')
    if args.stage == 'resume':
        gate = experiment.read(OUTPUT/'resume_gate.json')
        assert gate['status'] == 'passed' and gate['configuration_hash'] == key
        assert gate['plan_sha256'] == hashlib.sha256((ROOT/'plan.json').read_bytes()).hexdigest()
        assert gate['driver_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
    rows = experiment.read(experiment.PROMPTS/'prompts.json')['rows']
    tokens = load_file(str(experiment.PROMPTS/'prompts.safetensors'))
    selected = plan['validation'][args.worker:args.worker+1] if args.stage == 'validate' else plan['remaining'][args.worker::8]
    print(json.dumps(dict(stage=args.stage, worker=args.worker, gpu=torch.cuda.get_device_name(),
                         samples=selected, conda_environment='lowrank')), flush=True)
    with torch.inference_mode():
        model, hashes = experiment.load_model(config)
        experiment.set_arm(model, 'exact_sparse_page4', config)
        counts = experiment.matched.instrument_kernels()
        for position, index in enumerate(selected):
            row = rows[index]
            path = OUTPUT/args.stage/'exact_sparse_page4'/f'sample_{index:04d}.json'
            if path.exists() and experiment.read(path)['status'] == 'complete':
                saved = experiment.read(path)
                experiment.audit_result(saved, row, key, hashes, tokenizer, 'exact_sparse_page4')
            else:
                print('START', index, row['task'], flush=True)
                saved = experiment.generate_one(model, tokenizer, row, tokens, 'exact_sparse_page4',
                                                key, hashes, counts, path)
                experiment.audit_result(saved, row, key, hashes, tokenizer, 'exact_sparse_page4')
            if args.stage == 'validate':
                old = experiment.read(ARCHIVE/'exact_sparse_page4/evaluate'/path.name)
                comparison = dict(index=index, task=row['task'], configuration_hash=key,
                    old_gpu=old['gpu'], new_gpu=saved['gpu'],
                    tokens_equal=saved['result']['ids'] == old['result']['ids'],
                    score_equal=saved['result']['score'] == old['result']['score'],
                    first_argmax_equal=saved['first_argmax'] == old['first_argmax'],
                    prefill_logits_bitwise_equal=saved['prefill_logits_sha256'] == old['prefill_logits_sha256'],
                    original_score=old['result']['score'], new_score=saved['result']['score'],
                    original_seconds=old['result']['seconds'], new_seconds=saved['result']['seconds'])
                experiment.atomic_json(OUTPUT/'validate'/f'worker{args.worker}.json', comparison)
                print(json.dumps(comparison), flush=True)
            print('DONE', position+1, len(selected), index, flush=True)


if __name__ == '__main__':
    main()
