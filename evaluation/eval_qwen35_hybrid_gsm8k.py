"""GSM8K generation evaluation for immutable Qwen3.5 V and V+Wo banks."""

import argparse
from contextlib import nullcontext
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import torch

from basisserve.core.qwen35_gated_v_runtime import GatedVRuntime, factor_hash
from basisserve.core.qwen35_hybrid_output_runtime import HybridOutputRuntime
from evaluation.qwen35_hybrid_common import atomic_save, load_bank, load_model, sha256, verify_model_identity


def validate_gsm8k_samples(samples, expected_doc_ids):
    # lm-eval logs each question once per answer filter, not once overall.
    filters = {'strict-match', 'flexible-extract'}
    expected_doc_ids = set(expected_doc_ids)
    assert len(samples) == len(filters) * len(expected_doc_ids)
    assert {(s['doc_id'], s['filter']) for s in samples} == {
        (i, f) for i in expected_doc_ids for f in filters}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', default='results/q35_hybrid/model')
    parser.add_argument('--baseline', default='results/q35_hybrid/baseline_summary.json')
    parser.add_argument('--bank')
    parser.add_argument('--wo-bank')
    parser.add_argument('--output', required=True)
    parser.add_argument('--thinking', choices=('off', 'on'), required=True)
    parser.add_argument('--max-new-tokens', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-length', type=int, default=16384)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--seed', type=int, default=20260909)
    args = parser.parse_args()
    assert not Path(args.output).exists()
    assert args.bank or not args.wo_bank
    assert 0 < args.max_new_tokens < args.max_length
    assert args.batch_size > 0 and (args.limit is None or 0 < args.limit <= 1319)
    torch.set_num_threads(2)

    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import handle_non_serializable
    from transformers import AutoTokenizer

    started = time.monotonic()
    base = json.loads(Path(args.baseline).read_text())
    identity = {k: base[k] for k in ('model_revision', 'config_sha256', 'model_files_sha256')}
    verify_model_identity(args.model_path, identity)
    bank = load_bank(args.bank) if args.bank else None
    if bank is not None:
        assert bank['model_identity'] == identity
    provenance = {
        'model_identity': identity,
        'bank_sha256': sha256(args.bank) if args.bank else None,
        'factor_sha256': bank['factor_sha256'] if bank else None,
        'wo_bank_sha256': sha256(args.wo_bank) if args.wo_bank else None,
        'tokenizer_sha256': sha256(Path(args.model_path) / 'tokenizer.json'),
        'chat_template_sha256': sha256(Path(args.model_path) / 'chat_template.jinja'),
    }
    print(json.dumps({'command': sys.argv, 'args': vars(args), 'python': sys.executable,
                      'gpu': torch.cuda.get_device_name(), 'provenance': provenance}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = load_model(args.model_path, 'cuda:0')
    model.config.use_cache = True
    model.generation_config.max_new_tokens = None
    model.generation_config.do_sample = False
    context = (HybridOutputRuntime(model, bank['layers'], args.wo_bank) if args.wo_bank
               else GatedVRuntime(model, bank['layers']) if bank else nullcontext())
    with context:
        lm = HFLM(pretrained=model, tokenizer=tokenizer, backend='causal',
                  batch_size=args.batch_size, max_length=args.max_length,
                  add_bos_token=False, enable_thinking=args.thinking == 'on',
                  think_end_token='</think>' if args.thinking == 'on' else None)
        evaluation = lm_eval.simple_evaluate(
            model=lm, tasks=['gsm8k'], num_fewshot=5,
            limit=args.limit, log_samples=True, apply_chat_template=True,
            fewshot_as_multiturn=False,
            gen_kwargs={'max_gen_toks': args.max_new_tokens, 'do_sample': False, 'temperature': 0.0},
            random_seed=args.seed, numpy_random_seed=args.seed,
            torch_random_seed=args.seed, fewshot_random_seed=args.seed,
            bootstrap_iters=1000,
        )
    samples = evaluation['samples']['gsm8k']
    validate_gsm8k_samples(samples, range(args.limit or 1319))
    metrics = evaluation['results']['gsm8k']
    assert all(k in metrics for k in ('exact_match,strict-match', 'exact_match,flexible-extract'))
    if bank is not None:
        assert factor_hash(bank['layers']) == provenance['factor_sha256']
        assert sha256(args.bank) == provenance['bank_sha256']
    if args.wo_bank:
        assert sha256(args.wo_bank) == provenance['wo_bank_sha256']
    payload = {
        'status': 'complete', 'command': sys.argv, 'args': vars(args),
        'environment': 'lowrank', 'execution': 'local A100, two CPU threads',
        'python': sys.executable,
        'versions': {k: importlib.metadata.version(k) for k in ('torch', 'transformers', 'lm_eval')},
        'protocol': {'fewshot': 5, 'chat_template': True, 'fewshot_as_multiturn': False,
                     'thinking': args.thinking, 'max_new_tokens': args.max_new_tokens,
                     'seed': args.seed, 'sample_count': args.limit or 1319, 'greedy': True},
        'provenance': provenance, 'evaluation': evaluation,
        'elapsed_seconds': time.monotonic() - started,
    }
    atomic_save(args.output, json.loads(json.dumps(payload, default=handle_non_serializable, allow_nan=False)))
    print(json.dumps({'metrics': metrics, 'output': args.output, 'seconds': payload['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
