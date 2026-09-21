"""Non-thinking generation benchmarks and smoke for the Qwen3.5 vLLM adapter."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import torch
from transformers import AutoTokenizer

from basisserve.vllm import register
from evaluation.eval_qwen3_8b_r80_generation_vllm import LocalVLLMGenerationLM
from evaluation.qwen35_hybrid_common import atomic_save, load_bank, sha256, verify_model_identity
from evaluation.eval_qwen35_hybrid_gsm8k import validate_gsm8k_samples

# Bank authentication touches torch before engine creation. Spawn avoids
# inheriting its CPU thread pools; register in spawned interpreters as well.
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '2')
register()

TASKS = {
    'gsm8k': (5, 1024, 1319),
    'minerva_math500': (4, 4096, 500),
    'mbpp_plus_full': (0, 2048, 378),
    'ifeval': (0, 2048, 541),
}


class NonThinkingLM(LocalVLLMGenerationLM):
    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', default='results/q35_hybrid/model')
    p.add_argument('--model-manifest', default='results/q35_hybrid/baseline_summary.json')
    p.add_argument('--bank')
    p.add_argument('--wo-bank')
    p.add_argument('--wo-scope', choices=('all', 'full_attention', 'gdn'), default='all')
    p.add_argument('--output', required=True)
    p.add_argument('--task', choices=tuple(TASKS), default='gsm8k')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--limit', type=int)
    p.add_argument('--doc-ids', type=int, nargs='+')
    p.add_argument('--max-num-seqs', type=int, default=32)
    p.add_argument('--max-num-batched-tokens', type=int, default=4096)
    p.add_argument('--max-new-tokens', type=int)
    p.add_argument('--max-model-len', type=int, default=8192)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.45)
    p.add_argument('--kv-cache-gib', type=float)
    p.add_argument('--enforce-eager', action='store_true')
    p.add_argument('--seed', type=int, default=20260909)
    a = p.parse_args()
    fewshot, default_tokens, expected_samples = TASKS[a.task]
    if a.max_new_tokens is None:
        a.max_new_tokens = default_tokens
    assert not Path(a.output).exists()
    assert a.bank or not a.wo_bank
    assert a.wo_bank or a.wo_scope == 'all'
    assert 0 < a.max_new_tokens < a.max_model_len
    assert a.limit is None or 0 < a.limit <= expected_samples
    assert not a.doc_ids or (a.limit is None and not a.smoke and a.task != 'gsm8k')
    if a.doc_ids:
        assert len(set(a.doc_ids)) == len(a.doc_ids)
        assert all(0 <= i < expected_samples for i in a.doc_ids)
    torch.set_num_threads(2)
    base = json.loads(Path(a.model_manifest).read_text())
    identity = {k: base[k] for k in ('model_revision', 'config_sha256', 'model_files_sha256')}
    verify_model_identity(a.model_path, identity)
    bank = load_bank(a.bank) if a.bank else None
    if bank:
        assert bank['model_identity'] == identity
    provenance = {'model_identity': identity, 'v_bank_sha256': sha256(a.bank) if a.bank else None,
                  'v_factor_sha256': bank['factor_sha256'] if bank else None,
                  'wo_bank_sha256': sha256(a.wo_bank) if a.wo_bank else None,
                  'tokenizer_sha256': sha256(Path(a.model_path) / 'tokenizer.json'),
                  'chat_template_sha256': sha256(Path(a.model_path) / 'chat_template.jinja')}
    overrides = {'architectures': ['BasisServeQwen35HybridForCausalLM']}
    if a.bank:
        overrides.update(basisserve_v_bank=str(Path(a.bank).resolve()),
                         basisserve_v_bank_sha256=provenance['v_bank_sha256'])
    if a.wo_bank:
        overrides.update(basisserve_wo_bank=str(Path(a.wo_bank).resolve()),
                         basisserve_wo_bank_sha256=provenance['wo_bank_sha256'],
                         basisserve_wo_scope=a.wo_scope)
    from vllm import LLM, SamplingParams, TokensPrompt
    register()
    started = time.monotonic()
    print(json.dumps({'command': sys.argv, 'args': vars(a), 'python': sys.executable,
                      'provenance': provenance}), flush=True)
    engine = LLM(model=a.model_path, hf_overrides=overrides, dtype='bfloat16',
                 tensor_parallel_size=1, max_model_len=a.max_model_len,
                 max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.max_num_batched_tokens,
                 gpu_memory_utilization=a.gpu_memory_utilization,
                 kv_cache_memory_bytes=int(a.kv_cache_gib * 2**30) if a.kv_cache_gib else None,
                 enable_prefix_caching=False, enable_chunked_prefill=True,
                 enforce_eager=a.enforce_eager, seed=a.seed, disable_log_stats=False)
    tokenizer = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True)
    lm = NonThinkingLM(engine, tokenizer, max_length=a.max_model_len,
                       default_max_gen_toks=a.max_new_tokens)
    result = {'command': sys.argv, 'args': vars(a), 'provenance': provenance,
              'versions': {k: importlib.metadata.version(k) for k in ('torch', 'vllm', 'transformers', 'lm_eval')},
              'environment': os.environ.get('CONDA_DEFAULT_ENV'), 'python': sys.executable,
              'thinking': False,
              'cache_scope': 'latent V in standard-width zero-padded cache; no cache-memory reduction claim',
              'wo_scope': 'four logical source encoders and joint decoder on TP1; no distributed collective'}
    if a.task != 'gsm8k':
        result['versions'].update({k: importlib.metadata.version(k) for k in
                                   ('evalplus', 'math-verify', 'antlr4-python3-runtime')})
    if a.smoke:
        questions = ['What is 17 + 28? Give the answer after ####.',
                     'A shop has 12 boxes with 8 pencils each. It sells 19 pencils. How many remain? End with #### followed by the number.',
                     'If 3 tickets cost 21 dollars, how much do 11 tickets cost? End with #### followed by the number.',
                     'Compute 123 minus 48. End with #### followed by the number.']
        ids = [tokenizer.encode(lm.apply_chat_template([{'role': 'user', 'content': q}]),
                                add_special_tokens=False) for q in questions]
        outputs = engine.generate([TokensPrompt(prompt_token_ids=x) for x in ids],
                                  SamplingParams(max_tokens=32, temperature=0, logprobs=5))
        records = []
        for prompt, output in zip(ids, outputs, strict=True):
            completion = output.outputs[0]
            records.append({'prompt_token_ids': prompt, 'token_ids': list(completion.token_ids),
                            'text': completion.text, 'finish_reason': completion.finish_reason,
                            'chosen_logprobs': [float(lp[token].logprob) for token, lp in
                                                zip(completion.token_ids, completion.logprobs, strict=True)]})
        result['smoke_records'] = records
    else:
        import lm_eval
        from lm_eval.tasks import TaskManager
        from lm_eval.utils import handle_non_serializable
        evaluation = lm_eval.simple_evaluate(
            model=lm, tasks=[a.task], num_fewshot=fewshot, limit=a.limit,
            samples={a.task: a.doc_ids} if a.doc_ids else None,
            task_manager=TaskManager(include_path=str(Path(__file__).parent / 'tasks' / 'mbpp_plus_full')),
            confirm_run_unsafe_code=a.task == 'mbpp_plus_full',
            apply_chat_template=True, fewshot_as_multiturn=False, log_samples=True,
            gen_kwargs={'max_gen_toks': a.max_new_tokens, 'temperature': 0.0, 'do_sample': False},
            random_seed=a.seed, numpy_random_seed=a.seed, torch_random_seed=a.seed,
            fewshot_random_seed=a.seed, bootstrap_iters=1000)
        samples = evaluation['samples'][a.task]
        expected = len(a.doc_ids) if a.doc_ids else (a.limit or expected_samples)
        assert len({s['doc_id'] for s in samples}) == expected
        if a.doc_ids:
            assert {s['doc_id'] for s in samples} == set(a.doc_ids)
        if a.task == 'gsm8k':
            validate_gsm8k_samples(samples, expected)
        else:
            assert len(samples) == expected
        assert len(lm.generation_records) == expected
        assert all(r['original_prompt_tokens'] == r['retained_prompt_tokens'] for r in lm.generation_records)
        result['evaluation'] = json.loads(json.dumps(evaluation, default=handle_non_serializable, allow_nan=False))
        result['generation_records'] = lm.generation_records
        result['length_capped'] = sum(r['finish_reason'] == 'length' for r in lm.generation_records)
        result['closing_think_responses'] = len({s['doc_id'] for s in samples
                                                 if '</think>' in s['resps'][0][0]})
    if a.bank:
        assert sha256(a.bank) == provenance['v_bank_sha256']
    if a.wo_bank:
        assert sha256(a.wo_bank) == provenance['wo_bank_sha256']
    result.update(status='complete', elapsed_seconds=time.monotonic() - started)
    atomic_save(a.output, result)
    print(json.dumps({'output': a.output, 'seconds': result['elapsed_seconds'],
                      'metrics': result.get('evaluation', {}).get('results')}), flush=True)
    engine.llm_engine.engine_core.shutdown(timeout=30)


if __name__ == '__main__':
    main()
