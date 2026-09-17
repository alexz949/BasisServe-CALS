"""Llama V96 at 128k: Full-K reference evaluation."""
import argparse
from collections import Counter
import functools
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import sys
import time

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from evaluation import eval_k_routing_ruler as runtime
from evaluation.k_routing_config import routing_config, validate_residual_fisher_support
from evaluation.ruler_v1 import parse_tasks, sample_score
from evaluation.v96kl_common import read_json, write_json, save_tensors, sha256
from evaluation.deterministic_evaluation import NUMERICAL_POLICY, configure_deterministic_evaluation
from evaluation.fit_k_routing_streaming import verified


ARMS = ('full', 'ours')
FORMAL_ARMS = ('full',)
TASKS = ('niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_1',
         'niah_multikey_2', 'niah_multiquery', 'niah_multivalue', 'vt', 'fwe', 'qa_1', 'qa_2')
SMOKE_IDS = (0, 210)
SOURCES = (
    'evaluation/deterministic_evaluation.py',
    'evaluation/eval_llama_cal128.py', 'evaluation/eval_k_routing_ruler.py',
    'evaluation/k_routing_config.py', 'evaluation/ruler_v1.py', 'evaluation/v96kl_common.py',
    'evaluation/chunked_prefill_mlp.py', 'evaluation/llama_sink_recent_routing.py',
    'evaluation/llama_b16r16_k_offload.py', 'evaluation/llama_resident_prefill.py',
    'basisserve/core/c1_lrqk.py', 'basisserve/core/c1_shadowkv.py',
    'basisserve/core/c1_loki_attention.py', 'basisserve/core/c1_k_routing_sidecar.py',
    'basisserve/core/c1_v_conditional_k_router.py',
    'basisserve/checkpoint/gqa_vo_qwen3.py', 'basisserve/checkpoint/c1_lrqk_qwen3.py',
    'basisserve/checkpoint/c1_shadowkv_qwen3.py',
    'basisserve/kernels/compressed_v_decode_attention.py',
    'basisserve/kernels/indexed_sparse_decode_attention.py', 'basisserve/kernels/split_indexed_attention.py',
)


def prepare(args, identity, tokenizer):
    data = read_json(args.data/'manifest.json')
    assert data['status'] == 'complete'
    samples_per_task = data['protocol']['samples_per_task']
    assert samples_per_task > 0 and data['protocol']['sequence_length'] == 131072
    assert set(data['protocol']['tasks']) == set(TASKS)
    assert data['tokenizer_config_sha256'] == sha256(Path(identity['model'])/'tokenizer_config.json')
    tokens, rows = {}, []
    for task in parse_tasks(','.join(TASKS)):
        path = args.data/task.name/'validation.jsonl'
        assert sha256(path) == data['artifacts'][task.name]['sha256']
        sources = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(sources) == samples_per_task
        for ordinal, source in enumerate(sources):
            # Treat the official completion prefix as assistant text after the chat marker.
            text = tokenizer.apply_chat_template([dict(role='user', content=source['input'])],
                tokenize=False, add_generation_prompt=True) + str(source.get('answer_prefix', ''))
            ids = tokenizer(text, add_special_tokens=False)['input_ids']
            assert len(ids)+task.tokens_to_generate <= 131072
            index = len(rows)
            t = torch.tensor(ids, dtype=torch.int32)
            tokens[str(index)] = t
            rows.append(dict(index=index, task=task.name, ordinal=ordinal, input_tokens=len(ids),
                input_sha256=hashlib.sha256(t.numpy().tobytes()).hexdigest(), answers=source['outputs'],
                maximum_tokens=task.tokens_to_generate, match_type=task.match_type))
    save_tensors(args.output/'prompts.safetensors', tokens)
    write_json(args.output/'prompts.json', dict(status='complete', rows=rows,
        identity_sha256=sha256(args.identity), data_sha256=sha256(args.data/'manifest.json'),
        tokens_sha256=sha256(args.output/'prompts.safetensors'),
        template='native user chat, generation marker, then official assistant answer prefix'))
    print(dict(status='prepared', tasks=11, prompts=len(rows),
               minimum_tokens=min(r['input_tokens'] for r in rows),
               maximum_tokens=max(r['input_tokens'] for r in rows)), flush=True)


def inputs(args, identity):
    checkpoint = Path(identity['checkpoint'])
    manifest = read_json(checkpoint/'manifest.json')
    assert manifest['status'] == 'complete' and sha256(checkpoint/'manifest.json') == identity['manifest_sha256']
    assert sha256(Path(identity['model'])/'config.json') == identity['model_config_sha256']
    for entry in manifest['layers']:
        assert sha256(checkpoint/entry['file']) == entry['sha256']
    prompts = read_json(args.output/'prompts.json')
    assert prompts['identity_sha256'] == sha256(args.identity)
    assert prompts['data_sha256'] == sha256(args.data/'manifest.json')
    assert prompts['tokens_sha256'] == sha256(args.output/'prompts.safetensors')
    samples_per_task = read_json(args.data/'manifest.json')['protocol']['samples_per_task']
    assert len(prompts['rows']) == len(TASKS)*samples_per_task
    assert Counter(r['task'] for r in prompts['rows']) == dict.fromkeys(TASKS, samples_per_task)
    spec = dict(format='basisserve.llama_cal128_ruler.v1', numerical_policy=NUMERICAL_POLICY,
        identity_sha256=sha256(args.identity),
        prompts_sha256=sha256(args.output/'prompts.json'), dtype='bfloat16', sequence_length=131072,
        samples=len(prompts['rows']), samples_per_task=samples_per_task, arms=list(ARMS), rank_schedule=identity['layer_ranks'],
        generation='greedy; native EOS; official task caps; same compressed V in every arm',
        kernels='our Triton prefill and decode; actual dispatch counts checked per prompt',
        memory_policy=('V96 and B16R16 routing sidecars remain on GPU; B16R16 exact K is held '
                       'in pinned CPU memory and only selected K rows are fetched for decode'),
        ours=dict(base=16, residual=16, page_size=32, physical_group_budget=2048, sink=32, recent=64),
        source_sha256={name:sha256(Path(name)) for name in SOURCES})
    native_eos = read_json(Path(identity['model'])/'config.json')['eos_token_id']
    spec['eos_ids'] = sorted(native_eos if isinstance(native_eos, list) else [native_eos])
    spec['versions'] = {name:importlib.metadata.version(name) for name in ('torch', 'transformers', 'triton')}
    return manifest, prompts['rows'], load_file(str(args.output/'prompts.safetensors')), spec


def bank_for(args, identity, manifest, arm):
    bank, hashes = {}, {}
    if arm == 'ours':
        for entry in manifest['layers']:
            i = entry['layer']
            path = args.bank/f'layer_{i:03d}.safetensors'
            tensors, record = verified(path)
            p = record['protocol']
            validate_residual_fisher_support(p, 'llama')
            assert record['identity_sha256'] == sha256(args.identity) and record['layer'] == i
            assert record['v_rank'] == entry['ranks'][0]
            assert p['format'] == 'basisserve.k_router.streaming.v2' and not p['smoke']
            assert p['fit_ids'] == list(range(32)) and p['diagnostic_ids'] == list(range(32, 48))
            assert p['sequence_length'] == 131072 and p['fit_queries'] == 64 and p['diagnostic_queries'] == 32
            assert record['sweeps'] == 40 and record['pcg_iterations'] == 100
            loss = record['losses']['b16_r16']
            assert len(loss['sweeps']) == 40
            g, h, d = identity['hkv'], identity['hq'], identity['head_dim']
            shapes = dict(base_left_b16=(g, record['v_rank'], 16), base_right_b16=(g, 16, d),
                base_bias_b16=(g, d), residual_encoder_b16_r16=(g, d, 16), residual_query_b16_r16=(h, d, 16))
            assert set(tensors) == set(shapes)
            assert all(t.shape == shapes[n] and t.dtype == torch.float32 and torch.isfinite(t).all()
                       for n, t in tensors.items())
            bank[i] = tensors
            hashes[str(i)] = record['sha256']
    return bank, hashes


def instrument_kernels():
    counts = Counter()
    targets = (
        ('compressed_v_decode_attention', 'compressed_v_prefill_attention', 'prefill'),
        ('compressed_v_decode_attention', 'compressed_v_decode_attention_triton', 'dense_decode'),
        ('indexed_sparse_decode_attention', 'gqa_indexed_sparse_decode_attention_triton', 'indexed_decode'),
        ('split_indexed_attention', 'split_indexed_attention', 'split_decode'),
    )
    for module_name, name, label in targets:
        module = importlib.import_module('basisserve.kernels.'+module_name)
        original = getattr(module, name)

        def tracked(*args, _original=original, _label=label, **kwargs):
            if _label == 'prefill':
                assert args[0].shape[-1] != args[2].shape[-1], 'compact V must use Triton prefill'
            counts[_label] += 1
            return _original(*args, **kwargs)

        wrapped = functools.wraps(original)(tracked)
        for loaded in tuple(sys.modules.values()):
            if loaded is not None and getattr(loaded, '__name__', '').startswith(('basisserve.', 'evaluation.')):
                for attribute, value in tuple(vars(loaded).items()):
                    if value is original:
                        setattr(loaded, attribute, wrapped)
    return counts


def verify_dispatch(counts, arm, layers, generated):
    assert counts.get('prefill', 0) == layers
    expected = 'split_decode' if arm == 'ours' else 'indexed_decode' if arm == 'loki' else 'dense_decode'
    assert counts.get(expected, 0) == layers*(generated-1)
    assert sum(counts.values()) == layers*generated


def audit_saved(saved, row, spec, hashes, arm, tokenizer, smoke=False):
    assert saved['status'] == 'complete' and saved['protocol'] == spec and saved['sample'] == row
    assert saved['bank_sha256'] == hashes
    r = saved['result']
    assert saved['eos_ids'] == spec['eos_ids']
    assert saved['first_argmax'] == r['ids'][0]
    cap = min(4, row['maximum_tokens']) if smoke else row['maximum_tokens']
    eos = set(saved['eos_ids'])
    assert 0 < len(r['ids']) <= cap and not any(i in eos for i in r['ids'][:-1])
    assert r['stopped'] == (r['ids'][-1] in eos) and (r['stopped'] or len(r['ids']) == cap)
    assert tokenizer.decode(r['ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False) == r['prediction']
    assert sample_score(r['prediction'], row['answers'], row['match_type']) == r['score']
    verify_dispatch(r['kernel_calls'], arm, len(spec['rank_schedule']), len(r['ids']))
    if arm == 'ours' and len(r['ids']) > 1:
        assert len(r['routing']) == len(spec['rank_schedule'])
        assert all(s['exact_key_storage'] == 'pinned_cpu' and
                   s['routing_sidecar_storage'] == 'cuda' and
                   s['value_storage'] == 'cuda' and
                   s['selected_tokens_mean'] <= s['token_budget'] == 2048
                   for s in r['routing'])
    return r


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare', 'smoke', 'audit-smoke', 'evaluate', 'summarize'))
    for name in ('identity', 'data', 'bank', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, default='full')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=4)
    args = parser.parse_args()
    if args.stage == 'evaluate' and args.arm not in FORMAL_ARMS:
        print('SKIP FORMAL ARM', args.arm, flush=True)
        return
    configure_deterministic_evaluation()
    assert 0 <= args.shard < args.shards
    identity = read_json(args.identity)
    tokenizer = AutoTokenizer.from_pretrained(identity['model'], local_files_only=True)
    if args.stage == 'prepare':
        prepare(args, identity, tokenizer)
        return
    manifest, rows, tokens, spec = inputs(args, identity)
    if args.stage in ('audit-smoke', 'summarize'):
        smoke = args.stage == 'audit-smoke'
        selected = [rows[i] for i in (0, 7*spec['samples_per_task'])] if smoke else rows
        audited_arms = ARMS if smoke else FORMAL_ARMS
        results = {}
        for arm in audited_arms:
            _, hashes = bank_for(args, identity, manifest, arm)
            results[arm] = []
            for row in selected:
                path = args.output/arm/('smoke' if smoke else 'evaluate')/f"sample_{row['index']:03d}.json"
                result = audit_saved(read_json(path), row, spec, hashes, arm, tokenizer, smoke)
                if smoke:
                    assert len(result['ids']) > 1, 'smoke must exercise decode'
                results[arm].append(result)
        for arm in audited_arms:
            assert all(a['ids'][0] == b['ids'][0] for a, b in zip(results['full'], results[arm], strict=True))
        if smoke:
            write_json(args.output/'smoke_audit.json', dict(status='complete', protocol=spec, verified=len(ARMS)*len(SMOKE_IDS)))
        else:
            tasks = {task:{arm:100*sum(r['score'] for r, row in zip(results[arm], rows, strict=True)
                if row['task'] == task)/spec['samples_per_task'] for arm in FORMAL_ARMS} for task in TASKS}
            means = {arm:sum(t[arm] for t in tasks.values())/11 for arm in FORMAL_ARMS}
            write_json(args.output/'summary.json', dict(status='complete', protocol=spec, tasks=tasks,
                means=means, evaluated_arms=list(FORMAL_ARMS),
                verified_predictions=len(rows)*len(FORMAL_ARMS)))
            print(means, flush=True)
        return
    if args.stage == 'evaluate':
        gate = read_json(args.output/'smoke_audit.json')
        assert gate['status'] == 'complete' and gate['protocol'] == spec
    bank, hashes = bank_for(args, identity, manifest, args.arm)
    config = routing_config(identity, rope='native', sequence_length=131072)
    assert config.model_type == 'llama'
    model = runtime.load_evaluation_model(identity, config)
    runtime.install(model, Path(identity['checkpoint']), manifest, args.arm, bank)
    if args.arm == 'ours':
        from evaluation.llama_b16r16_k_offload import install as install_prefill
        install_prefill(model)
    else:
        from evaluation.llama_resident_prefill import install_full_or_ours
        install_full_or_ours(model)
    counts = instrument_kernels()
    eos = model.config.eos_token_id
    eos = sorted(set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id})
    selected = [rows[i] for i in (0, 7*spec['samples_per_task'])] if args.stage == 'smoke' else rows[args.shard::args.shards]
    for row in selected:
        path = args.output/args.arm/args.stage/f"sample_{row['index']:03d}.json"
        if path.exists():
            audit_saved(read_json(path), row, spec, hashes, args.arm, tokenizer, args.stage == 'smoke')
            continue
        t = tokens[str(row['index'])]
        assert len(t) == row['input_tokens'] and hashlib.sha256(t.numpy().tobytes()).hexdigest() == row['input_sha256']
        cap = min(4, row['maximum_tokens']) if args.stage == 'smoke' else row['maximum_tokens']
        counts.clear()
        torch.cuda.reset_peak_memory_stats()
        print('START', args.arm, row['index'], row['task'], len(t), flush=True)
        started = time.monotonic()
        ids, first, stats, stopped = runtime.generate(model, tokenizer, dict(row, input_ids=t.tolist()), args.arm, cap)
        verify_dispatch(counts, args.arm, len(manifest['layers']), len(ids))
        prediction = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        result = dict(ids=ids, prediction=prediction, stopped=stopped,
            score=sample_score(prediction, row['answers'], row['match_type']), routing=stats,
            kernel_calls=dict(counts), seconds=time.monotonic()-started,
            peak_gib=torch.cuda.max_memory_allocated()/2**30)
        saved = dict(status='complete', protocol=spec, sample=row, bank_sha256=hashes, result=result,
                     eos_ids=eos, command=shlex.join(sys.argv), python=sys.executable,
                     gpu=torch.cuda.get_device_name(), first_argmax=int(first.argmax()))
        audit_saved(saved, row, spec, hashes, args.arm, tokenizer, args.stage == 'smoke')
        write_json(path, saved)
        print('COMPLETE', args.arm, row['index'], result['score'], result['seconds'], flush=True)


if __name__ == '__main__':
    main()
