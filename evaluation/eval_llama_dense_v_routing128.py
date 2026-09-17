"""Dense-V128 Llama routing comparison on the shared 11 x 30 RULER prompts."""

import argparse
from collections import Counter
from dataclasses import replace
import functools
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import sys
import time
from types import MethodType

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from evaluation import eval_k_routing_ruler as runtime
from evaluation.deterministic_evaluation import NUMERICAL_POLICY, configure_deterministic_evaluation
from evaluation.fit_k_routing_streaming import verified
from evaluation.k_routing_config import routing_config, validate_residual_fisher_support
from evaluation.ruler_v1 import sample_score
from evaluation.v96kl_common import read_json, sha256, write_json


ARMS = ('ours', 'shadowkv', 'loki', 'lrqk')
LRQK_TOPK = 832
LOKI_TOPK = 856
TASKS = ('niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_1',
         'niah_multikey_2', 'niah_multiquery', 'niah_multivalue', 'vt', 'fwe', 'qa_1', 'qa_2')
SMOKE_IDS = (0, 210)
SOURCES = (
    'evaluation/deterministic_evaluation.py',
    'evaluation/eval_llama_dense_v_routing128.py',
    'evaluation/eval_k_routing_ruler.py',
    'evaluation/k_routing_config.py',
    'evaluation/ruler_v1.py',
    'evaluation/v96kl_common.py',
    'evaluation/chunked_prefill_mlp.py',
    'evaluation/llama_baseline_prefill.py',
    'evaluation/official_shadowkv_cpu.py',
    'evaluation/llama_prefill_memory.py',
    'evaluation/llama_sink_recent_routing.py',
    'basisserve/core/c1_lrqk.py',
    'basisserve/core/c1_shadowkv.py',
    'basisserve/core/c1_loki_attention.py',
    'basisserve/core/c1_k_routing_sidecar.py',
    'basisserve/core/c1_v_conditional_k_router.py',
    'basisserve/checkpoint/gqa_vo_qwen3.py',
    'basisserve/checkpoint/c1_lrqk_qwen3.py',
    'basisserve/checkpoint/c1_shadowkv_qwen3.py',
    'basisserve/kernels/compressed_v_decode_attention.py',
    'basisserve/kernels/indexed_sparse_decode_attention.py',
    'basisserve/kernels/split_indexed_attention.py',
)


def inputs(args, identity):
    checkpoint = Path(identity['checkpoint'])
    manifest = read_json(checkpoint/'manifest.json')
    assert manifest['status'] == 'complete'
    assert sha256(checkpoint/'manifest.json') == identity['manifest_sha256']
    assert sha256(Path(identity['model'])/'config.json') == identity['model_config_sha256']
    for entry in manifest['layers']:
        assert sha256(checkpoint/entry['file']) == entry['sha256']
    prompts = read_json(args.output/'prompts.json')
    assert prompts['identity_sha256'] == sha256(args.identity)
    assert prompts['data_sha256'] == sha256(args.data/'manifest.json')
    assert prompts['tokens_sha256'] == sha256(args.output/'prompts.safetensors')
    samples_per_task = read_json(args.data/'manifest.json')['protocol']['samples_per_task']
    assert len(prompts['rows']) == len(TASKS)*samples_per_task
    assert Counter(row['task'] for row in prompts['rows']) == dict.fromkeys(TASKS, samples_per_task)
    spec = dict(
        format='basisserve.llama_dense_v_routing128.v1',
        numerical_policy=NUMERICAL_POLICY,
        identity_sha256=sha256(args.identity),
        prompts_sha256=sha256(args.output/'prompts.json'),
        dtype='bfloat16', sequence_length=131072, samples=len(prompts['rows']), samples_per_task=samples_per_task,
        arms=list(ARMS), value_mode='original dense V128 cache and original Wo',
        rank_schedule=[identity['head_dim']] * len(identity['attention_layers']),
        generation='greedy; native EOS; official task caps',
        prefill='exact full causal attention with original dense V128',
        memory_policy='ours/Loki: resident K/V128; LRQK: prefill K/V on CPU, restored per layer at first decode; ShadowKV: official pinned-CPU historical V and selected-V transfers throughout decode',
        ours=dict(base=16, residual=16, page_size=32, physical_group_budget=2048,
                  sink=32, recent=64,
                  routing_feature='original dense V128; separately refitted Base16/Residual16'),
        shadowkv=dict(rank=160, chunk=8, routed=2048, outlier_chunks=48,
                      implementation='upstream ShadowKVCache_CPU and compiled upstream CUDA kernels',
                      extra_support='official aligned local tail and generated tokens',
                      upstream_sources={str(path): sha256(path) for path in (
                          Path('external/ShadowKV/models/kv_cache.py'),
                          Path('external/ShadowKV/models/tensor_op.py'),
                          *sorted(Path('external/ShadowKV/kernels').glob('shadowkv*.so')))}),
        loki=dict(pca_rank=32, topk_per_query_head=LOKI_TOPK, recent=0, forced_sink=0,
                  manifest_sha256=sha256(args.loki/'manifest.json')),
        lrqk=dict(rank=32, topk_per_query_head=LRQK_TOPK, recent=64,
                  prefill_iterations=2, decode_iterations=2, tolerance=0.01,
                  seed=0, state_dtype='bfloat16', solve_dtype='float32'),
        source_sha256={name: sha256(Path(name)) for name in SOURCES},
    )
    native_eos = read_json(Path(identity['model'])/'config.json')['eos_token_id']
    spec['eos_ids'] = sorted(native_eos if isinstance(native_eos, list) else [native_eos])
    spec['versions'] = {name: importlib.metadata.version(name)
                        for name in ('torch', 'transformers', 'triton')}
    return manifest, prompts['rows'], load_file(str(args.output/'prompts.safetensors')), spec


def bank_for(args, identity, manifest, arm):
    bank, hashes = {}, {}
    if arm == 'ours':
        for entry in manifest['layers']:
            layer = entry['layer']
            route_path = args.bank/f'layer_{layer:03d}.safetensors'
            tensors, record = verified(route_path)
            protocol = record['protocol']
            validate_residual_fisher_support(protocol, 'llama')
            assert record['identity_sha256'] == sha256(args.identity)
            assert record['layer'] == layer and record['v_rank'] == 128
            assert protocol['format'] == 'basisserve.k_router.streaming.v2'
            assert protocol['sequence_length'] == 131072 and not protocol['smoke']
            assert protocol['value_feature_mode'] == 'original dense V128'
            assert record['sweeps'] == 40 and record['pcg_iterations'] == 100
            shapes = dict(base_left_b16=(identity['hkv'], 128, 16),
                base_right_b16=(identity['hkv'], 16, 128),
                base_bias_b16=(identity['hkv'], 128),
                residual_encoder_b16_r16=(identity['hkv'], 128, 16),
                residual_query_b16_r16=(identity['hq'], 128, 16))
            assert set(tensors) == set(shapes)
            assert all(tensors[name].shape == shape for name, shape in shapes.items())
            bank[layer] = tensors
            hashes[str(layer)] = record['sha256']
    elif arm == 'loki':
        pca = read_json(args.loki/'manifest.json')
        assert pca['status'] == 'complete' and pca['rank'] == 32
        assert [row['layer'] for row in pca['layers']] == identity['attention_layers']
        for entry in pca['layers']:
            path = args.loki/entry['file']
            assert sha256(path) == entry['sha256']
            bank[entry['layer']] = load_file(str(path))
            hashes[str(entry['layer'])] = entry['sha256']
    return bank, hashes


@torch.inference_mode()
def dense_ours_forward(self, hidden_states, position_embeddings, attention_mask=None,
                       past_key_values=None, **kwargs):
    batch, length, _ = hidden_states.shape
    assert batch == 1
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    q = self.q_norm(self.q_proj(hidden_states).view(
        batch, length, self.num_attention_heads, self.head_dim)).transpose(1, 2)
    pre = self.k_norm(self.k_proj(hidden_states).view(
        batch, length, self.num_key_value_heads, self.head_dim)).transpose(1, 2)
    dense_v = self.v_proj(hidden_states).view(
        batch, length, self.num_key_value_heads, self.value_head_dim).transpose(1, 2)
    assert self.value_head_dim == self.head_dim == 128
    cos, sin = position_embeddings
    k = pre
    for start in range(0, length, 1024):
        stop = start + 1024
        qr, kr = apply_rotary_pos_emb(q[:, :, start:stop], k[:, :, start:stop],
                                     cos[:, start:stop], sin[:, start:stop])
        q[:, :, start:stop], k[:, :, start:stop] = qr, kr
    del pre, qr, kr
    previous = past_key_values.get_seq_length(self.layer_idx)
    assert previous == 0 or length == 1
    factors = self._routing_factors
    current = runtime.build_conditional_routing_sidecar(
        dense_v, k,
        base_left=factors['base_left_b16'], base_right=factors['base_right_b16'],
        base_bias=factors['base_bias_b16'],
        residual_encoder=factors['residual_encoder_b16_r16'], cos=cos, sin=sin)
    if previous:
        sidecar = past_key_values.sidecars[self.layer_idx]
        if sidecar.device != current.device:
            sidecar = sidecar.to(current.device)
        past_key_values.sidecars[self.layer_idx] = torch.cat((sidecar, current), 2)
    else:
        past_key_values.sidecars[self.layer_idx] = current.cpu()
    k, dense_v = past_key_values.update(k, dense_v, self.layer_idx)
    if previous == 0:
        output = runtime.compressed_v_prefill_attention(q, k, dense_v, scale=self.scaling)
    else:
        from evaluation.llama_sink_recent_routing import page_support
        from basisserve.kernels.split_indexed_attention import split_indexed_attention
        if attention_mask is not None:
            assert attention_mask.shape[-2] == 1
        group_heads = self.num_attention_heads // self.num_key_value_heads
        codes = torch.einsum('bhqd,hdr->bhqr', q.float(), self._routing_projector.float())
        codes = codes[:, :, 0].reshape(batch, self.num_key_value_heads, group_heads, -1)
        sidecar = past_key_values.sidecars[self.layer_idx]
        scores = (codes @ sidecar.float().transpose(-1, -2)) * self.scaling
        ids, valid = page_support(scores)
        selected = ids.masked_fill(~valid, -1).repeat_interleave(group_heads, dim=1)
        output = split_indexed_attention(q, k, dense_v, selected, scale=self.scaling)
        past_key_values.statistics[self.layer_idx] = dict(
            selected_tokens_mean=float(valid.sum(-1).float().mean()), sink_tokens=32,
            recent_tokens=64, token_budget=2048)
    del q, k, dense_v
    output = output.transpose(1, 2).contiguous().reshape(batch, length, -1)
    return self.o_proj(output), None


def install_dense_ours(model):
    from basisserve.checkpoint.c1_attention_layers import c1_attention_layers
    for _, module in c1_attention_layers(model):
        module.forward = MethodType(dense_ours_forward, module)


def install_prefill(model, arm):
    if arm == 'ours':
        from evaluation.chunked_prefill_mlp import install_chunked_prefill_norms
        install_chunked_prefill_norms(model)
    elif arm == 'shadowkv':
        from evaluation.official_shadowkv_cpu import install
        install(model)
    else:
        from evaluation.llama_baseline_prefill import install
        install(model, arm)


def install_prefill_cache_offload(model):
    """Free completed-layer caches while online factorization uses large temporaries."""
    for layer in model.model.layers:
        attention = layer.self_attn
        original = attention.forward

        @torch.inference_mode()
        def forward(self, hidden_states, *args, _original=original, **kwargs):
            cache = kwargs['past_key_values']
            previous = cache.get_seq_length(self.layer_idx)
            cached = cache.layers[self.layer_idx]
            if previous:
                cached.keys = cached.keys.to(hidden_states.device)
                cached.values = cached.values.to(hidden_states.device)
            result = _original(hidden_states, *args, **kwargs)
            assert cached.values.shape[-1] == 128
            if not previous:
                cached.keys = cached.keys.cpu()
                cached.values = cached.values.cpu()
            return result

        attention.forward = MethodType(forward, attention)


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
                assert args[0].shape[-1] == args[2].shape[-1] == 128
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
    expected = {'ours': 'split_decode', 'loki': 'indexed_decode',
                'shadowkv': 'dense_decode', 'lrqk': 'dense_decode'}[arm]
    assert counts.get(expected, 0) == layers*(generated-1)
    assert sum(counts.values()) == layers*generated


def audit_saved(saved, row, spec, hashes, arm, tokenizer, smoke=False):
    assert saved['status'] == 'complete' and saved['protocol'] == spec and saved['sample'] == row
    assert saved['bank_sha256'] == hashes
    result = saved['result']
    assert saved['eos_ids'] == spec['eos_ids'] and saved['first_argmax'] == result['ids'][0]
    cap = min(4, row['maximum_tokens']) if smoke else row['maximum_tokens']
    eos = set(saved['eos_ids'])
    assert 0 < len(result['ids']) <= cap and not any(i in eos for i in result['ids'][:-1])
    assert result['stopped'] == (result['ids'][-1] in eos)
    assert result['stopped'] or len(result['ids']) == cap
    assert tokenizer.decode(result['ids'], skip_special_tokens=True,
                            clean_up_tokenization_spaces=False) == result['prediction']
    assert sample_score(result['prediction'], row['answers'], row['match_type']) == result['score']
    verify_dispatch(result['kernel_calls'], arm, len(spec['rank_schedule']), len(result['ids']))
    assert result['cache_value_head_dim'] == 128
    if len(result['ids']) > 1:
        stats = result['routing']
        if arm == 'shadowkv':
            assert stats['rank'] == spec[arm]['rank'] and stats['routed_tokens'] == spec[arm]['routed']
            assert stats['outlier_tokens'] == spec[arm]['outlier_chunks']*spec[arm]['chunk']
        else:
            assert len(stats) == len(spec['rank_schedule'])
            for stat in stats:
                if arm == 'ours':
                    assert stat['token_budget'] == spec[arm]['physical_group_budget']
                    assert stat['sink_tokens'] == spec[arm]['sink'] and stat['recent_tokens'] == spec[arm]['recent']
                    assert stat['selected_tokens_mean'] <= stat['token_budget']
                else:
                    assert stat['selected_per_query_head'] == spec[arm]['topk_per_query_head']+spec[arm]['recent']
                    if arm == 'loki':
                        assert stat['recent_tokens'] == 0
                    else:
                        assert stat['decode_steps'] == len(result['ids'])-1
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('smoke', 'audit-smoke', 'evaluate', 'summarize'))
    for name in ('identity', 'data', 'bank', 'loki', 'output', 'dense-reference'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, default='ours')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=4)
    args = parser.parse_args()
    configure_deterministic_evaluation()
    assert 0 <= args.shard < args.shards
    identity = read_json(args.identity)
    tokenizer = AutoTokenizer.from_pretrained(identity['model'], local_files_only=True)
    manifest, rows, tokens, spec = inputs(args, identity)
    if args.stage in ('audit-smoke', 'summarize'):
        smoke = args.stage == 'audit-smoke'
        arms = (args.arm,) if smoke else ARMS
        selected = [rows[i] for i in (0, 7*spec['samples_per_task'])] if smoke else rows
        results = {}
        for arm in arms:
            _, hashes = bank_for(args, identity, manifest, arm)
            results[arm] = []
            for row in selected:
                path = args.output/arm/('smoke-v128' if smoke else 'evaluate')/f"sample_{row['index']:03d}.json"
                result = audit_saved(read_json(path), row, spec, hashes, arm, tokenizer, smoke)
                if smoke:
                    assert len(result['ids']) > 1
                results[arm].append(result)
        for arm in arms:
            for row, result in zip(selected, results[arm], strict=True):
                reference = read_json(args.output/'ours'/('smoke-v128' if smoke else 'evaluate')/
                                      f"sample_{row['index']:03d}.json")
                assert reference['sample'] == row and reference['protocol'] == spec
                assert reference['first_argmax'] == result['ids'][0]
        if smoke:
            write_json(args.output/args.arm/'smoke_audit.json',
                       dict(status='complete', protocol=spec, verified=len(SMOKE_IDS), arm=args.arm))
        else:
            tasks = {task: {arm: 100*sum(result['score'] for result, row in
                        zip(results[arm], rows, strict=True) if row['task'] == task)/spec['samples_per_task']
                    for arm in ARMS} for task in TASKS}
            means = {arm: sum(row[arm] for row in tasks.values())/11 for arm in ARMS}
            means_10 = {arm: sum(row[arm] for task, row in tasks.items()
                        if task != 'niah_single_3')/10 for arm in ARMS}
            write_json(args.output/'summary.json', dict(status='complete', protocol=spec,
                tasks=tasks, means=means, means_10_excluding_single_3=means_10,
                verified_predictions=len(rows)*len(ARMS)))
            print(means, flush=True)
        return
    if args.stage == 'evaluate':
        gate = read_json(args.output/args.arm/'smoke_audit.json')
        assert gate['status'] == 'complete' and gate['protocol'] == spec and gate['arm'] == args.arm
    bank, hashes = bank_for(args, identity, manifest, args.arm)
    config = routing_config(identity, rope='native', sequence_length=131072)
    assert config.model_type == 'llama'
    model = runtime.load_evaluation_model(identity, config)
    original_weights = [(layer.self_attn.v_proj.weight.detach().cpu().clone(),
                         layer.self_attn.o_proj.weight.detach().cpu().clone())
                        for layer in model.model.layers]
    if args.arm != 'shadowkv':
        runtime.install(model, Path(identity['checkpoint']), manifest, args.arm, bank, dense_v=True)
    for layer, (value, output) in zip(model.model.layers, original_weights, strict=True):
        attention = layer.self_attn
        assert attention.v_proj.weight.shape[0] == identity['hkv']*128
        assert attention.o_proj.weight.shape[1] == identity['hq']*128
        assert torch.equal(attention.v_proj.weight.detach().cpu(), value)
        assert torch.equal(attention.o_proj.weight.detach().cpu(), output)
    del original_weights
    if args.arm == 'loki':
        from basisserve.core import c1_loki_attention
        c1_loki_attention.c1_loki_recent_decode = functools.partial(
            c1_loki_attention.c1_loki_recent_decode, recent_tokens=0, top_k=LOKI_TOPK)
    if args.arm == 'lrqk':
        for layer in model.model.layers:
            attention = layer.self_attn
            attention._lrqk_config = replace(attention._lrqk_config, topk=LRQK_TOPK)
            assert attention._lrqk_config.recent == 64
    if args.arm == 'ours':
        install_dense_ours(model)
    install_prefill(model, args.arm)
    if args.arm == 'shadowkv':
        from evaluation.official_shadowkv_cpu import load_cache_class, generate as shadow_generate
        shadow_cache_class = load_cache_class()
    elif args.arm == 'lrqk':
        install_prefill_cache_offload(model)
    counts = instrument_kernels()
    eos = model.config.eos_token_id
    eos = sorted(set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id})
    selected = [rows[i] for i in (0, 7*spec['samples_per_task'])] if args.stage == 'smoke' else rows[args.shard::args.shards]
    for row in selected:
        directory = 'smoke-v128' if args.stage == 'smoke' else args.stage
        path = args.output/args.arm/directory/f"sample_{row['index']:03d}.json"
        if path.exists():
            audit_saved(read_json(path), row, spec, hashes, args.arm, tokenizer, args.stage == 'smoke')
            continue
        tensor = tokens[str(row['index'])]
        assert len(tensor) == row['input_tokens']
        assert hashlib.sha256(tensor.numpy().tobytes()).hexdigest() == row['input_sha256']
        cap = min(4, row['maximum_tokens']) if args.stage == 'smoke' else row['maximum_tokens']
        counts.clear()
        torch.cuda.reset_peak_memory_stats()
        print('START', args.arm, row['index'], row['task'], len(tensor), flush=True)
        started = time.monotonic()
        if args.arm == 'shadowkv':
            ids, first, stats, stopped = shadow_generate(model, tokenizer, tensor, cap, shadow_cache_class)
        else:
            ids, first, stats, stopped = runtime.generate(
                model, tokenizer, dict(row, input_ids=tensor.tolist()), args.arm, cap)
        verify_dispatch(counts, args.arm, len(manifest['layers']), len(ids))
        prediction = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        result = dict(ids=ids, prediction=prediction, stopped=stopped,
            score=sample_score(prediction, row['answers'], row['match_type']), routing=stats,
            kernel_calls=dict(counts), seconds=time.monotonic()-started,
            peak_gib=torch.cuda.max_memory_allocated()/2**30, cache_value_head_dim=128)
        saved = dict(status='complete', protocol=spec, sample=row, bank_sha256=hashes,
            result=result, eos_ids=eos, command=shlex.join(sys.argv), python=sys.executable,
            gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            first_argmax=int(first.argmax()))
        audit_saved(saved, row, spec, hashes, args.arm, tokenizer, args.stage == 'smoke')
        write_json(path, saved)
        print('COMPLETE', args.arm, row['index'], result['score'], result['seconds'],
              result['peak_gib'], flush=True)


if __name__ == '__main__':
    main()
