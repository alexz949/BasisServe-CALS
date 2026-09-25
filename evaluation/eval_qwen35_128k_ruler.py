"""Qwen3.5-9B 128K RULER (11 tasks x 100 examples) on the deployed uniform V192 + GDN Wo75 model.

Arms: ``full`` (exact causal attention over the compressed V192 cache) and ``b{B}r{R}`` (Base-B + Residual-R page routing
from the ``ours_b{B}r{R}`` bank, hard physical budget 2048 per KV group including the exact recent 64, no pinned sink). Both arms share the same
FlashAttention prefill and the same GDN Wo75 Private-AllGather output projections; only decode-time key selection differs.
Prompts follow the official RULER base-completion template (thinking disabled, no chat template).
"""
import argparse
import gc
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from basisserve.core.qwen35_gdn_private_ag_runtime import Qwen35PrivateAGRuntime, load_qwen35_gdn_private_ag_factors
from basisserve.core.qwen35_k_routing_runtime import Qwen35RoutingAttention
from evaluation.fit_k_routing_streaming import layer_file, verified
from evaluation.qwen35_hybrid_common import atomic_save, load_bank, load_model, sha256, verify_model_identity
from evaluation.ruler_v1 import parse_tasks, ruler_prompt, sample_score
from evaluation.v96kl_common import read_json

ARM_PATTERN = re.compile(r'^(full|dense|v_only|wo_only|lrqk|shadowkv|b(\d+)r(\d+))$')
TASKS = ('niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_1',
         'niah_multikey_2', 'niah_multiquery', 'niah_multivalue', 'vt', 'fwe', 'qa_1', 'qa_2')
LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)
SEQUENCE_LENGTH = 131072
# Official LongBench pred.py: chat models are better off without the chat wrapper on these datasets.
LONGBENCH_NO_CHAT_TASKS = ('trec', 'triviaqa', 'samsum', 'lsht', 'lcc', 'repobench-p')
SAMPLES_PER_TASK = 100
CODE = ('evaluation/eval_qwen35_128k_ruler.py', 'basisserve/core/qwen35_k_routing_runtime.py',
        'basisserve/core/qwen35_gated_v_runtime.py', 'basisserve/core/qwen35_gdn_private_ag_runtime.py',
        'basisserve/core/c1_conditional_page_attention.py', 'basisserve/core/compact_v_flash.py',
        'basisserve/core/c1_v_k_index.py')


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def read_inputs(args, tokenizer):
    manifest = read_json(args.data/'manifest.json')
    protocol = manifest['protocol']
    assert manifest['status'] == 'complete' and protocol['sequence_length'] == SEQUENCE_LENGTH
    assert manifest['tokenizer_config_sha256'] == sha256(args.model/'tokenizer_config.json')
    if args.benchmark == 'longbench':
        assert manifest['format'] == 'basisserve.longbench_v1.shadowkv9.v1' and manifest['benchmark'] == 'longbench_v1'
        tasks = tuple(protocol['tasks'])
        counts = {name: int(protocol['samples_per_task'][name]) for name in tasks}
    else:
        assert protocol['samples_per_task'] == SAMPLES_PER_TASK and protocol['model_template_type'] == 'base'
        assert tuple(protocol['tasks']) == TASKS
        tasks = TASKS
        counts = {name: SAMPLES_PER_TASK for name in tasks}
    # Tokenizing 1100 prompts of 128K tokens takes minutes; every shard reads the same frozen token ids instead.
    prompt_format = ('chat template (user = official LongBench prompt), add_generation_prompt, enable_thinking=False; '
                     'raw completion prompt for ' + ', '.join(LONGBENCH_NO_CHAT_TASKS)
                     if args.benchmark == 'longbench' else 'official RULER base completion')
    cache = args.data/('prompts_qwen35_longbench_chat.pt' if args.benchmark == 'longbench' else 'prompts_qwen35_base.pt')
    stamp = dict(manifest_sha256=sha256(args.data/'manifest.json'), tokenizer_config_sha256=manifest['tokenizer_config_sha256'],
                 tokenizer_sha256=sha256(args.model/'tokenizer.json'), prompt_format=prompt_format)
    cached = torch.load(cache, weights_only=False) if cache.exists() else None
    ruler_tasks = {task.name: task for task in parse_tasks(','.join(TASKS))} if args.benchmark == 'ruler' else {}
    rows = []
    for name in tasks:
        path = args.data/name/'validation.jsonl'
        assert sha256(path) == manifest['artifacts'][name]['sha256']
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(records) == counts[name]
        for ordinal, source in enumerate(records):
            if args.benchmark == 'longbench':
                maximum, match = int(source['max_gen']), str(source['metric'])
            else:
                maximum, match = ruler_tasks[name].tokens_to_generate, ruler_tasks[name].match_type
            if cached is not None:
                assert cached['stamp'] == stamp
                ids = cached['ids'][len(rows)]
            elif args.benchmark == 'longbench' and name not in LONGBENCH_NO_CHAT_TASKS:
                ids = tokenizer.apply_chat_template([dict(role='user', content=str(source['input']))], tokenize=True,
                    add_generation_prompt=True, return_dict=True, enable_thinking=False)['input_ids']
            else:
                ids = tokenizer(ruler_prompt(source), add_special_tokens=True)['input_ids']
            assert len(ids) + maximum <= SEQUENCE_LENGTH
            row = dict(index=len(rows), task=name, ordinal=ordinal, ids=ids, answers=source['outputs'],
                       match_type=match, maximum_tokens=maximum)
            if args.benchmark == 'longbench':
                row.update(all_classes=source.get('all_classes'), source_index=int(source['source_index']), _id=str(source['_id']))
            rows.append(row)
    assert len(rows) == sum(counts.values())
    if cached is None:
        # Shards start together; whichever finishes tokenizing first publishes the cache, the others verify it.
        try:
            atomic_save(cache, dict(stamp=stamp, ids=[row['ids'] for row in rows]))
        except FileExistsError:
            existing = torch.load(cache, weights_only=False)
            assert existing['stamp'] == stamp and existing['ids'] == [row['ids'] for row in rows]
    return rows


def install(model, args, bank, gdn):
    sources = {}
    if args.arm == 'dense':
        # Original dense V and Wo: nothing installed; native Qwen3.5 attention and GDN with the fla kernel.
        model.eval()
        return sources
    if args.arm == 'wo_only':
        # Attribution arm: original dense V (native attention), only the GDN Wo75 Private-AllGather installed.
        Qwen35PrivateAGRuntime(model, gdn).install()
        model.eval()
        return sources
    for layer in LAYERS:
        native = model.model.layers[layer].self_attn
        assert int(bank['schedule'][layer]) == 192
        factors = bank['layers'][layer]
        routing = None
        if args.ranks is not None:
            base_rank, residual_rank = args.ranks
            path = layer_file(args.routers, f'ours_b{base_rank}r{residual_rank}', layer)
            tensors, meta = verified(path)
            protocol = meta['protocol']
            assert meta['layer'] == layer and int(meta['v_rank']) == 192
            assert protocol['v_bank_sha256'] == sha256(args.v_bank) and protocol['gdn_bank_sha256'] == sha256(args.gdn_bank)
            assert protocol['base_rank'] == base_rank and protocol['residual_rank'] == residual_rank and protocol['page_size'] == args.page_size
            assert protocol['excluded_prefix_pages'] == 0 and protocol['excluded_recent_tokens'] == 64
            assert protocol['deployment_page_budget_tokens'] == 2048 and protocol['sequence_length'] == SEQUENCE_LENGTH
            assert not protocol['smoke']
            routing = {key: value.to(native.v_proj.weight.device) for key, value in tensors.items()}
            sources[path.name] = dict(sha256=sha256(path), objective=protocol['objective'],
                fit_windows=len(protocol['fit_ids']), sweeps=protocol['sweeps'], pcg_iterations=protocol['pcg_iterations'])
        model.model.layers[layer].self_attn = Qwen35RoutingAttention(native, factors['E_V'], factors['R_V'],
            arm='ours' if args.ranks is not None else ('full' if args.arm == 'v_only' else args.arm), factors=routing,
            base_rank=None if args.ranks is None else args.ranks[0], residual_rank=None if args.ranks is None else args.ranks[1],
            budget=args.ours_budget, lrqk_topk=args.lrqk_topk, loki_topk=args.loki_topk, shadowkv_budget=args.shadowkv_budget, page_size=args.page_size)
    if args.arm != 'v_only':
        # Attribution arm 'v_only': V192 installed, GDN Wo left original.
        Qwen35PrivateAGRuntime(model, gdn).install()
    model.eval()
    return sources


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('model', 'v-bank', 'gdn-bank', 'routers', 'data', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--arm', required=True, help="'full' or b{B}r{R}, e.g. b16r16, b24r8")
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=8)
    p.add_argument('--limit', type=int, default=10 ** 6)
    p.add_argument('--indices', help='comma-separated row indices for smoke runs (overrides sharding)')
    p.add_argument('--benchmark', choices=('ruler', 'longbench'), default='ruler')
    p.add_argument('--ours-budget', type=int, default=2048, help='page-routing physical tokens incl. recent 64 (no sink on Qwen3.5)')
    p.add_argument('--page-size', type=int, default=32, choices=(1, 2, 4, 8, 16, 32), help='routing page size; must match the router protocol')
    p.add_argument('--lrqk-topk', type=int, default=2048)
    p.add_argument('--loki-topk', type=int, default=2048)
    p.add_argument('--shadowkv-budget', type=int, default=2048, help='ShadowKV routed tokens; 48 outlier chunks x 8 and the local tail are extra')
    args = p.parse_args()
    match = ARM_PATTERN.match(args.arm)
    assert match, args.arm
    args.ranks = (int(match.group(2)), int(match.group(3))) if match.group(2) else None
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert 0 <= args.shard_index < args.num_shards and 0 < args.limit
    bank = load_bank(args.v_bank)
    assert bank['status'] == 'complete' and bank['method'] == 'uniform' and bank['nominal_v_rank'] == 192
    assert bank['encoder_sweeps'] == 12 and bank['encoder_cg'] == 16 and bank['wo_compression'] is False
    verify_model_identity(args.model, bank['model_identity'])
    gdn = load_qwen35_gdn_private_ag_factors(args.gdn_bank)
    assert gdn['status'] == 'complete' and len(gdn['layers']) == 24
    gp = gdn['protocol']
    assert gp['trajectory'] == 'native_dense' and gp['upstream_compression'] is None
    assert gp['rank_allocation'] == 'uniform' and gp['local_rank'] == 768 and gp['tp_size'] == 4
    assert gp['windows_sha256'] == bank['windows_sha256']
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = read_inputs(args, tokenizer)
    model = load_model(str(args.model), 'cuda:0')
    sources = install(model, args, bank, gdn)
    protocol = dict(format='basisserve.qwen35.routing_ruler.128k.v2', benchmark=args.benchmark, v_bank_sha256=sha256(args.v_bank),
        gdn_bank_sha256=sha256(args.gdn_bank), data_sha256=sha256(args.data/'manifest.json'),
        model_identity=bank['model_identity'], thinking=False,
        value_mode={'dense': 'dense original V and Wo', 'v_only': 'uniform V192, original GDN Wo', 'wo_only': 'original dense V, GDN Wo75'}.get(args.arm, 'uniform V192 + GDN Wo75'),
        prompt_format=('chat template (user = official LongBench prompt), add_generation_prompt, enable_thinking=False; '
                       'raw completion prompt for ' + ', '.join(LONGBENCH_NO_CHAT_TASKS)
                       if args.benchmark == 'longbench' else 'official RULER base completion'),
        sequence_length=SEQUENCE_LENGTH, samples=len(rows), arm=args.arm,
        samples_per_task={t: sum(r['task'] == t for r in rows) for t in dict.fromkeys(r['task'] for r in rows)},
        router_sources=sources, code_sha256={name: sha256(Path(name)) for name in CODE},
        numerical_policy=dict(model_dtype='bfloat16', cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False),
        checkpoint=dict(full_attention_v='uniform V192 ALS12 encoder-CG16 decoder-CG50',
                        gdn_wo='24 layers; TP4-private 1024->768 per source; ALS6'),
        ours=None if args.ranks is None else dict(base=args.ranks[0], residual=args.ranks[1], page_size=args.page_size,
            physical_group_budget=args.ours_budget, sink=0, recent=64, recent_inside_budget=True, maximum_support=args.ours_budget),
        lrqk=dict(rank=32, topk_per_query_head=args.lrqk_topk, recent=64, prefill_iterations=2, decode_iterations=2,
                  state_dtype='bfloat16', solve_dtype='float32', physical_gqa_union='uncapped'),
        shadowkv=dict(rank=160, chunk=8, routed=args.shadowkv_budget, outlier_chunks=48, extra_support='official local tail and generated tokens'),
        loki_topk=args.loki_topk,
        gdn_kernel='flash-linear-attention chunk_gated_delta_rule (Triton) via transformers fallback resolution',
        prefill='exact full causal FlashAttention with the V192 cache and GDN-Wo75 installed on every arm',
        selection='greedy; native EOS; official per-task generation caps')
    if args.benchmark == 'longbench':
        manifest = read_json(args.data/'manifest.json')
        protocol['longbench'] = dict(data_protocol=manifest['protocol'], model_tag=manifest['model_tag'],
            scoring='official LongBench eval.py per-sample scorer: max over references; first non-empty line for samsum/trec/triviaqa/lsht',
            generation='greedy, native EOS, official dataset2maxlen caps')
        for name in ('evaluation/longbench_metrics.py', 'evaluation/longbench_official/metrics.py',
                     'evaluation/longbench_official/dataset2prompt.json', 'evaluation/longbench_official/dataset2maxlen.json'):
            protocol['code_sha256'][name] = sha256(Path(name))
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    if args.indices:
        selected = [rows[int(x)] for x in args.indices.split(',')]
    else:
        selected = rows[args.shard_index::args.num_shards][:args.limit]
    for row in selected:
        path = args.output/args.arm/f'{row["index"]:04d}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved['status'] == 'complete' and saved['protocol'] == protocol
            continue
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        ids = torch.tensor([row['ids']], device='cuda', dtype=torch.long)
        prefill = model.model(ids, use_cache=True)
        cache = prefill.past_key_values
        logits = model.lm_head(prefill.last_hidden_state[:, -1:])[:, -1]
        assert torch.isfinite(logits).all()
        first_hash = tensor_hash(logits)
        prefill_seconds = time.monotonic() - started
        del prefill
        generated = []
        for step in range(row['maximum_tokens']):
            token = logits.argmax(-1)
            generated.append(token.item())
            if generated[-1] in eos or step + 1 == row['maximum_tokens']:
                break
            decoded = model.model(token[:, None], past_key_values=cache, use_cache=True)
            logits = model.lm_head(decoded.last_hidden_state[:, -1:])[:, -1]
            assert torch.isfinite(logits).all()
            del decoded
        prediction = tokenizer.decode(generated, skip_special_tokens=True)
        if args.benchmark == 'longbench':
            from evaluation.longbench_metrics import longbench_score
            score = longbench_score(row['task'], prediction, row['answers'], row.get('all_classes'))
        else:
            score = sample_score(prediction, row['answers'], row['match_type'])
        record = dict(status='complete', protocol=protocol, index=row['index'], task=row['task'], ordinal=row['ordinal'],
            input_tokens=len(row['ids']), input_ids_sha256=tensor_hash(ids), first_logits_sha256=first_hash,
            command=sys.argv, python=sys.executable, stopped=generated[-1] in eos,
            generated_ids=generated, prediction=prediction, answers=row['answers'], score=score,
            prefill_seconds=prefill_seconds, seconds=time.monotonic() - started,
            peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
        atomic_save(path, record)
        print(json.dumps(dict(index=row['index'], task=row['task'], score=score, prefill=round(prefill_seconds, 1),
                              seconds=round(record['seconds'], 1), peak_gib=round(record['peak_gib'], 1))), flush=True)
        del cache, logits, ids


if __name__ == '__main__':
    main()
