"""Paired Qwen3.5 64K RULER88 with original Wo and allocated compact V."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from basisserve.core.qwen35_k_routing_runtime import Qwen35RoutingAttention
from evaluation.assemble_qwen35_k_routing_v import LAYERS
from evaluation.qwen35_hybrid_common import atomic_save, load_bank, load_model, sha256, verify_model_identity
from evaluation.ruler_v1 import parse_tasks, ruler_prompt, sample_score


ARMS = ('full', 'exact_sparse', 'b16r16', 'b32r32', 'loki', 'lrqk', 'shadowkv')
TASKS = ('niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_1',
    'niah_multikey_2', 'niah_multiquery', 'niah_multivalue', 'vt', 'fwe', 'qa_1', 'qa_2')


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def prompt_ids(tokenizer, row, mode):
    if mode == 'completion':
        return tokenizer(ruler_prompt(row), add_special_tokens=True)['input_ids']
    # Official RULER templates put any answer prefix after the assistant marker.
    text = tokenizer.apply_chat_template([dict(role='user', content=str(row['input']))],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    text += str(row.get('answer_prefix', ''))
    return tokenizer(text, add_special_tokens=False)['input_ids']


def read_inputs(args, tokenizer):
    manifest = json.loads((args.data/'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['protocol']['sequence_length'] == 65536
    assert manifest['protocol']['samples_per_task'] == 8
    assert tuple(manifest['protocol']['tasks']) == TASKS
    assert manifest['tokenizer_config_sha256'] == sha256(args.model/'tokenizer_config.json')
    rows = []
    for task in parse_tasks(','.join(TASKS)):
        path = args.data/task.name/'validation.jsonl'
        assert sha256(path) == manifest['artifacts'][task.name]['sha256']
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(records) == 8
        for ordinal, source in enumerate(records):
            ids = prompt_ids(tokenizer, source, args.prompt_format)
            assert len(ids)+task.tokens_to_generate <= 65536
            rows.append(dict(index=len(rows), task=task.name, ordinal=ordinal,
                ids=ids, answers=source['outputs'], match_type=task.match_type,
                maximum_tokens=task.tokens_to_generate))
    assert len(rows) == 88
    return rows


def install(model, args, bank):
    sources = {}
    for layer in LAYERS:
        native = model.model.layers[layer].self_attn
        v_rank = bank['schedule'][layer]
        factors = (dict(E_V=torch.eye(256).repeat(4, 1, 1), R_V=torch.eye(256).repeat(4, 1, 1))
            if v_rank == 256 else bank['layers'][layer])
        routing, basis = None, None
        if args.arm in ('b16r16', 'b32r32'):
            path = args.routers/args.arm/f'l{layer:02d}.pt'
            payload = torch.load(path, map_location='cpu', weights_only=True)
            assert payload['status'] == 'complete' and payload['layer'] == layer and payload['v_rank'] == v_rank
            protocol = payload['protocol']
            assert protocol['v_bank_sha256'] == sha256(args.v_bank)
            assert protocol['fit_windows'] == 64 and protocol['diagnostic_windows'] == 16
            assert protocol['bcd_sweeps'] == 40 and protocol['pcg_iterations'] == 100
            assert protocol['routing_budget'] == 2048 and protocol['wo_compression'] is False
            routing = {key: value.to(native.v_proj.weight.device) for key, value in payload['tensors'].items()}
            sources[path.name] = sha256(path)
        elif args.arm == 'loki':
            path = args.loki/f'l{layer:02d}.pt'
            payload = torch.load(path, map_location='cpu', weights_only=True)
            assert payload['status'] == 'complete' and payload['layer'] == layer and payload['rank'] == 32
            assert payload['windows_sha256'] == bank['windows_sha256']
            assert payload['model_config_sha256'] == bank['model_identity']['config_sha256']
            basis = payload['projector'].to(native.v_proj.weight.device)
            sources[path.name] = sha256(path)
        model.model.layers[layer].self_attn = Qwen35RoutingAttention(native, factors['E_V'], factors['R_V'],
            arm=args.arm, factors=routing, loki_basis=basis)
    return sources


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--v-bank', type=Path, required=True)
    p.add_argument('--routers', type=Path, required=True)
    p.add_argument('--loki', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--prompt-format', choices=('completion', 'chat'), required=True)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=4)
    p.add_argument('--limit', type=int, default=88)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert 0 <= args.shard_index < args.num_shards and 0 < args.limit <= 88
    bank = load_bank(args.v_bank)
    assert bank['status'] == 'complete' and bank['method'] == 'twosided' and bank['nominal_v_rank'] == 192
    assert bank['encoder_sweeps'] == 12 and bank['encoder_cg'] == 16 and bank['wo_compression'] is False
    assert sum(bank['schedule'].values()) == 1536
    verify_model_identity(args.model, bank['model_identity'])
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = read_inputs(args, tokenizer)
    model = load_model(str(args.model), 'cuda:0')
    sources = install(model, args, bank)
    code = ('evaluation/eval_qwen35_k_routing_ruler.py', 'basisserve/core/qwen35_k_routing_runtime.py',
        'basisserve/core/qwen35_gated_v_runtime.py', 'basisserve/core/c1_lrqk.py',
        'basisserve/core/c1_shadowkv.py', 'basisserve/core/compact_v_flash.py', 'basisserve/core/c1_v_k_index.py',
        'basisserve/core/c1_conditional_page_attention.py')
    protocol = dict(v_bank_sha256=sha256(args.v_bank), data_sha256=sha256(args.data/'manifest.json'),
        model_identity=bank['model_identity'], prompt_format=args.prompt_format, thinking=False,
        sequence_length=65536, samples=88, arm=args.arm, sources=sources,
        code_sha256={name: sha256(Path(name)) for name in code},
        ours_budget=2048, page_size=32, pinned_prefix_pages=0, recent_tokens=64,
        page_selection='historical normalized page mass, then GQA max; recent64 inside hard budget; no pinned sink',
        loki=dict(rank=32, topk=2048, support='per query head; uncapped physical GQA union'),
        lrqk=dict(rank=32, topk=2048, recent=64, prefill_iterations=2, decode_iterations=2,
            state_dtype='bfloat16', solve_dtype='float32'),
        shadowkv=dict(rank=160, budget=2048, chunk=8, outliers=48), wo_compression=False,
        prefill='full FlashAttention on every arm', selection='greedy, native EOS, official task caps')
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    for row in rows[args.shard_index::args.num_shards][:args.limit]:
        path = args.output/args.arm/f'{row["index"]:03d}.json'
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
        prefill_seconds = time.monotonic()-started
        del prefill
        generated = []
        for step in range(row['maximum_tokens']):
            token = logits.argmax(-1)
            generated.append(token.item())
            if generated[-1] in eos or step+1 == row['maximum_tokens']:
                break
            decoded = model.model(token[:, None], past_key_values=cache, use_cache=True)
            logits = model.lm_head(decoded.last_hidden_state[:, -1:])[:, -1]
            assert torch.isfinite(logits).all()
            del decoded
        prediction = tokenizer.decode(generated, skip_special_tokens=True)
        score = sample_score(prediction, row['answers'], row['match_type'])
        record = dict(status='complete', protocol=protocol, index=row['index'], task=row['task'], ordinal=row['ordinal'],
            input_tokens=len(row['ids']), input_ids_sha256=tensor_hash(ids), first_logits_sha256=first_hash,
            generated_ids=generated, prediction=prediction, answers=row['answers'], score=score,
            prefill_seconds=prefill_seconds, seconds=time.monotonic()-started,
            peak_gib=torch.cuda.max_memory_allocated()/2**30)
        atomic_save(path, record)
        print(json.dumps(dict(index=row['index'], task=row['task'], score=score,
            seconds=record['seconds'], peak_gib=record['peak_gib'])), flush=True)
        del cache, logits, ids


if __name__ == '__main__':
    main()
