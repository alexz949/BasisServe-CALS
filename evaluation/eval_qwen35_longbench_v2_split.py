"""Paired LongBench-v2 CoT on complete 32K-64K inputs."""

import argparse
import gc
import hashlib
import json
import re
from pathlib import Path
import time

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from evaluation.eval_qwen35_k_routing_ruler import ARMS, tensor_hash
from evaluation.qwen35_split_routing import install
from evaluation.prepare_qwen35_longbench_v2 import render, chat_ids
from evaluation.qwen35_hybrid_common import atomic_save, load_bank, load_model, sha256, verify_model_identity


def parse_answer(text):
    text = text.strip().replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', text)
    if match is None:
        match = re.search(r'The correct answer is ([A-D])', text)
    return match.group(1) if match else None


def selected_rows(rows, smoke):
    return [min(rows, key=lambda r:r['prompt_tokens']), max(rows, key=lambda r:r['prompt_tokens'])] if smoke else rows


def read_inputs(args):
    manifest = json.loads((args.data/'manifest.json').read_text())
    protocol = dict(manifest['protocol'], official_answer_cap=manifest['protocol']['answer_cap'],
        answer_cap=args.answer_cap)
    assert manifest['status'] == 'complete'
    assert protocol['model_config_sha256'] == sha256(args.model/'config.json')
    assert protocol['tokenizer_config_sha256'] == sha256(args.model/'tokenizer_config.json')
    assert sha256(args.data/'samples.json') == manifest['samples_sha256']
    assert sha256(args.data/'tokens.safetensors') == manifest['tokens_sha256']
    for name, digest in protocol['official_templates'].items():
        assert sha256(Path('external/LongBench/prompts')/name) == digest
    rows = json.loads((args.data/'samples.json').read_text())
    tokens = load_file(str(args.data/'tokens.safetensors'))
    assert len(rows) == len(tokens) == protocol['samples']
    for index, row in enumerate(rows):
        tensor = tokens[f'sample_{index:03d}']
        assert row['index'] == index and len(tensor) == row['prompt_tokens']
        assert 32768 <= len(tensor) and len(tensor)+protocol['reasoning_cap'] <= 65536
        assert hashlib.sha256(tensor.numpy().tobytes()).hexdigest() == row['input_ids_sha256']
    return rows, tokens, protocol


def summarize(args, rows, data_protocol):
    arms = args.arms
    indices = [r['index'] for r in selected_rows(rows, args.smoke)]
    records = {}
    for arm in arms:
        records[arm] = []
        for index in indices:
            r = json.loads((args.output/arm/f'{index:03d}.json').read_text())
            assert r['status'] == 'complete' and r['sample'] == rows[index]
            assert r['protocol']['arm'] == arm
            assert r['parsed_answer'] == parse_answer(r['answer']['text'])
            assert r['score'] == int(r['parsed_answer'] == rows[index]['answer'])
            for stage, cap in (('reasoning', data_protocol['reasoning_cap']), ('answer', data_protocol['answer_cap'])):
                ids = r[stage]['generated_ids']
                assert 0 < len(ids) <= cap
                eos = r['protocol']['eos_token_ids']
                assert all(token not in eos for token in ids[:-1])
                assert len(ids) == cap or ids[-1] in eos
            if records[arm]:
                assert r['protocol'] == records[arm][0]['protocol']
            records[arm].append(r)
    for arm in arms:
        for r, dense in zip(records[arm], records['full'], strict=True):
            for key in ('input_ids_sha256', 'first_logits_sha256'):
                assert r['reasoning'][key] == dense['reasoning'][key]
            for key in ('v_bank_sha256', 'data_sha256', 'model_identity', 'budget', 'prefill', 'generation', 'eos_token_ids'):
                assert r['protocol'][key] == dense['protocol'][key]
            shared = r['protocol']['code_sha256'].keys() & dense['protocol']['code_sha256'].keys()
            for name in shared:
                assert r['protocol']['code_sha256'][name] == dense['protocol']['code_sha256'][name]
    scores = {}
    for arm, values in records.items():
        groups = {}
        for field in ('domain', 'difficulty'):
            groups[field] = {label:dict(n=sum(r['sample'][field] == label for r in values),
                accuracy=100*sum(r['score'] for r in values if r['sample'][field] == label)/sum(r['sample'][field] == label for r in values))
                for label in sorted({r['sample'][field] for r in values})}
        scores[arm] = dict(accuracy=100*sum(r['score'] for r in values)/len(values),
            unparsed=sum(r['parsed_answer'] is None for r in values), groups=groups)
    for arm in arms:
        scores[arm]['delta_vs_full_pp'] = scores[arm]['accuracy']-scores['full']['accuracy']
    atomic_save(args.output/'summary.json', dict(status='complete', scope='smoke' if args.smoke else 'formal',
        samples_per_arm=len(indices), scores=scores, data_protocol=data_protocol,
        protocols={a:v[0]['protocol'] for a,v in records.items()},
        first_logits_parity='bitwise for paired long-context reasoning stage; answer-stage inputs depend on reasoning',
        artifacts={str(args.output/a/f'{i:03d}.json'):sha256(args.output/a/f'{i:03d}.json') for a in arms for i in indices}))
    print(json.dumps(scores), flush=True)


def generate(model, tokenizer, ids, cap, eos):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    prefill = model.model(ids, use_cache=True)
    cache = prefill.past_key_values
    logits = model.lm_head(prefill.last_hidden_state[:, -1:])[:, -1]
    assert torch.isfinite(logits).all()
    first_hash = tensor_hash(logits)
    prefill_seconds = time.monotonic()-started
    del prefill
    generated = []
    for step in range(cap):
        generated.append(logits.argmax(-1).item())
        if generated[-1] in eos or step+1 == cap:
            break
        output = model.model(torch.tensor([[generated[-1]]], device='cuda'), past_key_values=cache, use_cache=True)
        logits = model.lm_head(output.last_hidden_state[:, -1:])[:, -1]
        assert torch.isfinite(logits).all()
        del output
    return dict(text=tokenizer.decode(generated, skip_special_tokens=True), generated_ids=generated,
        prompt_tokens=ids.shape[1], input_ids_sha256=tensor_hash(ids), first_logits_sha256=first_hash,
        prefill_seconds=prefill_seconds, seconds=time.monotonic()-started,
        peak_gib=torch.cuda.max_memory_allocated()/2**30)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('model', 'v-bank', 'routers', 'loki', 'data', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--arm', choices=ARMS, default='full')
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=4)
    p.add_argument('--answer-cap', type=int, default=512)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--summarize', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows, tokens, data_protocol = read_inputs(args)
    if args.summarize:
        summarize(args, rows, data_protocol)
        return
    assert 0 <= args.shard_index < args.num_shards
    bank = load_bank(args.v_bank)
    assert bank['status'] == 'complete' and bank['method'] == 'twosided'
    assert bank['nominal_v_rank'] == 192 and sum(bank['schedule'].values()) == 1536
    assert bank['wo_compression'] is False
    verify_model_identity(args.model, bank['model_identity'])
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = load_model(str(args.model), 'cuda:0')
    sources = install(model, args, bank)
    files = (str(Path(__file__).relative_to(Path.cwd())), 'evaluation/eval_qwen35_k_routing_ruler.py', 'evaluation/prepare_qwen35_longbench_v2.py', 'evaluation/qwen35_split_routing.py', 'basisserve/kernels/split_indexed_attention.py',
        'basisserve/core/qwen35_k_routing_runtime.py', 'basisserve/core/qwen35_gated_v_runtime.py',
        'basisserve/core/c1_lrqk.py', 'basisserve/core/c1_shadowkv.py', 'basisserve/core/compact_v_flash.py',
        'basisserve/core/c1_conditional_page_attention.py', 'basisserve/core/c1_v_k_index.py')
    protocol = dict(attention_backend='split indexed for page routers and Loki; original full/ShadowKV', arm=args.arm, sources=sources, model_identity=bank['model_identity'],
        v_bank_sha256=sha256(args.v_bank), data_sha256=sha256(args.data/'manifest.json'),
        code_sha256={name:sha256(Path(name)) for name in files},
        budget=dict(ours=2048, recent_inside_ours=64, pinned_sink=0, page_size=32,
            loki_topk_per_head=2048, lrqk_topk_per_head=2048, lrqk_extra_recent=64,
            shadowkv_routed=2048, shadowkv_outlier_chunks=48, shadowkv_chunk=8,
            shadowkv_short_prompt='full support when prompt fits routed/outlier/local capacity'),
        prefill='full causal attention with allocated V192; Wo uncompressed',
        generation=dict(mode='official two-stage CoT; answer stage omits document', sampling='greedy', native_thinking=False,
            reasoning_cap=data_protocol['reasoning_cap'], answer_cap=data_protocol['answer_cap'],
            official_answer_cap=data_protocol['official_answer_cap']))
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    protocol['eos_token_ids'] = sorted(eos)
    chosen = selected_rows(rows, args.smoke)
    for row in chosen[args.shard_index::args.num_shards]:
        path = args.output/args.arm/f"{row['index']:03d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved['status'] == 'complete' and saved['protocol'] == protocol
            continue
        ids = tokens[f"sample_{row['index']:03d}"].long()[None].cuda()
        reasoning = generate(model, tokenizer, ids, data_protocol['reasoning_cap'], eos)
        print(json.dumps(dict(arm=args.arm, index=row['index'], stage='reasoning',
            seconds=reasoning['seconds'], tokens=len(reasoning['generated_ids']))), flush=True)
        answer_ids = chat_ids(tokenizer, render(data_protocol['answer_template'], row, reasoning['text']))
        assert len(answer_ids)+data_protocol['answer_cap'] <= 65536
        answer = generate(model, tokenizer, torch.tensor([answer_ids], device='cuda'), data_protocol['answer_cap'], eos)
        parsed = parse_answer(answer['text'])
        record = dict(status='complete', protocol=protocol, sample=row, reasoning=reasoning, answer=answer,
            answer_input_ids=answer_ids, parsed_answer=parsed, score=int(parsed == row['answer']))
        atomic_save(path, record)
        print(json.dumps(dict(arm=args.arm, index=row['index'], domain=row['domain'],
            score=record['score'], parsed=parsed, seconds=reasoning['seconds']+answer['seconds'])), flush=True)


if __name__ == '__main__':
    main()
