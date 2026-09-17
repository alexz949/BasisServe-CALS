"""Original dense-V128 Full-K Llama-3.1-8B-Instruct on RULER-128K with TP2."""

import argparse
from collections import Counter
import functools
import importlib
import os
from pathlib import Path
import shlex
import sys
import time

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, DistributedConfig

from evaluation import eval_k_routing_ruler as runtime
from evaluation import eval_llama_cal128 as base
from evaluation.deterministic_evaluation import configure_deterministic_evaluation
from evaluation.k_routing_config import routing_config
from evaluation.ruler_v1 import sample_score
from evaluation.v96kl_common import read_json, sha256, write_json


ARMS = ('full',)
SMOKE_IDS = base.SMOKE_IDS
TASKS = base.TASKS


def inputs(args, identity):
    original = base.ARMS
    base.ARMS = ARMS
    try:
        manifest, rows, tokens, spec = base.inputs(args, identity)
    finally:
        base.ARMS = original
    spec.update(
        format='basisserve.llama_dense128_tp2_ruler.v1',
        rank_schedule=[identity['head_dim']] * len(identity['attention_layers']),
        generation='greedy; native EOS; official task caps; original dense V and Full-K',
        kernels='Triton prefill and dense Triton decode; dispatch counts checked per prompt',
        value_mode='original V128 and original Wo',
        distributed=dict(
            tensor_parallel_size=2,
            plan='Llama native QKV/MLP colwise and O/down rowwise',
            transport='NCCL host transport',
            nccl_p2p_disable=1,
            local_query_heads=identity['hq'] // 2,
            local_key_value_heads=identity['hkv'] // 2,
        ),
    )
    spec.pop('ours')
    spec['source_sha256']['evaluation/eval_llama_dense128.py'] = sha256(Path(__file__))
    spec['source_sha256']['evaluation/llama_prefill_memory.py'] = sha256(
        Path('evaluation/llama_prefill_memory.py'))
    return manifest, rows, tokens, spec


def instrument_kernels():
    counts = Counter()
    targets = (
        ('compressed_v_decode_attention', 'compressed_v_prefill_attention', 'prefill'),
        ('compressed_v_decode_attention', 'compressed_v_decode_attention_triton', 'dense_decode'),
        ('indexed_sparse_decode_attention', 'gqa_indexed_sparse_decode_attention_triton', 'indexed_decode'),
        ('split_indexed_attention', 'split_indexed_attention', 'split_decode'),
    )
    for module_name, name, label in targets:
        module = importlib.import_module('basisserve.kernels.' + module_name)
        original = getattr(module, name)

        def tracked(*args, _original=original, _label=label, **kwargs):
            counts[_label] += 1
            return _original(*args, **kwargs)

        wrapped = functools.wraps(original)(tracked)
        for loaded in tuple(sys.modules.values()):
            if loaded is not None and getattr(loaded, '__name__', '').startswith(('basisserve.', 'evaluation.')):
                for attribute, value in tuple(vars(loaded).items()):
                    if value is original:
                        setattr(loaded, attribute, wrapped)
    return counts


def load_tp2(identity):
    assert dist.is_initialized() and dist.get_world_size() == 2
    assert os.environ.get('NCCL_P2P_DISABLE') == '1'
    config = routing_config(identity, rope='native', sequence_length=131072)
    model = AutoModelForCausalLM.from_pretrained(
        identity['model'], config=config, dtype=torch.bfloat16,
        local_files_only=True, trust_remote_code=False, attn_implementation='sdpa',
        distributed_config=DistributedConfig(tp_size=2, tp_plan='auto'),
    ).eval()
    for layer in model.model.layers:
        attention = layer.self_attn
        attention.num_attention_heads = config.num_attention_heads // 2
        attention.num_key_value_heads = config.num_key_value_heads // 2
        attention.num_key_value_groups = attention.num_attention_heads // attention.num_key_value_heads
        attention.value_head_dim = attention.head_dim
        attention.q_norm = torch.nn.Identity()
        attention.k_norm = torch.nn.Identity()
        attention._routing_arm = 'full'
    from evaluation.llama_prefill_memory import install as install_prefill_memory
    install_prefill_memory(model)
    return model


def rank_state(ids, first, cap, device):
    state = torch.full((cap + 2,), -1, device=device, dtype=torch.long)
    state[0] = len(ids)
    state[1:1 + len(ids)] = torch.tensor(ids, device=device)
    state[-1] = int(first.argmax())
    gathered = [torch.empty_like(state) for _ in range(2)]
    dist.all_gather(gathered, state)
    assert torch.equal(gathered[0], gathered[1])


def run_distributed(args, identity, manifest, rows, tokens, spec, tokenizer):
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    assert dist.get_world_size() == 2
    if args.stage == 'evaluate':
        gate = read_json(args.output/'smoke_audit.json')
        assert gate['status'] == 'complete' and gate['protocol'] == spec
    model = load_tp2(identity)
    counts = instrument_kernels()
    eos = model.config.eos_token_id
    eos = sorted(set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id})
    selected = [rows[i] for i in (0, 7*spec['samples_per_task'])] if args.stage == 'smoke' else rows[args.shard::args.shards]
    for row in selected:
        path = args.output/'full'/args.stage/f"sample_{row['index']:03d}.json"
        skip = torch.tensor([int(path.exists()) if rank == 0 else 0], device=f'cuda:{rank}')
        dist.broadcast(skip, src=0)
        if int(skip):
            if rank == 0:
                base.audit_saved(read_json(path), row, spec, {}, 'full', tokenizer, args.stage == 'smoke')
            dist.barrier()
            continue
        token_tensor = tokens[str(row['index'])]
        assert len(token_tensor) == row['input_tokens']
        cap = min(4, row['maximum_tokens']) if args.stage == 'smoke' else row['maximum_tokens']
        counts.clear()
        torch.cuda.reset_peak_memory_stats()
        if rank == 0:
            print('START full', row['index'], row['task'], len(token_tensor), flush=True)
        started = time.monotonic()
        ids, first, stats, stopped = runtime.generate(
            model, tokenizer, dict(row, input_ids=token_tensor.tolist()), 'full', cap)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        base.verify_dispatch(counts, 'full', len(manifest['layers']), len(ids))
        collective_device = torch.device('cuda', rank)
        rank_state(ids, first, cap, collective_device)
        peak = torch.tensor([torch.cuda.max_memory_allocated()/2**30], device=collective_device)
        peaks = [torch.empty_like(peak) for _ in range(2)]
        dist.all_gather(peaks, peak)
        if rank == 0:
            prediction = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            result = dict(
                ids=ids,
                prediction=prediction,
                stopped=stopped,
                score=sample_score(prediction, row['answers'], row['match_type']),
                routing=stats,
                kernel_calls=dict(counts),
                seconds=elapsed,
                peak_gib_by_rank=[float(value) for value in peaks],
            )
            saved = dict(
                status='complete', protocol=spec, sample=row, bank_sha256={}, result=result,
                eos_ids=eos, command=shlex.join(sys.argv), python=sys.executable,
                gpu=torch.cuda.get_device_name(), first_argmax=int(first.argmax()),
            )
            base.audit_saved(saved, row, spec, {}, 'full', tokenizer, args.stage == 'smoke')
            write_json(path, saved)
            print('COMPLETE full', row['index'], result['score'], elapsed, flush=True)
        dist.barrier()
    dist.destroy_process_group()


def audit_or_summarize(args, rows, spec, tokenizer):
    smoke = args.stage == 'audit-smoke'
    selected = [rows[i] for i in (0, 7*spec['samples_per_task'])] if smoke else rows
    results = []
    for row in selected:
        stage = 'smoke' if smoke else 'evaluate'
        saved = read_json(args.output/'full'/stage/f"sample_{row['index']:03d}.json")
        result = base.audit_saved(saved, row, spec, {}, 'full', tokenizer, smoke)
        if smoke:
            assert len(result['ids']) > 1
        results.append(result)
    if smoke:
        write_json(args.output/'smoke_audit.json',
            dict(status='complete', protocol=spec, verified=len(SMOKE_IDS)))
        return
    tasks = {
        task: {'full': 100 * sum(result['score'] for result, row in zip(results, rows, strict=True)
                                  if row['task'] == task) / spec['samples_per_task']}
        for task in TASKS
    }
    means = {'full': sum(value['full'] for value in tasks.values()) / len(TASKS)}
    write_json(args.output/'summary.json', dict(
        status='complete', protocol=spec, tasks=tasks, means=means,
        verified_predictions=len(rows),
    ))
    lines = [
        '# Llama-3.1-8B-Instruct Dense V128 Full-K RULER 128K', '',
        f"11 tasks × {spec['samples_per_task']} prompts; native TP2 with NCCL P2P disabled.", '',
        '| Task | Dense |', '|---|---:|',
    ]
    lines.extend(f"| {task} | {tasks[task]['full']:.4f} |" for task in TASKS)
    lines.append(f"| Mean | {means['full']:.4f} |")
    text = '\n'.join(lines) + '\n'
    markdown = args.output/'summary.md'
    if markdown.exists():
        assert markdown.read_text() == text
    else:
        markdown.write_text(text)
    print('VERIFIED', means, flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare', 'smoke', 'audit-smoke', 'evaluate', 'summarize'))
    for name in ('identity', 'data', 'bank', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, default='full')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=2)
    args = parser.parse_args()
    configure_deterministic_evaluation()
    assert args.arm == 'full' and 0 <= args.shard < args.shards
    identity = read_json(args.identity)
    tokenizer = AutoTokenizer.from_pretrained(identity['model'], local_files_only=True)
    if args.stage == 'prepare':
        base.prepare(args, identity, tokenizer)
        return
    manifest, rows, tokens, spec = inputs(args, identity)
    if args.stage in ('smoke', 'evaluate'):
        run_distributed(args, identity, manifest, rows, tokens, spec, tokenizer)
    else:
        audit_or_summarize(args, rows, spec, tokenizer)


if __name__ == '__main__':
    main()
