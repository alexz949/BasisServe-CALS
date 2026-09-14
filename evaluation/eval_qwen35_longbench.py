"""Seven paired Qwen3.5 routing arms on the frozen LongBench100 pilot."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from evaluation.eval_longbench_c1_fourarm import official_scorer
from evaluation.eval_qwen35_k_routing_ruler import ARMS, install, tensor_hash
from evaluation.prepare_qwen35_longbench import TASKS
from evaluation.qwen35_hybrid_common import atomic_save, load_bank, load_model, sha256, verify_model_identity


def score(scorer, row, prediction):
    value = max(float(scorer.dataset2metric[row['task']](prediction, answer,
        all_classes=row['all_classes'])) for answer in row['answers'])
    assert 0 <= value <= 1
    return value


def read_inputs(args):
    manifest = json.loads((args.data/'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['protocol']['tasks'] == list(TASKS)
    assert manifest['protocol']['model_config_sha256'] == sha256(args.model/'config.json')
    assert manifest['protocol']['tokenizer_config_sha256'] == sha256(args.model/'tokenizer_config.json')
    assert sha256(args.data/'samples.json') == manifest['samples_sha256']
    assert sha256(args.data/'tokens.safetensors') == manifest['tokens_sha256']
    official = Path(manifest['protocol']['official_root'])
    for name, digest in manifest['protocol']['official_sha256'].items():
        assert sha256(official/'LongBench'/name) == digest
    rows = json.loads((args.data/'samples.json').read_text())
    tokens = load_file(str(args.data/'tokens.safetensors'))
    assert len(rows) == len(tokens) == 100
    for index, row in enumerate(rows):
        tensor = tokens[f'sample_{index:03d}']
        assert row['index'] == index and row['task'] == TASKS[index//20] and row['ordinal'] == index%20
        assert len(tensor) == row['prompt_tokens'] and len(tensor)+row['maximum_tokens'] <= 65536
        assert hashlib.sha256(tensor.numpy().tobytes()).hexdigest() == row['input_ids_sha256']
    return rows, tokens, official_scorer(official)


def summarize(args, rows, scorer):
    records = {}
    indices = [min((r for r in rows if r['task'] == task), key=lambda r:r['prompt_tokens'])['index']
        for task in TASKS] if args.smoke else list(range(100))
    for arm in ARMS:
        records[arm] = []
        for index in indices:
            r = json.loads((args.output/arm/f'{index:03d}.json').read_text())
            assert r['status'] == 'complete' and r['sample'] == rows[index]
            assert r['protocol']['arm'] == arm
            assert r['score'] == score(scorer, rows[index], r['prediction'])
            assert 0 < len(r['generated_ids']) <= rows[index]['maximum_tokens']
            if records[arm]:
                assert r['protocol'] == records[arm][0]['protocol']
            records[arm].append(r)
    reference = records['full']
    for arm in ARMS:
        for r, dense in zip(records[arm], reference, strict=True):
            assert r['input_ids_sha256'] == dense['input_ids_sha256']
            assert r['first_logits_sha256'] == dense['first_logits_sha256']
            if arm == 'shadowkv' and r['sample']['prompt_tokens']//8-4 <= 48+2048//8:
                assert r['generated_ids'] == dense['generated_ids']
            for key in ('v_bank_sha256', 'data_sha256', 'code_sha256', 'model_identity', 'budget', 'prefill'):
                assert r['protocol'][key] == dense['protocol'][key]
    scores = {}
    for arm, values in records.items():
        task_scores = {task: 100*sum(r['score'] for r in values if r['sample']['task'] == task)/
            sum(r['sample']['task'] == task for r in values) for task in TASKS}
        scores[arm] = dict(per_task=task_scores, mean=sum(task_scores.values())/5)
    for arm in ARMS:
        scores[arm]['delta_vs_full_pp'] = scores[arm]['mean']-scores['full']['mean']
    report = dict(status='complete', scope='smoke' if args.smoke else 'formal',
        samples_per_arm=len(indices), scores=scores,
        protocols={a:v[0]['protocol'] for a,v in records.items()},
        first_logits_parity='bitwise for every paired sample',
        artifacts={str(args.output/a/f'{i:03d}.json'):sha256(args.output/a/f'{i:03d}.json') for a in ARMS for i in indices})
    atomic_save(args.output/'summary.json', report)
    print(json.dumps(scores), flush=True)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('model', 'v-bank', 'routers', 'loki', 'data', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--arm', choices=ARMS, default='full')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=4)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--summarize', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows, tokens, scorer = read_inputs(args)
    if args.summarize:
        summarize(args, rows, scorer)
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
    files = (str(Path(__file__).relative_to(Path.cwd())), 'evaluation/eval_qwen35_k_routing_ruler.py',
        'basisserve/core/qwen35_k_routing_runtime.py', 'basisserve/core/qwen35_gated_v_runtime.py',
        'basisserve/core/c1_lrqk.py', 'basisserve/core/c1_shadowkv.py', 'basisserve/core/compact_v_flash.py',
        'basisserve/core/c1_conditional_page_attention.py', 'basisserve/core/c1_v_k_index.py')
    protocol = dict(arm=args.arm, sources=sources, model_identity=bank['model_identity'],
        v_bank_sha256=sha256(args.v_bank), data_sha256=sha256(args.data/'manifest.json'),
        code_sha256={name:sha256(Path(name)) for name in files},
        budget=dict(ours=2048, recent_inside_ours=64, pinned_sink=0, page_size=32,
            loki_topk_per_head=2048, lrqk_topk_per_head=2048, lrqk_extra_recent=64,
            shadowkv_routed=2048, shadowkv_outlier_chunks=48, shadowkv_chunk=8,
            shadowkv_short_prompt='full support when prompt fits routed/outlier/local capacity'),
        prefill='full causal attention with allocated V192; Wo uncompressed',
        generation='greedy, native EOS, thinking off, official task caps')
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    chosen = [min((r for r in rows if r['task'] == task), key=lambda r:r['prompt_tokens']) for task in TASKS] if args.smoke else rows
    for row in chosen[args.shard_index::args.num_shards]:
        path = args.output/args.arm/f"{row['index']:03d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved['status'] == 'complete' and saved['protocol'] == protocol
            continue
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ids = tokens[f"sample_{row['index']:03d}"].long()[None].cuda()
        started = time.monotonic()
        prefill = model.model(ids, use_cache=True)
        cache = prefill.past_key_values
        logits = model.lm_head(prefill.last_hidden_state[:, -1:])[:, -1]
        assert torch.isfinite(logits).all()
        first_hash = tensor_hash(logits)
        prefill_seconds = time.monotonic()-started
        del prefill
        generated = []
        for step in range(row['maximum_tokens']):
            generated.append(logits.argmax(-1).item())
            if generated[-1] in eos or step+1 == row['maximum_tokens']:
                break
            output = model.model(torch.tensor([[generated[-1]]], device='cuda'), past_key_values=cache, use_cache=True)
            logits = model.lm_head(output.last_hidden_state[:, -1:])[:, -1]
            assert torch.isfinite(logits).all()
            del output
        prediction = tokenizer.decode(generated, skip_special_tokens=True)
        record = dict(status='complete', protocol=protocol, sample=row, prediction=prediction,
            generated_ids=generated, input_ids_sha256=tensor_hash(ids), first_logits_sha256=first_hash,
            score=score(scorer, row, prediction), prefill_seconds=prefill_seconds,
            seconds=time.monotonic()-started, peak_gib=torch.cuda.max_memory_allocated()/2**30)
        atomic_save(path, record)
        print(json.dumps(dict(arm=args.arm, index=row['index'], task=row['task'],
            score=record['score'], seconds=record['seconds'])), flush=True)


if __name__ == '__main__':
    main()
