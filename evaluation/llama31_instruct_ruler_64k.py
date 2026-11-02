"""Paired 64K RULER: uniform C1-V96 full K versus Triton Base16/R8 routing."""

import argparse
import json
from pathlib import Path
import shlex
import sys
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM,AutoTokenizer
from accelerate import dispatch_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.llama31_instruct_c1_64k import MODEL,RUN,LENGTH,configure,install_chunked_mlp
from evaluation.llama31_instruct_ruler_data import TASKS
from evaluation.v96kl_common import sha256,read_json,write_json,tensor_hash
from evaluation.ruler_v1 import parse_tasks,sample_score
from basisserve.core.llama_c1_ruler import LlamaC1RulerAttention,LlamaC1RulerCache


def inputs(args,tokenizer):
    c1 = read_json(args.run/'c1/results.json')
    assert c1['status'] == 'complete' and len(c1['artifacts']) == 32
    assert c1['fit_config']['cache_rank_per_head'] == 96
    assert c1['fit_config']['model_config_sha256'] == sha256(args.model/'config.json')
    data = read_json(args.run/'ruler_data/manifest.json')
    assert data['status'] == 'complete'
    assert data['protocol']['tasks'] == TASKS.split(',')
    assert data['protocol']['sequence_length'] == LENGTH
    assert data['protocol']['samples_per_task'] == 30
    assert data['protocol']['model_template_type'] == 'meta-llama3'
    assert data['protocol']['model_config_sha256'] == sha256(args.model/'config.json')
    bank,router_hashes = {},{}
    for layer in range(32):
        item = c1['artifacts'][str(layer)]
        assert sha256(args.run/'c1'/item['file']) == item['sha256']
        path = args.run/'router'/f'layer_{layer:03d}.safetensors'
        record = read_json(path.with_suffix('.json'))
        assert record['status'] == 'complete'
        assert record['protocol']['c1_sha256'] == sha256(args.run/'c1/results.json')
        assert record['protocol']['query_count'] == 64 and record['protocol']['residual_rank'] == 8
        assert sha256(path) == record['sha256']
        bank[layer] = load_file(str(path))
        router_hashes[str(layer)] = record['sha256']
    rows = []
    for task in parse_tasks(TASKS):
        item = data['artifacts'][task.name]
        path = args.run/'ruler_data'/item['file']
        assert sha256(path) == item['sha256']
        sources = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(sources) == 30
        for source in sources:
            ids = tokenizer(source['input']+source.get('answer_prefix',''),add_special_tokens=False)['input_ids']
            assert len(ids)+task.tokens_to_generate <= LENGTH
            rows.append(dict(index=len(rows),task=task.name,ids=ids,answers=source['outputs'],
                match_type=task.match_type,maximum_tokens=task.tokens_to_generate))
    assert len(rows) == 330
    protocol = dict(model=str(args.model),model_config_sha256=sha256(args.model/'config.json'),
        c1_sha256=sha256(args.run/'c1/results.json'),router_sha256=router_hashes,
        data_sha256=sha256(args.run/'ruler_data/manifest.json'),sequence_length=LENGTH,
        samples_per_task=30,tasks=TASKS.split(','),dtype='bfloat16',value_rank=96,
        base_rank=16,residual_rank=8,query_count=64,page_size=32,physical_budget=2048,
        sink_tokens=32,recent_physical_pages=2,historical_pages=61,
        prefill='shared full-causal compressed_v_prefill_attention Triton',
        full_decode='compressed_v_decode_attention_triton',
        sparse_decode='gqa_page32_log_mass_triton + physical-page support + gqa_indexed_sparse_decode_attention_triton',
        generation='greedy, official task caps, model generation EOS; all 330 retained',
        source_sha256={p:sha256(ROOT/p) for p in (
            'evaluation/llama31_instruct_ruler_64k.py','basisserve/core/llama_c1_ruler.py',
            'basisserve/core/c1_strict_recent_decode.py','basisserve/core/c1_v_conditional_k_router.py',
            'basisserve/kernels/compressed_v_decode_attention.py',
            'basisserve/kernels/indexed_sparse_decode_attention.py')})
    return c1,bank,rows,protocol


def sample_metadata(row):
    return {k:v for k,v in row.items() if k != 'ids'} | dict(prompt_tokens=len(row['ids']),
        prompt_sha256=tensor_hash(torch.tensor(row['ids'],dtype=torch.int32)))


@torch.inference_mode()
def load_model(args,c1,bank):
    model = AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,
        local_files_only=True,attn_implementation='sdpa').eval()
    for index,layer in enumerate(model.model.layers):
        tensors = load_file(str(args.run/'c1'/c1['artifacts'][str(index)]['file']))
        layer.self_attn = LlamaC1RulerAttention(layer.self_attn,
            tensors['value_coordinate_encoders'],tensors['head_output_decoders'],bank[index])
    install_chunked_mlp(model.model)
    assert torch.cuda.device_count() == 2
    mapping = {'model.embed_tokens':0,'model.rotary_emb':0,'model.norm':1,'lm_head':1}
    mapping.update({f'model.layers.{i}':i//16 for i in range(32)})
    return dispatch_model(model,device_map=mapping,skip_keys=['past_key_values','past_key_value'])


@torch.inference_mode()
def generate_pair(model,tokenizer,row,cap):
    modules = [layer.self_attn for layer in model.model.layers]
    for module in modules:
        module.sparse = True
        module.prefill_calls = module.decode_calls = module.routing_calls = module.selected_max = 0
    length = len(row['ids'])
    cache = LlamaC1RulerCache(length+cap)
    out = model(input_ids=torch.tensor([row['ids']],device='cuda:0'),past_key_values=cache,
        use_cache=True,logits_to_keep=1)
    first = out.logits[0,-1].float().cpu()
    assert torch.isfinite(first).all()
    del out
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos,list) else [eos]) | {tokenizer.eos_token_id}
    results = {}
    for arm in ('full','sparse'):
        for module in modules:
            module.sparse = arm == 'sparse'
        cache.lengths = [length]*32
        cache._seen_tokens = length
        ids = [int(first.argmax())]
        while len(ids) < cap and ids[-1] not in eos:
            out = model(input_ids=torch.tensor([[ids[-1]]],device='cuda:0'),
                past_key_values=cache,use_cache=True,logits_to_keep=1)
            assert torch.isfinite(out.logits).all()
            ids.append(int(out.logits[0,-1].argmax()))
            del out
        prediction = tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        results[arm] = dict(ids=ids,prediction=prediction,stopped_on_eos=ids[-1] in eos,
            score=sample_score(prediction,row['answers'],row['match_type']))
    expected_steps = sum(len(result['ids'])-1 for result in results.values())
    assert all(m.prefill_calls == 1 and m.decode_calls == expected_steps for m in modules)
    assert all(m.routing_calls == len(results['sparse']['ids'])-1 for m in modules)
    assert all(m.selected_max <= 2048 for m in modules)
    audit = dict(prefill_kernel_calls=sum(m.prefill_calls for m in modules),
        decode_kernel_calls=sum(m.decode_calls for m in modules),
        sparse_kernel_calls=sum(m.routing_calls for m in modules),
        max_selected_tokens=max(m.selected_max for m in modules),shared_first_token=True)
    del cache
    return results,audit


def summarize(args,rows,protocol,tokenizer):
    eos = read_json(args.model/'generation_config.json')['eos_token_id']
    eos = set(eos if isinstance(eos,list) else [eos]) | {tokenizer.eos_token_id}
    records,hashes = [],{}
    for row in rows:
        path = args.run/'ruler/evaluate'/f"sample_{row['index']:03d}.json"
        saved = read_json(path)
        assert saved['status'] == 'complete' and saved['protocol'] == protocol
        assert saved['sample'] == sample_metadata(row)
        for result in saved['arms'].values():
            ids = result['ids']
            assert 0 < len(ids) <= row['maximum_tokens']
            assert not any(token in eos for token in ids[:-1])
            assert result['stopped_on_eos'] == (ids[-1] in eos)
            assert result['stopped_on_eos'] or len(ids) == row['maximum_tokens']
            assert result['prediction'] == tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            assert result['score'] == sample_score(result['prediction'],row['answers'],row['match_type'])
        assert saved['arms']['full']['ids'][0] == saved['arms']['sparse']['ids'][0]
        assert saved['kernel_audit']['prefill_kernel_calls'] == 32
        assert saved['kernel_audit']['sparse_kernel_calls'] == 32*(len(saved['arms']['sparse']['ids'])-1)
        assert saved['kernel_audit']['max_selected_tokens'] <= 2048
        records.append(saved)
        hashes[str(row['index'])] = sha256(path)
    scores = {task:{arm:100*sum(r['arms'][arm]['score'] for r in records
        if r['sample']['task'] == task)/30 for arm in ('full','sparse')} for task in TASKS.split(',')}
    means = {arm:sum(score[arm] for score in scores.values())/11 for arm in ('full','sparse')}
    write_json(args.run/'ruler/result.json',dict(status='complete',protocol=protocol,
        tasks=scores,means=means,paired_prompts=330,verified_predictions=660,input_sha256=hashes))
    text = '# Llama-3.1-8B-Instruct C1-V96: RULER 64K\n\n'
    text += '11 tasks × 30 prompts; Base16/R8, Query-Gram Q64; sink32 + last two physical Page32s + 61 historical pages.\n\n'
    text += '| Task | Full K / V96 | Sparse K / V96 |\n|---|---:|---:|\n'
    text += ''.join(f'| {task} | {score["full"]:.4f} | {score["sparse"]:.4f} |\n' for task,score in scores.items())
    text += f'| Mean | {means["full"]:.4f} | {means["sparse"]:.4f} |\n\n'
    text += 'Environment: basis. Commands: scripts/run_llama31i_ruler_64k.sh. All attention uses repository Triton kernels.\n'
    (args.run/'ruler/summary.md').write_text(text)
    print('verified 330 paired prompts',means,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('smoke','evaluate','summarize'))
    parser.add_argument('--model',type=Path,default=MODEL)
    parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--shard',type=int,default=0)
    args = parser.parse_args()
    assert 0 <= args.shard < 2
    configure()
    tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    c1,bank,rows,protocol = inputs(args,tokenizer)
    if args.stage == 'summarize':
        summarize(args,rows,protocol,tokenizer)
        return
    model = load_model(args,c1,bank)
    selected = [rows[0],rows[240]] if args.stage == 'smoke' else rows[args.shard::2]
    for row in selected:
        path = args.run/'ruler'/args.stage/f"sample_{row['index']:03d}.json"
        if path.exists():
            saved = read_json(path)
            assert saved['status'] == 'complete' and saved['protocol'] == protocol
            assert saved['sample'] == sample_metadata(row)
            continue
        started = time.monotonic()
        for device in range(2):
            torch.cuda.reset_peak_memory_stats(device)
        cap = min(4,row['maximum_tokens']) if args.stage == 'smoke' else row['maximum_tokens']
        arms,audit = generate_pair(model,tokenizer,row,cap)
        write_json(path,dict(status='complete',protocol=protocol,sample=sample_metadata(row),
            arms=arms,kernel_audit=audit,seconds=time.monotonic()-started,
            peak_gpu_bytes=[torch.cuda.max_memory_allocated(i) for i in range(2)],
            command=shlex.join(sys.argv),python=sys.executable))
        print(row['index'],row['task'],{k:r['score'] for k,r in arms.items()},
              f'seconds={time.monotonic()-started:.1f}',flush=True)


if __name__ == '__main__':
    main()
