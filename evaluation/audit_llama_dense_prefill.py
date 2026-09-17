"""Compare native single-GPU Full-K prefill with routing and archived TP2 tokens."""
import hashlib
import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from evaluation import eval_k_routing_ruler as runtime
from evaluation.deterministic_evaluation import configure_deterministic_evaluation
from evaluation.eval_llama_dense_v_routing128 import install_prefill_cache_offload, instrument_kernels
from evaluation.k_routing_config import routing_config
from evaluation.llama_prefill_memory import install
from evaluation.v96kl_common import read_json, save_tensors, sha256, write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--identity', type=Path, required=True)
    args = parser.parse_args()
    configure_deterministic_evaluation()
    assert torch.cuda.device_count() == 1
    evaluation, reference, output = args.evaluation, args.reference, args.output
    identity_path = args.identity
    identity = read_json(identity_path)
    prompts = read_json(evaluation/'prompts.json')
    assert prompts['identity_sha256'] == sha256(identity_path)
    assert prompts['tokens_sha256'] == sha256(evaluation/'prompts.safetensors')
    tokens = load_file(str(evaluation/'prompts.safetensors'))
    samples_per_task = sum(row['task'] == 'niah_single_1' for row in prompts['rows'])
    assert len(prompts['rows']) == samples_per_task*11
    selected = {0, 7*samples_per_task}
    comparisons = {}
    for row in prompts['rows']:
        index = row['index']
        name = f'sample_{index:03d}.json'
        ours, dense = read_json(evaluation/'ours/evaluate'/name), read_json(reference/name)
        assert ours['status'] == dense['status'] == 'complete'
        assert ours['sample'] == dense['sample'] == row
        assert ours['result']['cache_value_head_dim'] == 128
        comparisons[index] = (ours['first_argmax'], dense['first_argmax'])
        if ours['first_argmax'] != dense['first_argmax']:
            selected.add(index)
    config = routing_config(identity, rope='native', sequence_length=131072)
    model = runtime.load_evaluation_model(identity, config)
    # Use the original attention projections directly, without a routing adapter.
    for layer in model.model.layers:
        attention = layer.self_attn
        attention.num_attention_heads = config.num_attention_heads
        attention.num_key_value_heads = config.num_key_value_heads
        attention.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        attention.value_head_dim = attention.head_dim
        attention.q_norm = torch.nn.Identity()
        attention.k_norm = torch.nn.Identity()
        attention._routing_arm = 'full'
        assert attention.head_dim == 128
    install(model)
    install_prefill_cache_offload(model)
    counts = instrument_kernels()
    tokenizer = AutoTokenizer.from_pretrained(identity['model'], local_files_only=True)
    protocol = dict(identity_sha256=sha256(identity_path),
                    prompts_sha256=sha256(evaluation/'prompts.json'),
                    source_sha256=sha256(Path(__file__)),
                    value_mode='original V128 and Wo; native single-GPU Full-K prefill',
                    selected=sorted(selected), generation_cap=1)
    for index in sorted(selected):
        row = prompts['rows'][index]
        assert row['index'] == index
        tensor = tokens[str(index)]
        assert hashlib.sha256(tensor.numpy().tobytes()).hexdigest() == row['input_sha256']
        print('START native Full-K prefill', index, flush=True)
        counts.clear()
        with torch.inference_mode():
            ids, first, _, _ = runtime.generate(
                model, tokenizer, dict(row, input_ids=tensor.tolist()), 'full', 1)
        assert dict(counts) == {'prefill': 32}
        ours, dense = comparisons[index]
        top = first.topk(10)
        path = output/f'sample_{index:03d}.safetensors'
        save_tensors(path, {'first_logits': first})
        record = dict(protocol=protocol, sample=row, first_argmax=ids[0],
                      ours_first_argmax=ours, tp2_first_argmax=dense,
                      matches_ours=ids[0] == ours, matches_tp2=ids[0] == dense,
                      ours_logit=float(first[ours]), tp2_token_logit=float(first[dense]),
                      top_ids=top.indices.tolist(), top_logits=top.values.tolist(),
                      logits_sha256=sha256(path), kernel_calls=dict(counts))
        write_json(output/f'sample_{index:03d}.json', record)
        print('CHECK', index, 'native', ids[0], 'ours', ours, 'TP2', dense, flush=True)
        assert record['matches_ours'], record
    write_json(output/'summary.json', dict(status='complete', protocol=protocol,
               checked=len(selected), mismatching_tp2=sum(a != b for a, b in comparisons.values())))


if __name__ == '__main__':
    main()
