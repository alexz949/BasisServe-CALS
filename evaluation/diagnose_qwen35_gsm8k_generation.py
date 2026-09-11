"""Check actual long GSM8K prompts in the pre-existing HF compressed runtime."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer

from basisserve.core.qwen35_gated_v_runtime import GatedVRuntime
from basisserve.core.qwen35_hybrid_output_runtime import HybridOutputRuntime
from evaluation.qwen35_hybrid_common import atomic_save, load_bank, load_model, sha256, verify_model_identity


def aligned_stop_settings(model_eos_id, tokenizer_eos_id, tokenizer_eos_text, task_stops):
    """Match the native EOS plus task/chat stops used by the vLLM runner."""
    return {'eos_token_id': list(dict.fromkeys([model_eos_id, tokenizer_eos_id])),
            'stop_strings': list(dict.fromkeys([*task_stops, tokenizer_eos_text]))}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--doc-ids', default='0,2')
    p.add_argument('--max-new-tokens', type=int, default=128)
    a = p.parse_args()
    assert not Path(a.output).exists()
    torch.set_num_threads(2)
    pilot = json.loads(Path(a.pilot).read_text()); args = pilot['args']
    verify_model_identity(args['model_path'], pilot['provenance']['model_identity'])
    bank = load_bank(args['bank']) if args['bank'] else None
    if bank:
        assert sha256(args['bank']) == pilot['provenance']['v_bank_sha256']
    if args['wo_bank']:
        assert sha256(args['wo_bank']) == pilot['provenance']['wo_bank_sha256']
    tokenizer = AutoTokenizer.from_pretrained(args['model_path'], local_files_only=True)
    model = load_model(args['model_path'], 'cuda:0')
    context = (HybridOutputRuntime(model, bank['layers'], args['wo_bank']) if args['wo_bank']
               else GatedVRuntime(model, bank['layers']) if bank else nullcontext())
    samples = {r['doc_id']: r for r in pilot['evaluation']['samples']['gsm8k']}
    rows = []
    with context:
        for i in map(int, a.doc_ids.split(',')):
            row = samples[i]; prompt = row['arguments'][0][0]
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            vllm_ids = tokenizer.encode(row['resps'][0][0], add_special_tokens=False)[:a.max_new_tokens]
            assert vllm_ids
            full = torch.tensor([ids + vllm_ids[:-1]], device='cuda:0')
            hidden = model.model(full, use_cache=False).last_hidden_state[:, len(ids)-1:]
            logits = model.lm_head(hidden).float()[0]
            chosen = torch.tensor(vllm_ids, device='cuda:0')
            agreement = float((logits.argmax(-1) == chosen).float().mean())
            del hidden, logits, full
            inputs = torch.tensor([ids], device='cuda:0')
            stops = aligned_stop_settings(model.generation_config.eos_token_id,
                                          tokenizer.eos_token_id, tokenizer.eos_token,
                                          row['arguments'][0][1]['until'])
            generated = model.generate(inputs, attention_mask=torch.ones_like(inputs),
                                       do_sample=False, max_new_tokens=a.max_new_tokens,
                                       use_cache=True, pad_token_id=tokenizer.eos_token_id,
                                       tokenizer=tokenizer, **stops)
            output = tokenizer.decode(generated[0, len(ids):], skip_special_tokens=False)
            rows.append({'doc_id': i, 'prompt_tokens': len(ids), 'teacher_forced_top1_agreement': agreement,
                         'compared_tokens': len(vllm_ids), 'hf_cached_generation': output,
                         'stop_settings': stops,
                         'vllm_generation_prefix': tokenizer.decode(vllm_ids, skip_special_tokens=False)})
            print(json.dumps(rows[-1]), flush=True)
    atomic_save(a.output, {'command': sys.argv, 'environment': 'lowrank',
                          'pilot_sha256': sha256(a.pilot), 'rows': rows})


if __name__ == '__main__':
    main()
