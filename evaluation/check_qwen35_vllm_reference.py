"""Compare real vLLM smoke continuations with the existing HF hybrid runtime."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import torch

from basisserve.core.qwen35_gated_v_runtime import GatedVRuntime
from basisserve.core.qwen35_hybrid_output_runtime import HybridOutputRuntime
from evaluation.qwen35_hybrid_common import atomic_save, load_bank, load_model, sha256, verify_model_identity


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert not Path(args.output).exists()
    torch.set_num_threads(2)
    smoke = json.loads(Path(args.smoke).read_text())
    original = smoke['args']
    verify_model_identity(original['model_path'], smoke['provenance']['model_identity'])
    bank = load_bank(original['bank']) if original['bank'] else None
    if bank:
        assert sha256(original['bank']) == smoke['provenance']['v_bank_sha256']
    if original['wo_bank']:
        assert sha256(original['wo_bank']) == smoke['provenance']['wo_bank_sha256']
    model = load_model(original['model_path'], 'cuda:0')
    context = (HybridOutputRuntime(model, bank['layers'], original['wo_bank']) if original['wo_bank']
               else GatedVRuntime(model, bank['layers']) if bank else nullcontext())
    rows = []
    with context:
        for record in smoke['smoke_records']:
            prompt, generated = record['prompt_token_ids'], record['token_ids']
            assert generated
            ids = torch.tensor([prompt + generated[:-1]], device='cuda:0')
            hidden = model.model(ids, use_cache=False).last_hidden_state
            logits = model.lm_head(hidden[:, len(prompt)-1:]).float()[0]
            target = torch.tensor(generated, device=logits.device)
            logprobs = logits.log_softmax(-1).gather(1, target[:, None]).flatten()
            actual = torch.tensor(record['chosen_logprobs'], device=logits.device)
            difference = (logprobs - actual).abs()
            rows.append({'tokens': len(generated), 'top1_agreement': float((logits.argmax(-1) == target).float().mean()),
                         'mean_abs_chosen_logprob_difference': float(difference.mean()),
                         'max_abs_chosen_logprob_difference': float(difference.max()),
                         'hf_chosen_logprobs': logprobs.cpu().tolist(),
                         'vllm_chosen_logprobs': record['chosen_logprobs']})
            print(json.dumps({k: v for k, v in rows[-1].items() if not isinstance(v, list)}), flush=True)
    atomic_save(args.output, {'command': sys.argv, 'environment': 'lowrank',
                             'smoke_sha256': sha256(args.smoke), 'rows': rows,
                             'scope': 'teacher-forced HF reference on vLLM-generated prefixes; assess against dense backend difference'})


if __name__ == '__main__':
    main()
