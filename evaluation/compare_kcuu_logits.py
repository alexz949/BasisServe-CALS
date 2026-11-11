"""Compare first-token full-vocabulary logits from identical real-model inputs."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from basisserve.core.qwen3_8b_vllm_c1 import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ag', type=Path, required=True)
    parser.add_argument('--hybrid', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ag, hybrid = [json.loads(p.read_text()) for p in (args.ag, args.hybrid)]
    assert ag['status'] == hybrid['status'] == 'passed'
    assert ag['mode'] == 'ag' and hybrid['mode'] == 'hybrid'
    for key in ('source_sha256', 'model_revision', 'model_config_sha256', 'factors_sha256', 'prompt_sha256'):
        assert ag[key] == hybrid[key]
    tensors = []
    for result in (ag, hybrid):
        path = Path(result['logits_file'])
        assert file_sha256(path) == result['logits_sha256']
        tensor = load_file(str(path), device='cpu')['logits'].float()
        assert tensor.shape == (1, result['vocab_size']) and torch.isfinite(tensor).all()
        tensors.append(tensor)
    delta = tensors[0] - tensors[1]
    relative = (delta.norm() / tensors[0].norm()).item()
    result = dict(status='passed' if relative < 0.02 else 'failed', relative_l2=relative,
                  max_absolute_error=delta.abs().max().item(), finite=True,
                  top1_equal=bool(torch.equal(tensors[0].argmax(-1), tensors[1].argmax(-1))),
                  tolerance=0.02, scope='First-token logits on the same 17-token prompt; not a quality evaluation',
                  source_sha256=ag['source_sha256'])
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    assert relative < 0.02, result
    print(json.dumps(result))


if __name__ == '__main__':
    main()
