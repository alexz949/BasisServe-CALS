"""Export real decoder weights and train-only latent fixtures, without hashes."""

import argparse
from pathlib import Path
import sys
import gc
import shlex

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from evaluation.eval_qwen3_latent_a8 import load_fixed_model
from evaluation.eval_qwen3_kv4_fp8_ppl import write
from scripts.eval_svdllm_safetensors_ppl_accelerate import _token_ids


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/q3-a8-tp8")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prior = ROOT / "results/q3-kv4-fp8/formal"
    for rank in (64, 96):
        destination = args.output / f"r{rank}.pt"
        assert not destination.exists()
        model, tokenizer, projections, decoders, handles, schedule, protocol = load_fixed_model(prior, rank)
        layers, captured = [0, 18, 35], {}

        def capture(i, args):
            captured[i] = args[0].reshape(-1, 4096).detach().cpu()

        observers = [decoders[f"{i}.decoder"].register_forward_pre_hook(
            lambda m, x, i=i: capture(i, x)) for i in layers]
        train = _token_ids(tokenizer, "wikitext2", "train", None).reshape(-1)
        start = protocol["calibration_starts"][0]
        model.model(input_ids=train[start:start+256][None].cuda(), use_cache=False)
        payload = {}
        for i in layers:
            active = torch.tensor([q*128+j for q in range(32) for j in range(schedule[i][q//4])])
            decoder = decoders[f"{i}.decoder"]
            payload[i] = dict(latent=captured[i].index_select(1, active).contiguous(),
                decoder=decoder.weight.cpu().index_select(1, active).T.contiguous(),
                scale=decoder.input_scale.cpu(), widths=[4*r for r in schedule[i]])
            assert payload[i]["decoder"].shape == (sum(payload[i]["widths"]), 4096)
        torch.save(dict(rank=rank, layers=payload, protocol=protocol,
            sample_start=start, sample_length=256, source="WT2 train, fixed KV4, BF16 encoder/decoder"), destination)
        write(args.output / f"r{rank}_manifest.json", dict(rank=rank, command=shlex.join(sys.argv),
            widths={i:p["widths"] for i,p in payload.items()}, sample_start=start,
            sample_length=256, source="WT2 train", calibration="previous frozen per-layer scales", hashes=False))
        print("EXPORTED", destination, flush=True)
        for handle in handles + observers:
            handle.remove()
        del model, tokenizer, projections, decoders, captured, payload, decoder
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
