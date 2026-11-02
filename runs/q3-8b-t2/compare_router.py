"""Compare the refit Page4 B16R16 router with the published one.

Residual codes live in an arbitrary 16-dim basis, so compare the basis-invariant per-head score map
residual_query[h] @ residual_encoder[g(h)]^T (128 x 128 in dense K coordinates), and the fit page-Fisher NMSE per layer.
"""
import json
from pathlib import Path
import torch
from safetensors import safe_open

R = Path("/zpool-00/home/lz299/BasisServe-CALS/runs/q3-8b-t2")
HF = R / "hf/checkpoints/attention_c1/qwen3_8b_post_uniform_v96_128k_als6_retrievalmix/router_page4_b16r16"


def load(path):
    with safe_open(str(path), "pt") as handle:
        return {key: handle.get_tensor(key).float() for key in handle.keys()}, json.loads(handle.metadata().get("protocol", "{}")) if handle.metadata() else {}


def score_map(encoder, query):  # [groups,128,16], [heads,128,16] -> [heads,128,128]
    heads, groups = query.shape[0], encoder.shape[0]
    return torch.stack([query[h] @ encoder[h // (heads // groups)].T for h in range(heads)])


rel, nmse_ours, nmse_paper = [], [], []
for layer in range(36):
    ours, _ = load(R / f"refit/router/layer_{layer:03d}.safetensors")
    paper, _ = load(HF / f"layer_{layer:03d}.safetensors")
    if layer == 0:
        print("refit router keys:", sorted(ours))
    a = score_map(ours["residual_encoder_b16_r16"], ours["residual_query_b16_r16"])
    b = score_map(paper["residual_encoder_b16_r16"], paper["residual_query_b16_r16"])
    rel.append(((a - b).norm() / b.norm()).item())
    record = json.load(open(R / f"refit/router/layer_{layer:03d}.json")) if (R / f"refit/router/layer_{layer:03d}.json").exists() else {}
    paper_record = json.load(open(HF / f"layer_{layer:03d}.json"))
    nmse_paper.append(paper_record["losses"]["b16_r16"]["fit_page_fisher_nmse"])
    losses = record.get("losses", {})
    nmse_ours.append(losses.get("fit_page_fisher_nmse", float("nan")))
print("residual score map (Q_res · E_res^T) rel. Frobenius diff: min %.4f median %.4f max %.4f" % (min(rel), sorted(rel)[18], max(rel)))
print("per layer:", [round(x, 3) for x in rel])
print("fit page-Fisher NMSE  ours :", [round(x, 3) for x in nmse_ours])
print("fit page-Fisher NMSE  paper:", [round(x, 3) for x in nmse_paper])
