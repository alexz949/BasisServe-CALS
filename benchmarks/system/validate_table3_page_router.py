"""Check the page-size-general Table 3 router against the archived kernels and PyTorch.

Page scores: a Page1 build yields the kernel's own token scores, so every other page size must
equal their per-page log-sum-exp, and Page32 must equal the archived kernel bit for bit.
Selection: compared with a PyTorch top-k of the same GQA-max rule and with the archived selector.
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import torch

from table3_page_router import build

PAGE_SIZES = (4, 8, 16, 32)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def page_scores(module, tensors, length, page_size):
    scores = torch.empty(1, 8, 4, math.ceil(length / page_size), device="cuda")
    code = torch.empty(1, 8, 4, 16, device="cuda", dtype=torch.bfloat16)
    module.conditional_router_page_lse(*tensors, code, scores, 128 ** -.5, False)
    return scores


def select(module, mass, historical, page_size):
    selected = torch.empty(1, 8, 1984 // page_size, device="cuda", dtype=torch.long)
    support = torch.empty(1, 8, 2048, device="cuda", dtype=torch.long)
    module.select_full_group_max_pages_and_pack(mass, selected, support, historical, historical + 64)
    return selected, support


def reference_selection(mass, page_size):
    pinned = 32 // page_size
    group = torch.softmax(mass[..., pinned:], dim=-1).amax(dim=2)
    top = group.topk(1984 // page_size - pinned, dim=-1).indices + pinned
    fixed = torch.arange(pinned, device="cuda").expand(1, 8, -1)
    return torch.cat((fixed, top), dim=-1).sort(dim=-1).values, group


def milliseconds(function, iterations=200):
    for _ in range(50):
        function()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - start) / iterations


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    sys.path.insert(0, str(runtime))
    config = load_module("router_configuration", runtime / "benchmarks/system/bench_tp1_router_config.py")
    folder = runtime / "results/system_benchmarks/table3_page_router"
    torch.manual_seed(0)
    archived = config.build("w16u1")
    modules, builds = {}, {}
    for page_size in (1, *PAGE_SIZES):
        modules[page_size], builds[page_size] = build(config.SNAPSHOT, folder / f"p{page_size}", page_size,
                                                      value_dim=128, routed_tokens=1984, sink_tokens=32)

    scores = []
    for length in (1000, 4113, 24518):
        tensors = config.inputs(length)
        token = page_scores(modules[1], tensors, length, 1)
        for page_size in PAGE_SIZES:
            pages = math.ceil(length / page_size)
            padded = torch.full((1, 8, 4, pages * page_size), -torch.inf, device="cuda")
            padded[..., :length] = token
            expected = torch.logsumexp(padded.view(1, 8, 4, pages, page_size), dim=-1)
            actual = page_scores(modules[page_size], tensors, length, page_size)
            scores.append(dict(length=length, page_size=page_size,
                max_abs_difference_from_token_scores=float((actual - expected).abs().max()),
                bitwise_equal_to_archived=bool(torch.equal(
                    actual, page_scores(archived, tensors, length, 32))) if page_size == 32 else None))

    selections = []
    # 2100 tokens leaves fewer page groups than selections; "clustered" puts the mass in one span.
    for pattern, historical in (("random", 2100), ("random", 24518), ("random", 130129),
                                ("clustered", 130129)):
        for page_size in PAGE_SIZES:
            pages = math.ceil(historical / page_size)
            mass = 3 * torch.randn(1, 8, 4, pages, device="cuda")
            if pattern == "clustered":
                mass[..., pages // 3:pages // 3 + 3 * (1984 // page_size)] += 12
            # The general selector overwrites its input, so it always gets a copy.
            selected, support = select(modules[page_size], mass.clone(), historical, page_size)
            expected, group = reference_selection(mass, page_size)
            missed = (expected[..., None] != selected[..., None, :]).all(dim=-1)
            floor = torch.gather(group, 2, expected[..., 32 // page_size:] - 32 // page_size).amin(-1)
            chosen = torch.gather(group, 2, selected[..., 32 // page_size:] - 32 // page_size).amin(-1)
            tokens = (selected[..., None] * page_size + torch.arange(page_size, device="cuda")).flatten(-2)
            tokens = tokens.masked_fill(tokens >= historical, -1)
            recent = torch.arange(historical, historical + 64, device="cuda").expand(1, 8, -1)
            ties, _ = select(modules[page_size], torch.zeros_like(mass), historical, page_size)
            row = dict(pattern=pattern, historical=historical, page_size=page_size, pages=pages,
                selected_pages=selected.shape[-1], pages_missing_from_reference_top_k=int(missed.sum()),
                relative_score_gap_of_last_selected=float(((floor - chosen) / floor).abs().max()),
                ascending=bool((selected[..., 1:] > selected[..., :-1]).all()),
                support_matches_selected_pages=bool(torch.equal(support, torch.cat((tokens, recent), -1))),
                ties_take_lowest_pages=bool(torch.equal(
                    ties, torch.arange(selected.shape[-1], device="cuda").expand(1, 8, -1))))
            if page_size == 32:
                row["equal_to_archived_selector"] = bool(torch.equal(
                    selected, select(archived, mass, historical, 32)[0]))
            selections.append(row)

    timings = []
    length = 130129
    tensors = config.inputs(length)
    for name, module, page_size in (("archived", archived, 32), *(
            ("general", modules[p], p) for p in PAGE_SIZES)):
        mass = page_scores(module, tensors, length, page_size)
        scratch = mass.clone()
        code = torch.empty(1, 8, 4, 16, device="cuda", dtype=torch.bfloat16)
        selected, support = select(module, scratch, length, page_size)

        def copy_and_select():
            scratch.copy_(mass)
            module.select_full_group_max_pages_and_pack(scratch, selected, support, length, length + 64)

        timings.append(dict(router=name, page_size=page_size, historical=length,
            page_scores_ms=milliseconds(lambda: module.conditional_router_page_lse(
                *tensors, code, scratch, 128 ** -.5, False)),
            selection_ms=milliseconds(copy_and_select) - milliseconds(lambda: scratch.copy_(mass))))

    passed = (all(b["anchors_found"] == b["anchors_expected"] for b in builds.values())
        and all(row["max_abs_difference_from_token_scores"] < 1e-3
                and row["bitwise_equal_to_archived"] is not False for row in scores)
        and all(row["relative_score_gap_of_last_selected"] < 1e-5 and row["ascending"]
                and row["support_matches_selected_pages"] and row["ties_take_lowest_pages"]
                and row.get("equal_to_archived_selector", True) for row in selections))
    report = dict(status="passed" if passed else "failed", gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, builds=builds, page_scores=scores, selection=selections,
        per_layer_call_timing=timings,
        timing_scope="Wall time of one call for one layer (8 KV heads) into preallocated outputs; "
                     "selection time excludes restoring its input")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
