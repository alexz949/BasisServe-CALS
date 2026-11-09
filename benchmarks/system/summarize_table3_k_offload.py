"""Tables for the Table 3 Key-offload arms: per-run CSV, cohort medians, state inventory."""

import argparse
import csv
import json
from pathlib import Path
import statistics

GIB = 2 ** 30
ARMS = ["dense_resident", "basis_resident", "basis_resident_nobase", "basis_k_offload_nobase"]
METRICS = [
    ("attention_block_ms", "Attention block ms"),
    ("full_wall_ms", "Full-model wall ms"),
    ("full_cuda_ms", "Full-model CUDA ms"),
    ("decode_resident_gib", "Decode resident GiB"),
    ("decode_peak_allocated_gib", "Decode peak allocated GiB"),
    ("decode_peak_reserved_gib", "Decode peak reserved GiB"),
    ("host_pinned_key_gib", "Host pinned-Key GiB"),
]
PHASES = [
    ("model_loaded_gib", "Model and preallocated caches"),
    ("prefill_peak_allocated_gib", "Prefill peak allocated"),
    ("post_prefill_gib", "After prefill and placement"),
    ("decode_resident_gib", "Steady decode start"),
    ("decode_peak_allocated_gib", "Steady decode peak allocated"),
    ("decode_end_gib", "Steady decode end"),
    ("overall_peak_allocated_gib", "Overall peak allocated"),
    ("overall_peak_reserved_gib", "Overall peak reserved"),
    ("decode_peak_reserved_gib", "Decode peak reserved, allocator cache released after prefill"),
    ("decode_device_used_gib", "Device used in decode, allocator cache released"),
]


def mean(rows, key):
    return statistics.fmean(row[key] for row in rows)


def arm_name(result):
    """Arms on the page-size-general router carry their page size as a suffix."""
    general = "general" in result["configuration"].get("router", "")
    return result["arm"] + (f"_p{result['page_size']}" if general else "")


def arm_order(name):
    """Archived arms first, then each page-size pair with the larger page first."""
    base = max((arm for arm in ARMS if name.startswith(arm)), key=len)
    page_size = int(name[len(base) + 2:] or 0)
    return bool(page_size), -page_size, ARMS.index(base)


def run_row(result):
    measured, profile, inventory = result["measurement"], result["attention_profile"], result["inventory"]
    memory, released = measured["memory"], profile["memory"]
    host_keys = sum(row["backing_bytes"] for row in inventory["rows"]
                    if row["device"] == "host" and row["name"] == "host_key")
    row = dict(arm=arm_name(result), context=result["context_tokens"], cohort=result["cohort"],
        status=result["status"],
        attention_block_ms=mean(profile["components"], "attention_block"),
        full_wall_ms=measured["wall_median_ms"], full_cuda_ms=measured["cuda_median_ms"],
        model_loaded_gib=result["memory_model_loaded"]["allocated"] / GIB,
        prefill_peak_allocated_gib=memory["prefill"]["peak_allocated"] / GIB,
        post_prefill_gib=memory["prefill"]["allocated"] / GIB,
        decode_resident_gib=memory["conditioning"]["allocated"] / GIB,
        decode_peak_allocated_gib=memory["decode"]["peak_allocated"] / GIB,
        decode_end_gib=memory["decode"]["allocated"] / GIB,
        overall_peak_allocated_gib=max(v["peak_allocated"] for v in memory.values()) / GIB,
        overall_peak_reserved_gib=max(v["peak_reserved"] for v in memory.values()) / GIB,
        decode_peak_reserved_gib=released["decode"]["peak_reserved"] / GIB,
        decode_device_used_gib=released["decode"]["device_used"] / GIB,
        host_pinned_key_gib=host_keys / GIB,
        prefill_and_placement_s=measured["prefill_and_placement_seconds"],
        gpu=result["environment"]["gpu"], torch=result["environment"]["torch"])
    for name in profile["components"][0]:
        row[name + "_ms"] = mean(profile["components"], name)
    if profile["slot_steps"]:
        row["selected_keys_per_step"] = mean(profile["slot_steps"], "selected")
        row["fetched_keys_per_step"] = mean(profile["slot_steps"], "fetched")
        row["fetched_mib_per_step"] = mean(profile["slot_steps"], "fetched_bytes") / 2 ** 20
    return row


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows])


def number(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=Path, required=True,
                        help="arms/<phase> directory written by run_table3_k_offload.py")
    args = parser.parse_args()
    results = [json.loads(path.read_text()) for path in sorted(args.phase.glob("*/result.json"))]
    results = sorted((r for r in results if r["mode"] == "steady"),
                     key=lambda r: (arm_order(arm_name(r)), r["cohort"]))
    runs = [run_row(result) for result in results]
    arms = sorted({row["arm"] for row in runs}, key=arm_order)
    columns = sorted({key for row in runs for key in row}, key=lambda key: (
        key not in ("arm", "context", "cohort", "status"), key))
    with (args.phase / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(runs)

    contexts = sorted({row["context"] for row in runs})
    numeric = [key for key in columns if all(isinstance(row.get(key, 0.0), float) for row in runs)]
    medians = {}
    for context in contexts:
        for arm in arms:
            cohort = [row for row in runs if row["arm"] == arm and row["context"] == context
                      and row["status"] == "complete"]
            if cohort:
                medians[arm, context] = dict(arm=arm, context=context, cohorts=len(cohort), **{
                    key: statistics.median(row[key] for row in cohort)
                    for key in numeric if all(key in row for row in cohort)})
    with (args.phase / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["arm", "context", "cohorts", *numeric])
        writer.writeheader()
        writer.writerows(medians.values())

    inventory_rows = []
    for result in results:
        if result["cohort"] == min(r["cohort"] for r in results):
            for row in result["inventory"]["rows"]:
                inventory_rows.append(dict(arm=arm_name(result), context=result["context_tokens"], **row))
    with (args.phase / "inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory_rows[0]))
        writer.writeheader()
        writer.writerows(inventory_rows)

    text = ["# Table 3 matched Key-offload arms", ""]
    first = results[0]
    text += [f"Llama-3.1-8B-Instruct BF16, TP1/B1, dense V128, B16R16 full scan, 1984 routed plus 64 "
             f"recent tokens, "
             f"{first['environment']['gpu'].split(',')[1].strip()}, torch {first['environment']['torch']}, "
             f"conda env `{first['environment']['conda_environment']}`. "
             f"{first['conditioning_steps']} conditioning and {first['measured_steps']} measured decode "
             f"steps after a full prefill; prefill is excluded from decode timing.", "",
             "Arms: `dense_resident` and `basis_resident` are the archived Table 3 runtimes. "
             "`basis_resident_nobase` reads Base from the first 16 Value coordinates instead of a "
             "separate Base cache. `basis_k_offload_nobase` additionally keeps exact Keys in mapped "
             "pinned host memory and fetches selected Keys into persistent GPU slots. Arms without a "
             "suffix use the archived Page32 router; `_pN` arms use the page-size-general build of "
             "the same router with N-token pages.", ""]
    for context in contexts:
        present = [arm for arm in arms if (arm, context) in medians]
        text += [f"## {context} tokens: median over prompt cohorts", "",
                 table(["Arm", "Cohorts", *[label for _, label in METRICS]],
                       [[f"`{arm}`", medians[arm, context]["cohorts"],
                         *[number(medians[arm, context][key], 3 if key.endswith("ms") else 2)
                           for key, _ in METRICS]] for arm in present]), ""]
        dense = medians.get(("dense_resident", context))
        if dense:
            text += ["Speedup over `dense_resident` with the same timing definition:", "",
                     table(["Arm", "Attention block", "Full-model wall", "Full-model CUDA"],
                           [[f"`{arm}`", *[number(dense[key] / medians[arm, context][key])
                                           for key in ("attention_block_ms", "full_wall_ms", "full_cuda_ms")]]
                            for arm in present if arm != "dense_resident"]), ""]
        for control in [arm for arm in present if arm.startswith("basis_resident_nobase")]:
            resident = medians[control, context]
            candidate = control.replace("basis_resident", "basis_k_offload")
            offload = medians.get((candidate, context))
            if not offload:
                continue
            saved = resident["decode_resident_gib"] - offload["decode_resident_gib"]
            text += [f"`{candidate}` against its matched resident control (`{control}`):", "",
                     f"- Full-model wall latency: {number(offload['full_wall_ms'] / resident['full_wall_ms'], 3)}x "
                     f"({number(resident['full_wall_ms'], 3)} to {number(offload['full_wall_ms'], 3)} ms).",
                     f"- Attention block: {number(offload['attention_block_ms'] / resident['attention_block_ms'], 3)}x.",
                     f"- GPU decode resident memory: {number(resident['decode_resident_gib'])} to "
                     f"{number(offload['decode_resident_gib'])} GiB, {number(saved)} GiB less "
                     f"({number(100 * saved / resident['decode_resident_gib'], 1)}%).",
                     f"- The same {number(offload['host_pinned_key_gib'])} GiB of exact Keys now sits in pinned "
                     "host memory; total system memory is not reduced.", ""]
            if "fetched_keys_per_step" in offload:
                text += [f"- Key slots: {number(offload['selected_keys_per_step'], 0)} selected Keys per step "
                         f"over all layers, {number(offload['fetched_keys_per_step'], 0)} fetched from host "
                         f"({number(offload['fetched_mib_per_step'], 3)} MiB per step).", ""]
        text += ["Memory by phase, GiB (PyTorch allocated unless stated):", "",
                 table(["Phase", *[f"`{arm}`" for arm in present]],
                       [[label, *[number(medians[arm, context][key]) for arm in present]]
                        for key, label in PHASES]), ""]
        text += ["Attention-block components, ms per step summed over 32 layers (separate profiled run):", ""]
        names = [key for key in numeric if key.endswith("_ms") and key not in (
            "full_wall_ms", "full_cuda_ms")]
        text += [table(["Arm", *[name[:-3] for name in names]],
                       [[f"`{arm}`", *[number(medians[arm, context].get(name), 3) for name in names]]
                        for arm in present]), ""]
        text += ["Per cohort:", "",
                 table(["Arm", "Cohort", "Status", "Attention block ms", "Full wall ms", "Full CUDA ms",
                        "Decode resident GiB"],
                       [[f"`{row['arm']}`", row["cohort"], row["status"], number(row["attention_block_ms"], 3),
                         number(row["full_wall_ms"], 3), number(row["full_cuda_ms"], 3),
                         number(row["decode_resident_gib"])]
                        for row in runs if row["context"] == context]), ""]

    text += ["## State-level bytes", "",
             "Tensor inventory after the measured decode, lowest cohort, counted once per backing "
             "storage. Caches are preallocated for prompt plus generated tokens; active bytes scale "
             "the token-indexed states by the tokens actually stored.", ""]
    for result in results:
        if result["cohort"] != min(r["cohort"] for r in results):
            continue
        inventory = result["inventory"]
        groups = {}
        for row in inventory["rows"]:
            key = (row["device"], row["category"])
            group = groups.setdefault(key, dict(backing=0, active=0, names=[]))
            group["backing"] += row["backing_bytes"]
            group["active"] += row["active_bytes"]
            if row["name"] not in group["names"]:
                group["names"].append(row["name"] + (" (view)" if row["aliases"] and row[
                    "aliases"] != row["name"] else ""))
        ordered = sorted(groups.items(), key=lambda item: (item[0][0], -item[1]["backing"]))
        tracked = inventory["gpu_backing_bytes"] + inventory["unowned_gpu_bytes"]
        text += [f"### `{arm_name(result)}`, {result['context_tokens']} tokens "
                 f"({inventory['active_tokens']} stored of {inventory['capacity_tokens']} capacity)", "",
                 table(["Device", "State", "Tensors", "Allocated GiB", "Active GiB"],
                       [[device, category, ", ".join(group["names"]), number(group["backing"] / GIB, 4),
                         number(group["active"] / GIB, 4)] for (device, category), group in ordered]), "",
                 f"GPU tensors in this inventory: {number(inventory['gpu_backing_bytes'] / GIB, 3)} GiB; "
                 f"other live CUDA tensors: {number(inventory['unowned_gpu_bytes'] / GIB, 3)} GiB; "
                 f"PyTorch allocated at the same moment: {number(inventory['torch_allocated_bytes'] / GIB, 3)} GiB "
                 f"(difference {number((inventory['torch_allocated_bytes'] - tracked) / 2 ** 20, 1)} MiB, "
                 "allocator block rounding).", ""]
    check = args.phase / "placement_check.json"
    if check.exists():
        text += ["## Placement equivalence (teacher-forced)", "",
                 table(["Reference", "Candidate", "Steps", "Identical support", "Bitwise selected Keys",
                        "Caches identical", "Max attention diff", "Max logit diff", "Argmax agreement"],
                       [[f"`{row['reference']}`", f"`{row['candidate']}`", row["forced_steps"],
                         row["steps_with_identical_support"], row["steps_with_bitwise_identical_selected_keys"],
                         ", ".join(f"{name} {'yes' if same else 'no'}" for name, same in row["cache_identical"].items()),
                         number(row["attention_max_abs_difference"], 5), number(row["logits_max_abs_difference"], 5),
                         number(row["argmax_agreement"], 3)] for row in json.loads(check.read_text())]), ""]
    (args.phase / "SUMMARY.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
