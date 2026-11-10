#!/usr/bin/env bash
# Interconnect acceptance check for a 4-GPU TP4 benchmark node. About 3 minutes; the GPUs must be idle.
# Usage: bash benchmarks/system/check_interconnect.sh [output.log]
#   CUDA_VISIBLE_DEVICES  GPUs to test (default 0,1,2,3)
#   PYTHON                interpreter with torch+CUDA (default: the basis env if BASIS_ARTIFACTS is set, else python3)
# Prints PCIe topology, copy bandwidth, NCCL all-reduce with P2P on and off, then a PASS/FAIL verdict.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
out=${1:-results/interconnect_$(hostname)_$(date -u +%Y%m%dT%H%M%SZ).log}
mkdir -p "$(dirname "$out")"
default_py=python3
[[ -n "${BASIS_ARTIFACTS:-}" && -x "$BASIS_ARTIFACTS/envs/vllm/basis/bin/python" ]] && default_py=$BASIS_ARTIFACTS/envs/vllm/basis/bin/python
PY=${PYTHON:-$default_py}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=2
[[ -n "${BASIS_ARTIFACTS:-}" ]] && export LD_LIBRARY_PATH="$BASIS_ARTIFACTS/envs/vllm/basis/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
IFS=, read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"

{
echo "### host $(hostname) $(date -u +%FT%TZ) python=$PY CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
lscpu | grep -E '^(Model name|Socket\(s\)|NUMA node\(s\))'
"$PY" -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "nccl", ".".join(map(str, torch.cuda.nccl.version())))'
echo "### gpus"
nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,memory.used --format=csv
echo "### topology"
nvidia-smi topo -m
nvidia-smi topo -p2p r
echo "### pcie upstream chain (sysfs; current gen drops when idle, width does not)"
for bus in $(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader); do
    id=$(tr 'A-F' 'a-f' <<< "${bus:4}"); p=$(readlink -f "/sys/bus/pci/devices/$id" 2>/dev/null)
    echo "$id"
    while [[ -n "$p" && -e "$p/current_link_width" ]]; do
        echo "  $(basename "$p"): width $(cat "$p/current_link_width")/$(cat "$p/max_link_width") speed $(cat "$p/current_link_speed")/$(cat "$p/max_link_speed")"
        p=$(dirname "$p")
    done
done
echo "### copy bandwidth"
"$PY" benchmarks/system/check_interconnect_copy.py
port=29611
groups=("$CUDA_VISIBLE_DEVICES")
for ((i = 0; i < ${#gpus[@]}; i++)); do for ((j = i + 1; j < ${#gpus[@]}; j++)); do groups+=("${gpus[i]},${gpus[j]}"); done; done
for p2p in 0 1; do
    for g in "${groups[@]}"; do
        n=$(( $(tr -cd , <<< "$g" | wc -c) + 1 ))
        echo "### allreduce gpus=$g p2p_disable=$p2p"
        log=$(CUDA_VISIBLE_DEVICES=$g NCCL_P2P_DISABLE=$p2p NCCL_DEBUG=INFO \
              "$PY" -m torch.distributed.run --nproc_per_node="$n" --master_port=$port benchmarks/system/check_interconnect_allreduce.py 2>&1)
        grep -E '^RESULT' <<< "$log" || { echo "FAILED"; tail -20 <<< "$log"; }
        echo "TRANSPORT gpus=$g p2p_disable=$p2p : $(grep -oE 'via (P2P|SHM|NET)[A-Za-z/]*' <<< "$log" | sort -u | tr '\n' ' ')"
        port=$((port + 1))
    done
done
} 2>&1 | tee "$out"

echo "### verdict" | tee -a "$out"
"$PY" - "$out" <<'EOF' 2>&1 | tee -a "$out"
import json, re, sys
text = open(sys.argv[1]).read()
copy = json.loads(re.search(r"^RESULT copy (.*)$", text, re.M).group(1))
runs = [json.loads(m) for m in re.findall(r"^RESULT allreduce (.*)$", text, re.M)]
full = [r for r in runs if r["world"] == copy["gpus"]]
on = next(r for r in full if r["nccl_p2p_disable"] == "0"); off = next(r for r in full if r["nccl_p2p_disable"] == "1")
at = lambda run, size: next(x for x in run["results"] if x["bytes"] == size)
transport_on = re.search(r"^TRANSPORT gpus=\S+ p2p_disable=0 : (.*)$", text, re.M).group(1)
p2p_min = min(copy["p2p_GBs"].values())
agg = copy["concurrent_h2d_aggregate_GBs"]
checks = [
    ("peer access on every pair", all(copy["peer_access"].values()), str(all(copy["peer_access"].values()))),
    ("NCCL uses P2P when enabled", "P2P" in transport_on and "SHM" not in transport_on, transport_on.strip()),
    ("P2P copy >= 20 GB/s on every pair", p2p_min >= 20, f"min {p2p_min} GB/s"),
    ("4-GPU all-reduce 256 KiB <= 100 us (P2P on)", at(on, 256 << 10)["us"] <= 100, f"{at(on, 256 << 10)['us']} us"),
    ("4-GPU all-reduce 64 MiB busbw >= 10 GB/s (P2P on)", at(on, 64 << 20)["busbw_GBs"] >= 10, f"{at(on, 64 << 20)['busbw_GBs']} GB/s"),
]
for name, ok, value in checks:
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {value}")
print(f"info  concurrent host->device aggregate by GPU count: {agg} GB/s (flat = one shared upstream link)")
for label, run in (("P2P on", on), ("P2P off", off)):
    est = ", ".join(f"b{b} {64 * at(run, s)['us'] / 1000:.1f} ms" for b, s in ((32, 256 << 10), (64, 512 << 10), (128, 1 << 20)))
    print(f"info  decode all-reduce time per step, 64 x 4-GPU all-reduce, {label}: {est}")
print("OVERALL", "PASS" if all(ok for _, ok, _ in checks) else "FAIL")
EOF
echo "log: $out"
