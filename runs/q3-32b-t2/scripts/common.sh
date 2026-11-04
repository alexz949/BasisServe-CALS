# Shared paths for the Qwen3-32B Table-2 RULER pipeline (conda lowrank). Source from the repository root.
export PYTHONPATH=/home/sesterce/BasisServe-CALS
export P=$HOME/miniforge3/envs/lowrank/bin/python
export MODEL=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
export R=runs/q3-32b-t2
export WINDOWS=$R/windows/c4-retrieval-50-50/windows.safetensors     # 48 x 131072: fit 0-15 C4, 16-31 synthetic retrieval, held-out 32-47 C4
export DATA=$R/hf/datasets/qwen3-32b-ruler128k-1100/ruler            # RULER-128K, 11 tasks x 100, seed 42, base template
export IDENTITY=$R/identity/v96.json                                 # written by 02_fit_v96.sh (retrieval-mix V96)
export ROUTER=$R/router                                              # fit_k_routing_windowed output; bank = $ROUTER/ours_b16r16
mkdir -p $R/logs
