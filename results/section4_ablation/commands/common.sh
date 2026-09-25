# Section 4 ablations (Llama-3.1-8B-Instruct, identity B = V96 + retrieval-mix calibration): shared paths.
R=/home/Ubuntu/q3_8b_128k/repo; PY=/home/Ubuntu/miniconda3/envs/lowrank/bin/python; C=/home/Ubuntu/l31_retrieval_cal; W=/home/Ubuntu/l31_router_fit; S4=$W/section4; L=$S4/logs
ID=$C/identity/B.json; WIN=$C/c4_retrieval_50_50; DATA=$C/ruler1100
export PYTHONPATH=$R LD_LIBRARY_PATH=/home/Ubuntu/miniconda3/envs/lowrank/lib:${LD_LIBRARY_PATH:-} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say(){ printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
run8(){ local tag=$1; shift; local pids=() s; for s in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$s "$@" --window-shard $s > $L/${tag}_s$s.log 2>&1 & pids+=("$!"); done; local st=0 p; for p in "${pids[@]}"; do wait "$p" || st=1; done; return $st; }
