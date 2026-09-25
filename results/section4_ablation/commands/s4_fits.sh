#!/usr/bin/env bash
# Experiment 2 banks at page 4, no sink, same calibration windows and fitter settings as router_B_p4 (B16R16):
#   b16r0 (B16 only: moments -> merge -> base -> fit, no Fisher stage), b0r16 and b0r32 (R only: ... -> fisher -> fit).
set -uo pipefail; source /home/Ubuntu/l31_router_fit/section4/common.sh; cd $R; F=evaluation/fit_k_routing_windowed.py
fit_bank(){ local tag=$1 base=$2 res=$3; local OUT=$W/router_B_p4_$tag
  local A=(--identity $ID --windows $WIN/windows.safetensors --output $OUT --sequence-length 131072 --fit-count 32 --diagnostic-count 0 --sweeps 40 --pcg-iterations 100
           --rope native --teacher deployed --objective page_fisher --page-size 4 --pinned-pages 0 --base-rank $base --residual-rank $res --window-shards 8)
  [[ -f $OUT/ours_b${base}r${res}/layer_031.safetensors ]] && { say "=== [$tag] bank exists, skipping"; return 0; }
  say "=== [$tag] moments: 8 window shards"; run8 fit_${tag}_moments $PY -u $F moments "${A[@]}" || { say "$tag moments FAILED"; return 1; }
  say "=== [$tag] merge"; $PY -u $F merge "${A[@]}" > $L/fit_${tag}_merge.log 2>&1 || { say "$tag merge FAILED"; return 1; }
  say "=== [$tag] base"; $PY -u $F base "${A[@]}" > $L/fit_${tag}_base.log 2>&1 || { say "$tag base FAILED"; return 1; }
  if [[ $res -gt 0 ]]; then say "=== [$tag] fisher: 8 window shards"; run8 fit_${tag}_fisher $PY -u $F fisher "${A[@]}" || { say "$tag fisher FAILED"; return 1; }; fi
  say "=== [$tag] fit: 8 processes (layers s::8)"; local pids=() s
  for s in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$s $PY -u $F fit "${A[@]}" --layers $(seq -s, $s 8 31) > $L/fit_${tag}_fit_s$s.log 2>&1 & pids+=("$!"); done
  local st=0 p; for p in "${pids[@]}"; do wait "$p" || st=1; done; [[ $st -eq 0 ]] || { say "$tag fit FAILED"; return 1; }
  say "=== [$tag] bank done: $(ls $OUT/ours_b${base}r${res}/*.safetensors | wc -l)/32 layers"; }
fit_bank b16r0 16 0 || exit 1
fit_bank b0r16 0 16 || exit 1
fit_bank b0r32 0 32 || exit 1
say "=== FITS DONE ==="
