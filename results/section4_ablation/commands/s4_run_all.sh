#!/usr/bin/env bash
# Section 4 chain: exp1 (8 GPUs, ~5 min) -> fits (all GPUs, ~50 min) || exp3a sweep (GPU 0) -> ruler330 x3 (GPUs 0-7, ~1 h) || exp2 sweep (GPU 4) -> summaries.
set -uo pipefail; source /home/Ubuntu/l31_router_fit/section4/common.sh
$S4/s4_exp1.sh > $L/exp1_driver.log 2>&1; say "exp1 exit=$?"
WHICH=exp3a GPU_A=0 $S4/s4_sweeps.sh > $L/sweeps_exp3a_driver.log 2>&1 & sw=$!
$S4/s4_fits.sh > $L/fits_driver.log 2>&1; say "fits exit=$?"; wait $sw; say "sweep exp3a exit=$?"
WHICH=exp2 GPU_B=4 $S4/s4_sweeps.sh > $L/sweeps_exp2_driver.log 2>&1 & sw=$!
$S4/s4_ruler330.sh > $L/ruler330_driver.log 2>&1; say "ruler330 exit=$?"; wait $sw; say "sweep exp2 exit=$?"
$S4/s4_summaries.sh > $L/summaries_driver.log 2>&1; say "summaries exit=$?"
say "=== SECTION4 CHAIN DONE ==="
