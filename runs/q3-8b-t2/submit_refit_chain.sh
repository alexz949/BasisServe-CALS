#!/bin/bash
# Table 2 refit chain: V covariance capture -> ALS V96 -> router moments/base/fisher/residual fit. Each stage waits afterok on the previous one.
R=/zpool-00/home/lz299/BasisServe-CALS/runs/q3-8b-t2; B=/zpool-00/home/lz299/BasisServe-CALS/BasisKV; PY=/home/lz299/miniconda3/envs/lowrank/bin/python
W=$R/hf/calibration/qwen3-8b-post-128k/c4-retrieval-50-50/windows.safetensors
COMMON="--model /zpool-00/home/lz299/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 --config $R/yarn4 --device-map balanced --windows $W"
OUT=$R/refit
submit() { # name deps gpus cpus mem time array cmd
  local name=$1 deps=$2 gpus=$3 cpus=$4 mem=$5 time=$6 array=$7 cmd=$8 f=$R/$1.sbatch
  { echo "#!/bin/bash"; echo "#SBATCH -J q3t2-$name"; echo "#SBATCH -p athena,athena-mini"; echo "#SBATCH --exclude=node2"
    [ "$gpus" -gt 0 ] && echo "#SBATCH --gres=gpu:$gpus"; echo "#SBATCH -c $cpus"; echo "#SBATCH --mem=$mem"; echo "#SBATCH -t $time"
    [ -n "$array" ] && echo "#SBATCH --array=$array"; echo "#SBATCH -o $R/logs/${name}_%a.log"; [ -n "$deps" ] && echo "#SBATCH --dependency=afterok:$deps"
    echo "set -uo pipefail"; echo "export PYTHONPATH=/zpool-00/home/lz299/.cache/basiskv_env/pylib517:$B"; echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    echo "cd $B"; echo 'echo "node $(hostname) gpus=${CUDA_VISIBLE_DEVICES:-none} task=${SLURM_ARRAY_TASK_ID:-0} start $(date +%T)"'
    [ "$gpus" -gt 0 ] && echo "$PY -c 'import torch; [torch.zeros(1, device=f\"cuda:{i}\") for i in range(torch.cuda.device_count())]; print(\"cuda ok\", torch.cuda.device_count())' || exit 3"
    echo "$cmd"; echo 'status=$?; echo "exit $status $(date +%T)"; exit $status'; } > $f
  local id; id=$(sbatch --parsable $f) || exit 1; rm $f; echo "$name $id" | tee -a $R/jobids; LAST=${id%%;*}
}
LAST=232615
submit 01_snap        $LAST 3 6 100G 05:00:00 0-1 "$PY -m calibration.value $COMMON --output $OUT/snap --window-shard-index \$SLURM_ARRAY_TASK_ID --window-shard-count 2"
submit 02_snap_merge  $LAST 1 4 64G  01:00:00 ""  "$PY -m calibration.value $COMMON --output $OUT/snap --window-shard-count 2 --merge"
submit 03_fit         $LAST 1 4 48G  04:00:00 0-2 "$PY -m core.fit_value fit-shard --snapshot-dir $OUT/snap --output-dir $OUT/V96 --rank 96 --layer-shard-index \$SLURM_ARRAY_TASK_ID --layer-shard-count 3 --device cuda:0"
submit 04_fit_merge   $LAST 0 4 32G  00:30:00 ""  "$PY -m core.fit_value merge --output-dir $OUT/V96"
submit 05_moments     $LAST 3 6 100G 05:00:00 0-1 "$PY -m calibration.router moments $COMMON --value-factors $OUT/V96 --heldout-windows 0 --output $OUT/router_cap --base-rank 16 --residual-rank 16 --window-shard-index \$SLURM_ARRAY_TASK_ID --window-shard-count 2"
submit 06_moments_merge $LAST 1 4 64G 02:00:00 "" "$PY -m calibration.router moments $COMMON --value-factors $OUT/V96 --heldout-windows 0 --output $OUT/router_cap --base-rank 16 --residual-rank 16 --window-shard-count 2 --merge"
submit 07_base        $LAST 0 8 64G  03:00:00 ""  "$PY -m core.fit_router base --statistics-dir $OUT/router_cap --output-dir $OUT/router_cap/base --base-rank 16 --torch-num-threads 8"
submit 08_fisher      $LAST 3 6 64G  08:00:00 0-1 "$PY -m calibration.router fisher $COMMON --value-factors $OUT/V96 --heldout-windows 0 --output $OUT/router_cap --base-rank 16 --residual-rank 16 --window-shard-index \$SLURM_ARRAY_TASK_ID --window-shard-count 2"
submit 09_router_fit  $LAST 1 4 48G  06:00:00 0-3 "$PY -m core.fit_router fit --statistics-dir $OUT/router_cap --output-dir $OUT/router --residual-rank 16 --device cuda:0 --sweeps 40 --pcg-iterations 100 --layer-shard-index \$SLURM_ARRAY_TASK_ID --layer-shard-count 4"
