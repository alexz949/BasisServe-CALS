#!/usr/bin/env bash
# Queue one continuation when the authorized FA4 allocation starts running.
set -uo pipefail
job_id=$1
notify_thread=$2
cd /u/azhang35/BasisServe-CALS
result_root=results/rebuttal/gh200_tp4/fa4
exec 9>"$result_root/start-$job_id.lock"
flock -n 9 || exit 0
printf '%s\n' "$$" > "$result_root/start-$job_id.pid"
[[ -f "$result_root/start-$job_id.notified" ]] && exit 0
while :; do
    state=$(squeue -h -j "$job_id" -o '%T' 2>/dev/null)
    if [[ $state == RUNNING ]]; then
        message="Slurm 启动通知（用户已授权）：FA4 DiffKV 单卡 interactive 作业 $job_id 已开始，分配30分钟。用户要求在这次分配内持续检查、修复和复测。请立即读取 results/rebuttal/gh200_tp4/fa4/slurm-$job_id.log 及 smoke/bench-$job_id.json。scripts/fa4_allocation.sh 自动先跑 smoke 再跑8组对照；等 allocation-$job_id.initial_exit 出现后，才通过 srun --jobid=$job_id --overlap 在同一分配内执行后续测试，避免并发污染。使用 basis 环境，提交测试前说明实际命令。完成后 touch results/rebuttal/gh200_tp4/fa4/allocation-$job_id.release 提前释放。不要扩大到TP4正式测试、另申请GPU或提交GitHub。"
        for attempt in 1 2 3; do
            if timeout 30s codex queue --thread "$notify_thread" --message "$message"; then
                date -Is > "$result_root/start-$job_id.notified"
                echo "Queued allocation-start continuation for $job_id."
                exit 0
            fi
            sleep 10
        done
        echo "Could not queue allocation-start continuation for $job_id."
        exit 1
    fi
    if [[ -z $state ]]; then
        terminal=$(sacct -n -P -X -j "$job_id" -o State | head -n 1)
        case "$terminal" in
            COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*) exit 0 ;;
        esac
    fi
    sleep 30
done
