#!/usr/bin/env bash
# Usage: bash scripts/watch_slurm.sh JOB_ID [LOG_DIR] [INTERVAL_SECONDS] [CODEX_THREAD_ID]
# Prints status and new log tails; exits 0 on success, 1 on job failure.
set -uo pipefail

job_id=${1:-3565545}
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
log_dir=${2:-"$repo_root/results/rebuttal/gh200_tp4/tp4"}
interval=${3:-60}
notify_thread=${4:-}
if [[ ! $job_id =~ ^[0-9]+$ || ! $interval =~ ^[1-9][0-9]*$ ]]; then
    echo 'Usage: watch_slurm.sh JOB_ID [LOG_DIR] [positive INTERVAL_SECONDS]' >&2
    exit 2
fi
mkdir -p -- "$log_dir"
log_dir=$(cd -- "$log_dir" && pwd)
if [[ -n $notify_thread ]]; then
    if [[ ! $notify_thread =~ ^[0-9a-fA-F-]{36}$ ]] || ! command -v codex >/dev/null; then
        echo 'Codex notifications require a thread UUID and the codex CLI.' >&2
        exit 2
    fi
fi
exec 9>"$log_dir/watch-$job_id.lock"
if ! flock -n 9; then
    echo "Job $job_id already has a monitor in $log_dir" >&2
    exit 2
fi
printf '%s\n' "$$" > "$log_dir/watch-$job_id.pid"
trap 'echo "Monitor stopped; Slurm job was left running."; exit 130' INT TERM

declare -A offsets=()
shopt -s nullglob
show_logs() {
    local file size previous delta
    for file in "$log_dir"/*.log; do
        [[ ${file##*/} == watch-* ]] && continue
        size=$(stat -c %s -- "$file") || continue
        previous=${offsets[$file]:-0}
        (( size < previous )) && previous=0
        delta=$((size - previous))
        if (( delta > 0 )); then
            printf '\n--- %s: new log output ---\n' "${file##*/}"
            if (( delta > 65536 )); then
                echo '[Showing the last 64 KiB; full output remains in the original log.]'
                delta=65536
            fi
            tail -c "$delta" -- "$file"
            printf '\n'
        fi
        offsets[$file]=$size
    done
}

notify_completion() {
    [[ -z $notify_thread ]] && return 0
    local marker message attempt
    marker="$log_dir/watch-$job_id-$notify_thread.notified"
    [[ -f $marker ]] && return 0
    message="Slurm 自动通知（用户已授权）：作业 $job_id 已结束，状态=$state，退出码=$code，耗时=$elapsed。请读取 $log_dir 中该作业的日志和结果 JSON，向用户报告结果；若失败，检查失败原因。正式性能测试仍需 smoke 通过和用户批准命令清单，不要因这条通知自动启动正式测试或提交 GitHub。"
    for attempt in 1 2 3; do
        if timeout 30s codex queue --thread "$notify_thread" --message "$message"; then
            date -Is > "$marker"
            echo "Completion notification queued for Codex thread $notify_thread."
            return 0
        fi
        echo "Codex notification attempt $attempt failed."
        (( attempt < 3 )) && sleep 10
    done
    echo 'Completion notification was not delivered; see this monitor log.' >&2
    return 1
}

missing=0
echo "Monitoring Slurm job $job_id every ${interval}s; logs: $log_dir"
[[ -n $notify_thread ]] && echo "Completion notifications: Codex thread $notify_thread"
while :; do
    stamp=$(date '+%F %T %Z')
    queue=$(squeue -h -j "$job_id" -o '%T|%S|%R' 2>/dev/null)
    if [[ -n $queue ]]; then
        missing=0
        IFS='|' read -r state start reason <<< "$queue"
        printf '[%s] %s  start=%s  node/reason=%s\n' "$stamp" "$state" "$start" "$reason"
        show_logs
    else
        accounting=$(sacct -n -P -j "$job_id" --format=JobIDRaw,State,ExitCode,Elapsed,Start,End 2>/dev/null |
            awk -F '|' -v id="$job_id" '$1 == id { print; exit }')
        if [[ -n $accounting ]]; then
            missing=0
            IFS='|' read -r _ state code elapsed start end <<< "$accounting"
            printf '[%s] %s  exit=%s  elapsed=%s  start=%s  end=%s\n' "$stamp" "$state" "$code" "$elapsed" "$start" "$end"
            show_logs
            case "$state" in
                COMPLETED)
                    notify_completion || exit 2
                    [[ $code == '0:0' ]] && exit 0
                    exit 1 ;;
                FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|BOOT_FAIL*|PREEMPTED*|DEADLINE*|REVOKED*)
                    notify_completion || exit 2
                    exit 1 ;;
            esac
        else
            missing=$((missing + 1))
            printf '[%s] Slurm record unavailable (%s/10); retrying.\n' "$stamp" "$missing"
            if (( missing >= 10 )); then
                echo 'Monitor exiting: cannot determine job status.'
                exit 2
            fi
        fi
    fi
    sleep "$interval" &
    wait "$!"
done
