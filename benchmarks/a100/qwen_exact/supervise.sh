#!/usr/bin/env bash
set -uo pipefail
exec >> /mnt/basiskv/qwen-exact/logs/launcher.log 2>&1
date -Is
bash /mnt/basiskv/qwen-exact/setup/run.sh
qwen_exit_status=$?
printf 'Qwen job exit code: %s\n' "$qwen_exit_status"
date -Is
exit "$qwen_exit_status"
