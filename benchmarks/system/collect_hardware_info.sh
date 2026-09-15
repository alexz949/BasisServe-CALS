#!/usr/bin/env bash
set -euo pipefail
python -u -m benchmarks.system.hardware "$@"
