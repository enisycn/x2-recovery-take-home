#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${project_dir}/src/x2_recovery_ros:${PYTHONPATH:-}"
python_bin="${PYTHON_BIN:-/usr/bin/python3}"
"${python_bin}" -m x2_recovery_ros.evaluate "$@"
