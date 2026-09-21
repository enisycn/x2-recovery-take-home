#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=isaac_runtime.sh
source "${project_dir}/scripts/isaac_runtime.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CHECKPOINT [extra Isaac Lab arguments]" >&2
  exit 2
fi
checkpoint="$1"
shift

cd "${project_dir}"
export PYTHONPATH="${project_dir}/isaaclab_ext${PYTHONPATH:+:${PYTHONPATH}}"
run_isaac_offline "${ISAAC_PYTHON}" scripts/evaluate_isaac.py \
  --checkpoint "${checkpoint}" --kit_args "${HRS_KIT_OFFLINE_ARGS}" "$@"
