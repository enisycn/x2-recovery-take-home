#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ISAACLAB_ROOT:?Set ISAACLAB_ROOT to the Isaac Lab checkout; it will only be executed, never modified.}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CHECKPOINT [extra Isaac Lab arguments]" >&2
  exit 2
fi
checkpoint="$1"
shift

cd "${project_dir}"
PYTHONPATH="${project_dir}/isaaclab_ext${PYTHONPATH:+:${PYTHONPATH}}" \
  "${ISAACLAB_ROOT}/isaaclab.sh" play \
  --rl_library rsl_rl \
  --task HRS-X2-Recovery-Play-v0 \
  --external_callback x2_recovery_isaac.register \
  --num_envs 1 \
  --checkpoint "${checkpoint}" "$@"
