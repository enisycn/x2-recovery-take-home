#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ISAACLAB_ROOT:?Set ISAACLAB_ROOT to the Isaac Lab checkout; it will only be executed, never modified.}"

cd "${project_dir}"
PYTHONPATH="${project_dir}/isaaclab_ext${PYTHONPATH:+:${PYTHONPATH}}" \
  "${ISAACLAB_ROOT}/isaaclab.sh" train \
  --rl_library rsl_rl \
  --task HRS-X2-Recovery-v0 \
  --external_callback x2_recovery_isaac.register \
  --headless "$@"
