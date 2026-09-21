#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=isaac_runtime.sh
source "${project_dir}/scripts/isaac_runtime.sh"
cd "${project_dir}"
export PYTHONPATH="${project_dir}/isaaclab_ext${PYTHONPATH:+:${PYTHONPATH}}"
run_isaac_offline "${ISAAC_PYTHON}" scripts/render_isaac_gifs.py \
  --enable_cameras --kit_args "${HRS_KIT_OFFLINE_ARGS}" "$@"
