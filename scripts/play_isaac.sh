#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ISAAC_PYTHON:?Set ISAAC_PYTHON to the existing Isaac environment Python executable.}"
[[ -x "${ISAAC_PYTHON}" ]] || { echo "ISAAC_PYTHON is not executable: ${ISAAC_PYTHON}" >&2; exit 1; }
isaac_env_dir="$(cd "$(dirname "${ISAAC_PYTHON}")/.." && pwd)"
if [[ -d "${isaac_env_dir}/conda-meta" ]]; then
  unset VIRTUAL_ENV
  export CONDA_PREFIX="${isaac_env_dir}"
else
  unset CONDA_PREFIX
  export VIRTUAL_ENV="${isaac_env_dir}"
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CHECKPOINT [extra Isaac Lab arguments]" >&2
  exit 2
fi
checkpoint="$1"
shift

cd "${project_dir}"
PYTHONPATH="${project_dir}/isaaclab_ext${PYTHONPATH:+:${PYTHONPATH}}" \
  "${ISAAC_PYTHON}" -c 'from isaaclab.cli import cli; cli()' play \
  --rl_library rsl_rl \
  --task HRS-X2-Recovery-Play-v0 \
  --external_callback x2_recovery_isaac.register \
  --num_envs 1 \
  --checkpoint "${checkpoint}" "$@"
