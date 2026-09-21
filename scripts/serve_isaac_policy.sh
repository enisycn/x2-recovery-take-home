#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ISAAC_PYTHON:?Set ISAAC_PYTHON to the existing Isaac environment Python executable.}"
[[ -x "${ISAAC_PYTHON}" ]] || { echo "ISAAC_PYTHON is not executable: ${ISAAC_PYTHON}" >&2; exit 1; }

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 EXPORTED_POLICY_PT [extra server arguments]" >&2
  exit 2
fi
policy_path="$1"
shift

cd "${project_dir}"
PYTHONPATH="${project_dir}/isaaclab_ext${PYTHONPATH:+:${PYTHONPATH}}" \
  "${ISAAC_PYTHON}" scripts/isaac_policy_server.py \
  --policy "${policy_path}" --headless "$@"
