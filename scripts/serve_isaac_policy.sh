#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ISAACLAB_ROOT:?Set ISAACLAB_ROOT to the Isaac Lab checkout; it will only be executed, never modified.}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 EXPORTED_POLICY_PT [extra server arguments]" >&2
  exit 2
fi
policy_path="$1"
shift

cd "${project_dir}"
PYTHONPATH="${project_dir}/isaaclab_ext${PYTHONPATH:+:${PYTHONPATH}}" \
  "${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/isaac_policy_server.py \
  --policy "${policy_path}" --headless "$@"
