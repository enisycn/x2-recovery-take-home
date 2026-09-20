#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ISAACLAB_ROOT:?Set ISAACLAB_ROOT to the Isaac Lab checkout; it will only be executed, never modified.}"

urdf_path="${project_dir}/models/agibot_x2_urdf/X2_URDF-v1.3.0/x2_ultra_simple_collision.urdf"
output_dir="${project_dir}/assets/isaac"

if [[ ! -f "${urdf_path}" ]]; then
  echo "Missing ${urdf_path}; run scripts/fetch_agibot_model.sh first." >&2
  exit 1
fi

cd "${project_dir}"
PYTHONPATH="${project_dir}/isaaclab_ext${PYTHONPATH:+:${PYTHONPATH}}" \
  "${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/import_x2_urdf.py \
  --input "${urdf_path}" --output "${output_dir}" --headless
