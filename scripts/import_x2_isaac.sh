#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ISAAC_PYTHON:?Set ISAAC_PYTHON to the existing Isaac environment Python executable.}"
[[ -x "${ISAAC_PYTHON}" ]] || { echo "ISAAC_PYTHON is not executable: ${ISAAC_PYTHON}" >&2; exit 1; }

urdf_path="${project_dir}/models/agibot_x2_urdf/X2_URDF-v1.3.0/x2_ultra_simple_collision.urdf"
output_dir="${project_dir}/assets/isaac"
manifest_path="${project_dir}/models/agibot_x2_urdf/.hrs-source.json"
official_url="https://github.com/AgibotTech/agibot_x2_urdf.git"
pinned_commit="60c5de582c523cd188f563819e62d34cfdc3d2d0"

if [[ ! -f "${urdf_path}" ]]; then
  echo "Missing ${urdf_path}; run scripts/fetch_agibot_model.sh first." >&2
  exit 1
fi

if [[ -d "${project_dir}/models/agibot_x2_urdf/.git" ]]; then
  origin="$(git -C "${project_dir}/models/agibot_x2_urdf" remote get-url origin)"
  revision="$(git -C "${project_dir}/models/agibot_x2_urdf" rev-parse HEAD)"
  if [[ "${origin}" != "${official_url}" || "${revision}" != "${pinned_commit}" ]]; then
    echo "Refusing to import an unverified Git origin or revision." >&2
    exit 1
  fi
elif [[ -f "${manifest_path}" ]]; then
  /usr/bin/python3 -c '
import hashlib, json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
urdf = pathlib.Path(sys.argv[2])
allowed = (
    "https://x2-aimdk.agibot.com/",
    "https://github.com/AgibotTech/agibot_x2_urdf",
)
assert manifest["validation"] == "safe-official-archive-structure-v1"
assert str(manifest["source_url"]).startswith(allowed)
assert manifest["urdf_sha256"] == hashlib.sha256(urdf.read_bytes()).hexdigest()
' "${manifest_path}" "${urdf_path}" || {
    echo "Refusing an archive whose validation manifest no longer matches." >&2
    exit 1
  }
else
  echo "Refusing to import a model without verified Git metadata or an HRS source manifest." >&2
  exit 1
fi

cd "${project_dir}"
PYTHONPATH="${project_dir}/isaaclab_ext${PYTHONPATH:+:${PYTHONPATH}}" \
  "${ISAAC_PYTHON}" scripts/import_x2_urdf.py \
  --input "${urdf_path}" --output "${output_dir}" --headless --device cpu
