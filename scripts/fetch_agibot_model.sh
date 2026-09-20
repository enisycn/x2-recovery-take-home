#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${project_dir}/models/agibot_x2_urdf"

if [[ -d "${model_dir}/.git" ]]; then
  echo "Official model already present: ${model_dir}"
  exit 0
fi

mkdir -p "${project_dir}/models"
git clone --depth 1 https://github.com/AgibotTech/agibot_x2_urdf.git "${model_dir}"
echo "MuJoCo scene: ${model_dir}/X2_URDF-v1.3.0/scene.xml"
