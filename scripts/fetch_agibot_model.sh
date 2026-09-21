#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${project_dir}/models/agibot_x2_urdf"
official_url="https://github.com/AgibotTech/agibot_x2_urdf.git"
# The official repository had one commit when reviewed on 2026-09-21.
pinned_commit="77f43eb"

if [[ -d "${model_dir}/.git" ]]; then
  origin="$(git -C "${model_dir}" remote get-url origin)"
  revision="$(git -C "${model_dir}" rev-parse --short=7 HEAD)"
  if [[ "${origin}" == "${official_url}" && "${revision}" == "${pinned_commit}" ]]; then
    echo "Verified official model already present: ${model_dir}"
    exit 0
  fi
  echo "Refusing an existing model with an unexpected origin or revision: ${model_dir}" >&2
  exit 1
fi

mkdir -p "${project_dir}/models"
git -c protocol.file.allow=never -c core.hooksPath=/dev/null clone \
  --depth 1 --no-tags --no-recurse-submodules --branch main \
  "${official_url}" "${model_dir}"

origin="$(git -C "${model_dir}" remote get-url origin)"
revision="$(git -C "${model_dir}" rev-parse --short=7 HEAD)"
if [[ "${origin}" != "${official_url}" || "${revision}" != "${pinned_commit}" ]]; then
  echo "Downloaded model failed origin/revision verification; it will not be used." >&2
  rm -rf "${model_dir}"
  exit 1
fi

git -C "${model_dir}" fsck --no-dangling
echo "Isaac input: ${model_dir}/X2_URDF-v1.3.0/x2_ultra_simple_collision.urdf"
