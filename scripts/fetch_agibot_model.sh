#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${project_dir}/models/agibot_x2_urdf"
official_url="https://github.com/AgibotTech/agibot_x2_urdf.git"
# Official revision used for the submitted model and results.
pinned_commit="60c5de582c523cd188f563819e62d34cfdc3d2d0"

if [[ -d "${model_dir}/.git" ]]; then
  origin="$(git -C "${model_dir}" remote get-url origin)"
  revision="$(git -C "${model_dir}" rev-parse HEAD)"
  if [[ "${origin}" == "${official_url}" && "${revision}" == "${pinned_commit}" ]]; then
    echo "Verified official model already present: ${model_dir}"
    exit 0
  fi
  echo "Refusing an existing model with an unexpected origin or revision: ${model_dir}" >&2
  exit 1
fi

if [[ -e "${model_dir}" ]]; then
  echo "Refusing a pre-existing model directory without verified Git metadata: ${model_dir}" >&2
  exit 1
fi

mkdir -p "${model_dir}"
trap 'rm -rf -- "${model_dir}"' ERR
git -C "${model_dir}" -c core.hooksPath=/dev/null init -q
git -C "${model_dir}" remote add origin "${official_url}"
# Fetch the recorded revision directly so a later upstream main does not
# silently change the model or make a fresh checkout fail its pin check.
git -C "${model_dir}" -c protocol.file.allow=never -c core.hooksPath=/dev/null \
  fetch --depth 1 --no-tags origin "${pinned_commit}"
git -C "${model_dir}" -c core.hooksPath=/dev/null checkout -q --detach FETCH_HEAD
trap - ERR

origin="$(git -C "${model_dir}" remote get-url origin)"
revision="$(git -C "${model_dir}" rev-parse HEAD)"
if [[ "${origin}" != "${official_url}" || "${revision}" != "${pinned_commit}" ]]; then
  echo "Downloaded model failed origin/revision verification; it will not be used." >&2
  rm -rf "${model_dir}"
  exit 1
fi

git -C "${model_dir}" fsck --no-dangling
echo "Isaac input: ${model_dir}/X2_URDF-v1.3.0/x2_ultra_simple_collision.urdf"
