#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=isaac_runtime.sh
source "${project_dir}/scripts/isaac_runtime.sh"

cd "${project_dir}"
export PYTHONPATH="${project_dir}/isaaclab_ext${PYTHONPATH:+:${PYTHONPATH}}"

capture_file="$(mktemp "${TMPDIR:-/tmp}/hrs_x2_train.XXXXXX.log")"
cleanup() {
  rm -f "${capture_file}"
}
trap cleanup EXIT

set +e
run_isaac_offline "${ISAAC_PYTHON}" scripts/isaac_train_entry.py \
  --headless --kit_args "${HRS_KIT_OFFLINE_ARGS}" "$@" 2>&1 | tee "${capture_file}"
train_status=${PIPESTATUS[0]}
set -e

if (( train_status != 0 )); then
  echo "[HRS] Training failed; preserving the run directory without final artifacts." >&2
  exit "${train_status}"
fi

run_dir="$(sed -n 's/^\[HRS\] Offline training log: //p' "${capture_file}" | tail -n 1)"
if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
  echo "[HRS] Training finished but its run directory could not be resolved." >&2
  exit 1
fi

"${ISAAC_PYTHON}" scripts/finalize_training_run.py "${run_dir}"
