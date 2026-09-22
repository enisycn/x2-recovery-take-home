#!/usr/bin/env bash
# Common isolation for HRS-only Isaac processes. Source from a launcher script.

set -euo pipefail

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

# Native Kit settings are applied before the Python application is created.
export OMNI_CRASHREPORTER_ENABLED=0
export OMNI_CRASHREPORTER_SKIPOLDDUMPUPLOAD=1
export OMNI_CRASHREPORTER_PRESERVEDUMP=1
readonly HRS_KIT_OFFLINE_ARGS="--/crashreporter/enabled=false --/crashreporter/skipOldDumpUpload=true --/telemetry/enableAnonymousAppName=false --/telemetry/enableAnonymousData=false --/telemetry/enableSentry=false --/privacy/usage=false --/privacy/performance=false --/privacy/personalization=false"

run_isaac_offline() {
  local -a command_prefix=()

  # A private user+network namespace is a second barrier against uploads.  On
  # hosts that disallow unprivileged namespaces, Kit's explicit opt-out flags
  # above still apply and the launcher remains portable.
  if unshare -Urn true >/dev/null 2>&1; then
    command_prefix+=(unshare -Urn)
  fi

  "${command_prefix[@]}" "$@"
}
