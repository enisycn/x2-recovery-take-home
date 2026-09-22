#!/usr/bin/env bash
set -eo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="$(mktemp -d /tmp/hrs-x2-isaac-ros.XXXXXX)"
server_pid=""
launch_pid=""

cleanup() {
  local result=$?
  if [[ "${result}" -ne 0 ]]; then
    tail -n 30 "${runtime_dir}/launch.log" 2>/dev/null || true
    tail -n 20 "${runtime_dir}/server.log" 2>/dev/null || true
  fi
  for process_id in "${launch_pid}" "${server_pid}"; do
    if [[ -n "${process_id}" ]]; then
      kill -TERM -- "-${process_id}" 2>/dev/null || true
      for _ in $(seq 1 20); do
        kill -0 -- "-${process_id}" 2>/dev/null || break
        sleep 0.1
      done
      kill -KILL -- "-${process_id}" 2>/dev/null || true
      wait "${process_id}" 2>/dev/null || true
    fi
  done
  rm -rf "${runtime_dir}"
}
trap cleanup EXIT

: "${ISAAC_PYTHON:?Set ISAAC_PYTHON to the existing Isaac environment Python executable.}"
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 EXPORTED_POLICY_PT [server options]" >&2
  exit 2
fi
policy="$1"
shift
socket_path="${runtime_dir}/recovery.sock"

cd "${project_dir}"
setsid "${project_dir}/scripts/serve_isaac_policy.sh" "${policy}" --socket "${socket_path}" "$@" >"${runtime_dir}/server.log" 2>&1 &
server_pid="$!"
for _ in $(seq 1 480); do
  [[ -S "${socket_path}" ]] && break
  kill -0 "${server_pid}" 2>/dev/null || { cat "${runtime_dir}/server.log"; exit 1; }
  sleep 0.25
done
[[ -S "${socket_path}" ]] || { echo "Isaac socket did not appear" >&2; exit 1; }

source /opt/ros/humble/setup.bash
source "${HRS_ROS_INSTALL:-${project_dir}/install}/setup.bash"
set -u
export PYTHONNOUSERSITE=1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-94}"
export ROS_LOG_DIR="${runtime_dir}/ros_logs"
export FASTRTPS_DEFAULT_PROFILES_FILE="${project_dir}/config/fastdds_shm.xml"
mkdir -p "${ROS_LOG_DIR}"

setsid ros2 launch x2_recovery_ros x2_recovery.launch.py \
  backend:=isaac_ipc socket_path:="${socket_path}" timeout_sec:=10.0 >"${runtime_dir}/launch.log" 2>&1 &
launch_pid="$!"
for _ in $(seq 1 80); do
  ros2 service list --no-daemon 2>/dev/null | grep -Fxq /x2/start_recovery && break
  sleep 0.1
done

/usr/bin/python3 "${project_dir}/scripts/validate_ros_isaac_client.py"
# Keep observable telemetry and the real simulator timeout reason in the record.
rg 'status=.*left_knee_joint=.*rad|Recovery failed.*timeout' "${runtime_dir}/launch.log"
rg -q 'Recovery failed.*timeout' "${runtime_dir}/launch.log"
rg -q 'status=RUNNING.*left_knee_joint=.*rad' "${runtime_dir}/launch.log"
echo "ROS-Isaac runtime validation passed (CLI success, busy rejection, live telemetry, Isaac timeout)"
