#!/usr/bin/env bash
set -eo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="$(mktemp -d /tmp/hrs-x2-isaac-ros.XXXXXX)"
server_pid=""
launch_pid=""

cleanup() {
  for process_id in "${launch_pid}" "${server_pid}"; do
    if [[ -n "${process_id}" ]] && kill -0 "${process_id}" 2>/dev/null; then
      kill -TERM "${process_id}" 2>/dev/null || true
      wait "${process_id}" 2>/dev/null || true
    fi
  done
  rm -f /tmp/hrs_x2_recovery.sock
  rm -rf "${runtime_dir}"
}
trap cleanup EXIT

: "${ISAAC_PYTHON:?Set ISAAC_PYTHON to the existing Isaac environment Python executable.}"
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 EXPORTED_POLICY_PT" >&2
  exit 2
fi
policy="$1"

cd "${project_dir}"
env HRS_NICE="${HRS_NICE:-15}" HRS_CPUSET="${HRS_CPUSET:-<HOST_CPUSET>}" \
  "${project_dir}/scripts/serve_isaac_policy.sh" "${policy}" >"${runtime_dir}/server.log" 2>&1 &
server_pid="$!"
for _ in $(seq 1 160); do
  [[ -S /tmp/hrs_x2_recovery.sock ]] && break
  kill -0 "${server_pid}" 2>/dev/null || { cat "${runtime_dir}/server.log"; exit 1; }
  sleep 0.25
done
[[ -S /tmp/hrs_x2_recovery.sock ]] || { echo "Isaac socket did not appear" >&2; exit 1; }

source /opt/ros/humble/setup.bash
source "${project_dir}/install/setup.bash"
set -u
export PYTHONNOUSERSITE=1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-94}"
export ROS_LOG_DIR="${runtime_dir}/ros_logs"
export FASTRTPS_DEFAULT_PROFILES_FILE="${project_dir}/config/fastdds_shm.xml"
mkdir -p "${ROS_LOG_DIR}"

ros2 launch x2_recovery_ros x2_recovery.launch.py \
  backend:=isaac_ipc timeout_sec:=3.0 >"${runtime_dir}/launch.log" 2>&1 &
launch_pid="$!"
for _ in $(seq 1 80); do
  ros2 service list --no-daemon 2>/dev/null | grep -Fxq /x2/start_recovery && break
  sleep 0.1
done

timeout -k 2s 10s ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}' \
  >"${runtime_dir}/first.txt"
timeout -k 2s 10s ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}' \
  >"${runtime_dir}/busy.txt"
timeout -k 2s 10s ros2 topic echo /x2/joint_states sensor_msgs/msg/JointState \
  --once --no-daemon >"${runtime_dir}/joint.txt"

for _ in $(seq 1 100); do
  grep -Fq "Recovery failed" "${runtime_dir}/launch.log" 2>/dev/null && break
  sleep 0.1
done

grep -Fq "success=True" "${runtime_dir}/first.txt"
grep -Fq "success=False" "${runtime_dir}/busy.txt"
grep -Fq "Recovery failed" "${runtime_dir}/launch.log"
grep -Fq "left_knee_joint" "${runtime_dir}/joint.txt"

echo "FIRST REQUEST"
cat "${runtime_dir}/first.txt"
echo "CONCURRENT REQUEST"
cat "${runtime_dir}/busy.txt"
echo "LIVE JOINT SAMPLE"
sed -n '1,30p' "${runtime_dir}/joint.txt"
echo "FINAL STATUS"
grep -F "Recovery failed" "${runtime_dir}/launch.log"
echo "ROS-Isaac runtime validation passed"
