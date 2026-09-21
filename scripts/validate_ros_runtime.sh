#!/usr/bin/env bash
set -eo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="$(mktemp -d /tmp/hrs-x2-ros-validation.XXXXXX)"
launch_pid=""

cleanup() {
  stop_launch
  rm -rf "${runtime_dir}"
}
trap cleanup EXIT

source /opt/ros/humble/setup.bash
source "${project_dir}/install/setup.bash"
set -u
export PYTHONNOUSERSITE=1
base_domain_id="${ROS_DOMAIN_ID:-93}"
export ROS_DOMAIN_ID="${base_domain_id}"
export ROS_LOG_DIR="${runtime_dir}/ros_logs"
export FASTRTPS_DEFAULT_PROFILES_FILE="${project_dir}/config/fastdds_shm.xml"
mkdir -p "${ROS_LOG_DIR}"

wait_for_service() {
  local attempts
  for attempts in $(seq 1 50); do
    if ros2 service list --no-daemon 2>/dev/null | grep -Fxq "/x2/start_recovery"; then
      return 0
    fi
    sleep 0.1
  done
  echo "start_recovery service was not discovered" >&2
  return 1
}

wait_for_log() {
  local pattern="$1"
  local log_file="$2"
  local attempts
  for attempts in $(seq 1 120); do
    if grep -Fq "${pattern}" "${log_file}" 2>/dev/null; then
      return 0
    fi
    sleep 0.1
  done
  echo "timed out waiting for: ${pattern}" >&2
  return 1
}

stop_launch() {
  if [[ -n "${launch_pid}" ]] && kill -0 "${launch_pid}" 2>/dev/null; then
    kill -TERM "${launch_pid}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "${launch_pid}" 2>/dev/null || break
      sleep 0.1
    done
    kill -KILL "${launch_pid}" 2>/dev/null || true
    wait "${launch_pid}" 2>/dev/null || true
  fi
  launch_pid=""
}

success_log="${runtime_dir}/success.log"
ros2 launch x2_recovery_ros x2_recovery.launch.py >"${success_log}" 2>&1 &
launch_pid="$!"
wait_for_log "X2 recovery ready" "${success_log}"
wait_for_service

echo "SUCCESS CASE: first request"
timeout -k 2s 8s ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
echo "SUCCESS CASE: concurrent request"
timeout -k 2s 8s ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
echo "SUCCESS CASE: live joint sample"
timeout -k 2s 8s ros2 topic echo /x2/joint_states sensor_msgs/msg/JointState \
  --once --no-daemon
wait_for_log "Recovery succeeded" "${success_log}"
grep -F "Recovery succeeded" "${success_log}"
stop_launch

# Discovery leases from the just-stopped success nodes can briefly remain in
# the graph.  Use a fresh valid DDS domain so the timeout request is provably
# handled by the new zero-policy node rather than a retiring success node.
export ROS_DOMAIN_ID="$(( (base_domain_id + 1) % 233 ))"
failure_log="${runtime_dir}/failure.log"
ros2 launch x2_recovery_ros x2_recovery.launch.py \
  policy_mode:=zero timeout_sec:=0.5 >"${failure_log}" 2>&1 &
launch_pid="$!"
wait_for_log "X2 recovery ready" "${failure_log}"
wait_for_service

echo "FAILURE CASE: request"
timeout -k 2s 8s ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
wait_for_log "Recovery failed" "${failure_log}"
grep -F "Recovery failed" "${failure_log}"
stop_launch

echo "ROS runtime validation passed"
