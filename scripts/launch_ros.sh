#!/usr/bin/env bash
set -eo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
source "${project_dir}/install/setup.bash"
set -u
export ROS_LOG_DIR="${project_dir}/log/ros"
mkdir -p "${ROS_LOG_DIR}"
exec ros2 launch x2_recovery_ros x2_recovery.launch.py "$@"
