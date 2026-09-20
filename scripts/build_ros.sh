#!/usr/bin/env bash
set -eo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
set -u
cd "${project_dir}"
export PYTHONNOUSERSITE=1
/usr/bin/colcon build --symlink-install --base-paths src --event-handlers console_direct+
