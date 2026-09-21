#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ISAAC_PYTHON:?Set ISAAC_PYTHON to the Python executable in the Isaac Lab environment.}"

cd "${project_dir}"
env \
  -u AMENT_PREFIX_PATH \
  -u COLCON_PREFIX_PATH \
  -u ROS_DISTRO \
  -u ROS_VERSION \
  PYTHONPATH="${project_dir}/isaaclab_ext" \
  "${ISAAC_PYTHON}" -m pytest -q isaaclab_ext/test/test_reward_formulas.py
