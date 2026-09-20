# AgiBot X2 ground recovery

A deliberately small, reproducible take-home implementation for X2 ground recovery. It contains:

- a deterministic reduced-order environment that runs with NumPy only;
- a compact cross-entropy policy-search experiment and five-episode evaluation;
- an optional MuJoCo backend for the official AgiBot X2 Ultra v1.3.0 model;
- a ROS 2 Humble package with recovery and telemetry nodes.

The reduced-order backend exists so the complete control and ROS workflow can be exercised on a CPU-only machine. It is a training baseline, not evidence that the policy transfers to the physical robot. See `docs/task_requirements.md` for the requirement map and `reports/validation.md` for recorded results.

## Quick start

```bash
source /opt/ros/humble/setup.bash
./scripts/run_training.sh
./scripts/run_evaluation.sh
./scripts/build_ros.sh
source install/setup.bash
ros2 launch x2_recovery_ros x2_recovery.launch.py
```

Detailed setup and architecture notes will be completed alongside the validated implementation.

