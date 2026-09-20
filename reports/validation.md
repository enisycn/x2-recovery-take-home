# Validation record

Recorded on 20 September 2026. Results below distinguish executed checks from code that could not be run in the sandbox.

## Static and unit checks

```bash
/usr/bin/python3 -m compileall -q isaaclab_ext src/x2_recovery_ros/x2_recovery_ros scripts
bash -n scripts/*.sh
PYTHONPATH="$PWD/src/x2_recovery_ros" /usr/bin/python3 -m pytest -q src/x2_recovery_ros/test
ISAAC_PYTHON=/path/to/isaac/bin/python ./scripts/check_runtime_isolation.sh
```

Observed: Python and shell checks passed; `7 passed, 1 skipped`. The runtime-isolation check confirmed distinct Python executables: ROS imported `rclpy` but could not see Isaac Lab, while the Isaac interpreter imported Isaac Lab but could not see `rclpy`. The skipped test exercises a real Unix socket, and the execution sandbox rejects `AF_UNIX` creation with `EPERM`. The IPC path therefore still requires an end-to-end run on the target workstation.

The Isaac task and PPO registration loaded with the machine's Isaac Lab 3.0 Python environment:

```text
Gym tasks: HRS-X2-Recovery-v0, HRS-X2-Recovery-Play-v0
experiment: hrs_x2_recovery
num_envs: 2048
simulation dt: 0.005 s
decimation: 10
episode: 8.0 s
```

Validation ran on Ubuntu 22.04.5 with an AMD Ryzen AI 9 HX 370 (12 cores/24 threads), Python 3.10.12 and ROS 2 Humble. The installed simulator stack reported Isaac Sim 6.0.1, Isaac Lab 3.0.0 and RSL-RL 5.0.1. Importing the configuration emitted `no CUDA-capable device is detected`, so no Isaac physics rollout or PPO result is claimed.

## Reduced-order experiment

```bash
./scripts/run_training.sh
./scripts/run_evaluation.sh
```

The committed training run used seeded cross-entropy policy search: 18 iterations, population 40, eight elites, two rollouts per candidate and seed 7. The held-out evaluation is in `reports/evaluation.json`.

| Episode | Seed | Result | Steps | Return | Failure |
| ---: | ---: | --- | ---: | ---: | --- |
| 1 | 101 | success | 119 | 62.41 | |
| 2 | 102 | success | 120 | 62.50 | |
| 3 | 103 | success | 118 | 62.25 | |
| 4 | 104 | success | 118 | 62.34 | |
| 5 | 105 | failed | 120 | 7.52 | timeout before stable two-foot stance |

Episode 5 reached a two-foot, unsupported pose but accumulated only 0.30 s of stable time before the 6.0 s timeout; the harness requires 0.40 s. The 4/5 count is a CPU orchestration baseline, not an X2 rigid-body result.

## ROS 2 Humble

Fresh build:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select x2_recovery_ros
```

Observed: `1 package finished`.

Launch and request:

```bash
source install/setup.bash
ros2 launch x2_recovery_ros x2_recovery.launch.py
ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
```

Observed first response:

```text
success=True, message='Recovery accepted'
```

An immediate second call returned:

```text
success=False, message='Recovery already running'
```

Live telemetry was read from the running episode:

```text
name: [left_hip_pitch_joint, left_knee_joint, left_ankle_pitch_joint,
       right_hip_pitch_joint, right_knee_joint, right_ankle_pitch_joint]
position: [0.4282, 0.6322, -0.1244, 0.4282, 0.6322, -0.1244]
status: RUNNING
```

The node and telemetry log then reached `SUCCEEDED`; the recovery node reported 119 steps. With `policy_mode:=zero timeout_sec:=0.5`, it reported:

```text
Recovery failed in 10 steps: timeout before stable two-foot stance
status=FAILED
```

The sandbox blocks UDP sockets, so Fast DDS printed UDP transport warnings; shared-memory transport still carried the local service and topic data. CLI discovery was run with `--no-daemon` where supported.

## Blocked high-fidelity run

The official model checkout could not be fetched from the shell because outbound Git access is disabled, and no X2 URDF was already present. CUDA was unavailable to Isaac in this execution. Consequently:

- no X2 USD was generated;
- no PPO checkpoint or Isaac reward curve exists;
- no five-episode Isaac success count is reported;
- the ROS–Isaac socket server was syntax/API checked but not exercised end to end.

The exact commands to complete those checks on the intended workstation are in the README.

## GitHub submission state

The standalone local repository contains six meaningful commits and passes `git fsck`. It has no configured remote: GitHub CLI was unavailable, and browser access to create the repository was denied by the execution environment. The two commands required after creating the destination repository are recorded in the README; they preserve the complete local history.
