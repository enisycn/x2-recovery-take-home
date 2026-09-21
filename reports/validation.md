# Validation record

Updated on 21 September 2026. Results below distinguish executed checks from code that could not be run in the sandbox.

## Static and unit checks

```bash
/usr/bin/python3 -m compileall -q isaaclab_ext src/x2_recovery_ros/x2_recovery_ros scripts
bash -n scripts/*.sh
PYTHONPATH="$PWD/src/x2_recovery_ros" /usr/bin/python3 -m pytest -q src/x2_recovery_ros/test
ISAAC_PYTHON=/path/to/isaac/bin/python ./scripts/check_runtime_isolation.sh
ISAAC_PYTHON=/path/to/isaac/bin/python ./scripts/test_isaac_formulas.sh
```

Observed: Python and shell checks passed; ROS tests reported `7 passed, 1 skipped`, and the independent Isaac equation suite reported `5 passed`. The runtime-isolation check confirmed distinct Python executables: ROS imported `rclpy` but could not see Isaac Lab, while the Isaac interpreter imported Isaac Lab but could not see `rclpy`. The skipped test exercises a real Unix socket, and the execution sandbox rejects `AF_UNIX` creation with `EPERM`. The IPC path therefore still requires an end-to-end run on the target workstation.

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
./scripts/run_training.sh --seed 7 --iterations 60 --population 80
./scripts/run_evaluation.sh
```

The committed training run used seeded cross-entropy policy search: 60 iterations, population 80, eight elites, two rollouts per candidate and seed 7. Candidate selection used five separate validation seeds (90001–90005); the evaluation seeds below were not used for fitting or selection. The held-out evaluation is in `reports/evaluation.json`.

| Episode | Seed | Result | Steps | Return | Failure |
| ---: | ---: | --- | ---: | ---: | --- |
| 1 | 101 | success | 109 | 61.63 | |
| 2 | 102 | success | 110 | 61.69 | |
| 3 | 103 | success | 109 | 61.58 | |
| 4 | 104 | success | 104 | 61.12 | |
| 5 | 105 | success | 106 | 61.31 | |

All five episodes reached the CPU harness's stable, upright, two-foot, unsupported predicate within the 6.0 s timeout. The 5/5 count is a CPU orchestration baseline, not an X2 rigid-body result.

## ROS 2 Humble

Fresh build:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select x2_recovery_ros
```

Observed: `1 package finished`.

The repeatable runtime check launches both nodes, exercises success and timeout paths, and keeps Fast DDS on shared memory without opening a network port:

```bash
./scripts/validate_ros_runtime.sh
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
position: [0.2330, 0.3466, -0.0289, 0.2330, 0.3466, -0.0289]
status: RUNNING
```

The node and telemetry log then reached `SUCCEEDED`; the recovery node reported 109 steps. With `policy_mode:=zero timeout_sec:=0.5`, it reported:

```text
Recovery failed in 10 steps: timeout before stable two-foot stance
status=FAILED
```

The sandbox blocks network-interface inspection and prints benign `getifaddrs` warnings. `config/fastdds_shm.xml` disables UDP/TCP transports; the launch and CLI processes share one sandbox namespace, and the local service/topic checks passed without a network port.

## Blocked high-fidelity run

The official model checkout could not be fetched from the shell because outbound Git DNS is disabled, and no X2 URDF was already present. A browser fallback to GitHub's raw file host was rejected by the browser safety review, so it was not bypassed. CUDA was unavailable to Isaac in this execution. Consequently:

- no X2 USD was generated;
- no PPO checkpoint or Isaac reward curve exists;
- no five-episode Isaac success count is reported;
- the ROS–Isaac socket server was syntax/API checked but not exercised end to end.

The exact commands to complete those checks on the intended workstation are in the README.

## Repository state

The standalone local repository has meaningful staged commits and passes `git fsck`. It has no configured remote, and no existing robot workspace was read into or copied into this repository.
