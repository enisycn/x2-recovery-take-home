# AgiBot X2 ground recovery

This repository implements the HRS take-home as a self-contained external Isaac Lab task and ROS 2 Humble package. The main path is an AgiBot X2 Ultra v1.3.0 floating-base model in Isaac Lab/PhysX, trained with RSL-RL PPO. A small NumPy model remains as a clearly labelled CPU test harness for training plumbing and ROS interface validation.

The repository never installs files into an existing robot workspace. Isaac Lab is invoked as a runtime, the AgiBot source model and converted USD stay under this repository, ROS builds into this repository, and the ROS–Isaac bridge uses one local Unix socket.

The asset fetcher accepts only AgiBotTech's official HTTPS repository at reviewed commit `77f43eb`, disables submodules and Git hooks, verifies the origin and revision, and deletes a failed download before it can be imported. It only receives public model files; no upload or outward copy of local robot data is performed.

ROS and Isaac deliberately run in different processes and Python environments. ROS 2 Humble uses its system Python; Isaac uses the Python environment supplied with its simulator stack. Neither process imports the other framework. Their only shared contract is newline-delimited JSON over `/tmp/hrs_x2_recovery.sock`, so sourcing ROS cannot replace Isaac's Python dependencies and Isaac cannot pollute the ROS overlay.

## What is validated

| Item | Status |
| --- | --- |
| External Isaac Lab 3.0 task loads against the local API | Validated without starting physics |
| Official X2 download and repo-local URDF→USD conversion | Implemented; download unavailable in the execution sandbox |
| PPO training and strict five-episode Isaac evaluation | Implemented; not run because the sandbox exposes no CUDA device or X2 meshes |
| Reduced-order training experiment | Run; checkpoint and reward plot committed |
| Reduced-order five-episode evaluation | 5/5; explicitly not a rigid-body or hardware claim |
| ROS build, launch, acceptance/busy behavior, telemetry, success and timeout failure | Validated |
| ROS connection to the Isaac policy process | Implemented through local IPC; socket execution blocked by the sandbox |

See [the validation record](reports/validation.md) for commands and observed outputs and [the evidence map](docs/evidence.md) for the research behind each design choice.

## Repository layout

```text
isaaclab_ext/x2_recovery_isaac/   external Isaac Lab task and PPO config
src/x2_recovery_ros/              ROS 2 package and CPU test harness
scripts/                          asset, training, evaluation and launch entry points
reports/                          measured outputs and validation record
docs/                             requirement and research traceability
assets/isaac/                     generated USD, ignored by Git
models/agibot_x2_urdf/            official upstream checkout, ignored by Git
build install log logs/           local outputs, ignored by Git
```

## Setup

The main path targets Ubuntu, ROS 2 Humble, Isaac Sim 6.0 and Isaac Lab 3.0. The tested local checkout reports `3.0.0-beta2.patch1`. Use the Python environment in which Isaac Lab and RSL-RL are installed.

Validation used Ubuntu 22.04.5, ROS 2 Humble, Python 3.10.12 and an AMD Ryzen AI 9 HX 370 (12 cores/24 threads). Isaac Sim 6.0.1, Isaac Lab 3.0.0 and RSL-RL 5.0.1 were available for API checks, but the process could not access an NVIDIA/CUDA device. The reduced experiment therefore ran on CPU; no GPU model or training time is claimed.

```bash
git clone <submission-url> hrs_x2_take_home
cd hrs_x2_take_home

# Point at an existing Isaac Lab checkout. The scripts execute it read-only.
export ISAACLAB_ROOT=/absolute/path/to/IsaacLab

./scripts/fetch_agibot_model.sh
./scripts/import_x2_isaac.sh
```

The selected asset is AgiBot's official `X2_URDF-v1.3.0/x2_ultra_simple_collision.urdf`. AgiBot identifies v1.3.0 as the original flagship “X2 Ultra” model and v1.4.0 as the upgraded “X2 Ultra -N”; the take-home names X2 without the `-N` hardware identifier. Confirm the neck nameplate and switch models before hardware transfer if HRS uses the newer variant. Conversion merges fixed joints, keeps a floating base, enables self-collision, applies the Humanoid schema, and writes an instanceable USD to `assets/isaac/`. Mesh and inertial data are not hand-edited. Joint torque, speed and position limits originate in the official URDF and remain in the generated USD. Upstream assets are fetched at setup time under their Mulan PSL v2 license and are not committed here.

For the CPU harness only:

```bash
/usr/bin/python3 -m pip install -r requirements.txt
./scripts/run_training.sh --seed 7 --iterations 60 --population 80
./scripts/run_evaluation.sh
```

Verify the runtime boundary on a workstation that has both stacks:

```bash
ISAAC_PYTHON=/absolute/path/to/isaac/environment/bin/python \
  ./scripts/check_runtime_isolation.sh
```

## Isaac Lab environment

`HRS-X2-Recovery-v0` is a manager-based environment with 200 Hz PhysX simulation and 20 Hz policy decisions. Each reset places the pelvis 0.28 m above the floor and rotates it -90° about Y, then adds small pose, velocity and joint perturbations. In X2's documented FLU frame this points its forward/chest axis upward, keeping every episode on the back while preventing a single exact initial state.

The policy observes pelvis height, local linear and angular velocity, projected gravity, relative joint positions and velocities, binary foot contacts, and two previous actions. Uniform observation noise is enabled during training and disabled during evaluation.

The action is one bounded desired position per actuated joint. The articulation first contracts the hard position limits to 90%, then `scale=0.85` maps raw actions into the central 85% of those soft limits. An exponential moving average (`alpha=0.25`) filters the resulting target. One implicit PD actuator group uses `Kp=60`, `Kd=4`; effort and velocity limits are inherited from the USD. This compact design gives the policy full-body control while limiting violent target changes.

Episodes last 8 seconds. Time limit is the only training termination because ending at first upright contact would not teach the policy to remain standing. Evaluation applies a separate sustained success check.

Startup domain randomization is deliberately narrow:

- rigid-body mass × `[0.95, 1.05]`;
- pelvis centre of mass ±15 mm in X/Y and ±10 mm in Z;
- actuator stiffness and damping × `[0.90, 1.10]`;
- static friction `[0.70, 1.10]`, dynamic friction `[0.60, 1.00]`, restitution `[0.00, 0.05]`.

These ranges are a conservative first pass. They should be calibrated against a physical X2 before any hardware deployment.

## Reward

All terms are evaluated each 20 Hz policy step.

| Term | Weight | Purpose |
| --- | ---: | --- |
| Height-staged recovery progress | `+4.0` | Reward righting below 0.35 m, rising to 0.58 m, then upright standing |
| Upright exponential | `+2.0` | Align the pelvis vertical axis with gravity |
| Pelvis height exponential | `+1.5` | Reach the 0.68 m standing target |
| Both feet in contact | `+1.5` | Establish the required two-foot support |
| No other body support | `-1.0` per contacting body | Prevent kneeling, hand or torso-supported false positives |
| Standing still | `+2.0` | Reduce root linear and angular speed near the target |
| Action rate L2 | `-0.015` | Smooth desired positions |
| Joint velocity L2 | `-2e-4` | Discourage fast motion |
| Joint torque L2 | `-2e-6` | Discourage excessive effort |
| Soft joint-limit violation | `-0.20` | Keep motion away from mechanical limits |

The staged task reward follows HoST's height-dependent righting/rising/standing decomposition. Smooth actions, speed regularization, explicit contact checks and sustained stability follow the hardware concerns reported by HoST and FRASA. He et al.'s real-world getting-up study further motivates the simplified collision model and the explicit post-training collision inspection. Exact equations and numerical checks are in [the formula audit](docs/formula_audit.md); source connections are in [the evidence map](docs/evidence.md).

## PPO training and evaluation

The RSL-RL setup uses 2,048 environments, 32 steps per environment, 1,500 maximum iterations, ELU MLPs `[512, 256, 128]`, learning rate `3e-4`, clip `0.2`, discount `0.99`, GAE lambda `0.95`, five learning epochs and four mini-batches. The seed is 42.

```bash
./scripts/train_isaac.sh --num_envs 2048 --seed 42

# Play a checkpoint; Isaac Lab also exports policy.pt and policy.onnx.
./scripts/play_isaac.sh logs/rsl_rl/hrs_x2_recovery/<run>/model_<iteration>.pt

# Exactly five reproducible, perturbed back-lying starts: seeds 101–105.
./scripts/evaluate_isaac.sh \
  logs/rsl_rl/hrs_x2_recovery/<run>/exported/policy.pt
```

A recovery counts only after all checks hold continuously for 0.5 seconds:

- pelvis height ≥ 0.62 m;
- projected-gravity XY norm ≤ 0.15 and Z component ≤ -0.98;
- root linear speed ≤ 0.20 m/s;
- root angular speed ≤ 0.35 rad/s;
- each foot contact force ≥ 15 N;
- every other body contact force < 15 N.

The evaluator writes `reports/isaac_evaluation.json`. No such file is committed yet because the high-fidelity run was blocked; a missing result is preferable to fabricated evidence.

The committed CPU harness uses seeded cross-entropy search over a four-synergy, three-phase controller. The completed run used 60 iterations, 80 candidates per iteration, eight elites and two seeded rollouts per candidate. It produced [a checkpoint](src/x2_recovery_ros/artifacts/recovery_policy.npz), [a reward plot](reports/training_reward.png), and [a five-episode report](reports/evaluation.json). All fixed evaluation seeds 101–105 passed in 104–110 steps. This 5/5 result validates orchestration and metrics only; it is not evidence about X2 rigid-body dynamics.

## ROS 2

Build and launch both nodes:

```bash
./scripts/build_ros.sh
source install/setup.bash
ros2 launch x2_recovery_ros x2_recovery.launch.py
```

The recovery node returns from `/x2/start_recovery` before a worker starts. An atomic gate covers both pending and running states, so a second request is rejected. It publishes a transient-local, reliable status and timestamped joint states. The telemetry node logs every status change and one selected joint once per second.

```bash
ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
ros2 topic echo /x2/recovery_status std_msgs/msg/String \
  --qos-durability transient_local
ros2 topic echo /x2/joint_states sensor_msgs/msg/JointState
```

Run the complete local success, busy-request, telemetry and timeout smoke test with:

```bash
./scripts/validate_ros_runtime.sh
```

The validation script loads `config/fastdds_shm.xml`, which disables UDP/TCP and keeps DDS traffic in host-local shared memory.

The default launch uses the deterministic CPU harness so the ROS contract can be tested without a GPU. Timeout behavior is directly configurable:

```bash
ros2 launch x2_recovery_ros x2_recovery.launch.py \
  policy_mode:=zero timeout_sec:=0.5
```

For the Isaac policy, run the simulator process in the Isaac Python environment and ROS in the Humble environment. This avoids mixing incompatible Python runtimes:

```bash
# Terminal 1: Isaac environment
export ISAACLAB_ROOT=/absolute/path/to/IsaacLab
./scripts/serve_isaac_policy.sh /absolute/path/to/exported/policy.pt

# Terminal 2: ROS Humble environment
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch x2_recovery_ros x2_recovery.launch.py backend:=isaac_ipc

# Terminal 3
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
```

The local server owns the Isaac environment and policy. For each request it resets one episode, streams all simulator joint positions over `/tmp/hrs_x2_recovery.sock`, and returns strict success or timeout. The ROS node converts those events to the required topics. The socket is local, single-client and mode `0600`; it does not touch another robot project or expose a network port.

## Limits and next experiments

The main open item is an actual Isaac training curve and five-episode result. After that run, the first useful improvements are to inspect collision geometry at reset, tune the stand-height threshold from the imported model, plot each reward component, and run ablations for staged reward, contact penalties and randomization. Before hardware work, actuator gains, delay, friction, mass and centre-of-mass ranges must be identified from the X2; torque, thermal and self-collision safety need separate validation. No result in this repository is presented as proof of safe hardware transfer.

This is a complete standalone local Git repository with meaningful staged commits. No remote is configured. If a submission repository is requested later, preserve the local history with:

```bash
git remote add origin git@github.com:<account>/hrs_x2_take_home.git
git push -u origin main
```
