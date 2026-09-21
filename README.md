# AgiBot X2 ground recovery

This repository implements the HRS take-home as a self-contained external Isaac Lab task and ROS 2 Humble package. The main path is an AgiBot X2 Ultra v1.3.0 floating-base model in Isaac Lab/PhysX, trained with RSL-RL PPO. A small NumPy model remains as a clearly labelled CPU test harness for training plumbing and ROS interface validation.

The repository never installs files into an existing robot workspace. Isaac Lab is invoked through an explicitly selected, existing Python interpreter, the AgiBot source model and converted USD stay under this repository, ROS builds into this repository, and the ROS–Isaac bridge uses one local Unix socket.

The asset fetcher accepts only AgiBotTech's official HTTPS repository at reviewed commit `60c5de582c523cd188f563819e62d34cfdc3d2d0`, disables submodules and Git hooks, verifies the origin and revision, and deletes a failed download before it can be imported. It only receives public model files; no upload or outward copy of local robot data is performed.

ROS and Isaac deliberately run in different processes and Python environments. ROS 2 Humble uses its system Python; Isaac uses the Python environment supplied with its simulator stack. Neither process imports the other framework. Their only shared contract is newline-delimited JSON over `/tmp/hrs_x2_recovery.sock`, so sourcing ROS cannot replace Isaac's Python dependencies and Isaac cannot pollute the ROS overlay.

## What is validated

| Item | Status |
| --- | --- |
| External Isaac Lab 3.0 task and 32 exact-link contact sensors | Validated in live PhysX runs |
| Official X2 download and repo-local URDF→USD conversion | Validated at pinned upstream commit; 39 links, 38 joints, 50 collision elements and 49 mesh references |
| PPO training and strict five-episode Isaac evaluation | Real X2 training completed; measured checkpoint, plot and five-episode report are included below |
| Isaac Sim/PhysX startup and stepping | Validated without changing the existing environment |
| Reduced-order training experiment | Run; checkpoint and reward plot committed |
| Reduced-order five-episode evaluation | 5/5; explicitly not a rigid-body or hardware claim |
| ROS build, launch, acceptance/busy behavior, telemetry, success and timeout failure | Validated |
| ROS connection to the Isaac policy process | Implemented through mode-0600 local IPC and the exported Isaac policy |

See [the validation record](reports/validation.md) for commands and observed outputs, [the root-cause record](docs/root_cause.md) for environment findings, and [the evidence map](docs/evidence.md) for the research behind each design choice.

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

Validation used Ubuntu 22.04.5, ROS 2 Humble, Python 3.10.12, an AMD Ryzen AI 9 HX 370 (12 cores/24 threads), Isaac Sim 6.0.1, Isaac Lab 3.0.0 and RSL-RL 5.0.1. The real-X2 PPO run used the available CUDA device with 256 parallel environments. Every HRS process ran at nice level 15 on the explicitly selected CPU set `<HOST_CPUSET>`; no system, driver, Conda, ROS, Isaac, or existing robot-project setting was modified.

```bash
git clone <submission-url> hrs_x2_take_home
cd hrs_x2_take_home

# Select an existing Isaac environment. The scripts do not modify it.
export ISAAC_PYTHON=/absolute/path/to/isaac/environment/bin/python

./scripts/fetch_agibot_model.sh
./scripts/import_x2_isaac.sh
```

If browser policy blocks GitHub but the official ZIP has been downloaded manually from AgiBot's [SDK page](https://x2-aimdk.agibot.com/zh-cn/latest/get_sdk/index.html), stage it without executing archive content:

```bash
./scripts/stage_agibot_archive.py /absolute/path/to/agibot_x2_urdf.zip
./scripts/import_x2_isaac.sh
```

The staging script rejects path traversal, symlinks, oversized archives, an unexpected robot identity, incomplete kinematics, and missing mesh references. It records archive and URDF SHA-256 hashes before the importer accepts the model.

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

The installed Isaac/PhysX stack can be checked on CPU without changing that environment:

```bash
"$ISAAC_PYTHON" scripts/probe_isaac_cpu.py --headless --device cpu
```

## Isaac Lab environment

`HRS-X2-Recovery-v0` is a manager-based environment with 200 Hz PhysX simulation and 20 Hz policy decisions. Each supine reset places the pelvis 0.190 m above the floor and rotates it -90° about Y with the installed runtime's scalar-last `XYZW=(0,-0.7071,0,0.7071)` convention. In X2's documented FLU frame this points its forward/chest axis upward. A collision-hull audit of all 50 official collision elements measures 0.18030 m from pelvis to the lowest supine point; 20,000 samples over the bounded reset jitter retain at least 6.3 mm of floor clearance. Root and joint velocities start at zero, so the robot begins resting on its back instead of falling into the floor. The reproducible calculation is in `scripts/audit_x2_geometry.py` and `reports/x2_geometry_audit.json`.

The floor is one repo-local 200 m × 200 m kinematic cuboid shared by all clones. This size covers the complete 2,048-environment grid at 2.5 m spacing. An earlier 20 m floor covered only the centre of a 512-environment run and allowed most robots to fall below the scene; those checkpoints are diagnostic only and are excluded from the final result.

The installed Isaac Lab 3.0 beta backend leaves the same-timestamp projected-gravity cache valid after a root-pose write. The task therefore uses a local reset wrapper that invalidates gravity, heading and body-frame root-velocity buffers after the standard reset. A five-seed live probe verifies that the data buffer and the policy observation are identical immediately after every reset, velocity is zero, and the supine projected-gravity Z component remains within ±0.0047. The record is in `reports/x2_reset_probe.json`; no simulator installation file is patched.

The policy observes pelvis height, local linear and angular velocity, projected gravity, relative joint positions and velocities, binary foot contacts, and two previous actions. Uniform observation noise is enabled during training and disabled during evaluation.

The action is one bounded relative position per actuated joint: `q_target=clip(q_current+0.25*clip(action,-1,1), soft_limits)`. This is HoST's position-increment equation at its final `β=0.25` bound and is consistent with FRASA's integrated desired-joint command. A zero-initialized policy holds its current pose; it does not command the midpoint of every asymmetric joint range. The articulation contracts hard position limits once to 98%, keeping the official zero-endpoint knees within about 0.024 rad of full extension. Implicit PD gains reflect joint load: leg/waist `Kp=120, Kd=6`, ankle `80/5`, arm `40/3`, and wrist/head `15/1.5`. Effort and velocity limits are inherited from the USD.

A separate PhysX reachability probe writes the collision-audited straight pose using the tensor API's scalar-last quaternion convention. It measured a 0.67465 m pelvis height, gravity `[0.00539, 0.00003, -0.99999]` in the body frame, 202/207 N on the feet, zero other-body contact, and 0.75 s of consecutive strict stance before the uncontrolled open-loop pose tipped forward. This verifies that the model, frame convention, contact sensors, floor placement and action limits admit the required stance; the learned policy must supply active balance. The complete trajectory is in `reports/x2_standing_probe.json`.

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
| `exp(pelvis height)-1` | `+5.0` | HumanUP Stage-I dense rise objective |
| `exp(head height)-1` | `+5.0` | HumanUP Stage-I whole-body rise objective |
| Positive pelvis vertical velocity | `+1.0` | Continuous form of HumanUP's height-increase indicator |
| `exp(-projected gravity z)` | `+0.25` | HumanUP upright objective |
| Signed upright target | `+2.5` | HoST task-orientation weight; reject inverted high poses |
| Both feet near standing | `+2.5` | Establish the required two-foot support |
| Other support near standing | `-2.0` per body | Permit transitional pushes, then reject hand, knee or torso support |
| HoST post-task angular speed | `+10.0` | Stabilize rotation above 0.62 m |
| HoST post-task planar speed | `+10.0` | Stabilize translation above 0.62 m |
| HoST post-task orientation | `+10.0` | Make the final pelvis vertical |
| HoST post-task target height | `+10.0` | Hold the 0.68 m standing target |
| Action-rate L2 | `-0.10` | Smooth desired positions |
| Joint acceleration / velocity / torque L2 | `-1e-7 / -1e-4 / -6e-7` | HumanUP weak discovery regularization |
| Root angular / linear speed L2 | `-0.10 / -0.10` | Bound body motion |
| Soft joint-limit violation | `-1.0` | Keep motion inside imported limits |

Training also uses two published exploration ideas. HumanUP's Stage-I standing-pose mixture starts at 50% and reaches zero after 16,000 policy steps. HoST's upward pelvis-force curriculum starts at 200 N and reaches zero after 24,000 policy steps. Both are disabled in the play/evaluation configuration, and the last 450 iterations of the measured 1,200-iteration run are fully unassisted. Exact equations, adaptations and numerical checks are in [the formula audit](docs/formula_audit.md); every source-to-code connection is in [the evidence map](docs/evidence.md).

## PPO training and evaluation

The reusable RSL-RL configuration supports 2,048 environments and 1,500 iterations on a larger GPU. The measured work used 256 environments: one 500-iteration base run, then two 500-iteration branches from its checkpoint. The height-only branch is retained as a measured failed ablation; the orientation-gated branch is the final policy. Each iteration collected 32 steps per environment. The final policy's training lineage contains 8.192 million simulated policy steps; all three experiments total 12.288 million. Every run used seed 42, ELU MLPs `[512, 256, 128]`, learning rate `3e-4`, clip `0.2`, discount `0.99`, GAE lambda `0.95`, five learning epochs and four mini-batches.

```bash
./scripts/train_isaac.sh --num_envs 2048 --seed 42 --device cuda:0

# Device-isolated or CPU-only machine: slower, but uses the same X2 task.
./scripts/train_isaac.sh --num_envs 64 --seed 42 --device cpu

# Run the deterministic five-episode viewer without --headless.
./scripts/play_isaac.sh logs/rsl_rl/hrs_x2_recovery/<run>/model_<iteration>.pt

# Exactly five reproducible, perturbed back-lying starts: seeds 101–105.
./scripts/evaluate_isaac.sh \
  logs/rsl_rl/hrs_x2_recovery/<run>/model_<iteration>.pt
```

A recovery counts only after all checks hold continuously for 0.5 seconds:

- pelvis height ≥ 0.62 m;
- projected-gravity XY norm ≤ 0.15 and Z component ≤ -0.98;
- root linear speed ≤ 0.20 m/s;
- root angular speed ≤ 0.35 rad/s;
- each foot contact force ≥ 15 N;
- every other body contact force < 15 N.

The evaluator writes `reports/isaac_evaluation.json` and exports TorchScript and ONNX policies under `reports/exported/`. The first 500-iteration report is retained as `reports/isaac_evaluation_500.json`; it records the measured local optimum that motivated the second phase.

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
# Terminal 1: existing Isaac environment
export ISAAC_PYTHON=/absolute/path/to/isaac/environment/bin/python
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

The simulator results establish reproducible software behavior only. The next useful experiments are a two-stage pose curriculum, a short reference-motion seed, and ablations for the staged reward, height gate and domain randomization. Before hardware work, actuator gains, delay, friction, mass and centre-of-mass ranges must be identified from the X2; torque, thermal and self-collision safety need separate validation. No result in this repository is presented as proof of safe hardware transfer.

This is a complete standalone local Git repository with meaningful staged commits. No remote is configured. If a submission repository is requested later, preserve the local history with:

```bash
git remote add origin git@github.com:<account>/hrs_x2_take_home.git
git push -u origin main
```
