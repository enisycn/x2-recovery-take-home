# AgiBot X2 ground recovery

**22 September diagnostic update:** The [no-training review](docs/no_training_review_2026-09-22.md)
supersedes the earlier claim that curriculum is the sole remaining problem.
It confirms stale contact history across resets, zero action increments at
upright heights of 0.62 m and above, and episode-boundary errors in the
diagnostic/imitation tools. The recorded 0/5 evaluation remains a failed
experiment; this repository is not a completed successful recovery controller.
No new training was run during that review.

This repository implements the HRS take-home as a self-contained external Isaac Lab task and ROS 2 Humble package. The main path is an AgiBot X2 Ultra v1.3.0 floating-base model in Isaac Lab/PhysX, trained with RSL-RL PPO. A small NumPy model remains as a clearly labelled CPU test harness for training plumbing and ROS interface validation.

The repository never installs files into an existing robot workspace. Isaac Lab is invoked through an explicitly selected, existing Python interpreter, the AgiBot source model and converted USD stay under this repository, ROS builds into this repository, and the ROS–Isaac bridge uses one local Unix socket.

The asset fetcher accepts only AgiBotTech's official HTTPS repository at reviewed commit `60c5de582c523cd188f563819e62d34cfdc3d2d0`, disables submodules and Git hooks, verifies the origin and revision, and deletes a failed download before it can be imported. It only receives public model files; no upload or outward copy of local robot data is performed. Isaac launchers disable crash uploads and telemetry and use a private network namespace when the host permits unprivileged namespaces.

ROS and Isaac deliberately run in different processes and Python environments. ROS 2 Humble uses its system Python; Isaac uses the Python environment supplied with its simulator stack. Neither process imports the other framework. Their only shared contract is newline-delimited JSON over `/tmp/hrs_x2_recovery.sock`, so sourcing ROS cannot replace Isaac's Python dependencies and Isaac cannot pollute the ROS overlay.

## What is validated

| Item | Status |
| --- | --- |
| External Isaac Lab 3.0 task and one 32-body contact view | Validated in live PhysX runs |
| Official X2 download and repo-local URDF→USD conversion | Validated at pinned upstream commit; 39 links, 38 joints, 50 collision elements and 49 mesh references |
| HumanUP-history PPO and strict five-episode Isaac evaluation | Completed: 54.5 M normalized observations in the selected checkpoint, checkpoint/plot/I-O audit committed, strict result 0/5 with measured curriculum-gap analysis |
| Isaac Sim/PhysX startup and stepping | Validated without changing the existing environment |
| Reduced-order training experiment | Run; checkpoint and reward plot committed |
| Reduced-order five-episode evaluation | 5/5; explicitly not a rigid-body or hardware claim |
| ROS build, launch, acceptance/busy behavior, telemetry, success and timeout failure | Validated |
| ROS connection to the Isaac policy process | Validated through mode-0600 local IPC: accepted request, busy rejection, 31-joint telemetry and measured `FAILED` timeout |

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

Validation used Ubuntu 22.04.5, ROS 2 Humble, Python 3.10.12, an AMD Ryzen AI 9 HX 370 (12 cores/24 threads), Isaac Sim 6.0.1, Isaac Lab 3.0.0, RSL-RL 5.0.1 and an NVIDIA GeForce RTX 5080 Laptop GPU. The final real-X2 PPO run used 3,000 parallel environments. Every HRS process ran at nice level 15 on the explicitly selected CPU set `<HOST_CPUSET>`; no system, driver, Conda, ROS, Isaac, or existing robot-project setting was modified.

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

`HRS-X2-Recovery-v0` is a manager-based environment with 100 Hz PhysX simulation and 20 Hz policy decisions, matching FRASA's published training/control ratio. Each supine reset places the pelvis 0.190 m above the floor and rotates it -90° about Y with the installed runtime's scalar-last `XYZW=(0,-0.7071,0,0.7071)` convention. In X2's documented FLU frame this points its forward/chest axis upward. A collision-hull audit of all 50 official collision elements measures 0.18030 m from pelvis to the lowest supine point; 20,000 samples over the bounded reset jitter retain at least 6.3 mm of floor clearance. Root and joint velocities start at zero, so the robot begins resting on its back instead of falling into the floor. The reproducible calculation is in `scripts/audit_x2_geometry.py` and `reports/x2_geometry_audit.json`.

The floor is one repo-local 200 m × 200 m kinematic cuboid shared by all clones. This size covers the complete 4,096-environment grid at 2.5 m spacing. An earlier 20 m floor covered only the centre of a 512-environment run and allowed most robots to fall below the scene; those checkpoints are diagnostic only and are excluded from the final result.

The installed Isaac Lab 3.0 beta backend leaves the same-timestamp projected-gravity cache valid after a root-pose write. The task therefore uses a local reset wrapper that invalidates gravity, heading and body-frame root-velocity buffers after the standard reset. A five-seed live probe verifies that the data buffer and the policy observation are identical immediately after every reset, velocity is zero, and the supine projected-gravity Z component remains within ±0.0047. The record is in `reports/x2_reset_probe.json`; no simulator installation file is patched.

The selected actor uses HumanUP's temporal proprioceptive representation. Current proprioception has 98 values: base angular velocity, roll/pitch, 31 relative joint positions, 31 joint velocities and 31 previous actions. A ten-step history contributes 980 values and a 70-value Stage-I extrinsics slot is zero under nominal dynamics, for 1,148 stored observations. HumanUP's encoder compresses the history to 20 values; the policy MLP therefore receives 118 values and outputs 31 joint commands. The critic receives all 1,148 values. This is temporal compression, not an autoencoder or diffusion model.

The action is one bounded relative position per actuated joint: `q_target=clip(q_at_policy_step+0.25*tanh(action), soft_limits)`. The target is computed once at 20 Hz and held for all five physics substeps. This fixes an audited bug where the same increment was previously re-added at every substep. Smooth `tanh` bounding also prevents PPO from hiding arbitrarily large network outputs behind a hard clip. A zero action holds the current pose. The articulation contracts hard position limits once to 98%, keeping the official zero-endpoint knees within about 0.024 rad of full extension. Implicit PD gains reflect joint load: leg/waist `Kp=120, Kd=6`, ankle `80/5`, arm `40/3`, and wrist/head `15/1.5`. Effort and velocity limits are inherited from the USD.

A separate PhysX reachability probe writes the collision-audited straight pose using the tensor API's scalar-last quaternion convention and holds the same joint target through the policy action map. Its first controlled sample measured a 0.67460 m pelvis height, gravity `[0.00368, 0.00013, -0.99999]` in the body frame, 195/215 N on the feet and zero other-body contact. It held every strict stance condition for 0.65 s before the fixed pose, which has no base-state feedback, tipped forward. This verifies that the model, frame convention, contact sensors, floor placement and action limits admit the required stance; the learned policy must supply active balance. The complete trajectory is in `reports/x2_standing_probe.json`.

HumanUP episodes last 10 seconds and end on timeout, root linear speed above 2.5 m/s, pelvis height outside `[0,1.2]` m, or a non-finite root state. Success is not a termination because ending at first upright contact would not teach the policy to remain standing. Evaluation applies a separate sustained success check.

The reported take-home experiment uses the official nominal mass/inertia, fixed floor friction and configured gains. Domain randomization is deliberately disabled until nominal recovery succeeds; its ranges would otherwise add an unmeasured sim-to-real objective to the required simulation experiment.

## Reward

All terms are evaluated each 20 Hz policy step.

| Term | Weight | Purpose |
| --- | ---: | --- |
| HumanUP Stage-I block | published 24 code-release weights | Discover contact-rich righting and rising with weak regularization |
| Upright-gated normalized pelvis height | `+40.0` | Preserve an ascent gradient without accepting an inverted bridge |
| Final sagittal-leg / remaining-joint pose | `+50.0` each | Extend the legs and align the whole body near standing |
| Signed upright target | `+2.5` | Reject inverted high poses |
| Both feet / non-foot support | `+10.0 / -10.0` | Load both feet and remove other support near standing |
| HoST post-task angular / planar speed | `+50.0` each | Brake motion above 0.58 m |
| HoST post-task orientation / target height | `+10.0` each | Hold the upright 0.68 m target |
| FRASA-style strict-state proximity | `+100.0` | Supply a dense gradient around all evaluation boundaries |
| Exact instantaneous stance | `+200.0` in rise phases | Make remaining in the complete target set valuable |

HumanUP and HoST motivate the nine-pose curriculum, which spans straight standing to near-supine coupled root/joint states. The faithful discovery environment mixes 10–20% reference starts with true supine states; focused rise phases deliberately hold one or a few adjacent stages. The selected checkpoint's final logged phase used only `reference_2`, which the five-episode audit identified as an incomplete curriculum rather than hiding it. All evaluation forces true supine starts and disables assistance, noise and domain randomization. Exact equations, source adaptations and numerical checks are in [the formula audit](docs/formula_audit.md) and [literature decision record](docs/literature_decision_record.md).

## PPO training and evaluation

The selected HumanUP-history checkpoint is PPO iteration 750 and its actor normalizer has seen 54,504,000 observations. Each update uses 3,000 parallel X2 environments and 24 steps per environment. The policy uses PPO clip `0.2`, gamma `0.99`, GAE lambda `0.95`, five epochs, four mini-batches, adaptive KL target `0.008`, learning rate `2e-4` in the base schedule, and `[512,256,128]` ELU actor/critic heads. Focused curriculum stages lower exploration standard deviation and entropy after the discovery stage. The committed reward plot covers the selected final logged segment, iterations 617–765.

```bash
./scripts/train_isaac.sh --phase humanup_rise \
  --max_iterations 400 --num_envs 3000 --seed 46 --device cuda:0

# Device-isolated or CPU-only machine: slower, but uses the same X2 task.
./scripts/train_isaac.sh --num_envs 64 --seed 42 --device cpu

# Run the deterministic five-episode viewer without --headless.
./scripts/play_isaac.sh reports/checkpoints/x2_humanup_selected_model750.pt \
  --output reports/isaac_evaluation_humanup.json --device cuda:0

# Exactly five reproducible, perturbed back-lying starts: seeds 101–105.
./scripts/evaluate_isaac.sh \
  reports/checkpoints/x2_humanup_selected_model750.pt \
  --output reports/isaac_evaluation_humanup.json --device cuda:0
```

A recovery counts only after all checks hold continuously for 0.5 seconds:

- pelvis height ≥ 0.58 m;
- projected-gravity XY norm ≤ 0.15 and Z component ≤ -0.98;
- root linear speed ≤ 0.25 m/s;
- root angular speed ≤ 0.35 rad/s;
- each foot contact force ≥ 15 N;
- every other body contact force < 15 N.

The selected HumanUP checkpoint's fixed-seed true-supine evaluation produced **0/5 successful recoveries**. Across seeds 101–105, maximum pelvis height remained 0.1894–0.1905 m, maximum upright score was 0.3297–0.4355, and the longest continuous two-foot contact was 2.35–5.00 s. Terminal non-foot floor force was 128–174 N. The exact per-seed states are in [the HumanUP Isaac evaluation](reports/isaac_evaluation_humanup.json). [The reference-2 I/O audit](reports/x2_humanup_model750_trajectory_ref2_audit.json) separately verifies 1,148 finite observations, 31 finite actions, a 20-value history latent, bounded targets, no soft-limit violation, and a 0.25 s strict-stance frontier from that curriculum state.

The result isolates a curriculum coverage problem. The official model can stand: the independent PhysX reachability probe held every strict condition for 0.65 s near 0.674 m. The same RMA policy reaches 0.597 m and satisfies the full strict predicate for 0.25 s from `reference_2`, but the selected final training phase sampled only that pose and did not include the lower bridge-to-supine transition. A 7.2-million-transition stage-2-to-4 experiment did not solve `reference_3` or `reference_4`. Stage-3-only training then used 14.4 million transitions and a further 21.6-million-transition extension; the deterministic stage-3 peak was 0.4210 m before the extension and 0.4145 m after it, with zero exact-stance reward in both. The longer checkpoint is therefore rejected. The evidence-backed correction is a dynamically valid discovery/retargeted trajectory followed by HumanUP-style imitation/refinement; weakening the success predicate or calling the reference-start behavior a ground recovery would be misleading.

The evaluator writes `reports/isaac_evaluation_humanup.json` and exports a verified 1,148-input/31-output TorchScript graph to `reports/exported_humanup/policy.pt`. The generic RSL-RL exporter cannot represent the custom history encoder, so the repository uses an explicit tensor-only deployment wrapper and checks the serialized graph against the source actor with zero observed error.

Two labelled GIFs make the distinction visual: [the final PPO attempt](reports/gifs/x2_final_policy_attempt.gif) is a real supine rollout and is explicitly marked unsuccessful; [the standing reachability probe](reports/gifs/x2_standing_reachability_only.gif) starts from an imposed straight stance and is explicitly marked as a model/action-path check rather than a recovery result. They are generated directly from Isaac by `scripts/render_isaac_gifs.sh`.

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
./scripts/serve_isaac_policy.sh reports/exported_humanup/policy.pt

# Terminal 2: ROS Humble environment
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch x2_recovery_ros x2_recovery.launch.py backend:=isaac_ipc

# Terminal 3
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
```

The local server owns the true-supine HumanUP Isaac environment and verified 1,148-input policy. For each request it resets one episode, streams all simulator joint positions over `/tmp/hrs_x2_recovery.sock`, and returns strict success or timeout. The ROS node converts those events to the required topics. The socket is local, single-client and mode `0600`; it does not touch another robot project or expose a network port.

## Limits and next experiments

The simulator results establish reproducible software behavior only. The measured policy is not ready for hardware: it scored 0/5 and still uses heavy non-foot support. The next useful experiment is HumanUP-style motion discovery followed by a dedicated imitation/refinement stage, then ablations for the height gate and reference curriculum. Before hardware work, actuator gains, delay, friction, mass and centre-of-mass ranges must be identified from the X2; torque, thermal and self-collision safety need separate validation. No result in this repository is presented as proof of safe hardware transfer.

This is a complete standalone local Git repository with meaningful staged commits. No remote is configured. If a submission repository is requested later, preserve the local history with:

```bash
git remote add origin git@github.com:<account>/hrs_x2_take_home.git
git push -u origin main
```
