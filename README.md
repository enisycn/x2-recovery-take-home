# HRS — AgiBot X2 recovery

Standalone Isaac Lab / PhysX PPO experiment and ROS 2 Humble recovery service. The task uses the official X2 Ultra v1.3.0 URDF with a floating base, self-collisions, imported joint/actuator limits and a flat floor.

**Validated result: 5/5 real Isaac/PhysX recoveries from back-lying.** The selected `symmetric_v3` PPO checkpoint reaches standing and holds every strict condition for **8.62–8.90 consecutive seconds**, remaining upright at the end of all five 10-second episodes. No reference reset, lifting force or root teleport is used during evaluation.

[Watch the successful GIF](reports/gifs_symmetric_v3/x2_final_policy_attempt.gif) · [Five-episode results](reports/symmetric_v3_evaluation.json) · [Checkpoint](reports/checkpoints/x2_symmetric_v3_model200.pt) · [Reward curve](reports/symmetric_v3_training_reward.png)

The selected checkpoint is iteration 200, trained from scratch with 3,000 parallel robots and 19,296,000 transitions; the training log records 7 min 48 s to that checkpoint. Earlier failed runs are retained as history, not included in the 5/5 result. [Current design](docs/symmetric_v3.md), [shared formulas](docs/simple_v2.md), [source/configuration provenance](reports/configs/symmetric_v3_model200/provenance.json) and [validation](reports/validation.md).

## Isolation and assets

Use the existing Isaac interpreter; no packages, drivers or settings are installed into the user's robot environment. HRS assets, ROS build, logs and code stay in this repository. Preserve the CPU allocation:

```bash
# Set this to your existing Isaac interpreter; below is the tested local path.
export ISAAC_PYTHON=<USER_HOME>/miniconda3/envs/codex/bin/python
export HRS_CPUSET=<HOST_CPUSET>
export HRS_NICE=15
```

Tested stack: Isaac Lab `3.0.0-beta2.patch1`, Isaac Sim 6.0.1, RSL-RL 5.0.1, ROS 2 Humble and RTX 5080 Laptop GPU. Other installations may require API adaptation, especially the scalar-last XYZW tensor convention used here.

The official asset is pinned to AgiBotTech commit `60c5de582c523cd188f563819e62d34cfdc3d2d0`:

```bash
./scripts/fetch_agibot_model.sh
./scripts/import_x2_isaac.sh
```

Fetch only if assets are missing. The fetcher disables Git hooks/submodules and verifies the origin/revision. No downloaded repository scripts are executed. The launchers disable Kit telemetry/crash uploads and use a private network namespace where supported. The official model stays under its upstream Mulan PSL v2 license.

## Train and evaluate

```bash
./scripts/train_isaac.sh --phase symmetric_v3 --num_envs 3000 \
  --max_iterations 201 --seed 43 --device cuda:0

./scripts/evaluate_isaac.sh reports/checkpoints/x2_symmetric_v3_model200.pt \
  --environment symmetric_v3 --device cuda:0 \
  --output reports/symmetric_v3_evaluation.json

./scripts/render_isaac_gifs.sh --checkpoint reports/checkpoints/x2_symmetric_v3_model200.pt \
  --environment symmetric_v3 --headless --device cuda:0 \
  --output_dir reports/gifs_symmetric_v3
```

`symmetric_v3` uses 122 continuous/contact/history inputs, `[512,256,128]` ELU actor/critic MLPs and eight symmetric movement commands mapped into 31 bounded joint-position targets. PPO clip is 0.2. PhysX runs at 200 Hz and the policy at 50 Hz. Half the training resets are true back-lying; auxiliary squat/sitting starts make up the other half. No lifting assistance is used. The old 1,148-input relative-action and 168-input independent-action checkpoints are incompatible with this policy contract.

The [formula/source table](docs/simple_v2.md) states which HumanUP formulas are used and which X2 choices are adaptations. The primary research is [HumanUP (RSS 2025)](https://www.roboticsproceedings.org/rss21/p063.html) and [HoST (RSS 2025)](https://www.roboticsproceedings.org/rss21/p064.html); this compact baseline is not claimed as a complete reproduction of either method.

Evaluation runs seeds 101–105 from zero-velocity, collision-audited back-lying starts, with no auxiliary reference or force. All conditions must overlap for at least 0.5 s: pelvis ≥0.58 m, gravity XY norm ≤0.15 and Z ≤−0.98, linear/angular speed ≤0.25 m/s / 0.35 rad/s, both feet ≥15 N against the floor, and every other body <15 N. The full episode is observed to record whether the robot remains standing. A reset cannot count as recovery.

Each run retains environment/agent YAML. Evaluation exports a verified tensor-only TorchScript policy in `reports/exported_symmetric_v3/`. Keep the checkpoint, config and environment option together.

## ROS 2

ROS and Isaac use separate processes/interpreters connected by a local mode-0600 Unix socket. No ROS packages are imported into Isaac's Python.

```bash
./scripts/build_ros.sh

# Terminal 1: Isaac
./scripts/serve_isaac_policy.sh reports/exported_symmetric_v3/policy.pt \
  --environment symmetric_v3 --device cuda:0

# Terminal 2: ROS
export ROS_DOMAIN_ID=94
export FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/config/fastdds_shm.xml"
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch x2_recovery_ros x2_recovery.launch.py \
  backend:=isaac_ipc timeout_sec:=10.0

# Terminal 3: ROS
export ROS_DOMAIN_ID=94
export FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/config/fastdds_shm.xml"
ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
ros2 topic echo /x2/recovery_status std_msgs/msg/String --qos-durability transient_local
ros2 topic echo /x2/joint_states sensor_msgs/msg/JointState
```

The service acknowledges acceptance before execution. Pending/running attempts reject a second request. Joint states come from the simulator and telemetry logs a selected joint at 1 Hz. The [real Isaac integration check](reports/ros_isaac_symmetric_v3_validation.txt) accepted the request, rejected a concurrent request, received 84 samples of 31 joints and returned `SUCCEEDED`. Failure and timeout are reported explicitly. `timeout_sec` is configurable.

The default launch backend is a **reduced CPU harness** for interface testing. Its success does not establish real X2 recovery. To exercise acceptance, busy rejection, telemetry and timeout without a GPU:

```bash
./scripts/validate_ros_runtime.sh
```

## Evidence and submission

- [Requirements](docs/task_requirements.md), [literature decisions](docs/literature_decision_record.md), [validation](reports/validation.md).
- [Live 200 Hz preflight](reports/simple_v2_preflight_0.005.json) and [400 Hz comparison](reports/simple_v2_preflight_0.0025.json): finite input/output, selective sensor reset, live control authority and pre-reset capture. A fixed standing pose lasts only 0.70 / 0.80 s; these are reachability tests, not learned recovery.
- Formula/contact and ROS unit checks: `PYTHONPATH=isaaclab_ext:src/x2_recovery_ros "$ISAAC_PYTHON" -m pytest -q isaaclab_ext/test/test_reward_formulas.py src/x2_recovery_ros/test`.

This is a separate Git repository with staged development history and no configured remote. No result here is a hardware deployment claim.

The submission ZIP also includes `history.bundle`. After extracting it, recover the full local history with `git clone hrs_x2_take_home/history.bundle hrs_x2_with_history`. Assets remain pinned external downloads; no files from the user's other robot project are included.
