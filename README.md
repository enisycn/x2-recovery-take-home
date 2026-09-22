# HRS — AgiBot X2 recovery

Isaac Lab / PhysX PPO recovery and a ROS 2 Humble interface using the official X2 Ultra v1.3.0 URDF. The robot has a floating base, self-collisions, imported joint/actuator limits and a flat floor.

**Result: 5/5 recoveries, with relaxed arms.** The selected `relaxed_v4` checkpoint remains in strict two-foot stance for **8.54–8.94 consecutive seconds** through the end of every 10-second episode. All five also maintain the relaxed-arm criterion for the entire final two seconds. There is no lift force, root teleport or reference reset in the final training experiment, evaluation or ROS runtime.

[GIF](reports/gifs_relaxed_v4/x2_final_policy_attempt.gif) · [Five-episode results](reports/relaxed_v4_evaluation.json) · [Checkpoint](reports/checkpoints/x2_relaxed_v4_model450.pt) · [Reward curve](reports/relaxed_v4_training_reward.png) · [PDF audit](docs/pdf_compliance_audit.md)

GitHub submission is **pending**. Meaningful local commit history is preserved; no remote or upload exists. The PDF's request for ongoing pushes from the beginning was not met because GitHub work was deferred. The local ZIP is a prepared deliverable, not a submitted GitHub repository.

## Setup and isolation

Tested: Ubuntu 22.04.5, Isaac Lab `3.0.0-beta2.patch1`, Isaac Sim 6.0.1, RSL-RL 5.0.1, ROS 2 Humble, RTX 5080 Laptop GPU. Isaac uses the existing Python 3.12 environment; ROS uses system Python 3.10 in a separate process. No packages, drivers or CPU-isolation settings were changed in the user's robot environment.

```bash
export ISAAC_PYTHON=<USER_HOME>/miniconda3/envs/codex/bin/python
export HRS_CPUSET=<HOST_CPUSET>
export HRS_NICE=15
```

HRS code, assets, builds and logs stay in this repository. The official URDF is pinned to AgiBotTech commit `60c5de582c523cd188f563819e62d34cfdc3d2d0`, under its upstream Mulan PSL v2 license. Fetch and import only if local assets are missing:

```bash
./scripts/fetch_agibot_model.sh
./scripts/import_x2_isaac.sh
```

The fetcher disables hooks/submodules and verifies origin/revision; downloaded scripts are not executed. Isaac launchers disable Kit telemetry/crash uploads and use a private network namespace where supported. The full floating articulation and official mass/limits are retained; conversion merges fixed links. The simulator convention here is scalar-last XYZW. Other Isaac versions may need API adaptation.

## Environment and rewards

The policy receives **122 inputs**: pelvis height; base-frame linear/angular velocity; projected gravity; 31 joint positions and velocities; two foot and 32 body contact indicators; two previous 8-value actions. Eight symmetric commands map into the 31 absolute joint-position targets using `centre + span*tanh(action)`, clipped to imported soft limits. This action-space reduction is a disclosed simplification. Remaining joints hold neutral targets. [Mapping and base design](docs/symmetric_v3.md).

Every `relaxed_v4` reset starts supine at pelvis height 0.190 m, with zero velocity and small seeded pose perturbations. Collision geometry was audited for floor clearance. PhysX runs at 200 Hz; policy/PD targets at 50 Hz. Episode timeout is 10 s; nonfinite states, root height outside [0,1.2] m or root linear speed over 2.5 m/s terminate the episode. Success does not end training early. There is no noise, domain randomization or external assistance in this nominal experiment.

Weights below multiply the reward terms; Isaac's RewardManager also multiplies by the 0.02-s policy timestep. `g_z` is projected gravity Z and `z` is pelvis height.

| Term | Weight | Purpose / definition |
| --- | ---: | --- |
| Signed height | 40 | `clip(z/.68,0,1) * clip((1-g_z)/2,0,1)` discourages inverted high poses |
| Head height | 5 | Clipped exponential head-height progress, target 1.2 m |
| Upright | 5 | `exp(-g_z)` |
| Both feet | 5 | Two current ground contacts, gated by height/orientation |
| Other ground support | -1 | Penalizes non-foot support near standing height |
| Balance | 10 | Low root linear/angular velocity near upright standing |
| Strict stance | 20 | Full recovery predicate described below |
| Relaxed arms | 40 | Supported-upright gate times `exp(-mean(arm error²)/2)`; shoulder pitch target 0, elbow target -0.15 rad |
| Action change | -0.005 | Squared successive action difference |
| Torque | -1e-6 | Squared joint torque |
| Joint limits | -1 | Soft-limit violation |
| Safety termination | -10 | Failure termination, excluding timeout |

[Exact shared formulas](docs/simple_v2.md), [arm reward and trials](docs/relaxed_v4.md). The recovery design draws on [HumanUP, RSS 2025](https://www.roboticsproceedings.org/rss21/p063.html), the post-task upper-body posture objective on [HoST, Table VI(d)](https://arxiv.org/html/2502.08378v1), and symmetry reduction on [FRASA](https://arxiv.org/abs/2410.08655v3). X2-specific weights, targets and gates are adaptations, not a full reproduction of those systems.

## Train, evaluate and render

Actor/critic: ELU MLPs [512,256,128], normalized observations, 8 outputs. PPO: clipping 0.2, gamma 0.99, GAE lambda 0.95, 5 epochs, 4 minibatches, 3000 environments and 32 steps/update. The final stage uses seed 47, fixed learning rate 1e-4, initial action std 0.10 and entropy 0.001. [Exact configurations and lineage](reports/configs/relaxed_v4_model450/provenance.json).

The selected iteration-450 checkpoint follows exploratory pretraining and balance refinement. Those earlier stages used 50% auxiliary squat/standing starts and are explicitly distinguished from the **100% supine final experiment**. Parent checkpoints and their YAML/source hashes are included. The observation normalizer has seen 43,488,000 samples across the selected lineage; iteration numbers resume across stages.

Reproduce the final 51-update experiment from its supplied parent:

```bash
./scripts/train_isaac.sh --phase relaxed_v4 --num_envs 3000 \
  --max_iterations 51 --seed 47 --device cuda:0 \
  --checkpoint reports/checkpoints/x2_relaxed_v4_parent_model400.pt \
  --reset_optimizer --action_std_override 0.10 \
  --learning_rate_override 0.0001 --learning_schedule fixed --entropy_coef 0.001

./scripts/evaluate_isaac.sh reports/checkpoints/x2_relaxed_v4_model450.pt \
  --environment relaxed_v4 --device cuda:0 --output reports/relaxed_v4_evaluation.json

./scripts/render_isaac_gifs.sh --checkpoint reports/checkpoints/x2_relaxed_v4_model450.pt \
  --environment relaxed_v4 --headless --device cuda:0 --output_dir reports/gifs_relaxed_v4
```

To run a new experiment from scratch, omit checkpoint/optimizer/std overrides and set a suitable iteration budget. This is not claimed to reproduce the warm-start result. The final reward curve covers iterations 400–450 of the final stage (resumed training; early values include episode-statistic warmup). The [auxiliary pretraining curve](reports/relaxed_v4_pretraining_reward.png) covers iterations 0–300; [earlier stages and failed trials](docs/relaxed_v4.md) are reported separately. Evaluation exports a tensor-only TorchScript model, verified against the actor with maximum error 0.0.

Success requires pelvis >=0.58 m, gravity XY norm <=0.15 and Z <=-0.98, root linear speed <=0.25 m/s, angular speed <=0.35 rad/s, each foot ground force >=15 N and every other body's ground force <15 N, continuously for 0.5 s. Evaluation observes the whole episode and captures terminal state before automatic reset. The extra arm check requires all four target errors <=0.30 rad while strict stance holds. Seeds are 101–105. Current filtered ground forces exclude self-collisions and stale history.

Earlier failures included inverted height exploitation, reset/contact measurement defects and forward-arm posture plateaus. Rewards alone did not make every checkpoint better: fresh iteration 250 failed recovery, whereas the selected final model passes both criteria. Failed reports remain in `reports/arm_posture_trials/`; the former successful forward-arm v3 checkpoint is preserved. Further work would test wider initial states, domain randomization and hardware constraints. Five nominal simulations do not establish general robustness or hardware readiness.

## ROS 2

```bash
./scripts/build_ros.sh

# Terminal 1: Isaac server
./scripts/serve_isaac_policy.sh reports/exported_relaxed_v4/policy.pt \
  --environment relaxed_v4 --device cuda:0

# Terminal 2: recovery and telemetry nodes
export ROS_DOMAIN_ID=94
export FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/config/fastdds_shm.xml"
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch x2_recovery_ros x2_recovery.launch.py backend:=isaac_ipc timeout_sec:=10.0

# Terminal 3: source ROS and set the same domain/profile as Terminal 2
ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
ros2 topic echo /x2/recovery_status std_msgs/msg/String --qos-durability transient_local
ros2 topic echo /x2/joint_states sensor_msgs/msg/JointState
```

The recovery node returns acceptance before timer-dispatched execution, rejects concurrent requests and publishes IDLE/RUNNING/SUCCEEDED/FAILED plus 31 simulator joint positions and timestamps. A mode-0600 local Unix socket connects it to Isaac. The telemetry node logs status and one joint at 1 Hz. The v4 server waits for both strict stance and relaxed arms for 0.5 s before success. Timeout is configurable at launch or with `ros2 param set /x2_recovery timeout_sec 0.2` before an attempt.

[Fresh build](reports/ros_fresh_build_v4.txt) and [actual Isaac integration record](reports/ros_isaac_relaxed_v4_validation.txt) verify literal CLI success, immediate busy rejection, live telemetry and a 0.2-s timeout producing FAILED. Re-run with:

```bash
./scripts/validate_ros_isaac_runtime.sh reports/exported_relaxed_v4/policy.pt \
  --environment relaxed_v4 --device cuda:0
PYTHONPATH=isaaclab_ext:src/x2_recovery_ros "$ISAAC_PYTHON" -m pytest -q \
  isaaclab_ext/test/test_reward_formulas.py src/x2_recovery_ros/test
```

The default launch backend is a reduced CPU interface harness; select `backend:=isaac_ipc` for actual X2 physics. CPU harness tests are labelled separately. [Validation record](reports/validation.md), [PDF compliance](docs/pdf_compliance_audit.md), [requirement map](docs/task_requirements.md).

The submission ZIP includes `history.bundle`. Recover the genuine local history after extraction with `git clone hrs_x2_take_home/history.bundle hrs_x2_with_history`. Pinned external assets are not bundled; files from the user's other robot project are not included.
