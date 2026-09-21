# Validation record

Updated on 21 September 2026. Every result below was executed on the target workstation. No result from the reduced CPU harness is presented as an X2 rigid-body result.

## System and isolation

Validation used Ubuntu 22.04.5, ROS 2 Humble, Python 3.10.12, Isaac Sim 6.0.1, Isaac Lab 3.0.0, RSL-RL 5.0.1 and an NVIDIA GeForce RTX 5080 Laptop GPU. HRS processes ran at nice level 15 on CPU set `<HOST_CPUSET>`. `scripts/check_runtime_isolation.sh` confirmed that system Python imports ROS but not Isaac Lab, while the selected Isaac Python 3.12.13 imports Isaac Lab but not `rclpy`. The two processes exchange newline JSON through a local mode-0600 Unix socket.

The official AgiBot X2 Ultra v1.3.0 simplified-collision URDF is pinned at upstream commit `60c5de582c523cd188f563819e62d34cfdc3d2d0`. The imported floating-base model has 39 links, 38 joints, 50 collision elements, 49 mesh references and 41.966521 kg total mass. Geometry, live reset, curriculum and standing-reachability reports are committed alongside this record.

## PPO experiment

```bash
ISAAC_PYTHON=<USER_HOME>/miniconda3/envs/codex/bin/python \
HRS_CPUSET='<HOST_CPUSET>' HRS_NICE=15 \
./scripts/train_isaac.sh \
  --max_iterations 400 --num_envs 3000 --device cuda:0 \
  --seed 46 --run_name v5b_full_contact_balanced
```

Observed: exit code 0 after 400 optimizer iterations and 14,400,000 simulator transitions. The run sustained about 20,000–22,000 transitions/s, with roughly 1.5–1.8 s collection and 0.09 s PPO learning per iteration, and used about 5 GB of GPU memory. The 400 updates took about 12 minutes after startup. The final actor has 168 observations and 31 actions; its checkpoint contains finite actor/critic tensors and a 14.4-million-sample observation normalizer. The fixed-seed action I/O audit reports no non-finite observation, no target outside the imported soft limits, action standard deviation 0.332–0.467, and 6.7% of deterministic raw actions outside ±1 before the smooth `tanh` map (versus 87% in the discarded hard-clip branch). The committed outputs are:

- `reports/checkpoints/x2_recovery_model_399.pt`;
- `reports/isaac_training_reward.png` and its scalar CSV;
- `reports/x2_policy_io_audit.json`;
- TorchScript and ONNX exports in `reports/exported/`.

## Strict five-episode Isaac evaluation

```bash
./scripts/evaluate_isaac.sh \
  reports/checkpoints/x2_recovery_model_399.pt \
  --output reports/isaac_evaluation.json --device cuda:0
```

Evaluation disables standing starts, lift assistance, observation noise and domain randomization. Each seed begins in a reproducibly perturbed, collision-audited supine pose. Success requires 0.5 continuous seconds with pelvis height at least 0.62 m, projected-gravity XY norm at most 0.15 and Z at most -0.98, root speeds at most 0.20 m/s and 0.35 rad/s, at least 15 N on each foot, and less than 15 N on every other body.

| Episode | Seed | Result | Max pelvis (m) | Max upright | Longest two-foot contact (s) | Max strict stance (s) |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 101 | failed | 0.1896 | 0.9862 | 5.00 | 0.00 |
| 2 | 102 | failed | 0.1905 | 0.9858 | 4.80 | 0.00 |
| 3 | 103 | failed | 0.1894 | 0.9863 | 4.95 | 0.00 |
| 4 | 104 | failed | 0.1905 | 0.9864 | 5.00 | 0.00 |
| 5 | 105 | failed | 0.1902 | 0.9867 | 5.50 | 0.00 |

Final result: **0/5 successful recoveries**. The robot learned a reproducible feet-loaded sitting behavior: both feet remain in contact for 4.8–5.5 s and upright score reaches about 0.986. It never raises the pelvis above the roughly 0.190 m initial supine height; terminal pelvis is 0.0679–0.0688 m and terminal non-foot support is 274–282 N. The failure is therefore a stable seated local optimum rather than a frame, floor, contact-sensor, limit or success-detector error.

The investigation corrected five concrete causes before selecting this checkpoint: head-height/upward-velocity reward exploits, per-substep accumulation of a relative action, a curriculum measured in aggregate transitions, a hard action clip that produced saturated network output, and foot-only contact observations. Nine collision-audited reference poses, whole-body contacts, smooth `tanh` actions and weak bilateral/sagittal regularization improved the result from transient foot contact to stable two-foot sitting. Assisted follow-up runs reached high-pelvis states, but lost them as force vanished. HumanUP's evidence-supported next step is motion discovery followed by a dedicated imitation/refinement stage at a much larger update budget. Relaxing the predicate would conceal the failure and was not done; the assignment explicitly permits zero successes when explained.

## ROS 2 build and runtime

Fresh build:

```bash
./scripts/build_ros.sh
```

Observed: `1 package finished`.

The deterministic CPU harness validates both ROS terminal paths:

```bash
./scripts/validate_ros_runtime.sh
```

Observed: the first request returned `success=True, message='Recovery accepted'`; the concurrent request returned `success=False, message='Recovery already running'`; live joint states were received; the normal scenario reached `SUCCEEDED` in 109 steps; and `policy_mode:=zero timeout_sec:=0.5` reached `FAILED` in 10 steps.

The exported PPO policy was then exercised through the actual Isaac server:

```bash
ISAAC_PYTHON=<USER_HOME>/miniconda3/envs/codex/bin/python \
HRS_CPUSET='<HOST_CPUSET>' HRS_NICE=15 \
./scripts/validate_ros_isaac_runtime.sh reports/exported/policy.pt
```

Observed: the first `/x2/start_recovery` request was accepted; an immediate second request was rejected as busy; `/x2/joint_states` returned a timestamped sample with all 31 simulator joints; and the real Isaac episode reached `FAILED` after 60 policy steps with `timeout before 0.5 s strict stable stance`. The command ended with `ROS-Isaac runtime validation passed`.

## Automated checks

```bash
/usr/bin/python3 -m compileall -q isaaclab_ext src/x2_recovery_ros/x2_recovery_ros scripts
bash -n scripts/*.sh
PYTHONPATH="$PWD/src/x2_recovery_ros" /usr/bin/python3 -m pytest -q src/x2_recovery_ros/test
ISAAC_PYTHON=<USER_HOME>/miniconda3/envs/codex/bin/python ./scripts/test_isaac_formulas.sh
ISAAC_PYTHON=<USER_HOME>/miniconda3/envs/codex/bin/python ./scripts/check_runtime_isolation.sh
```

Observed: compilation and shell syntax passed; ROS tests reported `8 passed`; the independent formula suite reported `14 passed`; and runtime isolation passed. The repository also passes `git diff --check` and `git fsck` after the final artifact commit.
