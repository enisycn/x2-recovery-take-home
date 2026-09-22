# Validation record

Updated 22 September 2026. CPU-harness results are labelled separately and are never presented as X2 rigid-body evidence.

## Isolation and asset

Validation used Ubuntu 22.04.5, ROS 2 Humble, Python 3.10.12, Isaac Sim 6.0.1, Isaac Lab 3.0.0, RSL-RL 5.0.1 and an NVIDIA GeForce RTX 5080 Laptop GPU. Every HRS command ran at nice level 15 on CPU set `<HOST_CPUSET>`. System Python imports ROS but not Isaac; the selected Isaac Python imports Isaac but not `rclpy`. No external robot workspace, system package, runtime, driver or CPU-isolation setting was modified.

The official X2 Ultra v1.3.0 simplified-collision URDF is pinned at upstream commit `60c5de582c523cd188f563819e62d34cfdc3d2d0`. The imported floating-base model has 39 links, 38 joints, 50 collision elements, 49 mesh references and 41.966521 kg total mass. The fetcher disables hooks/submodules, verifies origin and revision, and never uploads local data. Isaac launchers disable telemetry/crash uploads and use a private network namespace when supported.

The geometry audit gives 0.1803007 m from pelvis to the lowest point in the supine pose. The 0.190 m reset and bounded jitter retain at least 6.3 mm clearance. Five live resets verify zero initial velocity and fresh projected-gravity observations.

## Contact and policy I/O

One recursive PhysX sensor resolves all 32 X2 rigid bodies in a fixed order. It filters the collision partner to `/World/ground`; all support rewards and success checks use `force_matrix_w_history`. This correction is necessary because the unfiltered net-force buffer also contains articulation self-collisions.

The selected actor has a 1,148-value observation contract: 98 current proprioceptive values, 70 zero Stage-I extrinsics and ten 98-value historical states. HumanUP's temporal encoder produces a 20-value latent; the actor MLP consumes 118 values and outputs 31 actions. The reference-2 audit measured finite observations/actions, finite checkpoint tensors, a normalizer count of 54,504,000, no target outside the imported soft joint limits, and no non-finite state.

The generic RSL-RL exporter bypassed the custom history encoder and was rejected after a `1148 x 118` shape error. `HumanUpInferenceModule` explicitly exports the trained normalizer, history encoder and actor MLP. The reloaded `reports/exported_humanup/policy.pt` has a verified `1148→31` interface and zero maximum error against the source graph on the export probe.

## PPO and curriculum evidence

The selected checkpoint is `reports/checkpoints/x2_humanup_selected_model750.pt` (SHA-256 `c81dc5382c2fdc477801d85dcbb36d63554edcddb08633d8b299da5645cc3aca`). It uses PPO clip 0.2, gamma 0.99, GAE lambda 0.95, five epochs, four mini-batches, a `[512,256,128]` ELU head and HumanUP's 10-step history encoder. The committed reward plot and CSV cover iterations 617–765 of the selected run.

The saved final-phase configuration has `reference_min_stage=reference_max_stage=2`. From that pose, the deterministic policy reaches 0.5972 m and satisfies the complete strict predicate for at most 0.25 s. Behavioral cloning is separately labelled: standing and shallow-rise demonstrations are strict successes, while the reference-2 data has zero strict successes and is marked partial. It reduced held-out action MSE but did not manufacture a successful deeper trajectory.

A 100-update, 7.2-million-transition follow-up mixed stages 2–4. It still failed deterministic reference 3 and 4 audits, peaking at 0.3944 m and 0.3145 m. A subsequent stage-3-only 14.4-million-transition run improved its deterministic peak to 0.4210 m without reaching the 0.58 m stance boundary. Extending that same focused stage by another 300 updates and 21.6 million transitions increased the normalizer count to 97,704,000 but reached only 0.4145 m and still produced zero exact-stance reward. The longer ablation was therefore rejected as the selected checkpoint. This confirms that more PPO updates on the same reset distribution do not by themselves close the lower-stage gap; tensor shape, frame, NaN, limits and premature success termination have been checked separately.

## Strict five-episode Isaac evaluation

Command:

```bash
ISAAC_PYTHON=<USER_HOME>/miniconda3/envs/codex/bin/python \
HRS_CPUSET='<HOST_CPUSET>' HRS_NICE=15 \
./scripts/evaluate_isaac.sh \
  reports/checkpoints/x2_humanup_selected_model750.pt \
  --output reports/isaac_evaluation_humanup.json --device cuda:0
```

Evaluation forces reference-start probability to zero and disables assistance, observation noise and domain randomization. Seeds 101–105 each begin in a fresh perturbed supine state. Success requires 0.5 continuous seconds with pelvis height at least 0.58 m, projected-gravity XY norm at most 0.15 and Z at most -0.98, root speed at most 0.25 m/s and 0.35 rad/s, at least 15 N floor force on both feet, and less than 15 N floor force on every other body.

Result: **0/5**. Maximum pelvis height was 0.1894–0.1905 m, maximum upright score 0.3297–0.4355, and maximum continuous two-foot contact 2.35–5.00 s. Terminal non-foot support was 128.39–173.74 N. Every seed failed because it never reached the minimum pelvis height. The per-seed report is `reports/isaac_evaluation_humanup.json`; no threshold was relaxed after observing the result.

The separate straight-standing reachability probe held every strict condition for 0.65 s at about 0.674 m. It proves the asset, floor, frames, contacts, limits and success predicate admit a valid stance. It does not count as recovery because it starts upright.

## ROS 2 validation

Commands:

```bash
./scripts/build_ros.sh
./scripts/validate_ros_runtime.sh
./scripts/validate_ros_isaac_runtime.sh reports/exported_humanup/policy.pt
```

Observed after a fresh build: the first `/x2/start_recovery` call returned `success=True`; a concurrent call returned `success=False` with `Recovery already running`; timestamped 31-joint telemetry was received; the deterministic CPU contract scenario reached `SUCCEEDED`; and `policy_mode:=zero timeout_sec:=0.5` reached `FAILED`. The ROS-to-Isaac check loaded the verified 1,148-input HumanUP TorchScript, used the mode-0600 Unix socket, accepted the first request, rejected the busy request, published live simulator joint states and reached `FAILED` after 60 policy steps at the configured 3.0 s timeout. ROS and Isaac stayed in their separate Python environments.

The CPU harness scores 5/5 only in its reduced NumPy model. That validates orchestration and metrics, not X2 dynamics or hardware readiness.
