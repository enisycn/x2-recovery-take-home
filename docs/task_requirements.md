# Task requirement map

| PDF requirement | Implementation and evidence |
| --- | --- |
| Choose an X2 URDF and simulator | Official X2 Ultra v1.3.0 simplified-collision URDF; Isaac Lab 3.0/PhysX; fetch and conversion scripts |
| Floating base and flat floor | `X2_CFG` leaves the root free; `X2RecoverySceneCfg` supplies a 20 m plane |
| Start on the back without intersection | 0.28 m pelvis height, +90° Y rotation and narrow reset jitter; visual check remains required after asset import |
| Respect joint and actuator limits | URDF limits preserved in USD; actions map to 90% soft limits; simulator enforces effort and speed limits |
| Observations, actions, reward and ending | `env_cfg.py`, `mdp.py`, README design and reward tables |
| RL experiment and checkpoint/plot | PPO experiment is configured; sandbox-blocked run is disclosed; reduced harness checkpoint and plot are committed |
| Recovery Trigger service | `/x2/start_recovery`; callback accepts before timer dispatch |
| Reject a second request | `AttemptGate` covers pending and running states; CLI validation recorded |
| Status and joint-state topics | Required names and message types; simulator values and ROS timestamps |
| Configurable timeout | `timeout_sec` parameter and launch argument |
| Telemetry node | Logs status and configured joint at 1 Hz |
| Connect ROS to simulator | `isaac_policy_server.py` + local Unix IPC; reduced simulator is the default test harness; `check_runtime_isolation.sh` verifies that ROS and Isaac cannot import each other's stack |
| Build and one launch | `scripts/build_ros.sh`; `x2_recovery.launch.py` starts both nodes |
| Five simulation episodes | Isaac evaluator uses seeds 101–105 for five reproducible perturbed back-lying starts; reduced fallback report is committed and labelled |
| Upright, stable, both feet, no other support | Explicit force, height, tilt and speed predicate held for 0.5 s |
| Success/failure and ROS command record | `reports/validation.md` |
| Meaningful development history | Separate scaffold, experiment, ROS, Isaac and evidence commits |

The HRS work is a nested standalone Git repository. Its assets, builds, logs and simulator socket remain inside the HRS workspace or `/tmp`; no existing Isaac/ROS robot repository is modified.
