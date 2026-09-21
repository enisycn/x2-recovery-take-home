# Task requirement map

| PDF requirement | Implementation and evidence |
| --- | --- |
| Choose an X2 URDF and simulator | Official X2 Ultra v1.3.0 simplified-collision URDF; Isaac Lab 3.0/PhysX; fetch and conversion scripts |
| Floating base and flat floor | `X2_CFG` leaves the root free; `X2RecoverySceneCfg` supplies a repo-local 20 m × 20 m collision floor |
| Start on the back without intersection | Collision-audited 0.190 m pelvis height, -90° Y rotation, zero initial velocity and narrow seeded jitter; 20,000 sampled starts retain ≥6.3 mm floor clearance; five live resets verify fresh frame observations |
| Respect joint and actuator limits | URDF limits preserved in USD; actions map across one 98% soft-limit range; simulator enforces official effort and speed limits |
| Observations, actions, reward and ending | `env_cfg.py`, `mdp.py`, README design and reward tables |
| RL experiment and checkpoint/plot | 500-iteration base plus two measured 500-iteration branches; final-policy lineage is 1,000 iterations/8.192 million steps, and all checkpoints plus the branched reward plot are retained |
| Recovery Trigger service | `/x2/start_recovery`; callback accepts before timer dispatch |
| Reject a second request | `AttemptGate` covers pending and running states; CLI validation recorded |
| Status and joint-state topics | Required names and message types; simulator values and ROS timestamps |
| Configurable timeout | `timeout_sec` parameter and launch argument |
| Telemetry node | Logs status and configured joint at 1 Hz |
| Connect ROS to simulator | `isaac_policy_server.py` + mode-0600 local Unix IPC; `check_runtime_isolation.sh` verifies that ROS and Isaac cannot import each other's stack |
| Build and one launch | `scripts/build_ros.sh`; `x2_recovery.launch.py` starts both nodes |
| Five simulation episodes | Isaac evaluator runs seeds 101–105 from reproducible perturbed back-lying starts and records per-check metrics; reduced fallback is separately labelled |
| Upright, stable, both feet, no other support | Explicit force, height, tilt and speed predicate held for 0.5 s |
| Success/failure and ROS command record | `reports/validation.md` |
| Meaningful development history | Separate scaffold, experiment, ROS, Isaac and evidence commits |

The HRS work is a nested standalone Git repository. Its assets, builds, logs and simulator socket remain inside the HRS workspace or `/tmp`; no existing Isaac/ROS robot repository is modified.
