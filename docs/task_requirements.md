# Task requirement map

| PDF requirement | Implementation and evidence |
| --- | --- |
| Choose an X2 URDF and simulator | Official X2 Ultra v1.3.0 simplified-collision URDF; Isaac Lab 3.0/PhysX; fetch and conversion scripts |
| Floating base and flat floor | `X2_CFG` leaves the root free; `X2RecoverySceneCfg` supplies a repo-local 200 m × 200 m collision floor that covers the complete 4,096-environment grid |
| Start on the back without intersection | Collision-audited 0.190 m pelvis height, -90° Y rotation, zero initial velocity and narrow seeded jitter; 20,000 sampled starts retain ≥6.3 mm floor clearance; five live resets verify fresh frame observations |
| Respect joint and actuator limits | URDF limits preserved in USD; eight symmetric commands map into 31 absolute motor targets clipped to the 98% soft limits; simulator enforces official effort and speed limits |
| Observations, actions, reward and ending | `simple_cfg.py`, `synergy_action.py`, `mdp.py`, `docs/symmetric_v3.md` and `docs/simple_v2.md` |
| RL experiment and checkpoint/plot | Symmetric PPO with 3,000 environments; selected iteration-200 checkpoint has 19,296,000 transitions; checkpoint, exact config/source hashes, scalar CSV, reward plot and verified 122→8 TorchScript export are retained |
| Recovery Trigger service | `/x2/start_recovery`; callback accepts before timer dispatch |
| Reject a second request | `AttemptGate` covers pending and running states; CLI validation recorded |
| Status and joint-state topics | Required names and message types; simulator values and ROS timestamps |
| Configurable timeout | `timeout_sec` parameter and launch argument |
| Telemetry node | Logs status and configured joint at 1 Hz |
| Connect ROS to simulator | `isaac_policy_server.py` + mode-0600 local Unix IPC; runtime validation exercises service acceptance, busy rejection, joint states and a measured terminal state while keeping ROS and Isaac in separate Python processes |
| Build and one launch | `scripts/build_ros.sh`; `x2_recovery.launch.py` starts both nodes |
| Five simulation episodes | Selected symmetric_v3 checkpoint evaluated on seeds 101–105 from reproducible perturbed back-lying starts: **5/5**, with 8.62–8.90 s consecutive strict stance extending to the end of every episode; `reports/symmetric_v3_evaluation.json` |
| Upright, stable, both feet, no other support | Explicit force, height, tilt and speed predicate held for 0.5 s |
| Success/failure and ROS command record | Fresh ROS build plus CPU success/timeout paths and the real Isaac-policy **SUCCEEDED** path recorded in `reports/validation.md` and `reports/ros_isaac_symmetric_v3_validation.txt` |
| Meaningful development history | Separate scaffold, experiment, ROS, Isaac and evidence commits |

The HRS work is a nested standalone Git repository. Its assets, builds, logs and simulator socket remain inside the HRS workspace or `/tmp`; no existing Isaac/ROS robot repository is modified.
