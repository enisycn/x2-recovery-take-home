# Task requirement map

This file translates the supplied take-home PDF into checkable repository work.

| Requirement | Planned evidence |
| --- | --- |
| Choose an X2 URDF and simulator | Official AgiBot X2 Ultra v1.3.0; MuJoCo backend plus fetch script |
| Floating base on a flat floor | `MujocoRecoveryEnv`; reduced-order baseline mirrors base pose variables |
| Reset lying on its back, collision-safe | Backend reset checks and unit tests |
| Respect joint and actuator limits | Limits read from MuJoCo; normalized/clipped actions |
| Define observations, actions, reward, ending | README environment section and environment source |
| Run PPO or another RL algorithm | Seeded cross-entropy policy search; checkpoint and reward plot |
| Recovery service | `/x2/start_recovery`, `std_srvs/srv/Trigger` |
| Reject concurrent request | Non-blocking worker with a locked busy flag |
| Recovery status | `/x2/recovery_status`, `std_msgs/msg/String` |
| Live joint states | `/x2/joint_states`, `sensor_msgs/msg/JointState` |
| Configurable timeout | ROS parameter and launch configuration |
| Telemetry node | Status and selected-joint logging |
| One launch file | `x2_recovery.launch.py` |
| Five evaluation episodes | `reports/evaluation.json` and validation report |
| Upright, stable, two feet, no other support | Explicit success predicate and tests |
| Fresh colcon build and CLI checks | Recorded commands and outputs in `reports/validation.md` |
| Meaningful commit history | Separate scaffold, simulation, ROS, and validation commits |

## Scope decision

The high-fidelity target is MuJoCo with AgiBot's official MJCF/URDF assets. The current workstation has ROS 2 Humble but no MuJoCo Python package, no working GPU driver, and no outbound Git access. A reduced-order NumPy environment is therefore used for the runnable training experiment and ROS integration. Results from that backend are labelled as baseline results and are not presented as X2 hardware performance.

