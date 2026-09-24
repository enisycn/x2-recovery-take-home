# Assignment compliance map

Every required item is implemented and linked below. The selected result is a trained PPO policy, not the scripted baseline.

## Simulation and reinforcement learning

| Requirement | Status | Implementation and evidence |
| --- | --- | --- |
| Choose an AgiBot X2 URDF and simulator | Complete | Official X2 Ultra v1.3.0 at pinned upstream commit `60c5de5`; Isaac Lab 3.0 / Isaac Sim 6.0.1 / PhysX. `scripts/fetch_agibot_model.sh`, `scripts/import_x2_isaac.sh`. |
| Floating base on flat floor | Complete | Floating 31-joint articulation and repository-local 200 m collision floor. `x2_robot_cfg.py`, `simple_cfg.py`. |
| Supine, non-intersecting episode reset | Complete | Every final episode uses true-supine reset at measured pelvis height 0.190 m with zero velocity. Geometry and reset tests are in `test_reward_formulas.py`. |
| Collision and joint/actuator limits | Complete | Self-collision, recursive 32-body ground contact and imported URDF limits with one 98% soft margin. Geometry, inertia and axes are unchanged. |
| Observation and action spaces | Complete | 122 observations; eight bilateral absolute actions mapped to 31 joint targets. README and `synergy_action.py`. |
| Reward and episode termination | Complete | Term-by-term formulas, weights and intent are in README and `mdp.py`. Timeout and safety terminations are documented separately from success. |
| PPO training experiment | Complete | RSL-RL PPO, 3000 environments, 32 steps/environment, final seed 47. Exact hyperparameters and command are in README and config snapshots. |
| Checkpoint and reward plot | Complete | `x2_relaxed_v4_model450.pt`, parent400 checkpoint, reward CSV and PNG under `reports/`. Every successful supported training run also writes its own graph, CSV and manifest; see `docs/artifact_locations.md`. |

## ROS 2 integration and interfaces

| Requirement | Status | Implementation and evidence |
| --- | --- | --- |
| Python ROS 2 package and one launch file | Complete | `src/x2_recovery_ros`; both nodes in `x2_recovery.launch.py`; fresh `colcon` record in `ros_fresh_build_v4.txt`. |
| Connect ROS to real simulator episode | Complete | Default `isaac_ipc` backend connects ROS Python 3.10 to Isaac Python 3.12 through a mode-0600 local Unix socket. |
| Accept before execution | Complete | Trigger callback acquires the gate and returns `success=true`; timer dispatch starts the worker afterward. Measured acceptance remained below 1 ms in repeated validation. |
| Reject second request while running | Complete | `AttemptGate` covers pending and running states; actual concurrent request returned busy. |
| Publish status | Complete | `/x2/recovery_status`, `std_msgs/msg/String`: `IDLE`, `RUNNING`, `SUCCEEDED`, `FAILED`. |
| Publish simulator joint state | Complete | `/x2/joint_states`, `sensor_msgs/msg/JointState`: 31 names, measured positions and ROS timestamps while running. |
| Configurable unsuccessful timeout | Complete | `timeout_sec` launch/ROS parameter. The 0.2 s test reached `FAILED` after 10 real simulator policy steps. |
| Telemetry node | Complete | Subscribes to both topics and logs current status plus `left_knee_joint` at 1 Hz. |

## Validation and delivery

| Requirement | Status | Implementation and evidence |
| --- | --- | --- |
| Five simulation episodes | Complete | Seeds 101-105, 500 steps and 10 s each in `relaxed_v4_evaluation.json`. Result: 5/5. |
| Upright, both feet, no other support | Complete | Success requires height, projected-gravity uprightness, low root velocities, >=15 N on each foot and <15 N on all other bodies for 0.5 s continuously. |
| Report failures | Complete | No failure occurred in the submitted five episodes; all `failure_reason` values are empty. Historical failures and fixes are in `development_history.md`. |
| Fresh ROS 2 build | Complete | One-package clean build: 1 package finished in 0.89 s. |
| One launch command | Complete | `ros2 launch x2_recovery_ros x2_recovery.launch.py timeout_sec:=10.0`. |
| CLI request starts recovery | Complete | Literal `ros2 service call /x2/start_recovery ...` produced `Recovery accepted` and `RUNNING -> SUCCEEDED`. |
| Live joint telemetry | Complete | 100 samples in the success trial; 31 simulator joints and timestamps. |
| Busy rejection | Complete | Concurrent trial accepted the first and rejected the second request. |
| FAILED on timeout | Complete | `timeout_sec=0.2` produced `RUNNING -> FAILED` and 10 joint samples. |
| Commands and outcomes in repository | Complete | `docs/commands.md` gives ordered runnable commands; `docs/validation.md`, `docs/test_matrix.md` and the three validation reports record outcomes and distinguish pytest from system-level assignment validation. |
| README contents | Complete | Setup/dependencies, model/import/compute, environment, rewards, RL settings/results, ROS responsibilities, results and limitations are included, with one link to the command guide. `docs/parameter_provenance.md` separates paper, framework, robot/task and local parameters. |
| Commit history | Complete | Thirty-one logical commits retain the experiment and correction sequence. |

## Direct validation commands

```bash
./scripts/build_ros.sh

./scripts/serve_isaac_policy.sh reports/exported_relaxed_v4/policy.pt \
  --environment relaxed_v4 --device cuda:0

source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch x2_recovery_ros x2_recovery.launch.py timeout_sec:=10.0

ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
ros2 topic echo /x2/recovery_status std_msgs/msg/String --qos-durability transient_local
ros2 topic echo /x2/joint_states sensor_msgs/msg/JointState

./scripts/validate_ros_isaac_runtime.sh \
  reports/exported_relaxed_v4/policy.pt \
  --environment relaxed_v4 --device cuda:0
```
