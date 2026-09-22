# Live ROS 2 meeting demo

The required user-initiated command is the Trigger service call. Recovery status and joint states must be observed from the nodes; do not manually publish fake status or joint messages.

## Terminal 1: Isaac policy server

```bash
export REPO=/path/to/hrs_x2_take_home
cd "$REPO"
export ISAAC_PYTHON=/path/to/isaac/python
./scripts/serve_isaac_policy.sh \
  reports/exported_relaxed_v4/policy.pt \
  --environment relaxed_v4 --device cuda:0
```

Wait for:

```text
Isaac policy server ready: /tmp/hrs_x2_recovery.sock
```

## Terminal 2: both ROS nodes in one launch

```bash
cd "$REPO"
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=94
export FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/config/fastdds_shm.xml"
ros2 launch x2_recovery_ros x2_recovery.launch.py timeout_sec:=10.0
```

This launch starts `x2_recovery` and `x2_telemetry`. Expected initial status: `IDLE`.

## Terminal 3: inspect interfaces

```bash
source /opt/ros/humble/setup.bash
source "$REPO/install/setup.bash"
export ROS_DOMAIN_ID=94
export FASTRTPS_DEFAULT_PROFILES_FILE="$REPO/config/fastdds_shm.xml"

ros2 node list
ros2 service type /x2/start_recovery
ros2 topic type /x2/recovery_status
ros2 topic type /x2/joint_states
ros2 topic info /x2/joint_states -v
```

Expected types:

```text
/x2/start_recovery   std_srvs/srv/Trigger
/x2/recovery_status  std_msgs/msg/String
/x2/joint_states     sensor_msgs/msg/JointState
```

## Start recovery and watch live outputs

Run the status watcher in a spare terminal:

```bash
ros2 topic echo /x2/recovery_status std_msgs/msg/String \
  --qos-durability transient_local
```

Start recovery:

```bash
ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
```

Expected immediate response:

```text
success=True, message='Recovery accepted'
```

Expected status sequence:

```text
RUNNING
SUCCEEDED
```

Read one complete simulator joint message:

```bash
ros2 topic echo /x2/joint_states sensor_msgs/msg/JointState --once
```

The message must contain a ROS timestamp, 31 joint names and 31 measured positions.

## Busy rejection

Issue two requests concurrently:

```bash
first=/tmp/x2_request_first.txt
second=/tmp/x2_request_second.txt
ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}' >"$first" &
ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}' >"$second" &
wait
cat "$first"
cat "$second"
```

One request is accepted. The other must report:

```text
success=False, message='Recovery already running'
```

## Timeout to FAILED

Do this only when no attempt is running:

```bash
ros2 param set /x2_recovery timeout_sec 0.2
ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
```

Expected status sequence:

```text
RUNNING
FAILED
```

Restore the normal value:

```bash
ros2 param set /x2_recovery timeout_sec 10.0
```

## Graphs and result files to open

- `reports/relaxed_v4_training_reward.png`: final training-stage reward plot.
- `reports/relaxed_v4_pretraining_reward.png`: earlier training lineage plot.
- `reports/relaxed_v4_evaluation.json`: seeds 101-105 and 5/5 result.
- `reports/gifs_relaxed_v4/x2_final_policy_attempt.gif`: final visual result.
- `reports/ros_isaac_relaxed_v4_validation.txt`: recorded live ROS-Isaac outcomes.

For a newly completed training run, `scripts/train_isaac.sh` also creates `reward.png`, `reward.csv` and `run_manifest.json` inside that run's timestamped `logs/rsl_rl/...` directory. `docs/artifact_locations.md` explains how to find the newest run.

## Stop the demo

Press `Ctrl+C` in Terminal 2, then Terminal 1. The interactive inspection terminal can remain open.
