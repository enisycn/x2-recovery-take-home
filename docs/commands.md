# Reproduce the selected X2 recovery result

Run each block from the repository root. Isaac and ROS use separate Python environments; the commands below do not change CPU affinity or the configuration of other robot projects.

## 1. Install and import

Tested: Ubuntu 22.04.5, RTX 5080 Laptop GPU (16 GB), Isaac Sim 6.0.1, Isaac Lab 3.0.0-beta2.patch1, RSL-RL 5.0.1, ROS 2 Humble. Install [Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/install_python.html), [Isaac Lab](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-beta2.patch1), [RSL-RL](https://github.com/leggedrobotics/rsl_rl/releases/tag/v5.0.1) and [ROS 2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) first.

```bash
git clone https://github.com/enisycn/x2-recovery-take-home.git hrs_x2_take_home
cd hrs_x2_take_home
export ISAAC_PYTHON="$(command -v python)"  # after activating the Isaac Python 3.12 environment
"$ISAAC_PYTHON" -m pip install -r requirements.txt
./scripts/fetch_agibot_model.sh
./scripts/import_x2_isaac.sh
```

Set `ISAAC_PYTHON` in each new Isaac terminal. The model is pinned; generated USD stays local in `assets/isaac/` and is not needed in Git. The scripts use `--viz none` for headless runs and `--viz kit` for graphical playback.

## 2. Train once from random weights (optional)

This is the exact selected configuration: 3,000 environments, 1,500 uninterrupted PPO updates, only supine resets, and the same rewards from update 0. There is **no** `--checkpoint` option.

```bash
./scripts/train_isaac.sh --phase relaxed_v4 --num_envs 3000 \
  --max_iterations 1500 --seed 47 --device cuda:0 \
  --run_name fixed_rewards_single1500 \
  --stability_refinement --posture_refinement \
  --shoulder_command_variance 1 --shoulder_target_ratio 0.15 \
  --action_std_override 0.8 --learning_schedule adaptive --entropy_coef 0
```

Each completed run writes `reward.png`, `reward.csv`, `run_manifest.json`, settings and checkpoints under `logs/rsl_rl/hrs_x2_relaxed_v4/<timestamp>_<run_name>/`. Training prints iteration, reward, steps/s and ETA. For a second-terminal summary refreshed every five seconds, run `python3 scripts/watch_training.py` from the repository root. This watcher reads the active log; it does not start another training process.

Locate the run you just completed without relying on the latest-run pointer after another experiment:

```bash
experiment=logs/rsl_rl/hrs_x2_relaxed_v4
run_dir="$experiment/$(cat "$experiment/LATEST_RUN.txt")"
cat "$run_dir/run_manifest.json"
xdg-open "$run_dir/reward.png"
ls -lh "$run_dir"/model_*.pt
```

`LATEST_RUN.txt` points to the most recently completed run. Its directory is timestamped and includes the value of `--run_name`. The submitted result is the distinct [selected graph](../reports/selected_supine_training_reward.png) and [selected checkpoint](../reports/checkpoints/x2_supine_single1500.pt); a new run never overwrites those files automatically.

## 3. Play the supplied checkpoint

```bash
seed=101  # use 101, 102, 103, 104 or 105
./scripts/play_isaac.sh reports/checkpoints/x2_supine_single1500.pt \
  --environment relaxed_v4 --device cuda:0 --viz kit --start-delay 2 \
  --seeds "$seed" --output "/tmp/x2_play_seed_${seed}.json"
```

The two-second frozen preview shows the supine reset pose; physics and the episode clock start afterward. For all five episodes, replace `--seeds "$seed"` with `--seeds 101 102 103 104 105`. Playback and evaluation use the checkpoint path explicitly: they do not select the newest checkpoint automatically.

## 4. Evaluate the supplied checkpoint

```bash
./scripts/evaluate_isaac.sh reports/checkpoints/x2_supine_single1500.pt \
  --environment relaxed_v4 --device cuda:0 --output /tmp/x2_evaluation.json
"$ISAAC_PYTHON" -m json.tool /tmp/x2_evaluation.json
```

Expect `successes=5/5` and review `successful_recoveries`, each `standing_at_episode_end`, `final_strict_stable_s` and `failure_reason`. The evaluator also exports a TorchScript policy for ROS. Its export check alone does not establish recovery success.

To evaluate your **new** training run instead, read `latest_checkpoint` from its manifest and keep the result in that run directory:

```bash
new_checkpoint="$("$ISAAC_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["latest_checkpoint"])' "$run_dir/run_manifest.json")"
./scripts/evaluate_isaac.sh "$run_dir/$new_checkpoint" \
  --environment relaxed_v4 --device cuda:0 --output "$run_dir/evaluation.json"
```

## 5. Build and run ROS 2

Build the package after setup and again after changing ROS source:

```bash
./scripts/build_ros.sh
```

**Terminal 1 — Isaac policy server:**

```bash
cd /path/to/hrs_x2_take_home
export ISAAC_PYTHON=/path/to/your/isaac/environment/bin/python
./scripts/serve_isaac_policy.sh reports/exported_relaxed_v4/policy.pt \
  --environment relaxed_v4 --device cuda:0
```

**Terminal 2 — both ROS nodes in one launch:**

```bash
cd /path/to/hrs_x2_take_home
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=94
export FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/config/fastdds_shm.xml"
ros2 launch x2_recovery_ros x2_recovery.launch.py timeout_sec:=10.0
```

Prepare Terminals 3–5 with the same `cd`, `source` and `export` commands from Terminal 2. Start the watchers before requesting recovery:

```bash
# Terminal 3: status
ros2 topic echo /x2/recovery_status std_msgs/msg/String --qos-durability transient_local
# Terminal 4: timestamped simulator joint positions
ros2 topic echo /x2/joint_states sensor_msgs/msg/JointState
# Terminal 5: start request
ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
```

The first request returns `success=True`; a second request while `RUNNING` returns `false`. The live status changes to `SUCCEEDED` after a stable hold. To test timeout, set `ros2 param set /x2_recovery timeout_sec 0.2` in a prepared ROS terminal and call the service again; expect `FAILED`. Restore `10.0` afterward. [Recorded checks](ros_live_validation.md) and [validation summary](validation.md) show the outcomes.

For a one-command integration recheck after building ROS, close other Isaac sessions and run:

```bash
./scripts/validate_ros_isaac_runtime.sh reports/exported_relaxed_v4/policy.pt \
  --environment relaxed_v4 --device cuda:0
```
