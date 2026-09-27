# Commands, in order

Run commands from the root of this repository. Training is optional when validating the submitted checkpoint and ROS interface: the checkpoint and exported policy are already in `reports/`.

## 1. One-time setup

Clone this repository using its GitHub URL, then enter it:

```bash
git clone https://github.com/enisycn/x2-recovery-take-home.git hrs_x2_take_home
cd hrs_x2_take_home
```

Install [Isaac Sim 6.0.1](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/install_python.html), [Isaac Lab 3.0.0-beta2.patch1](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-beta2.patch1), [RSL-RL 5.0.1](https://github.com/leggedrobotics/rsl_rl/releases/tag/v5.0.1), and [ROS 2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) in their respective environments. The tested host was Ubuntu 22.04.5 with an NVIDIA RTX 5080 Laptop GPU (16 GB). Isaac used Python 3.12; ROS used system Python 3.10. Keep those environments separate.

Activate the Isaac environment and record its Python executable:

```bash
export ISAAC_PYTHON="$(command -v python)"
"$ISAAC_PYTHON" -c 'import importlib.util as u; assert all(u.find_spec(x) for x in ("isaacsim", "isaaclab", "rsl_rl"))'
"$ISAAC_PYTHON" -m pip install -r requirements.txt
./scripts/fetch_agibot_model.sh
./scripts/import_x2_isaac.sh
```

`ISAAC_PYTHON` must point to the Isaac environment in each new terminal that runs Isaac. Training, evaluation, policy-server and import launchers select `--viz none` for headless execution; graphical playback uses `--viz kit`. The deprecated `--headless` CLI flag is not passed by these launchers. The fetcher checks the official source and exact recorded Git revision. The imported USD remains local under `assets/isaac/`.

Install the ROS package dependencies in a ROS terminal:

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
```

## 2. Train (optional)

This starts a new PPO experiment from random actor/critic weights. Every episode starts supine with neutral joints and zero velocity; there are no auxiliary poses, reset schedule or lift assistance. Do not add `--checkpoint` when training from scratch. Reward gates stay active; they control when rewards apply, not the initial pose.

```bash
./scripts/train_isaac.sh --phase relaxed_v4 --num_envs 3000 \
  --max_iterations 500 --seed 47 --device cuda:0 \
  --run_name supine_dense_stance_scratch \
  --stability_refinement --action_std_override 0.8 --entropy_coef 0
```

After that run completes, continue for 500 further updates with the same reward and all-supine resets. Keep the learned standard deviations and optimizer state:

```bash
experiment=logs/rsl_rl/hrs_x2_relaxed_v4
parent_dir="$experiment/$(cat "$experiment/LATEST_RUN.txt")"
./scripts/train_isaac.sh --phase relaxed_v4 --num_envs 3000 \
  --max_iterations 500 --seed 47 --device cuda:0 \
  --run_name supine_dense_stance_continue --stability_refinement \
  --checkpoint "$parent_dir/model_499.pt" --entropy_coef 0
```

Training prints progress after each completed PPO iteration: iteration number, mean reward, mean episode length (policy steps), reward terms, elapsed time and ETA. Output is unbuffered. No completed episodes means the mean episode metrics are not available yet. The training console prints the temporary log path as `[HRS] Live console log:`. For a compact view refreshed every five seconds, run this in a second terminal while one training run is active:

```bash
cd /path/to/hrs_x2_take_home
python3 scripts/watch_training.py
```

The monitor shows only the latest complete iteration, including steps per second, mean episode metrics and supported-stance reward terms; it strips terminal formatting codes. `Ctrl+C` in this second terminal stops only the monitor. The newest temporary training log is selected once at startup; to choose a specific run, append its printed console log path to the command. When the launcher finishes, it saves the console output as `training.log` inside the run directory, removes the temporary copy, and the monitor stops without inferring success. TensorBoard metrics and checkpoints also remain in the run directory.

Every **completed** training run writes a checkpoint, `reward.png`, `reward.csv`, and `run_manifest.json` to its timestamped directory. The folder hierarchy is `logs/rsl_rl/hrs_x2_relaxed_v4/<timestamp>_<run_name>/`; the run is inside the experiment folder, not directly under `rsl_rl` and not under `/tmp`.

New runs do not overwrite the submitted plot or checkpoint in `reports/`. Change `--run_name` to label another experiment. `--max_iterations 500` is a chosen experiment budget, not a guarantee of recovery. The selected recipe uses learning rate 3e-4 (adaptive), initial action standard deviation 0.8 and entropy coefficient 0. The first 500-update run scored 0/5; the same-preset continuation reached 5/5 after 1,000 total updates. The final checkpoint index is 998 because RSL-RL repeats the parent index on resume. Its result must be evaluated separately; the recorded 5/5 result belongs to the supplied checkpoint. The historical mixed-start lineage is documented separately in [development history](development_history.md).

Find the latest completed run:

```bash
experiment=logs/rsl_rl/hrs_x2_relaxed_v4
run_dir="$experiment/$(cat "$experiment/LATEST_RUN.txt")"
cat "$run_dir/run_manifest.json"
xdg-open "$run_dir/reward.png"
ls -lh "$run_dir"/model_*.pt
```

The directory name is `timestamp_run_name`, using the label supplied with `--run_name`. `LATEST_RUN.txt` stores the name of the last run whose finalization completed, regardless of its label. Another completed run changes that pointer; keep the full directory path to revisit a particular experiment.

Evaluate the new checkpoint and keep its report and exported ROS policy in that same run directory:

```bash
new_checkpoint="$("$ISAAC_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["latest_checkpoint"])' "$run_dir/run_manifest.json")"
./scripts/evaluate_isaac.sh "$run_dir/$new_checkpoint" \
  --environment relaxed_v4 --device cuda:0 \
  --output "$run_dir/evaluation.json"
```

The exported policy is `$run_dir/exported_relaxed_v4/policy.pt`. Check `export.succeeded` and the recovery result in `evaluation.json`; export alone does not prove recovery success. To serve this new policy, use its absolute path with `serve_isaac_policy.sh`. Evaluation remains an explicit step after training.

The submitted reward plot is [here](../reports/relaxed_v4_training_reward.png). See [output locations](artifact_locations.md) for the full layout.

## 3. Play and evaluate the supplied checkpoint

Playback needs `--viz kit` to open the Isaac viewer in this version. Set `seed` to 101, 102, 103, 104 or 105 for a single episode:

```bash
seed=101
./scripts/play_isaac.sh reports/checkpoints/x2_supine_model998.pt \
  --environment relaxed_v4 --device cuda:0 --viz kit --start-delay 2 \
  --seeds "$seed" --output "/tmp/x2_play_seed_${seed}.json"
```

`--start-delay 2` shows the frozen supine reset pose for two wall-clock seconds before each episode. Only the viewer updates during this preview; physics, policy inference and the 10-second episode clock have not started. Set it to `0` to start immediately. Headless evaluation uses no preview.

The viewer closes after evaluation finishes. A 10-second simulation episode can run faster than wall-clock time. The supplied path selects `model998` explicitly; it does not automatically choose the latest training checkpoint. Seeds change small initial supine root-pose perturbations; the model weights, neutral joint angles and zero initial velocities remain the same.

Play all five seeds sequentially in one viewer session:

```bash
./scripts/play_isaac.sh reports/checkpoints/x2_supine_model998.pt \
  --environment relaxed_v4 --device cuda:0 --viz kit --start-delay 2 \
  --seeds 101 102 103 104 105 --output /tmp/x2_play_all_seeds.json
```

Evaluation runs those five fixed seeds headlessly and writes a JSON result:

```bash
./scripts/evaluate_isaac.sh reports/checkpoints/x2_supine_model998.pt \
  --environment relaxed_v4 --device cuda:0 \
  --output /tmp/x2_evaluation.json
```

The recorded result is [5/5](../reports/relaxed_v4_evaluation.json). To evaluate a checkpoint from your own run, replace the checkpoint path with one shown in its `run_manifest.json`.

At completion, expect one result line per seed and `successes=N/5 report=/tmp/x2_evaluation.json`. Inspect `successful_recoveries`, `total_episodes`, and each episode's `success`, `standing_at_episode_end` and `failure_reason` in that JSON. `export.succeeded` checks the policy export separately. Evaluation does not generate a training reward graph.

```bash
"$ISAAC_PYTHON" -m json.tool /tmp/x2_evaluation.json
```

## 4. Build and run ROS 2

Build once from the repository root:

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

**Terminal 2 — both ROS nodes with one launch:**

```bash
cd /path/to/hrs_x2_take_home
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=94
export FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/config/fastdds_shm.xml"
ros2 launch x2_recovery_ros x2_recovery.launch.py timeout_sec:=10.0
```

**Terminals 3–5 — prepare each ROS inspection terminal:**

```bash
cd /path/to/hrs_x2_take_home
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=94
export FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/config/fastdds_shm.xml"
```

Start both watchers **before** the request so the 10-second episode is visible live:

```bash
# Terminal 3: status
ros2 topic echo /x2/recovery_status std_msgs/msg/String --qos-durability transient_local

# Terminal 4: measured simulator joints
ros2 topic echo /x2/joint_states sensor_msgs/msg/JointState

# Terminal 5: accepted start request
ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
```

For the busy-request and `FAILED` timeout checks, use the exact steps in [live validation](ros_live_validation.md). Recorded outputs are in [validation](validation.md).
