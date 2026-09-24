# Commands, in order

Run commands from the root of this repository. Training is optional for a quick demo: the submitted checkpoint and exported ROS policy are already in `reports/`.

## 1. One-time setup

Clone this repository using its GitHub URL, then enter it:

```bash
git clone https://github.com/OWNER/REPO.git hrs_x2_take_home
cd hrs_x2_take_home
```

Replace `OWNER/REPO` with the URL shown by GitHub's **Code** button. Install [Isaac Sim 6.0.1](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/install_python.html), [Isaac Lab 3.0.0-beta2.patch1](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-beta2.patch1), [RSL-RL 5.0.1](https://github.com/leggedrobotics/rsl_rl/releases/tag/v5.0.1), and [ROS 2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) in their respective environments. The tested host was Ubuntu 22.04.5 with an NVIDIA RTX 5080 Laptop GPU (16 GB). Isaac used Python 3.12; ROS used system Python 3.10. Keep those environments separate.

Activate the Isaac environment and record its Python executable:

```bash
export ISAAC_PYTHON="$(command -v python)"
"$ISAAC_PYTHON" -c 'import importlib.util as u; assert all(u.find_spec(x) for x in ("isaacsim", "isaaclab", "rsl_rl"))'
"$ISAAC_PYTHON" -m pip install -r requirements.txt
./scripts/fetch_agibot_model.sh
./scripts/import_x2_isaac.sh
```

`ISAAC_PYTHON` must point to the Isaac environment in each new terminal that runs Isaac. The fetcher checks the official source and exact recorded Git revision. The imported USD remains local under `assets/isaac/`.

Install the ROS package dependencies in a ROS terminal:

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
```

## 2. Train (optional)

This re-runs the submitted final 51-update stage from its supplied parent checkpoint. It uses the GPU and 3,000 environments; a smaller environment count is a new experiment, not a reproduction of the recorded result.

```bash
./scripts/train_isaac.sh --phase relaxed_v4 --num_envs 3000 \
  --max_iterations 51 --seed 47 --device cuda:0 \
  --checkpoint reports/checkpoints/x2_relaxed_v4_parent_model400.pt \
  --reset_optimizer --action_std_override 0.10 \
  --learning_rate_override 0.0001 --learning_schedule fixed \
  --entropy_coef 0.001
```

Every **completed** training run writes a checkpoint, `reward.png`, `reward.csv`, and `run_manifest.json` to its timestamped directory. Find the latest:

```bash
experiment=logs/rsl_rl/hrs_x2_relaxed_v4
run_dir="$experiment/$(cat "$experiment/LATEST_RUN.txt")"
cat "$run_dir/run_manifest.json"
xdg-open "$run_dir/reward.png"
ls -lh "$run_dir"/model_*.pt
```

The submitted reward plot is [here](../reports/relaxed_v4_training_reward.png). See [output locations](artifact_locations.md) for the full layout.

## 3. Play and evaluate the supplied checkpoint

Play opens the Isaac viewer. Evaluation runs five fixed seeded episodes headlessly and writes a JSON result.

```bash
./scripts/play_isaac.sh reports/checkpoints/x2_relaxed_v4_model450.pt \
  --environment relaxed_v4 --device cuda:0

./scripts/evaluate_isaac.sh reports/checkpoints/x2_relaxed_v4_model450.pt \
  --environment relaxed_v4 --device cuda:0 \
  --output /tmp/x2_evaluation.json
```

The recorded result is [5/5](../reports/relaxed_v4_evaluation.json). To evaluate a checkpoint from your own run, replace the checkpoint path with one shown in its `run_manifest.json`.

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

For the busy-request and `FAILED` timeout checks, use the exact steps in [live demo](live_demo.md). Recorded outputs are in [validation](validation.md).
