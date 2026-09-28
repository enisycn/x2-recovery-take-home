# Training outputs and file locations

The supported training command is `scripts/train_isaac.sh`. After every successful run it creates a timestamped directory and automatically writes the reward graph, raw reward data and a machine-readable file index. A failed or interrupted run is kept for diagnosis but is not labelled complete and may lack the final graph.

## Per-run directory

For experiment `hrs_x2_relaxed_v4`, one run has this layout:

```text
logs/rsl_rl/hrs_x2_relaxed_v4/
├── LATEST_RUN.txt
└── YYYY-MM-DD_HH-MM-SS[_run_name]/
    ├── events.out.tfevents.*       TensorBoard source data
    ├── model_*.pt                  RSL-RL checkpoints
    ├── params/
    │   ├── agent.yaml              PPO and network configuration
    │   └── env.yaml                simulator and task configuration
    ├── reward.png                  automatic mean-reward graph
    ├── reward.csv                  values used by the graph
    ├── run_manifest.json           index and latest checkpoint name
    ├── training.log                saved console output (new runs)
    ├── evaluation.json             explicit evaluation step
    └── exported_relaxed_v4/policy.pt  exported policy from that evaluation
```

`LATEST_RUN.txt` contains the most recently completed directory name. These run directories are local and gitignored because TensorBoard files and model checkpoints are large and machine-generated. The compact, reviewed submission artifacts are under `reports/`.

Find and open the latest output:

```bash
experiment=logs/rsl_rl/hrs_x2_relaxed_v4
run_dir="$experiment/$(cat "$experiment/LATEST_RUN.txt")"
cat "$run_dir/run_manifest.json"
xdg-open "$run_dir/reward.png"
tensorboard --logdir logs/rsl_rl --port 6006
```

To finish an older successful run that predates automatic finalization:

```bash
"$ISAAC_PYTHON" scripts/finalize_training_run.py \
  logs/rsl_rl/hrs_x2_relaxed_v4/YYYY-MM-DD_HH-MM-SS_run_name
```

## Curated submission artifacts

| File or directory | Meaning |
| --- | --- |
| `docs/submission_report.pdf` | Four-page English summary of the method, reward curve, five-episode evaluation and ROS integration. |
| `reports/selected_supine_training_reward.png` and `.csv` | Complete selected all-supine lineage: 3,860 PPO updates across continuation stages. |
| `reports/relaxed_v4_training_reward.png` | Initial 2,404 selected PPO updates, through the intermediate supported-stance checkpoint. |
| `reports/relaxed_v4_training_reward.csv` | Values behind that initial curve. |
| `reports/forward_arm_refinement_reward.png` and `.csv` | Final 250-update reward curve and values. |
| `reports/configs/supine_model3847/` | Final resolved environment and PPO settings plus selected lineage provenance. |
| `reports/checkpoints/x2_supine_model3847.pt` | Selected policy checkpoint. |
| `reports/checkpoints/x2_supine_model2397.pt` | Intermediate supported-stance parent for the final arm refinement. |
| `reports/checkpoints/x2_supine_parent_model499.pt` | Supine-only 500-update parent used for the next 500 updates. |
| `reports/relaxed_v4_evaluation.json` | Five seeded evaluation episodes and success checks. |
| `reports/gifs_relaxed_v4/x2_final_policy_attempt.gif` | Visual rollout from the selected checkpoint. |
| `reports/videos/x2_recovery_supine_to_standing.mp4` | Two-second supine preview followed by the recorded ten-second episode; English on-screen labels. |
| `reports/exported_relaxed_v4/policy.pt` | TorchScript policy served to ROS. |
| `reports/*validation.txt` | Recorded test, build and ROS-Isaac outcomes. |

Evaluation and GIF rendering are explicit steps because they launch additional simulator episodes. They do not run after every training job. The command guide stores a new run's evaluation and export under its run directory; an explicit `--output` can choose another permanent directory. They need not be stored in `/tmp`:

```bash
./scripts/evaluate_isaac.sh CHECKPOINT \
  --environment relaxed_v4 --device cuda:0 \
  --output reports/my_evaluation.json

./scripts/render_isaac_gifs.sh \
  --checkpoint CHECKPOINT --environment relaxed_v4 \
  --viz none --device cuda:0 --output_dir reports/my_gifs
```

## ROS runtime output

ROS topics are live streams rather than training artifacts. The required recorded evidence is `reports/ros_isaac_relaxed_v4_validation.txt`; normal ROS process logs are written by ROS 2 below `~/.ros/log/`. See `docs/ros_live_validation.md` for the exact launch, service and topic commands.
