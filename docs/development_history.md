# Development history

The submitted checkpoint is `reports/checkpoints/x2_supine_model2397.pt`. Every episode in its training lineage starts on the robot's back. We tested other approaches, but did not combine their weights with this policy. Each score below is the same five-episode, true-supine simulator evaluation; success requires a stable, unsupported two-foot stance.

| Experiment | What changed | Result | Decision |
| --- | --- | --- | --- |
| Early mixed-start prototype | Some training resets started near sitting or standing. | 5/5 from supine at evaluation. | Excluded: its training starts were mixed. |
| Original rewards, supine only | Trained from random weights with height and upright rewards but less stance feedback. | 0/5 after 500 updates; a separate 2,000-update run also scored 0/5. | The robot rose or bounced without settling. |
| Revised stance rewards, supine only | Added continuous stance, support, slowing and leg-pose feedback. Trained from random weights. | 0/5 at 500 updates; **5/5 after continuing the same policy to 1,000 total**. | This is the start of the submitted policy's lineage. The arms were still too far forward. |
| One uninterrupted run with all final rewards | Enabled both stance and arm rewards from the first update; random weights, supine resets, 1,000 updates. | 0/5 at its 500 checkpoint; **4/5 at 1,000**. Seed 104 fell, and the successful episodes had elbows around -1.22 rad instead of the -0.15 rad target. | Useful simpler experiment, but less reliable and less natural than the submitted policy. Not selected. |
| Arm refinement of the successful stance policy | Continued the 1,000-update policy with a supported-standing arm objective. | **5/5**, all ending upright with the neutral-arm check passing. | Submitted checkpoint. |

Why use continuation stages? The first 1,000 updates taught recovery with stance rewards. The later updates targeted the arm pose after recovery was reliable. A continuation loads the actor, critic, optimizer and learned action standard deviations; it is not a fresh training run. The submitted checkpoint inherited **2,404 PPO updates** in total. Its filename `model2397.pt` is an RSL-RL checkpoint index, not that total. No imitation data, lift force or mixed-start weights entered this lineage.

The single-run check shows that 1,000 uninterrupted updates **can** learn recovery in four of five tested starts, but did not match the selected policy in this run. It does not prove that staged rewards are universally necessary: these are individual training runs, not a controlled multi-seed ablation. Mean training reward alone was not used to choose a model.

Evidence: [submitted evaluation](../reports/relaxed_v4_evaluation.json), [submitted training curve](../reports/relaxed_v4_training_reward.png), [single-run curve](../reports/experiments/single_run_1000/reward.png), [single-run 500 evaluation](../reports/experiments/single_run_1000/evaluation_500.json), [single-run 1,000 evaluation](../reports/experiments/single_run_1000/evaluation_1000.json), [single-run settings](../reports/experiments/single_run_1000/runtime_settings.json), and [original-reward 2,000-update analysis](original_rewards_experiment.md). The complete selected training commands are in [commands](commands.md).
