# Development history

This page separates the earlier mixed-start policy from the submitted policy. An *update* is one PPO training iteration; the five evaluation episodes are separate tests. **The first 500 + 500 updates of the submitted policy are not the earlier pretraining run.**

| Experiment | Training starts | Updates | Evaluation from a real supine reset | What it showed |
| --- | --- | ---: | --- | --- |
| Historical model450 | Mixed supine and near-upright/sitting starts; final 51 updates supine | About 450 | 5/5 | Passed from supine, but training used other poses. Not used in the final model. |
| Original rewards, from scratch | Only supine | 500 | 0/5 | Rose or bridged, but did not settle on both feet. |
| Same original rewards, separate longer run | Only supine | 2,000 | 0/5 | More updates alone did not help. Not part of the final model. |
| Revised stance rewards, from scratch | Only supine | 500 | 0/5 | Brief near-stance, no 0.5 s hold. **Final lineage starts here.** |
| Continue the same policy and rewards | Only supine | 1,000 total | 5/5 | Recovered and ended standing; arms still forward. |
| Refine the standing arm pose | Only supine | 2,404 total | 5/5 | Standing and neutral-arm check pass. Submitted model2397. |

The revised setup added smoother feedback for stance, two-foot support, slowing down, leg pose and saturated actions. These changes and the longer training budget were combined, so the experiments do not identify one reward as the sole cause of improvement. After recovery worked at 1,000 updates, a supported-standing arm objective improved the final pose. The physical success thresholds stayed the same.

After the first 1,000 updates learned recovery, 1,404 further selected updates refined the arms, giving 2,404 in total. Each continuation kept the same policy and training state. `model2397.pt` is the checkpoint filename; its number is not the total update count. Exact stage commands are in [commands](commands.md). **The original-reward 2,000-update run failed; successful standing first appeared at 1,000 updates in the revised-reward lineage.**

Historical model450 passed the five true-supine evaluation episodes, but its *training* starts were mixed. The submitted model starts from random weights and every *training* episode starts supine. No imitation dataset, mixed-start weights or lift force enters its lineage.

Evidence: [five final episodes](../reports/relaxed_v4_evaluation.json), [full selected reward curve](../reports/relaxed_v4_training_reward.png), [recorded stage commands](commands.md), [reward formulas and failure diagnosis](design_and_evidence.md), and [original-reward 2,000-update test](original_rewards_experiment.md). Earlier experiments remain in the chronological Git commits; the final model and its configuration snapshots identify the selected result.
