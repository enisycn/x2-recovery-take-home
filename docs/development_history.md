# Development history

This page separates the earlier mixed-start policy from the submitted policy. An *update* is one PPO training iteration; the five evaluation episodes are separate tests. **The first 500 + 500 updates of the submitted policy are not the earlier pretraining run.**

| Experiment | Training starts | Updates | Evaluation from a real supine reset | What it showed |
| --- | --- | ---: | --- | --- |
| Historical model450 | Mixed supine and near-upright/sitting starts; final 51 updates supine | About 450 | 5/5 | It could recover from supine, but its weights had learned from other start poses. We did not use these weights in the submitted policy. |
| Original rewards, from scratch | Only supine | 500 | 0/5 | Height improved, but the robot jumped or bridged instead of settling on both feet. |
| Same original rewards, longer independent run | Only supine | 2,000 | 0/5 | More updates alone did not fix that reward setup. This run is not an ancestor of the final policy. |
| Revised stance rewards, from scratch | Only supine | 500 | 0/5 | It briefly approached a valid stance but did not hold all checks for 0.5 s. **This starts the selected policy's lineage.** |
| Continue that same policy and reward setup | Only supine | 1,000 total | 5/5 | It recovered and ended standing in all five episodes; the arms were still forward. |
| Refine the standing arm pose | Only supine | 2,404 total | 5/5 | It retained recovery and passed the separate neutral-arm check throughout the final two seconds. This is submitted model2397. |

The revised setup added smoother feedback for stance, two-foot support, slowing down, leg pose and saturated actions. These changes and the longer training budget were combined, so the experiments do not identify one reward as the sole cause of improvement. After recovery worked at 1,000 updates, a supported-standing arm objective improved the final pose. The physical success thresholds stayed the same.

After the first 1,000 updates learned recovery, 1,404 further selected updates refined the arms, giving 2,404 in total. Each continuation kept the same policy and training state. `model2397.pt` is the checkpoint filename; its number is not the total update count. Exact stage commands are in [commands](commands.md). **The original-reward 2,000-update run failed; successful standing first appeared at 1,000 updates in the revised-reward lineage.**

Historical model450 passed the five true-supine evaluation episodes, but its *training* starts were mixed. The submitted model starts from random weights and every *training* episode starts supine. No imitation dataset, mixed-start weights or lift force enters its lineage.

Evidence: [five final episodes](../reports/relaxed_v4_evaluation.json), [full selected reward curve](../reports/relaxed_v4_training_reward.png), [recorded stage commands](commands.md), [reward formulas and failure diagnosis](design_and_evidence.md), and [original-reward 2,000-update test](original_rewards_experiment.md). Earlier experiments remain in the chronological Git commits; the final model and its configuration snapshots identify the selected result.
