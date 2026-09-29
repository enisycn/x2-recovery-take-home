# Development history

The **submitted** result is [`x2_supine_single1500.pt`](../reports/checkpoints/x2_supine_single1500.pt): one uninterrupted 1,500-update PPO run from random weights, with the final reward preset active from the start and every reset supine. Its [own reward curve](../reports/selected_supine_training_reward.png), [five-episode evaluation](../reports/relaxed_v4_evaluation.json) and [video](../reports/videos/x2_recovery_supine_to_standing.mp4) match that checkpoint. The selected run passed 5/5 required and 5/5 extra small-pose-variation episodes.

| Experiment | Observation | Lesson / status |
| --- | --- | --- |
| Mixed-start prototype | Near-sitting and upright starts supplied balance experience; a policy evaluated successfully. | Excluded from the selected run because the requested task starts on the back. |
| Original supine reward | 500 updates and a separate 2,000-update test scored 0/5; height alone could be earned while bouncing or falling. | Add dense foot-supported stance, leg-pose and motion feedback. |
| Dense-stance continuation | 0/5 at 500, then 5/5 at 1,000 after continuing the same policy. | Demonstrated that true-supine PPO can learn recovery, but this was a resumed lineage. |
| Supported-rise and arm refinements | A later all-supine continuation reached 5/5 with gentler foot contact at the height peak and an optional forward-arm pose. | The many stages complicated checkpoint-to-graph provenance. Its old `model3847` checkpoint and graphs are retained as historical artifacts. |
| Fixed-reward single 1,500 with forward arms | Random initialization, the supported forward-arm reward present from update 0, and one training process; 5/5 required plus 5/5 extra seeds. | **Selected for submission.** Graph, video, checkpoint, evaluation and ROS export all derive from this run. |
| Other single-run variants | A neutral-arm 1,500-update run passed 5/5 but briefly raised both feet around its height peak; a later 2,000-update extension scored 4/5. | Kept as comparisons, not substituted into the selected graph or video. |

The selected robot keeps both feet in contact at the height peak and maintains strict two-foot stance for 8.80–9.00 s at episode end. Four required episodes contain a separate 0.02–0.04 s upright airborne interval; see [validation](validation.md). The optional neutral-arm diagnostic in evaluator JSON is not an HRS success criterion. Neither the evidence above nor rising reward proves broad robustness, real-hardware transfer or that one reward term alone caused the improvement.

Older experiments, their failed evaluations, reward curves and original configurations remain under `reports/experiments/` and `reports/configs/`. They explain design choices without being mixed into the selected single-run result. [Commands](commands.md) give a new one-run reproduction; [parameter provenance](parameter_provenance.md) identifies paper, framework and local choices.
