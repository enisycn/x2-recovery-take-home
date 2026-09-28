# Fixed-reward supine recovery comparison

This separate PPO experiment started from random weights with the final stance and arm rewards enabled from update 1. All training resets were supine. It ran for 1,000 updates and then resumed for 500 more without changing the reward settings, actor, critic, optimizer or learned action standard deviations. The optimizer's checkpoint learning rate was preserved on resume. The two launcher calls are one **fixed reward objective**, not one uninterrupted process.

| Checkpoint | Required seeds 101–105 | Additional seeds 106–110 | Final neutral-arm check |
| --- | --- | --- | --- |
| 500 updates | 0/5 | Not tested | — |
| 1,000 updates | 4/5 | Not tested | Did not meet target |
| 1,500 updates | **5/5**, all standing at the end | **5/5**, all standing at the end | **0/10**; elbow error about 1.01 rad |
| Submitted staged policy | **5/5** | **5/5** | **10/10**; elbow error about 0.10 rad |

Recovery success is the [documented unsupported two-foot stance check](../../../docs/validation.md); the neutral-arm check is an additional posture preference. These ten small supine reset variations show that fixed rewards can solve the required recovery task. They do not measure broad robustness, and the comparison is not a multi-training-seed ablation. We retained the submitted staged policy because it also recovers and has a more natural final arm pose.

The [reward plot](reward.png) combines the two consecutive training logs, with the 20-update mean calculated within each segment. The episode-reward logger resets at the 1,000-update resume, causing the apparent drop at the boundary; the trained policy does not reset. The [CSV](reward.csv), [five required episodes](evaluation_101_105.json), [five additional episodes](evaluation_106_110.json), [submitted policy on the additional episodes](submitted_evaluation_106_110.json), and [runtime settings](resume_settings.json) are included. The 500- and 1,000-update evaluations are in [the earlier experiment record](../single_run_1000/).

Reproduce the training from the repository root after [setup](../../../docs/commands.md). The exact initial run used these settings:

```bash
./scripts/train_isaac.sh --phase relaxed_v4 --num_envs 3000 \
  --max_iterations 1000 --seed 47 --device cuda:0 \
  --run_name all_final_rewards_scratch1000 \
  --stability_refinement --posture_refinement \
  --shoulder_command_variance 1 --shoulder_target_ratio 0.15 \
  --action_std_override 0.8 --learning_schedule adaptive --entropy_coef 0
```

Continue from that run's `model_999.pt` for 500 updates with the same reward settings:

```bash
parent_dir="logs/rsl_rl/hrs_x2_relaxed_v4/$(cat logs/rsl_rl/hrs_x2_relaxed_v4/LATEST_RUN.txt)"
./scripts/train_isaac.sh --phase relaxed_v4 --num_envs 3000 \
  --max_iterations 500 --seed 47 --device cuda:0 \
  --run_name all_final_rewards_to1500 \
  --stability_refinement --posture_refinement \
  --shoulder_command_variance 1 --shoulder_target_ratio 0.15 \
  --checkpoint "$parent_dir/model_999.pt" \
  --learning_schedule adaptive --entropy_coef 0
```

The resulting final file is `model_1498.pt` because RSL-RL repeats the loaded checkpoint index on continuation. The actual total is 1,000 + 500 = **1,500 PPO updates**. Evaluate the final file with `./scripts/evaluate_isaac.sh PATH/TO/model_1498.pt --environment relaxed_v4 --device cuda:0 --seeds 101 102 103 104 105 --output evaluation.json`. A fresh retrain may differ; select by evaluation, not by reward height alone.
