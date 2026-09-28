# Development history

The submitted checkpoint is [`x2_supine_model3847.pt`](../reports/checkpoints/x2_supine_model3847.pt). Its weights descend from random initialization and **every training reset is supine**. A continuation resumes the actor, critic and optimizer; it is not a new from-scratch run. The selected ancestry contains 3,860 PPO updates, whereas `3847` is an RSL-RL file index affected by resume boundaries. No mixed-start weights, imitation data or lift force entered this lineage.

| Experiment | Change and observation | Decision |
| --- | --- | --- |
| Mixed-start prototype | Some training resets were near sitting or standing; evaluation recovered 5/5. | Excluded from the selected all-supine ancestry. |
| Original supine rewards | Random-weight training emphasized height/upright; 0/5 at 500 and in a separate 2,000-update experiment. The robot rose or bounced without sustained foot-only stance. | Add continuous stance, support, slowing and leg-pose feedback. |
| Dense stance, supine only | Random-weight policy: 0/5 at 500; 5/5 after continuing the same policy to 1,000 updates. | Foundation of the submitted lineage. |
| Fixed-reward alternative | The same final recovery/arm reward objective from the first update: 0/5 at 500, 4/5 at 1,000, 5/5 at 1,500 and 5/5 on five additional starts. Arms bent forward, but at the seed-101 height peak (0.778 m) both feet were airborne. | Valid recovery comparison, not selected for its less supported rise. |
| Supported-rise arm refinement | Continued the successful dense-stance lineage. The intermediate `model2397` scored 5/5 but followed our optional neutral-arm target, leaving the arms close to the torso. | Keep its supported rise and change the local arm objective. |
| Forward-arm refinement | Continued `model2397` with a supported-upright physical arm target and shoulder/elbow command shaping. The selected 200 + 300 + 156 + 300 + 250 + 250 updates retained 5/5 on seeds 101–105 and 5/5 on 106–110. Final shoulder pitch is about -0.39 to -0.44 rad and elbow about -0.85 to -0.87 rad. Peak height is 0.671–0.687 m with both feet contacting the floor at the peak in all five required episodes. | Submitted `model3847`. |

The neutral-arm target (shoulder 0, elbow -0.15 rad) was our design preference, not an HRS requirement. Its old diagnostic remains visible in evaluator JSON but is excluded from recovery success. The new forward target (-0.22/-1.17 rad) and its weights are likewise local choices, not paper constants. HoST motivates a distinct post-recovery posture objective; it does not prescribe these X2 angles or coefficients.

The 1,500-update result shows that staged rewards are not required to solve the basic recovery check. The selected continuation addressed a different visual/control preference after stable recovery was available. The 10/10 outcome tests only small supine pose perturbations in nominal flat-floor simulation; it does not establish broad robustness. Mean training return is useful for diagnosing learning but cannot replace the physical stance criteria.

Evidence: [required five episodes](../reports/relaxed_v4_evaluation.json), [additional five](../reports/relaxed_v4_extra_evaluation.json), [complete selected training graph](../reports/selected_supine_training_reward.png), [final refinement graph](../reports/forward_arm_refinement_reward.png), [fixed-reward comparison](../reports/experiments/fixed_rewards_1500/README.md), [original 2,000-update analysis](original_rewards_experiment.md), and [training commands](commands.md).
