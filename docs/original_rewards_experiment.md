# Original rewards, supine-only training

This experiment asks whether the original reward set can recover from the
floor with a longer fixed budget, without later reward additions. It starts
from random actor/critic weights and keeps all settings fixed for 2,000 PPO
updates. Every episode starts supine; there are no reference starts, external
forces, demonstrations or pretrained weights.

The 12 reward terms and their parameters match the saved configuration of
the historical mixed-start pretraining run (`posture_from_start`, 22 September)
and the original supine-only scratch run (`supine_from_scratch`, 27 September).
The local reward function bodies also match historical revision `d0f45b0`.
The reset distribution is deliberately supine-only: this tests the old rewards,
not a replay of mixed-start pretraining.

| Original term | Weight |
| --- | ---: |
| Signed pelvis height | 40 |
| Head height | 5 |
| Upright orientation | 5 |
| Two-foot support near standing | 5 |
| Other-body support near standing | -1 |
| Standing still | 10 |
| Strict stance | 20 |
| Action change | -0.005 |
| Joint torque | -0.000001 |
| Joint limits | -1 |
| Failure termination | -10 |
| Supported relaxed arms | 40 |

The later successful supine lineage added stance proximity, near-stance motion
cost, joint-speed cost, leg-posture shaping and action-saturation cost; it also
changed foot, balance and action-change weights. It reached 5/5 stance after
1,000 updates, then received a focused shoulder-command objective and further
training to reach an intermediate neutral-arm result after 2,404 inherited
updates. A separate historical checkpoint then received forward-arm refinement.
The currently submitted forward-arm policy is a different run trained from
random weights with its final rewards fixed from update 0. None of those
additional terms are active in this original-reward comparison run; neither
are the controlled-rise or load-transfer presets.

## Measured result

The completed run collected 192 million transitions and saved `model_1999.pt`
after exactly 2,000 updates (indices 0–1,999). Final mean training reward was
409.0; the curve plateaued near 400 from roughly update 750 onward.
Deterministic evaluation produced **0/5 recoveries**, with no strict stance in
any episode and no episode ending in a standing posture.

| Seed | Steps | Maximum pelvis height (m) | Longest strict stance (s) | Outcome |
| --- | ---: | ---: | ---: | --- |
| 101 | 500 | 0.8664 | 0.00 | Speed/contact/posture criteria never overlapped |
| 102 | 500 | 0.8624 | 0.00 | Speed/contact/posture criteria never overlapped |
| 103 | 500 | 0.8642 | 0.00 | Speed/contact/posture criteria never overlapped |
| 104 | 170 | 0.8469 | 0.00 | Safety termination at 3.4 simulated seconds |
| 105 | 500 | 0.8679 | 0.00 | Speed/contact/posture criteria never overlapped |

The robot gained height but did not settle. The linear-speed criterion passed
in only 1.0–1.6% of evaluated steps, and the angular-speed criterion in
0.4–1.0%. Those fractions include the initial transition from rest. Raw policy
standard deviations grew from 1.0 to 3.10–5.52 across the eight action channels;
these are pre-`tanh` values, not joint-angle deviations. Evaluation sampled no
Gaussian noise, so the failed rollout is also a failure of the deterministic
mean policy, not only of noisy training actions.

A diagnostic seed-101 rollout showed repeated ascent/descent, with vertical
speed between -2.21 and +2.23 m/s, knee targets switching across 0.024–2.300 rad,
and 5.18 s with every body-ground force below 15 N. This last measure is a
low-contact-force indicator, not an exact geometric flight detector. Height,
head-height and upright rewards contributed +442.71 of the +443.67 return;
balance contributed only +0.0017 and strict stance zero. The selected policy,
evaluated under these same original reward terms, received +1,299.75 and held
strict stance for 9.12 s. Thus the old objective still values the observed
stable trajectory more highly; the failed training found a suboptimal motion
cycle rather than demonstrating that bouncing is the global reward optimum.
See the [measured trajectory comparison](../reports/experiments/original_rewards2000/rollout_comparison.png)
and [diagnostic totals](../reports/experiments/original_rewards2000/rollout_diagnostics.json).

Increasing the original experiment's budget to 2,000 did not solve recovery
in this run. Keep the selected, separately validated supine-only policy: its
dense-stance ancestor reached 5/5 at 1,000 updates, and the later arm refinement
reached 5/5 with neutral arms after 2,404 inherited updates. A separate historical checkpoint continued that policy with forward-arm rewards. The currently selected single-run checkpoint has no parent. This experiment is
retained as a failed comparison and was not promoted to the ROS policy.

Evidence: [evaluation](../reports/experiments/original_rewards2000/evaluation.json),
[training curve](../reports/experiments/original_rewards2000/reward.png),
[raw curve data](../reports/experiments/original_rewards2000/reward.csv),
[run settings](../reports/experiments/original_rewards2000/experiment.json) and
[configuration comparison](../reports/experiments/original_rewards2000/configuration_audit.json).
The checkpoint and recorded rollout remain in the local run directory named
in the experiment record; they do not replace the supplied submission artifacts.

## Reproduction

Prepare the Isaac terminal as described in [commands.md](commands.md):

```bash
./scripts/train_isaac.sh --phase relaxed_v4 --num_envs 3000 \
  --max_iterations 2000 --seed 47 --device cuda:0 \
  --run_name original_rewards_supine2000 \
  --action_std_override 1.0 --entropy_coef 0.005 \
  --learning_rate_override 0.0003 --learning_schedule adaptive
```

Do not add a checkpoint or reward-refinement flag. At 32 rollout steps per
environment, 2,000 updates collect 192 million transitions. Initial exploration
std 1.0 and entropy coefficient 0.005 match the original reward experiment,
not the later dense-stance/arm continuation settings.

This is a budget extension of the original supine-only experiment, not an
isolated reward ablation against the selected model. The successful dense-stance
lineage also used initial std 0.8 and entropy coefficient 0.0. A difference in
outcome therefore cannot be attributed solely to the additional reward terms.

After completion:

```bash
experiment=logs/rsl_rl/hrs_x2_relaxed_v4
run_dir="$experiment/$(cat "$experiment/LATEST_RUN.txt")"
./scripts/evaluate_isaac.sh "$run_dir/model_1999.pt" \
  --environment relaxed_v4 --device cuda:0 \
  --output "$run_dir/evaluation.json"
```

The evaluation keeps the existing physical success checks and reports
neutral-arm posture separately. Check recovery count, stance at episode end,
the last two seconds of arm posture, and video before choosing a checkpoint.
Reward alone cannot establish recovery. A single training seed can show this
configuration works or fails in that run; it cannot prove later reward terms
are universally necessary or unnecessary.
