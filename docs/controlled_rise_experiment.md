# Controlled-rise experiment

The previous 500-update scratch policy repeatedly rose and collapsed. In its
seed-101 rollout, height/orientation/leg rewards contributed +379.07, the three
stability terms +2.77, and near-standing motion cost -13.48. That cost was zero
for 7.54 of 10 seconds. The selected, successful policy earned a much higher
return under the same reward; the cycle was a suboptimal learned behavior.

`--controlled_rise` is a separate, fixed reward preset for testing this failure.
It retains true supine resets, the 122 observations, eight action channels,
joint/actuator limits, 200 Hz physics, 50 Hz policy and all success thresholds.
There is no pretraining, reference motion, assistance or curriculum.

## Changes from the previous scratch trial

Let `z` be pelvis height, `g_z` projected gravity, `v` world linear velocity,
`w` world angular velocity, and `c(x)=clip(x,0,1)`. Contact means current
body-to-floor force of at least 15 N, excluding self-collision.

| Term | New expression or setting | Weight |
| --- | --- | ---: |
| Pelvis height | `c(z/.68) c((1-g_z)/2) (.25+.75*any_ground_contact) exp(-(max(abs(v_z)-.3,0)/.7)^2)` | 40 |
| Settling stance | `c((z-.25)/.40) c(-g_z)^2 (.25+.75*both_feet) exp(-.5*norm(v)^2/.8^2-.5*norm(w)^2/2^2-.5*nonfoot_contacts)` | 40 |
| Motion cost | `c((z-.20)/.38) c((1-g_z)/2)^2 (norm(v)^2+.05*norm(w)^2)` | -4 |
| Action change | Same raw action difference squared; previously -0.02 | -0.2 |
| Head height | Same function; previously 5 | 2 |
| Standing leg pose | Same function; previously 20 | 10 |

Other terms remain as in `X2PostureRefinementEnvCfg`, with shoulder-command
variance 1. Height credit remains available through hand push-off, but is
reduced during flight and high vertical speed. The settling term is informative
below 0.60 m and before perfect upright stance. Costs do not prohibit motion
at the floor. These are memoryless scores, not potential-difference shaping;
they do not mathematically rule out every cyclic local optimum.

[HoST, Table VI](https://arxiv.org/html/2502.08378v2#A1.T6) motivates separate
velocity objectives during and after rising, alongside action regularization.
The formulas, widths and weights here are local X2 choices informed by measured
rollouts. This experiment does not reproduce HoST's multiple critics or force
curriculum, and the paper does not guarantee success in 500 PPO updates.

## Run and evaluate

Prepare the Isaac terminal as described in [commands.md](commands.md), then run:

```bash
./scripts/train_isaac.sh --phase relaxed_v4 --controlled_rise \
  --num_envs 3000 --max_iterations 500 --seed 47 --device cuda:0 \
  --run_name controlled_rise_scratch500 \
  --action_std_override 0.8 --entropy_coef 0 \
  --learning_rate_override 0.0003 --learning_schedule adaptive

experiment=logs/rsl_rl/hrs_x2_relaxed_v4
run_dir="$experiment/$(cat "$experiment/LATEST_RUN.txt")"
./scripts/evaluate_isaac.sh "$run_dir/model_499.pt" \
  --environment relaxed_v4 --device cuda:0 \
  --output "$run_dir/evaluation.json"
```

Five deterministic episodes use seeds 101-105. Evaluation uses the same
physical environment; training reward changes cannot relax its success checks.
`model_499.pt` is the checkpoint after 500 updates numbered 0-499. At 3,000
environments and 32 steps per update, the budget is 48 million transitions.
The per-run folder retains parameters, source revision, reward CSV/plot,
checkpoint and evaluation; it does not overwrite the submitted evidence.

Numerical tests check support versus flight, inversion rejection, braking
below the old height cutoff and denser settling feedback while strict success
continues to reject moving/crouched states. Physical validation is required
before selecting this experiment for submission.

## First result: 0/5 after 500 updates

The controlled-rise run completed 500 updates in 15 min 27 s (excluding
startup), then scored 0/5 on seeds 101-105. Maximum pelvis height was
0.387-0.426 m, below the unchanged 0.58 m requirement. Both feet touched the
floor for 89.0-94.8% of the episode, but another body part provided support
for over 99% of each episode. Final non-foot force was 160-170 N at the most
loaded non-foot body. The recorded policy settles into a bent, hand-supported
pose. Less motion did not establish recovery.

The learned action standard deviations averaged 0.066 (hip/knee about 0.02),
down from initial 0.8. This combination made exploration narrow. These data
do not isolate a single coefficient as the cause. The trial is retained as
unsuccessful and does not replace the selected model2248.

## Follow-up: load transfer, same 500-update budget per run

**Implemented but not run.** The next requested experiment instead tests the
[original reward set for 2,000 updates](original_rewards_experiment.md).

The binary existence of foot contact did not mean the feet carried the
robot. `--load_transfer` replaces the height term's support factor with:

```text
foot_share = sum(upward floor force on feet) / max(sum(upward floor force on all bodies), 1 N)
transfer = clip((pelvis_height - 0.20) / 0.38, 0, 1)
support_factor = 1 - transfer + transfer * foot_share
```

The remaining height/orientation/vertical-speed factors are unchanged. This
allows floor-level push-off and progressively favors foot load as height
increases. It is a fixed function of robot state, not a training curriculum.
Non-foot support cost now starts at 0.25 m, reaching its full weight -3 at
0.58 m. Action-change weight returns to -0.02 and PPO entropy coefficient is
0.01 to reduce the pressure toward early loss of exploration. These are X2
engineering choices, not paper constants. The follow-up changes several
related terms and is not a single-variable ablation.

```bash
./scripts/train_isaac.sh --phase relaxed_v4 --load_transfer \
  --num_envs 3000 --max_iterations 500 --seed 47 --device cuda:0 \
  --run_name load_transfer_scratch500 \
  --action_std_override 0.8 --entropy_coef 0.01 \
  --learning_rate_override 0.0003 --learning_schedule adaptive
```

Use the evaluation command above after this run completes. This starts new
random weights; it does not resume the failed controlled-rise checkpoint.
