# Development history

Superseded engineering outcomes remain visible in chronological commits. Early result-labelled commits are historical and are superseded by stricter measurements.

## Main failures and fixes

| Failure | Root cause | Fix |
| --- | --- | --- |
| Outer clones fell below scene | Local floor did not cover the clone grid | 200 m local floor |
| Reset began with a fall | Provisional pelvis height ignored collision geometry | 0.190 m measured reset |
| Standing probe appeared inverted | WXYZ literal passed to XYZW writer | Explicit quaternion convention tests |
| Only pelvis contact detected | Nested imported hierarchy did not match sibling glob | Recursive 32-body exact sensor |
| Contact persisted across reset | Historical force buffer was stale | Current ground-filtered force and local reset |
| Orientation stayed stale after reset | Same-timestamp derived-state cache | Invalidate root-derived buffers |
| Knees could not straighten | Joint range contracted twice | One 98% soft-limit margin |
| Corrective actions vanished near stance | Relative action brake suppressed authority | Absolute unsuppressed targets |
| Dataset/evaluator crossed episodes | Auto-reset samples appended after `done` | First-episode active mask and pre-reset snapshot |
| Validation MSE leaked | Adjacent rows from one episode entered both splits | Episode-level split |
| Reference-start policy failed from the floor | Reset references omitted the lower supine-to-rise transition | Evaluate from real supine starts and train the required transition |
| Robot formed an inverted bridge | Height-only reward | Signed height, upright and support criteria |
| V3 held arms forward | Final arm pose absent from objective | Supported-upright relaxed-arm reward |
| First arm reward did not move arms | Speed gate switched reward off during motion | Keep speed in balance reward, not arm gate |
| Fine-tuning retained forward arms | Existing local optimum | Train terminal pose from initialization |

## Curriculum disclosure

Earlier pretraining mixed true-supine starts with auxiliary upright-root squat/sitting starts. These were reset states, not expert trajectories. A historical reference sequence did not cover the true lower supine-to-bridge transition, explaining why a policy could work from a reference pose yet score 0/5 from the floor. The historical final 51-update experiment used supine resets, but inherited the earlier mixed-start weights. The selected model2397 descends from a new, independent random-weight lineage with every training reset supine.

## Imitation tooling disclosure

Historical behavioral-cloning tools were corrected to stop at the first episode boundary, store episode offsets, split train/validation by whole episode and clear obsolete Adam moments after replacing actor weights. The selected relaxed-v4 policy does not use those datasets or checkpoints.

## Recorded continuation settings

The historical model450 was obtained by continuing the mixed-start pretraining lineage. The recorded final stage used only supine starts, but its weights did not start from random initialization. The main command guide now reproduces the separate supine-only dense-stance experiment. This historical continuation command loads the historical parent from Git history and must not be labelled training from scratch:

```bash
./scripts/train_isaac.sh --phase relaxed_v4 --num_envs 3000 \
  --max_iterations 51 --seed 47 --device cuda:0 \
  --checkpoint /path/to/historical/model400.pt \
  --reset_optimizer --action_std_override 0.10 \
  --learning_rate_override 0.0001 --learning_schedule fixed \
  --entropy_coef 0.001
```

The current supine reset has a dedicated function with no reference-pose parameters. Fresh runs also begin with an episode clock of zero. Historical curriculum functions remain for earlier variants and development evidence; they are not the reset used by relaxed_v4.


## Supine-only scratch experiments

The original 500-update scratch run scored 0/5: height/upright shaping rewarded
ballistic motion while strict supported stance was never visited. The historical
model450 had mixed-start pretraining; its result did not establish successful
training from random weights with supine-only resets.

| Experiment | Initialization | Updates | Five-seed result |
| --- | --- | ---: | --- |
| Original scratch | Random, std 1.0, entropy 0.005 | 500 | 0/5; no strict stance. |
| Support-gated refinement | Original scratch model499, std 0.3 | 100 | 0/5; no strict stance. |
| Action-resolution refinement | Original scratch model499, std 0.3 | 300 | 0/5; saturation decreased, unstable rise remained. |
| Dense-stance scratch | Random, std 0.8, entropy 0 | 500 | 0/5; at most 0.04 s strict stance. |
| Same-preset continuation (parent) | Dense-stance model499; critic/optimizer/std retained | +500 (1,000 total) | **5/5; 8.78–9.02 s strict stance at episode end. Arms forward.** |
| Positive posture refinement (rejected) | Stance model998, std 0.15, fixed 1e-4 | +200 | 3/5; still fails neutral-arm check. |

All of these experiments reset every episode supine. None uses the mixed-start
model450, expert actions, reference starts or lift forces. The current optional
preset removes the contact gate from height shaping, broadens training-only
stance proximity, scores leg posture, and regularizes ineffective saturated
commands. Its success criterion is unchanged. These combined interventions do
not isolate the contribution of a single reward. Exact run configurations,
checkpoints and results remain in their individual log directories; see
[design and evidence](design_and_evidence.md) for formulas and measured failures.


The first two 500-update runs form the stance parent. The earlier failed combined posture refinements are separate branches and are not inherited by the final model.
A separate negative posture-cost experiment achieved brief success in 2/3
development screens but ended standing in 0/3; this was not the canonical
five-seed evaluation. Neither posture experiment is selected.

The original ROS server also required neutral arms, whereas the evaluator's
HRS recovery result required physical stance only. The new checkpoint exposed
that mismatch: it stood but ROS timed out. Both now use the same unchanged
strict stance predicate; the additional arm-pose check remains explicit in
evaluation. No height, orientation, velocity, contact or hold threshold was relaxed.

The earlier published artifacts are available at commit
[`4c97e60`](https://github.com/enisycn/x2-recovery-take-home/tree/4c97e60/reports).
The selected lineage source snapshots include the uncommitted changes that were
present during training; commit IDs alone are not presented as exact source evidence.

## Neutral arms from the supine-only lineage

All continuations retain actor, critic, optimizer and learned per-channel std.
None loads mixed-start weights or demonstration data.

| Stage | Selected updates | Measured outcome |
| --- | ---: | --- |
| Focused shoulder command, width denominator 9 | 300 | 3/3 development episodes end standing; shoulders still 1.11–1.20 rad. |
| Same reward, fixed rate 1e-4 | 454 to model1750 | 3/3 end standing; shoulders 0.46–0.49 rad. |
| Rejected later tail of that run | 146 additional, not inherited | Model1896 ends standing in only 1/3 development episodes. |
| Resume model1750 with adaptive KL schedule | 300 | Model2049: 3/3 standing; shoulders 0.35–0.37 rad, still outside 0.30 rad tolerance. |
| Shoulder denominator narrowed to 1 | 200 | Model2248: 5/5 recovery and neutral arms; shoulders still slightly behind the neutral pose. |
| Shoulder target shifted from ratio 0.25 to 0.15 | 150 | **Model2397: 5/5 recovery, 5/5 neutral arms, shoulder error at most 0.116 rad.** |

The complete selected ancestry has 2,404 updates: 500 + 500 + 300 + 454 + 300 + 200 + 150.
RSL-RL repeats the loaded index at each continuation, producing filename model2397.
The full 600-update fixed-rate attempt remains in its run log; its last 146 updates
are a rejected branch, not hidden training in the selected policy. The main curve
contains every update inherited by the selected weights, including the 0/5 start.

The original shoulder raw output was saturated, so nearby actions produced nearly
identical position targets. The focused raw-command score supplied a gradient in
reward beyond that saturation without overriding deployed actions. The broad
score later distinguished small pose errors weakly; narrowing its denominator
increased that distinction while retaining the same target and maximum reward.
The physical success and 0.30-rad arm tolerances were not relaxed.

In [RSL-RL 5.0.1](https://github.com/leggedrobotics/rsl_rl/blob/v5.0.1/rsl_rl/algorithms/ppo.py),
KL adjusts the learning rate only with the adaptive schedule. The fixed-rate tail
regressed; resuming the earlier checkpoint with adaptation preserved stance in
all six development screens. This is evidence from these runs, not an ablation
proving KL alone caused the regression. PPO clipping is not a strict trust-region
guarantee; see [OpenAI's PPO explanation](https://spinningup.openai.com/en/latest/algorithms/ppo.html).

Final model2397 measurements: maximum final-window shoulder error 0.116 rad, elbow error
0.104 rad, arm-joint position range 0.00111 rad and velocity RMS 0.0196 rad/s. The preceding model2248 had 0.233 rad worst shoulder error; the change is measured, not inferred from appearance.
These are 50 Hz simulation measurements, not hardware performance claims.

## Fixed final rewards from scratch: 500-update check

A separate 3,000-environment run starts actor and critic from random weights and keeps the entire final reward configuration fixed for all 500 updates. Every reset remains supine. Seed 47, initial std 0.8, entropy 0 and adaptive learning rate starting at 3e-4 match the earlier scratch setup. No checkpoint is loaded. The final `model_499.pt` represents 500 updates / 48 million environment steps; training took 17 min 19 s excluding simulator startup.

```bash
./scripts/train_isaac.sh --phase relaxed_v4 --num_envs 3000 \
  --max_iterations 500 --seed 47 --device cuda:0 \
  --run_name supine_final_rewards_scratch500 \
  --stability_refinement --posture_refinement --shoulder_command_variance 1 \
  --action_std_override 0.8 --entropy_coef 0 \
  --learning_rate_override 0.0003 --learning_schedule adaptive
```

Seeds 101-105 scored **0/5**. All five ran the full 500 evaluation steps without a safety termination, reaching 0.718-0.762 m pelvis height but zero time satisfying all strict stance conditions together. Orientation and base speed remained unstable. The final training mean return was 398.02, demonstrating that rising reward alone does not establish recovery. Per-seed criteria, checkpoint hash and source revision are recorded in `reports/supine_experiments.json`; the complete local run preserves the reward plot, checkpoint, configuration, evaluation and seed-101 video.

This single run shows that 500 updates were insufficient for this fixed configuration and seed. It does not establish a universal minimum budget or prove that another reward/exploration setup cannot succeed faster. The selected model2397 and its verified 5/5 result remain separate from this 500-update trial.

## Controlled-rise reward experiment

A separate fixed 500-update scratch run reduced ballistic height credit and
made slowing-down feedback denser. It scored 0/5: the policy settled into a
hand-supported pose below the required height. Full commands, formulas and
limitations are in [the experiment note](controlled_rise_experiment.md), with
the [five-episode report](../reports/experiments/controlled_rise500/evaluation.json)
and [training curve](../reports/experiments/controlled_rise500/reward.png).
The proposed load-transfer follow-up was implemented but not run.

## Original rewards for 2,000 updates

A fresh 3,000-environment, supine-only run retained the original 12 reward terms
and original PPO settings for all 2,000 updates. It scored **0/5**: pelvis height
reached 0.847–0.868 m, but no strict stance occurred. The mean reward plateaued
near 400; the robot continued rising and falling without settling. This budget
extension did not solve the original experiment. It is not an isolated reward
ablation against the selected model, which also used different exploration
settings. See [the comparison and reproduction commands](original_rewards_experiment.md).
The selected model2397, exported ROS policy and validated results remain separate from this original-reward trial.

## Final shoulder target adjustment

From the fully validated model2248, 150 more PPO updates changed only the existing supported shoulder-command reward's target ratio from 0.25 to 0.15. The reward's weight, denominator and gate, the robot dynamics, observations, actions, termination and evaluator thresholds stayed fixed. The optimizer and learned action std were retained. Five fresh physical episodes again scored 5/5, all standing at the end, with worst shoulder error 0.116 rad rather than 0.233 rad. This is a small local posture refinement, not a new reward term or a demonstration. The seven-stage parent hash/configuration audit is saved with model2397.
