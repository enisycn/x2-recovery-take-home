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

Earlier pretraining mixed true-supine starts with auxiliary upright-root squat/sitting starts. These were reset states, not expert trajectories. A historical reference sequence did not cover the true lower supine-to-bridge transition, explaining why a policy could work from a reference pose yet score 0/5 from the floor. The historical final 51-update experiment used supine resets, but inherited the earlier mixed-start weights. The currently selected model998 belongs to a new, independent random-weight lineage with every training reset supine.

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
| Same-preset continuation (selected) | Dense-stance model499; critic/optimizer/std retained | +500 (1,000 total) | **5/5; 8.78–9.02 s strict stance at episode end. Arms forward.** |
| Positive posture refinement (rejected) | Selected model998, std 0.15, fixed 1e-4 | +200 | 3/5; still fails neutral-arm check. |

All of these experiments reset every episode supine. None uses the mixed-start
model450, expert actions, reference starts or lift forces. The current optional
preset removes the contact gate from height shaping, broadens training-only
stance proximity, scores leg posture, and regularizes ineffective saturated
commands. Its success criterion is unchanged. These combined interventions do
not isolate the contribution of a single reward. Exact run configurations,
checkpoints and results remain in their individual log directories; see
[design and evidence](design_and_evidence.md) for formulas and measured failures.


The complete selected curve combines the two 500-update runs. The failed
posture refinement is a branch and is not included in that curve or checkpoint.
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
