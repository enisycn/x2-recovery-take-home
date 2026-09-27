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

Earlier pretraining mixed true-supine starts with auxiliary upright-root squat/sitting starts. These were reset states, not expert trajectories. A historical reference sequence did not cover the true lower supine-to-bridge transition, explaining why a policy could work from a reference pose yet score 0/5 from the floor. The final environment default, final 51-update experiment, evaluation and ROS runtime use only true supine resets.

## Imitation tooling disclosure

Historical behavioral-cloning tools were corrected to stop at the first episode boundary, store episode offsets, split train/validation by whole episode and clear obsolete Adam moments after replacing actor weights. The selected relaxed-v4 policy does not use those datasets or checkpoints.

## Recorded continuation settings

The supplied model450 was obtained by continuing the mixed-start pretraining lineage. The recorded final stage used only supine starts, but its weights did not start from random initialization. The main command guide now selects a fresh, supine-only experiment. This historical continuation command loads the supplied parent and must not be labelled training from scratch:

```bash
./scripts/train_isaac.sh --phase relaxed_v4 --num_envs 3000 \
  --max_iterations 51 --seed 47 --device cuda:0 \
  --checkpoint reports/checkpoints/x2_relaxed_v4_parent_model400.pt \
  --reset_optimizer --action_std_override 0.10 \
  --learning_rate_override 0.0001 --learning_schedule fixed \
  --entropy_coef 0.001
```

The current supine reset has a dedicated function with no reference-pose parameters. Fresh runs also begin with an episode clock of zero. Historical curriculum functions remain for earlier variants and development evidence; they are not the reset used by relaxed_v4.


## Supine-only scratch run and stabilization

A separate 500-update scratch run used supine-only resets throughout and scored 0/5 at the final model499 checkpoint. Height and upright shaping could reward airborne high states while strict supported stance remained absent. Mean action std grew from 1.0 to 2.08045. The optional stability refinement targets the observed ballistic behavior with supported-height shaping, near-stance motion regularization and reduced exploration. It reuses only this supine-trained actor; it does not load the mixed-start model450. After 100 additional updates, model598 also scored 0/5, so this experiment is not a demonstrated fix. This combined intervention does not isolate the effect of any individual change. See [design and evidence](design_and_evidence.md) for formulas, measured results and provenance.
