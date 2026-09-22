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
| Robot formed an inverted bridge | Height-only reward | Signed height, upright and support criteria |
| V3 held arms forward | Final arm pose absent from objective | Supported-upright relaxed-arm reward |
| First arm reward did not move arms | Speed gate switched reward off during motion | Keep speed in balance reward, not arm gate |
| Fine-tuning retained forward arms | Existing local optimum | Train terminal pose from initialization |

## Curriculum disclosure

Earlier pretraining mixed true-supine starts with auxiliary upright-root squat/sitting starts. These were reset states, not expert trajectories. A historical reference sequence did not cover the true lower supine-to-bridge transition, explaining why a policy could work from a reference pose yet score 0/5 from the floor. The final environment default, final 51-update experiment, evaluation and ROS runtime use only true supine resets.

## Imitation tooling disclosure

Historical behavioral-cloning tools were corrected to stop at the first episode boundary, store episode offsets, split train/validation by whole episode and clear obsolete Adam moments after replacing actor weights. The selected relaxed-v4 policy does not use those datasets or checkpoints.

## GitHub timing
