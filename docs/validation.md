# Validation

## Policy

`reports/relaxed_v4_evaluation.json` records model998 in five deterministic 10 s episodes, seeds 101–105. All five recover and remain in unsupported two-foot stance at the end; all complete 500 steps without safety termination. Final uninterrupted strict stance lasts 8.78–9.02 s.

The separate neutral-arm check fails for all five: the arms remain forward. HRS recovery and ROS success use the unchanged physical stance predicate, not this extra arm preference.

The selected lineage began with random weights and trained for 500 + 500 updates using supine resets only. The same dense reward preset was used in both runs; no reset curriculum, reference starts, lift assistance, imitation, observation noise or dynamics randomization was used.

## ROS 2

`reports/ros_fresh_build_v4.txt` records a clean one-package `colcon build`.

`reports/ros_isaac_relaxed_v4_validation.txt` records the actual Isaac IPC path:

- first Trigger request accepted before execution;
- second request rejected while pending/running;
- live 31-joint telemetry with timestamps;
- literal successful CLI episode;
- timeout changed to 0.2 s, producing `FAILED` after the real simulator timed out.

The recovery states are `IDLE`, `RUNNING`, `SUCCEEDED` and `FAILED`.

## Tests

`reports/unit_tests_v4.txt` records 29 passing tests across reward formulas, reset geometry, action mapping, success criteria and ROS session behavior. These are fast regression checks: 21 Isaac task/formula tests, four reduced-order harness tests and four ROS session/IPC tests. Tests support implementation correctness; the five real Isaac episodes establish physical behavior. The current expanded suite has 33 passing checks, recorded in `reports/unit_tests_stance.txt`. See `docs/test_matrix.md` for the distinction and coverage map.

## Supine-only scratch setup check (27 September 2026)

The dedicated reset was checked in the actual Isaac/PhysX environment with 32 robots, five seeds and policy-step counters 0 and 1,000,000: all 320 resets remained supine at pelvis height 0.190 ± 0.001 m, with neutral joints, zero initial root/joint velocities and zero episode clocks. The active configuration has no curriculum terms, auxiliary-pose parameters or lift assistance. A fresh PPO runner had iteration 0, an empty optimizer state, initial action standard deviation 1.0 and finite 32 × 8 inference outputs. No PPO learning was executed. The training CLI rejects historical phases and auxiliary reset/handoff options before launching Isaac. The 29 regression tests passed again. These setup checks predate the subsequent dense-stance 1,000-update experiment; its separate five-episode report establishes successful supine-only training.

The historical model450 was also re-evaluated for all five seeds with the dedicated reset: 5/5 recovered, and the complete episode records matched the earlier reset implementation. This is a regression check of the existing checkpoint, not a result for the new scratch experiment.

The ROS server previously required neutral arms as well as stance. A model998 trial therefore timed out despite standing. That additional check was removed from ROS success to match the evaluator and HRS task; it remains a separately reported evaluation limitation. The rerun below uses the identical height, orientation, speed, support and 0.5 s hold thresholds.
