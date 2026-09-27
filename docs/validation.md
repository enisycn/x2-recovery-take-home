# Validation

## Policy

`reports/relaxed_v4_evaluation.json` records five deterministic 10 s episodes from true supine resets with seeds 101-105. All five:

- satisfy the strict recovery predicate continuously for at least 0.5 s;
- stand at episode end;
- satisfy the relaxed-arm predicate;
- satisfy both predicates through the final two seconds;
- complete 500 policy steps without safety termination.

Strict duration is 8.54-8.94 s; combined relaxed duration is 7.78-8.86 s. Evaluation disables reference starts, lift assistance, observation noise and domain randomization.

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

`reports/unit_tests_v4.txt` records 29 passing tests across reward formulas, reset geometry, action mapping, success criteria and ROS session behavior. These are fast regression checks: 21 Isaac task/formula tests, four reduced-order harness tests and four ROS session/IPC tests. Tests support implementation correctness; the five real Isaac episodes establish physical behavior. The current expanded suite has 32 passing checks, recorded in `reports/unit_tests_stance.txt`. See `docs/test_matrix.md` for the distinction and coverage map.

## Supine-only scratch setup check (27 September 2026)

The dedicated reset was checked in the actual Isaac/PhysX environment with 32 robots, five seeds and policy-step counters 0 and 1,000,000: all 320 resets remained supine at pelvis height 0.190 ± 0.001 m, with neutral joints, zero initial root/joint velocities and zero episode clocks. The active configuration has no curriculum terms, auxiliary-pose parameters or lift assistance. A fresh PPO runner had iteration 0, an empty optimizer state, initial action standard deviation 1.0 and finite 32 × 8 inference outputs. No PPO learning was executed. The training CLI rejects historical phases and auxiliary reset/handoff options before launching Isaac. The 29 regression tests passed again. These checks establish the new experiment's setup, not recovery success from scratch.

The supplied model450 was also re-evaluated for all five seeds with the dedicated reset: 5/5 recovered, and the complete episode records matched the earlier reset implementation. This is a regression check of the existing checkpoint, not a result for the new scratch experiment.
