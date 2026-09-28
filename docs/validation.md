# Validation

## Policy

`reports/relaxed_v4_evaluation.json` records `x2_supine_model3847` in five deterministic 10 s episodes, seeds 101–105. `reports/relaxed_v4_extra_evaluation.json` adds seeds 106–110. All ten complete 500 steps without safety termination.

Seeds 101–105 all recover and remain standing at episode end for 8.88–9.00 s. Seeds 106–110 also score 5/5 with 8.88–8.90 s final strict stance. In the required five, height peaks at 0.671–0.687 m with both feet touching the floor at each peak; this contrasts with the fixed-reward 1,500-update alternative's airborne seed-101 height peak. Terminal shoulder pitch is about -0.39 to -0.44 rad and elbow about -0.85 to -0.87 rad. Those arm measurements are descriptive, not HRS success criteria.

The selected lineage began with random weights and has 3,860 selected PPO updates with exclusively supine resets. The first 1,000 learn stance with dense feedback. The intermediate 2,404-update checkpoint adds a neutral-arm preference. Final continuations replace that local preference with a forward-arm target while retaining the supported stance. No mixed-start pretraining, imitation, lift assistance or physical success-threshold change enters this lineage. The evaluator JSON still reports a historical optional neutral-arm diagnostic; failure of that diagnostic does not affect the 5/5 recovery count.

## ROS 2

`reports/ros_isaac_relaxed_v4_validation.txt` records a fresh one-package `colcon build` and the selected model3847 runtime check.

`reports/ros_isaac_relaxed_v4_validation.txt` records the actual Isaac IPC path:

- first Trigger request accepted before execution;
- second request rejected while pending/running;
- live 31-joint telemetry with timestamps;
- literal successful CLI episode;
- timeout changed to 0.2 s, producing `FAILED` after the real simulator timed out.

The recovery states are `IDLE`, `RUNNING`, `SUCCEEDED` and `FAILED`. The selected model's successful CLI episode completed after 81 simulation steps; the 0.2 s timeout trial failed after 10 steps. ROS ends after the physical 0.5 s success hold. Final stance and arm posture are established by the separate full 10 s evaluation, not this shorter service episode.

## Tests

`reports/unit_tests_v4.txt` records 29 passing tests across reward formulas, reset geometry, action mapping, success criteria and ROS session behavior. These are fast regression checks: 21 Isaac task/formula tests, four reduced-order harness tests and four ROS session/IPC tests. Tests support implementation correctness; the five real Isaac episodes establish physical behavior. The current expanded suite has 39 passing checks, recorded in `reports/unit_tests_stance.txt`. See `docs/test_matrix.md` for the distinction and coverage map.

## Supine-only scratch setup check (27 September 2026)

The dedicated reset was checked in the actual Isaac/PhysX environment with 32 robots, five seeds and policy-step counters 0 and 1,000,000: all 320 resets remained supine at pelvis height 0.190 ± 0.001 m, with neutral joints, zero initial root/joint velocities and zero episode clocks. The active configuration has no curriculum terms, auxiliary-pose parameters or lift assistance. A fresh PPO runner had iteration 0, an empty optimizer state, initial action standard deviation 1.0 and finite 32 × 8 inference outputs. No PPO learning was executed. The training CLI rejects historical phases and auxiliary reset/handoff options before launching Isaac. The 29 regression tests passed again. These setup checks predate the subsequent dense-stance 1,000-update experiment; its separate five-episode report establishes successful supine-only training.

The historical model450 was also re-evaluated for all five seeds with the dedicated reset: 5/5 recovered, and the complete episode records matched the earlier reset implementation. This is a regression check of the existing checkpoint, not a result for the new scratch experiment.

The ROS server previously required neutral arms as well as stance. A model998 trial therefore timed out despite standing. That additional check was removed from ROS success to match the evaluator and HRS task; it remains a separately reported evaluation metric. The current runtime validation uses the identical height, orientation, speed, support and 0.5 s hold thresholds.
