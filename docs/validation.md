# Validation of the selected policy

The selected policy is [`x2_supine_single1500.pt`](../reports/checkpoints/x2_supine_single1500.pt). It was trained from random weights in one uninterrupted 1,500-update PPO run with exclusively supine resets and fixed reward terms. Its [graph](../reports/selected_supine_training_reward.png), [CSV](../reports/selected_supine_training_reward.csv), [run settings](../reports/configs/supine_single1500/) and [video](../reports/videos/x2_recovery_supine_to_standing.mp4) have matching provenance; they do not describe an earlier resumed policy.

## Physical recovery rule

The evaluator runs each episode for all 500 policy steps (10 s), retaining the terminal state before reset. Success requires at least 0.5 continuous seconds with pelvis ≥0.58 m; projected-gravity XY norm ≤0.15 and Z ≤−0.98; root linear/angular speed ≤0.25 m/s and 0.35 rad/s; ≥15 N ground force on **each** foot; and <15 N on every other body. The robot must also stand at episode end. Brief hand or body support during the rise is allowed, and does not count as final success.

| Seed | Recovery | Continuous strict stance at end | Both feet at height peak |
| --- | --- | ---: | --- |
| 101 | Yes | 8.96 s | No |
| 102 | Yes | 8.94 s | No |
| 103 | Yes | 8.96 s | No |
| 104 | Yes | 8.88 s | No |
| 105 | Yes | 8.86 s | No |

The [machine-readable five-episode record](../reports/relaxed_v4_evaluation.json) is **5/5**. The [additional seeds 106–110](../reports/relaxed_v4_extra_evaluation.json) are also 5/5, with 8.84–8.88 s strict stance at end. All ten complete without a safety termination. Evaluation seeds are local reproducibility choices, each changing only small supine root-pose perturbations. They are not ten independent training seeds or a robustness claim.

The controller briefly raises both feet at the height peak, then lands and remains in two-foot stance. The airborne peak is an observable limitation and is excluded from the strict-standing duration. No failures occurred in the required five episodes. Training reward alone would not establish these outcomes.

## ROS 2 and regression checks

The [fresh `colcon` build](../reports/ros_fresh_build_v4.txt) passed. The [live Isaac–ROS log](../reports/ros_isaac_relaxed_v4_validation.txt) uses the TorchScript export of this same checkpoint. The CLI `Trigger` request returned `success=True` before execution; a second request while running was rejected; 31 measured joints with timestamps streamed during recovery; the normal episode reached `SUCCEEDED`; and a 0.2 s timeout produced `FAILED`. ROS stops after the physical 0.5 s success hold, whereas the five-episode evaluation checks the full 10 s stance. [Terminal commands](commands.md) reproduce the check.

Fast formula, reset and ROS-session regression tests are recorded in [`reports/unit_tests_stance.txt`](../reports/unit_tests_stance.txt). They check code behavior; real PhysX episodes establish the physical result. Earlier failed policies and the separately resumed successful lineage remain documented in [development history](development_history.md).
