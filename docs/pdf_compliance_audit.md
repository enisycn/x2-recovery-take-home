# HRS PDF compliance audit — 22 September 2026

Source: the user's two-page `HRS_Take_Home_Task.pdf` (AgiBot X2 Ground Recovery). Both pages were read as text and rendered locally; embedded links were not opened. The PDF has no JavaScript or forms. No software or model was downloaded for this audit.

## Requirement-level findings

| PDF section | Status / evidence |
| --- | --- |
| Official AgiBot X2, floating base, flat floor, collisions and limits | Implemented; pinned official URDF, local USD conversion, self collision, actuator limits and full 31-joint articulation. Eight coupled action groups are disclosed as an action-space simplification. |
| Every episode starts resting on its back without floor intersection | Corrected for the submitted `relaxed_v4` environment: reference reset probability is zero in training, evaluation and ROS. The final experiment is entirely supine. Earlier exploratory pretraining mixed supine and auxiliary squat/standing starts; this deviation is explicitly retained in the lineage rather than presented as PDF-compliant training. Geometry uses clearance so gravity settles initial contact without interpenetration. |
| Observations, actions, rewards, episode endings | Defined in the code and method documents; no success-at-reset or stale-contact shortcuts. |
| RL experiment, checkpoint, reward plot | Real GPU PPO training. Selected model/config, source hashes, verified export and measured reward curve are included. |
| Two ROS nodes and one launch file | Python recovery and telemetry nodes; a single launch file starts both. Isaac runs in a separate process connected by a local Unix socket to preserve ROS/Isaac Python isolation. |
| Trigger acceptance before work, busy rejection | Timer dispatch after callback return; pending/running attempt gate. Real integration test measures acceptance and immediate second-request rejection. |
| Required status/joint interfaces | Exact PDF names and message types; 31 simulator joint positions and ROS timestamps. Telemetry logs status and a selected joint. |
| Configurable timeout -> FAILED | Node reads timeout parameter for each episode. Final integration test sets 0.2 s and checks real Isaac timeout, not only the reduced CPU harness. |
| Fresh colcon build, launch, literal CLI request, live telemetry | A new empty `tmp/ros_fresh_build` and `tmp/ros_fresh_install` were built. Integration uses this install, `ros2 service call`, status transitions and recorded joint telemetry. |
| Five episodes and documented success/failures | Five seeded true-supine, full 10-second Isaac evaluations; strict stance measured before automatic reset. The user's additional relaxed-arm objective is evaluated separately. Failed trial reports and their interpretations remain available. PDF itself accepts zero successes with an explanation; we require actual success. |
| README setup, model changes, resources, environment, reward weights, RL, failures, improvements, ROS | Documented in README and directly linked technical/evidence documents. |
| Meaningful commit history | Preserved in the local standalone Git repository and submission `history.bundle`; no squashing into one final commit. |
| GitHub repository created at start; ongoing pushes; repository submission | **Not satisfied.** GitHub work was deferred at the user's request. No remote exists or upload has occurred. Creating a repository later and pushing the genuine existing history can complete repository submission, but cannot retroactively satisfy the earlier ongoing-push timing requirement. No dates or history are falsified. |

## Boundary of the result

The physics validations are nominal simulation tests, not hardware or broad robustness claims. The selected model uses the official mass/limits and no lifting assistance. Domain-randomized sim-to-real deployment and arbitrary fallen postures are outside this take-home scope. GitHub publication remains a separate delivery action; the local ZIP is not represented as a submitted GitHub repository.

## Final evidence

Selected model: `x2_relaxed_v4_model450.pt`; [five-episode report](../reports/relaxed_v4_evaluation.json) passes recovery and relaxed-arm stance on all five seeds. [Real ROS record](../reports/ros_isaac_relaxed_v4_validation.txt) passes literal CLI success, busy rejection, live telemetry and Isaac timeout. [Fresh build](../reports/ros_fresh_build_v4.txt), [29 unit tests](../reports/unit_tests_v4.txt), [configuration/lineage](../reports/configs/relaxed_v4_model450/provenance.json). GitHub delivery remains pending as stated above.

Source PDF SHA-256: `2503770be75c8be84e6aa98b02c567a0b5fa75c37071fd6a842078739b478ca9`.
