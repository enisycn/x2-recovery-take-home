> Current delivery: [relaxed_v4](relaxed_v4.md), with relaxed arms and 100% supine final resets. Earlier variants below are historical.

> Current selected result: **symmetric_v3 checkpoint 200, 5/5 true-supine recoveries**. See [current method](symmetric_v3.md), [shared formulas](simple_v2.md) and [final evaluation](../reports/symmetric_v3_evaluation.json). Earlier HumanUP/history and brake analyses below describe historical configurations, not the selected controller.

# Evidence-to-design map

Sources were selected in this order: official robot assets and documentation, peer-reviewed recovery research, primary preprints for recent work, and official simulator/ROS documentation. The papers motivate design choices; they do not prove that the selected X2 weights are optimal.

## Primary sources

1. Huang et al., **Learning Humanoid Standing-up Control across Diverse Postures**, Robotics: Science and Systems 2025, DOI 10.15607/RSS.2025.XXI.064. [Proceedings](https://www.roboticsproceedings.org/rss21/p064.html) and [official implementation](https://github.com/InternRobotics/HoST). HoST supplies height-stage completion rewards and cross-robot guidance for a near-vertical, approximately 60%-body-weight pull.
2. He et al., **Learning Getting-Up Policies for Real-World Humanoid Robots**, Robotics: Science and Systems 2025, DOI 10.15607/RSS.2025.XXI.063. [Proceedings](https://www.roboticsproceedings.org/rss21/p063.html). HumanUP supplies dense discovery objectives, mixed reference starts and the evidence-backed path from discovery to motion imitation/refinement.
3. Gaspard et al., **FRASA: An End-to-End Reinforcement Learning Agent for Fall Recovery and Stand Up of Humanoid Robots**, arXiv:2410.08655, revision 3, 2025. [Paper](https://arxiv.org/abs/2410.08655). It supports 20 Hz decisions, 100 Hz physical control and a sustained physical-stability test.
4. Schulman et al., **Proximal Policy Optimization Algorithms**, arXiv:1707.06347, 2017. [Paper](https://arxiv.org/abs/1707.06347).
5. Hwangbo et al. (ETH Zurich), **Learning Agile and Dynamic Motor Skills for Legged Robots**, *Science Robotics* 4(26), 2019, DOI 10.1126/scirobotics.aau5872. [ETH record](https://www.research-collection.ethz.ch/items/63b8fb50-299f-4995-bd1d-63a9f2b7ba75). This quadruped work supports only contact-rich self-righting, position targets and simulator-to-hardware engineering.
6. Eßer et al. (Fraunhofer IML and MIT), **Action Space Design in Reinforcement Learning for Robot Motor Skills**, CoRL 2024, PMLR 270, 2025. [Proceedings](https://proceedings.mlr.press/v270/esser25a.html). It shows that action representation materially changes learning and that joint-position targets are a defensible choice; it does not establish one universal action space.
7. Peng, Bao and Zhou (University College London), **Gait-Conditioned Reinforcement Learning with Multi-Phase Curriculum for Humanoid Locomotion**, IEEE-RAS Humanoids 2025, DOI 10.1109/HUMANOIDS65713.2025.11203058. [UCL manuscript](https://discovery.ucl.ac.uk/10219246/1/2025_Humanoid_Conference_Paper-13.pdf). This supports progressive task difficulty but is not direct fall-recovery validation.
8. Xu et al., **Unified Humanoid Fall-Safety Policy from a Few Demonstrations**, arXiv:2511.07407, 2025. [Preprint](https://arxiv.org/abs/2511.07407). It scopes later fall/impact work only.
9. AgiBotTech, **official AgiBot X2 URDF models**. [Repository](https://github.com/AgibotTech/agibot_x2_urdf). The scripts pin reviewed commit `60c5de582c523cd188f563819e62d34cfdc3d2d0` and select X2 Ultra v1.3.0 simplified collision geometry.
10. AgiBot, **X2 coordinate system and joint limits**. [Coordinate system](https://x2-aimdk.agibot.com/en/latest/about_agibot_X2/coordinate_system.html) and [joint limits](https://x2-aimdk.agibot.com/en/latest/about_agibot_X2/joint_name_and_limit.html).
11. NVIDIA, **Isaac Sim URDF importer** and **Isaac Lab manager-based RL**. [Importer](https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/ext_isaacsim_asset_importer_urdf.html), [RL concepts](https://isaac-sim.github.io/IsaacLab/develop/source/concepts/reinforcement_learning.html), [manager-based environment](https://isaac-sim.github.io/IsaacLab/main/source/how-to/create_manager_base_env.html).
12. Open Robotics, **ROS 2 Humble interfaces and QoS**. [Interfaces](https://docs.ros.org/en/humble/Concepts/Basic/About-Interfaces.html) and [QoS](https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html).

## Current-literature check (22 September 2026)

Two newer recovery papers were also checked so the design is not frozen at the
2025 baseline. Hou et al., **Robust Fall Recovery for Armless Bipedal-Wheeled
Robots Via Force-Guided Learning**, IEEE Robotics and Automation Letters 2026
([DOI](https://doi.org/10.1109/LRA.2026.3701481),
[preprint](https://arxiv.org/abs/2606.14270)), combines a height-dependent
auxiliary force, stage-wise rewards and teacher/student distillation. It
supports the earlier HoST-style force curriculum as a discovery tool, but does
not justify leaving force enabled during evaluation. The X2 result therefore
keeps assistance disabled and reports the learned behavior unchanged.

Wu et al., **StableMimic** ([arXiv:2608.02385](https://arxiv.org/abs/2608.02385),
submitted 3 August 2026) trains recovery around several human get-up references
and blends dedicated tracking and recovery experts. It is a recent,
non-peer-reviewed preprint, so it is treated as supporting evidence for the
HumanUP motion-extraction/Stage-II direction rather than copied into the
submission. In particular, it does not create demonstration data for X2; a
successful X2 trajectory still has to be collected or retargeted and validated.

## Traceability

| Design decision | Evidence and measured reason | Repository implementation |
| --- | --- | --- |
| Use official X2 Ultra v1.3.0 simplified collision geometry | AgiBot distinguishes the original X2 Ultra from the later Ultra-N; HRS names X2 without the newer hardware identifier | pinned fetch/staging checks and `import_x2_urdf.py`; hardware nameplate must be checked before transfer |
| Encode a non-intersecting supine reset | AgiBot's FLU frame plus the official collision meshes determine orientation and clearance | scalar-last `XYZW=(0,-0.7071,0,0.7071)`, 0.190 m pelvis, zero speed; `x2_geometry_audit.json` and `x2_reset_probe.json` |
| Keep a floating, self-colliding model on a local floor | NVIDIA's importer exposes floating humanoid and self-collision options; local geometry avoids an online Nucleus dependency | repo-local USD and 200 m collision cuboid covering all 4,096 configured clones |
| Measure full support state | Net body forces include X2 self-collisions, while the assignment requires floor support | one exact recursive 32-body PhysX view filtered against `/World/ground`; reward and evaluation use ordered ground-only forces |
| Use relative joint-position targets | HoST, HumanUP and ETH use position-level control; the MIT/Fraunhofer study shows action choice must be tested per task | `q_cmd=clip(q+0.25*tanh(a), soft_limits)`, sampled once at 20 Hz and held over five 100 Hz physics steps |
| Gate height by signed upright orientation | HumanUP and HoST use height/upright task families; measured X2 branches exploited raw head height, upward velocity and ungated base height | linear normalized pelvis height times `clip(-g_z,0,1)^2`; separate signed upright target |
| Shape foot loading and final unsupported contact | The assignment requires both feet and no other support; FRASA uses contact and physical-stability checks | two-foot reward from low height, non-foot penalty near standing, and an unchanged exact-force evaluator |
| Activate completion terms near standing | HoST separates rising from post-task stability objectives | height-gated target-height, planar-speed, angular-speed and orientation exponentials above 0.58 m |
| Use adjacent reference poses | HumanUP mixes useful postures for discovery and HoST stages task difficulty; the selected checkpoint's reference-2-only final phase failed all true-supine starts | nine coupled root/joint poses from 0.092 to 0.680 m, with focused adjacent-stage training and a separate true-supine evaluator |
| Keep assistance out of the result | HoST uses upward assistance for exploration, while the assignment asks whether the learned controller recovers | optional morphology-scaled force remains disabled in every reported evaluation |
| Use 100 Hz physics and 20 Hz decisions | FRASA reports this ratio; it retains five contact solves per decision while removing an unsupported 200 Hz cost | `sim.dt=0.01`, `decimation=5` |
| Do not terminate on success | HumanUP uses broad safety endings and standing must be maintained rather than briefly touched | 10 s timeout plus 2.5 m/s, `[0,1.2]` m and finite-state safety checks; evaluation separately requires every stance condition for 0.5 s |
| Train nominal dynamics first | The requested result is nominal simulation recovery and the policy did not yet solve it; adding uncertainty would confound that result | fixed official mass/inertia, floor friction and gains in the selected experiment; randomization code is disabled |
| Use GPU-parallel PPO with temporal compression | HumanUP's RMA encoder handles contact-history dependence; PPO supplies clipped updates | 3,000 environments, 24 steps/update, 1,148 critic inputs, 98 current plus 20 history-latent actor inputs, `[512,256,128]` ELU heads, clip 0.2 |
| Keep ROS request acceptance short and status durable | ROS services are short request/response calls; transient-local QoS retains the latest status | callback sets an atomic pending/running gate, a timer dispatches work, and the status publisher is reliable/transient-local |
| Separate ROS and Isaac runtimes | The installed ROS Humble and Isaac environments use incompatible Python ABIs | newline JSON over a mode-0600 local Unix socket; verified imports and end-to-end service/telemetry path |
| Keep fall/impact safety outside the claim | Recent fall-safety work treats impact mitigation and recovery as separate connected problems | explicit hardware limitations; no hardware-readiness claim |

## What the evidence does and does not establish

This is an evidence-derived take-home implementation, not a reproduction of a cited system. HoST uses a more specialized architecture and adapts exploration from learning progress. HumanUP performs large-scale discovery, extracts a motion and adds an imitation/refinement stage. FRASA uses a smaller sagittal controller and a different RL algorithm. This repository uses one PPO actor/critic, a complete 31-joint X2 model and 400 updates, so its result must be judged by the recorded five-episode evaluation.

The selected policy approaches standing from curriculum `reference_2` but scores 0/5 from true supine starts. The independent standing probe proves the imported model and strict predicate admit a valid stance; it does not prove that this policy can reach it. The measured follow-up that mixed stages 2–4 also failed stages 3 and 4, so the evidence supports adjacent-stage mastery and then HumanUP-style motion imitation/refinement. Domain randomization becomes useful only after nominal recovery succeeds and its ranges are identified from the actual robot.
