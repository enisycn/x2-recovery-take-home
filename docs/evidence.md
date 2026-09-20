# Evidence-to-design map

Sources were selected in this order: official robot assets, peer-reviewed robotics papers, the latest available primary preprint version, and official Isaac Lab/ROS documentation. The papers motivate the design; they do not establish that the chosen X2 parameters are optimal.

## Primary sources

1. Huang et al., **Learning Humanoid Standing-up Control across Diverse Postures**, Robotics: Science and Systems 2025, DOI 10.15607/RSS.2025.XXI.064. [Proceedings and paper](https://www.roboticsproceedings.org/rss21/p064.html).
2. Gaspard et al., **FRASA: An End-to-End Reinforcement Learning Agent for Fall Recovery and Stand Up of Humanoid Robots**, arXiv:2410.08655, version 3 revised 4 November 2025. [Abstract and revision history](https://arxiv.org/abs/2410.08655).
3. Schulman et al., **Proximal Policy Optimization Algorithms**, arXiv:1707.06347, 2017. [Paper](https://arxiv.org/abs/1707.06347).
4. AgiBotTech, **official AgiBot X2 URDF models**. [Repository](https://github.com/AgibotTech/agibot_x2_urdf).
5. AgiBot, **X2 coordinate-system and joint-limit documentation**. [Coordinate system](https://x2-aimdk.agibot.com/en/latest/about_agibot_X2/coordinate_system.html) and [joint limits](https://x2-aimdk.agibot.com/en/latest/about_agibot_X2/joint_name_and_limit.html).
6. NVIDIA, **Isaac Sim URDF importer**. [Official importer documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/ext_isaacsim_asset_importer_urdf.html).
7. NVIDIA, **Isaac Lab reinforcement learning and manager-based task documentation**. [RL concepts](https://isaac-sim.github.io/IsaacLab/develop/source/concepts/reinforcement_learning.html) and [manager-based environment guide](https://isaac-sim.github.io/IsaacLab/main/source/how-to/create_manager_base_env.html).
8. Open Robotics, **ROS 2 Humble interfaces and QoS**. [Topics, services and actions](https://docs.ros.org/en/humble/Concepts/Basic/About-Interfaces.html) and [QoS settings](https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html).

## Traceability

| Design decision | Evidence | Repository implementation |
| --- | --- | --- |
| Use the official X2 Ultra v1.3.0 model and simplified collision URDF | AgiBot identifies v1.3.0 as the original flagship X2 Ultra and v1.4.0 as X2 Ultra -N; the assignment does not identify an -N robot | `fetch_agibot_model.sh`, `import_x2_urdf.py`; confirm the physical nameplate before transfer |
| Initialize supine rather than prone | AgiBot documents an FLU body frame: X forward, Y left, Z up; -90° about Y maps the forward/chest axis upward | `X2_CFG.init_state.rot=(0.7071, 0, -0.7071, 0)` |
| Convert URDF into a repo-local, floating, self-colliding humanoid USD | Isaac Sim 6 documents `URDFImporter`/`URDFImporterConfig`; official importer preserves robot structure and exposes humanoid/floating-base options | `UrdfConverterCfg(fix_base=False, robot_type="Humanoid", self_collision=True)` |
| Split recovery shaping by pelvis height into righting, rising and standing | HoST divides standing-up training into three stages using base height and reports robust real-robot stand-up | `staged_recovery_progress()` thresholds at 0.35 m and 0.58 m |
| Reward both feet and penalize every other supporting body | The assignment requires two feet with no other support; FRASA explicitly rewards collision-free target behavior and uses physical stability checks | `both_feet_contact()`, `unsupported_contacts()`, `strict_success()` |
| Smooth and bound position targets | HoST identifies smoothness regularization and implicit motion-speed bounds as necessary to reduce oscillatory and violent hardware motion; FRASA sends integrated desired positions to servo control | EMA joint-position-to-limits action, action-rate and joint-velocity penalties |
| Use 20 Hz policy control with faster physics | FRASA reports a 20 Hz decision process and 100 Hz physical control; faster simulation substeps preserve contact resolution | 200 Hz PhysX with `decimation=10` |
| Require a sustained stable pose | FRASA checks all state errors for 0.5 s before declaring physical stability | ten consecutive 20 Hz strict-success steps |
| Randomize friction, mass/CoM and actuator gains conservatively | HoST and FRASA both randomize contact, inertial and actuator properties for transfer; Isaac Lab exposes these as event terms | startup material, mass, pelvis CoM and gain events |
| Add action, velocity, torque and joint-limit regularization | HoST groups task, style, regularization and post-task rewards; its hardware constraints motivate low-speed, low-effort motion | reward table in `RewardsCfg` |
| Use PPO with clipped updates, GAE and normalized MLP observations | PPO is a standard on-policy clipped objective; Isaac Lab recommends RSL-RL for GPU-parallel robot learning and supplies humanoid PPO examples | `X2RecoveryPPORunnerCfg` |
| Keep service acceptance short and execute asynchronously | ROS documents services as short request/response interactions; the task explicitly mandates `Trigger` rather than an action | service callback sets a pending flag; timer starts the worker after the response |
| Make status durable for late subscribers | ROS 2 transient-local durability retains the last sample for compatible late subscribers | reliable, transient-local `/x2/recovery_status` publisher |
| Separate ROS Humble Python from Isaac Python | This is an engineering constraint observed locally, not a research claim; the runtimes use different Python ABIs | newline JSON over a mode-0600 Unix socket |

## Adaptations and limits

HoST uses a multi-critic architecture and curriculum forces; this take-home keeps one PPO critic and a dense staged reward to reduce implementation surface. FRASA trains a smaller sagittal controller with Cross-Q; this implementation instead controls every imported X2 joint through bounded targets because the assignment permits PPO and requires the complete X2 state. Domain-randomization ranges are intentionally narrower than the cited work and must be replaced with measured X2 uncertainty before hardware transfer.
