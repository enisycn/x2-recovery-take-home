# Parameter and evidence provenance

This document separates four kinds of evidence:

- **PAPER**: a published method or qualitative design principle.
- **FRAMEWORK**: an RSL-RL or Isaac Lab implementation/default, rather than a research claim.
- **ROBOT/TASK**: a value fixed by the official X2 model or by the assignment contract.
- **LOCAL**: an X2-specific engineering choice supported by geometry audit, failure analysis or the final evaluation. A local value must not be presented as a paper constant.

The cited papers motivate the method. They do not establish that their exact values transfer to AgiBot X2. The complete serialized settings remain in `reports/configs/supine_model2397/{agent,env}.yaml`; this page covers every material value that was selected or interpreted for the submitted controller.

## Primary sources used in the implementation

- **PPO** — Schulman et al. (2017), [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). Supplies the clipped surrogate objective and repeated minibatch/epoch update structure.
- **GAE** — Schulman et al. (2015), [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438). Supplies the exponentially weighted advantage estimator and its bias/variance interpretation.
- **HumanUP** — He et al. (2025), [Learning Getting-Up Policies for Real-World Humanoid Robots](https://doi.org/10.15607/RSS.2025.XXI.063). Motivates accurate whole-body collision, height/upright/feet task terms, PPO MLP control and discovery/refinement curriculum thinking.
- **HoST** — Huang et al. (2025), [Learning Humanoid Standing-up Control across Diverse Postures](https://doi.org/10.15607/RSS.2025.XXI.064). Motivates posture-conditioned standing-up objectives, smoothness constraints and a curriculum pull during exploration. The final X2 controller uses no pull force.
- **FRASA** — Gaspard et al. (2025), [FRASA: An End-to-End Reinforcement Learning Agent for Fall Recovery and Stand Up of Humanoid Robots](https://arxiv.org/abs/2410.08655v3). Motivates a low-dimensional bilateral action space, previous-action observations, upright target proximity and action variation penalties.
- **RSL-RL** — [official configuration reference](https://github.com/leggedrobotics/rsl_rl/blob/main/docs/guide/configuration.rst). Defines the actual PPO/MLP implementation and its software defaults.
- **Official X2 model** — [AgibotTech/agibot_x2_urdf at pinned commit `60c5de5`](https://github.com/AgibotTech/agibot_x2_urdf/tree/60c5de582c523cd188f563819e62d34cfdc3d2d0). Supplies geometry, mass, inertia, joint axes and hard actuator/joint limits.

DAgger, DAPG, GAIL and DeepMimic are comparison methods only; the selected checkpoint does not use them.

## Neural network

| Item | Submitted value | Provenance and interpretation |
| --- | --- | --- |
| Actor input | 122 observations | **LOCAL/TASK.** Constructed from measurable X2 proprioception and contact state; dimensional audit is below. |
| Actor MLP | `122 → 512 → 256 → 128 → 8` | **FRAMEWORK + LOCAL.** RSL-RL provides the MLP implementation and uses ELU by default. The widths are an X2 capacity choice, not copied from PPO. PPO's paper experiment used two 64-unit tanh layers; FRASA used 384/256 with CrossQ, so neither is claimed as the source of these widths. |
| Critic MLP | `122 → 512 → 256 → 128 → 1` | **FRAMEWORK + LOCAL.** Separate value network, consistent with actor-critic PPO. It predicts scalar `V(s)` and is discarded for deployment. |
| Activation | ELU after each hidden layer | **FRAMEWORK.** RSL-RL default/robotics convention. No paper-specific X2 claim. |
| Actor output | Eight Gaussian means plus eight learned standard deviations | **PPO/FRAMEWORK.** Continuous stochastic actor during training. Deployment uses the deterministic mean. |
| Observation normalization | Separate running mean/variance for actor and critic | **FRAMEWORK.** Enabled locally because input scales mix metres, radians, velocities and binary contacts. |
| Actor parameters | 228,232 linear parameters + 8 learned standard-deviation parameters | **MEASURED.** Counted from `model2397.pt`; normalization statistics are buffers, not learned policy weights. |
| Critic parameters | 227,329 | **MEASURED.** Counted from the checkpoint. |
| Initial standard deviation | `0.8` at random initialization | **LOCAL.** Runtime override; retained learned values on continuation. The serialized class default is 1.0; runtime settings record the actual override. |
| Final mean standard deviation | `0.0638` | **MEASURED.** Learned exploration scale in model2397; deterministic evaluation does not sample it. |

The first two runs use 500 updates each with the same dense stance reward. Later all-supine continuations add the supported shoulder objective; the last three use adaptive KL-based step sizing. The penultimate stage narrows the shoulder-reward denominator from 9 to 1, and the final 150 updates move its target ratio from 0.25 to 0.15. Actor, critic, optimizer and learned std are retained. The full selected ancestry contains 2,404 updates; rejected branches are identified in development history. This is not a single-factor ablation.

The final actor has no autoencoder, decoder, reconstruction loss, diffusion model, CNN, RNN, privileged latent, behavior-cloning loss or expert dataset. HumanUP's history/RMA-style model was implemented and evaluated historically, but it achieved 0/5 from strict true-supine resets and is not part of the selected policy.

## PPO and optimization

| Parameter | Value | Source class | Why this value |
| --- | ---: | --- | --- |
| Algorithm | PPO clipped surrogate | **PAPER:** Schulman et al. (2017) | Stable on-policy continuous control with multiple optimization passes over a rollout. |
| Clip `epsilon` | 0.2 | **PAPER + FRAMEWORK** | The PPO paper explicitly uses and studies `0.2`; also the RSL-RL default. This clips the policy probability ratio, not the physical joint action. |
| Discount `gamma` | 0.99 | **PAPER + FRAMEWORK** | PPO continuous-control tables use `0.99`; preserves delayed get-up reward. |
| GAE `lambda` | 0.95 | **PAPER + FRAMEWORK:** Schulman et al. (2015, 2017) | Standard bias/variance compromise for advantage estimates. |
| Learning epochs | 5 | **FRAMEWORK** | RSL-RL default. PPO establishes repeated epochs but its MuJoCo table used 10, so `5` is not claimed as a paper constant. |
| Minibatches | 4 | **FRAMEWORK** | RSL-RL default. With 96,000 rollout transitions, each minibatch is about 24,000 samples. |
| Value-loss coefficient | 1.0 | **FRAMEWORK** | RSL-RL default balancing actor and critic objectives. |
| Clipped value loss | enabled | **FRAMEWORK/PPO family** | RSL-RL implementation choice to constrain critic updates. |
| Entropy coefficient | 0 | **LOCAL** | No entropy bonus in this selected experiment; stochastic action sampling still provides exploration. |
| Desired KL | 0.01 | **PAPER + FRAMEWORK** | RSL-RL uses this to adapt learning rate only with the adaptive schedule. PPO clipping itself does not guarantee a strict KL bound. |
| Gradient norm clip | 1.0 | **FRAMEWORK** | RSL-RL stability default, not a PPO theorem. |
| Optimizer | Adam | **PAPER + FRAMEWORK** | PPO's algorithm description recommends minibatch SGD, usually Adam; RSL-RL implements Adam here. |
| Learning rate | `3e-4` initially; `1e-4` initial value for refinements | **FRAMEWORK + LOCAL** | Initial stance runs use adaptive KL sizing; shoulder continuation first uses fixed rate, then adaptive after regression. The adaptive rate varies during training. |
| Rollout length | 32 steps/environment | **LOCAL** | At 50 Hz this is 0.64 s per environment per PPO update; chosen for contact transition coverage and GPU throughput. |
| Parallel environments | 3000 | **LOCAL/COMPUTE** | Fits the 16 GB RTX 5080 Laptop GPU while producing 96,000 transitions/update. No paper mandates 3000. |
| Learning passes/update | `5 × 4 = 20` minibatch updates | **DERIVED** | Five epochs over four minibatches. |
| Seed | 47 for final stage | **LOCAL/REPRODUCIBILITY** | Experiment identifier; evaluation uses independent seeds 101-105. |
| Parent | `supine_model2248` | **LOCAL/LINEAGE** | Earlier all-supine shoulder checkpoint; exact parent path and hash are in the final provenance. |
| Optimizer reset | disabled | **LOCAL** | Continuation retains learned actor/critic and Adam state. |
| Selected checkpoint | iteration 2397; 2,404 selected updates | **MEASURED** | RSL-RL repeats the loaded index at each continuation. |

The raw `agent.yaml` records the dataclass before checkpoint loading and optional runtime overrides. The selected continuation retains the checkpoint's learned standard deviations. `reports/configs/supine_model2397/runtime_settings.json` records that runtime distinction explicitly.

## Simulation and robot parameters

| Parameter | Value | Provenance |
| --- | ---: | --- |
| Robot | AgiBot X2 Ultra v1.3.0 | **ROBOT:** official pinned URDF. |
| Floating mass | 41.966521 kg | **ROBOT/MEASURED:** imported articulation audit. |
| Movable joints / rigid bodies | 31 / 32 | **ROBOT/MEASURED.** |
| Hard joint, effort and speed limits | Official URDF values | **ROBOT.** The project does not enlarge them. |
| Soft joint margin | 0.98 of hard range | **LOCAL.** A 2% numerical margin; larger generic margins prevented knee/elbow extension. |
| Self-collision | enabled | **TASK/LOCAL.** Required for credible contact-rich recovery. |
| PhysX step | 0.005 s = 200 Hz | **LOCAL.** Stable contact resolution within available compute. HumanUP used 1000 Hz simulation but 50 Hz low-level control; its exact simulation rate was not copied. |
| Policy step | decimation 4 = 0.02 s = 50 Hz | **PAPER-INFORMED + LOCAL.** Matches HumanUP's 50 Hz low-level control rate and local timing tests. |
| Episode | 10 s = 500 policy steps | **TASK/LOCAL.** Long enough to rise and verify maintained stance. |
| Floor | local 200 m × 200 m × 0.1 m cuboid | **LOCAL/GEOMETRY.** Covers the complete cloned grid; avoids a remote Nucleus asset. |
| Ground friction | static 0.9, dynamic 0.8, restitution 0 | **LOCAL NOMINAL.** No paper claim and no final randomization. |
| Initial pelvis height | 0.190 m | **LOCAL/MEASURED.** Collision hull extends 0.1803007 m below the pelvis, leaving at least millimetres of clearance. |
| Initial orientation | XYZW `[0,-sqrt(1/2),0,sqrt(1/2)]` | **GEOMETRY/API.** A -90° rotation about world Y lays the back down in the installed scalar-last writer convention. |
| Reset position jitter | x/y ±0.04 m, z ±0.001 m | **LOCAL.** Small seeded variation without floor intersection. |
| Reset angle jitter | roll/pitch ±0.005 rad, yaw ±0.02 rad | **LOCAL.** |
| Reset velocity | zero | **TASK/LOCAL.** Final evaluation starts at rest. |
| PD gains | legs/waist 120/6; ankles 80/5; arms 40/3; wrist/head 15/1.5 (`Kp/Kd`) | **LOCAL.** Grouped by load; official URDF effort/speed limits remain authoritative. |
| Armature | 0.01 | **LOCAL/FRAMEWORK.** Numerical actuator model setting. |
| Observation noise / dynamics randomization | disabled in final run/evaluation | **LOCAL SCOPE.** The result is nominal simulation, not a robustness or sim-to-real claim. |

## Observation provenance

The 122-dimensional policy vector is:

| Term | Dim. | Scale/source |
| --- | ---: | --- |
| Pelvis height | 1 | Metres; HumanUP height-task principle, exact X2 signal local. |
| Body-frame base linear velocity | 3 | Standard proprioception; raw scale. |
| Body-frame base angular velocity | 3 | Standard proprioception; raw scale. |
| Projected gravity | 3 | `R(q)^T [0,0,-1]`; HumanUP explicitly uses projected gravity for uprightness. Avoids Euler wrap and removes irrelevant yaw. |
| Joint position relative to default | 31 | Official X2 joint order. |
| Joint velocity relative to default | 31 | Multiplied by 0.10; local scale balancing. |
| Left/right foot contact | 2 | Current ground-filtered force ≥15 N. Assignment semantics; threshold local. |
| Whole-body ground contact | 32 | One current bit for every imported rigid body. Local fix for observational ambiguity and nested-contact mapping. |
| Previous two eight-value actions | 16 | FRASA observes past actions; exactly two steps (40 ms) is local. |
| **Total** | **122** | `1+3+3+3+31+31+2+32+16`. |

## Action provenance

FRASA shows that bilateral symmetry can reduce the search space, and HumanUP discusses symmetry as a useful but limiting bias. The exact X2 action map is local:

```text
q_target = centre + span * tanh(action)
q_target = clamp(q_target, imported soft joint limits)
```

| Action | Joint group | Centre rad | Span rad |
| ---: | --- | ---: | ---: |
| 0 | left/right hip pitch | -0.90 | 1.40 |
| 1 | left/right knee | 1.15 | 1.15 |
| 2 | left/right ankle pitch | -0.15 | 0.70 |
| 3 | left/right shoulder pitch | -0.50 | 2.00 |
| 4 | left/right elbow | -0.80 | 0.80 |
| 5 | waist pitch | 0.00 | 0.30 |
| 6 | left/right ankle roll | 0.00 | 0.15 |
| 7 | left/right hip roll | 0.00 | 0.15 |

The centres and spans are **LOCAL** values bounded by official URDF limits. The final policy does not directly command torque; it supplies position targets to the grouped PD actuators.

## Reward provenance

Isaac RewardManager multiplies each term by its weight and the 0.02 s policy timestep. Paper citations support the form or idea; every X2 weight and gate remains local unless stated otherwise.

| Term | Weight | Submitted formula/behavior | Provenance |
| --- | ---: | --- | --- |
| Signed pelvis height | 40 | `clip(z/0.68,0,1) * clip((1-g_z)/2,0,1)` | HumanUP motivates height/upright progress; the signed gate is **LOCAL**, added after the inverted-bridge exploit. |
| Head height | 5 | `exp(clip(h_head,0,1.2))-1` | Formula form from HumanUP Stage I; X2 target and weight **LOCAL**. |
| Upright | 5 | `exp(-g_z)` | Formula from HumanUP Stage I; weight **LOCAL**. |
| Both feet | 10 | two current foot contacts × height gate × upright gate² | HumanUP/assignment motivate standing on both feet; exact force/gates/weight **LOCAL**. |
| Other support | -1 | count of non-foot ground contacts × high-pelvis gate | **TASK + LOCAL.** Encodes “without support from other body parts” while allowing transitional pushes. |
| Balance | 20 | high/upright gate × `exp(-(‖v‖+‖ω‖)^2/0.25)` | Smoothness/stability is paper-informed; formula and weight **LOCAL**. |
| Strict stance | 20 | binary complete success predicate | **TASK + LOCAL.** Gives the evaluator's full target during training. |
| Relaxed arms | 40 | supported-upright gate × `exp(-arm_mse/2)`; shoulder 0, elbow -0.15 rad | HoST motivates post-task motion/posture constraints; target, variance, gate and weight **LOCAL**. |
| Action change | -0.02 | `‖a_t-a_(t-1)‖²` | PPO robotics/FRASA/HumanUP smoothness principle; exact weight **LOCAL**. |
| Torque | -1e-6 | `‖tau‖²` | HumanUP regularization family; exact X2 weight **LOCAL**. |
| Joint limit | -1 | soft-limit violation | Safety regularization; exact weight/margin **LOCAL**. |
| Failure | -10 | non-timeout safety termination | **LOCAL.** Timeout is excluded so “still trying at 10 s” is distinct from a numerical/unsafe failure. |

The selected dense preset additionally uses stance proximity 20, leg pose 20, near-stance motion -2, raw-action saturation -0.5 and joint-speed cost -0.0005. Exact formulas, gates and training widths are in [design and evidence](design_and_evidence.md); all are local X2 adaptations. Training widths do not change the strict validator. The selected `--posture_refinement` adds weight 80 times the supported gate and `exp(-(a_shoulder-atanh(r))^2/v)`. The target ratio `r=.25` initially, the denominator `v` narrows from 9 to 1, and the last 150 updates use `r=.15` (`--shoulder_target_ratio 0.15`). The X2 target, width, gate and weight are **LOCAL**. The target change moves the measured shoulder pitch toward zero; it does not use demonstrations or replace network output at deployment.

No paper establishes `40`, `15 N`, `0.58 m`, `0.30 rad` or any other X2-specific reward/success number.

## Termination and evaluation provenance

| Check | Value | Source |
| --- | --- | --- |
| Episode timeout | 10 s | **TASK/LOCAL.** Truncation, not safety failure. |
| Root linear-speed termination | >2.5 m/s | **HumanUP-informed.** Kept as a broad unsafe-state guard. |
| Pelvis-height termination | outside `[0,1.2]` m | **HumanUP-informed + LOCAL X2 bound.** |
| Non-finite state | any NaN/Inf in floating-base state | **LOCAL safety.** |
| Contact termination | none | **TASK/LOCAL.** Whole-body contact is necessary during recovery. |
| Success height | ≥0.58 m | **LOCAL**, chosen below 0.68 m nominal standing height. |
| Upright | `‖g_xy‖≤0.15` and `g_z≤-0.98` | **LOCAL** projected-gravity tolerance. |
| Stable speed | linear ≤0.25 m/s; angular ≤0.35 rad/s | **LOCAL** stability tolerance. |
| Foot support | each foot ≥15 N | **TASK semantics + LOCAL threshold.** |
| Other support | every other body <15 N | **TASK semantics + LOCAL threshold.** |
| Hold time | all conditions continuous for ≥0.5 s | **LOCAL**, prevents one-frame success. |
| Final arm tolerance | shoulder and elbow errors ≤0.30 rad for the final two seconds | **LOCAL**, used as an additional posture report rather than the assignment's base recovery predicate. |

The five final episodes use seeds 101-105 and examine all 500 policy steps. Those measurements, rather than the training return, establish the submitted 5/5 claim.

On resume, RSL-RL staggers the initial episode counters; poses still reset supine. Subsequent episodes and all validation episodes use the full 500-step limit. This changes neither the reset-pose distribution nor the physical success checks.
