# HumanUP paper and code audit

## Sources and integrity

- Paper: Xialin He, Runpei Dong, Zixuan Chen, and Saurabh Gupta, *Learning Getting-Up Policies for Real-World Humanoid Robots*, RSS 2025 / arXiv:2502.12152v2.
- Local source: `references/humanup_rss2025.pdf`
- PDF SHA-256: `5a594169d416cf9903bf79224a3461e70b69b20800ad360b2591c4426b7a8cc4`
- PDF inspection: 15 pages, unencrypted, no forms, no embedded JavaScript.
- Official implementation: `RunpeiDong/HumanUP`, pinned commit `7516e0f27e6f4d1e7365cf64ea577a78247bd8cb`.
- Audited raw files were downloaded as inert text and were never executed. The X2 implementation was written locally against the existing Isaac Lab environment.

## What the method actually is

HumanUP is two-stage reinforcement learning. It is not behavioral cloning and it does not require a human motion dataset.

1. Stage I trains a PPO discovery policy from a canonical lying pose, with 10--20% standing resets, simplified collision, fixed flat terrain, weak regularization, and task rewards. The paper reports about 5 billion simulation transitions.
2. A successful Stage-I state trajectory is retained and stretched from a sub-second motion to 8 seconds.
3. Stage II trains another PPO policy to track that state trajectory. The tracking objective combines joint-position tracking (weight 8) with body/head tracking, stronger regularization, full collision, randomized initial poses, terrain, dynamics, and observation noise.

The repository contains optional 1500 N head-drag utilities, but the published G1 discovery configuration sets `drag_robot_up = False`. Therefore head force is disabled in the faithful X2 configuration. The earlier assisted X2 experiment is retained only as a failed ablation and is not presented as HumanUP reproduction.

## Neural network audit

For the 23-action G1 used in the paper:

- Current proprioception `s_t`: 74 values = 3 base angular velocity + 2 roll/pitch + 23 joint positions + 23 joint velocities + 23 previous actions.
- Environment latent slot `z_t`: 54 values = 4 mass + 1 friction + 46 motor-strength + 3 base-linear-velocity values. It is zero in Stage I when domain randomization is disabled.
- History: ten 74-value proprioceptive states.
- Full critic observation: `74 + 54 + 10*74 = 868`.
- History actor: each historical state is projected `74 -> 30`; temporal Conv1d layers are `30 -> 20` (kernel 4, stride 2) then `20 -> 10` (kernel 2); the flattened 30 values become a 20-value latent.
- Actor backbone: current proprioception plus history latent, `94 -> 512 -> 256 -> 128 -> 23`.
- Critic: full observation, `868 -> 512 -> 256 -> 128 -> 1`.

For the 31-action X2, only dimensions dictated by the robot change:

- Current proprioception: `3 + 2 + 3*31 = 98`.
- Stage-I latent slot: `4 + 1 + 2*31 + 3 = 70`, filled with zeros.
- Full critic observation: `98 + 70 + 10*98 = 1148`.
- History latent remains 20.
- Actor backbone: `118 -> 512 -> 256 -> 128 -> 31`.
- Critic: `1148 -> 512 -> 256 -> 128 -> 1`.

The original X2 actor used a flat 168-value observation and the same `512/256/128` hidden widths. Its layer widths were not the central problem. It lacked HumanUP's explicit ten-step temporal encoder and was trained with far less experience. The first 500-update run used 18 million transitions, about 278 times fewer than the paper's Stage-I budget.

## PPO settings copied from the release

| Setting | HumanUP / X2 faithful mode |
|---|---:|
| rollout | 24 steps per environment |
| environments in paper | 4096 |
| actor/critic hidden widths | 512, 256, 128 |
| activation | ELU |
| initial action standard deviation | 1.0 |
| clip ratio | 0.2 |
| clipped value loss | enabled |
| learning epochs | 5 |
| minibatches | 4 |
| learning rate | 2e-4, adaptive |
| entropy coefficient | 0.01 |
| discount / GAE lambda | 0.99 / 0.95 |
| desired KL | 0.008 |
| gradient-norm limit | 1.0 |

The local smoke run verified the constructed actor input is 118, critic input is 1148, history encoder output is 20, action output is 31, and all 24 reward terms remain finite in Isaac Lab.

The faithful discovery configuration keeps HumanUP's `init_std=1.0` and entropy coefficient `0.01`. A separate, explicitly named X2 bridge-pose teacher uses `init_std=0.5` and zero entropy while keeping clip, rollout, batch, optimizer, history encoder, and backbone settings unchanged. The split is intentional: HumanUP's Stage-I exploration is designed to discover a fast unsafe motion, whereas the bridge teacher must retain a narrow already-valid stance long enough to expand its basin of attraction.

## Reward audit

The Stage-I implementation follows the official code weights: base-height exponential 5, head-height exponential 5, positive height change 1, increased foot force 1, standing on both feet 2.5, orientation penalty -1, body-upright exponential 0.25, foot-height 2.5, foot-distance 2, body symmetry -1, waist symmetry -1, foot orientation -0.5, slip -1, termination -500, joint error -0.03, base linear velocity -0.1, angular velocity -0.1, joint velocity -1e-4, action rate -0.1, torque -1e-6, joint-limit -5, torque-limit -0.1, energy -1e-4, and acceleration -1e-7.

The appendix and code differ in two places: the PDF lists joint-position error -0.75 and torque -6e-7, while the released discovery config uses -0.03 and -1e-6. The executable X2 configuration follows the official code release and records this discrepancy rather than silently mixing values.

The G1 height caps are robot-specific. X2 uses collision-audited standing caps of 0.68 m pelvis height and 1.18 m head height. Body and joint names are mapped to the pinned official X2 URDF. These are explicit morphology adaptations; reward equations and weights remain those of the released Stage-I configuration.

## Consequence for the training plan

A Stage-II imitation run is valid only after Stage I produces a complete successful state trajectory. Copying the nine static bridge poses with behavioral cloning would teach pose snapshots without the contact sequence, velocities, or transitions needed to rise. The compute-efficient X2 plan therefore keeps HumanUP's network and all 24 Stage-I reward terms, then adds the HRS success predicate and HoST final-state stabilization terms only in the named teacher phases. It expands a verified standing policy through collision-audited bridge poses. A successful full supine rollout then becomes the Stage-II tracking reference.
