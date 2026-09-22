# Stable final arm posture (v4)

The v3 policy met the recovery criterion, but ended seed 101 with both shoulder-pitch joints near -1.918 rad. Its reward did not specify a final arm pose. The v3 checkpoint and reports are retained as the successful recovery baseline.

## Evidence and adaptation

HoST, RSS 2025, explicitly includes an **Upper Body Posture** post-task reward in Table VI(d):

`exp(-0.1 * ||p_upper - p_upper_target||²) * 1(base_height > H_stage2)`.

See [the primary paper, Section IV-B and Table VI(d)](https://arxiv.org/html/2502.08378v1#S4.SS2), [RSS proceedings](https://www.roboticsproceedings.org/rss21/p064.html), and the [official implementation](https://github.com/InternRobotics/HoST). The paper supports the gated posture objective; it does not establish that particular gains or targets succeed on X2.

Our X2 adaptation adds exactly one reward to v3:

`r_arm = 20 * I_strict * exp(-mean((q_arm - q_target)²) / 2)`.

Four actual joint positions are used: left/right shoulder pitch target **0 rad**, left/right elbow target **-0.15 rad** (small natural bend). The mean is over these four joints; this is equivalent to coefficient 0.125 on their squared-error sum. `I_strict` is the existing instantaneous full two-foot stance criterion, including upright body, low root speed, and no other ground support. This is a stronger gate than the paper's height-only gate. The positive reward cannot be gained while lying down. It imposes no arm penalty during recovery. Physics, action mapping, observations and failure limits are unchanged. Arm motion remains learned PPO control, with no post-processing to force a pose.

The policy still has 122 inputs, ELU layers 512/256/128, and 8 outputs mapped into the original 31-joint X2. Fine-tuning starts from `x2_symmetric_v3_model200.pt` with fresh optimizer, fixed learning rate 1e-4, initial action std 0.2, entropy 0.001, seed 44, 3000 environments and 32 steps per rollout. Back-lying starts remain 50% of training resets. Saved iteration numbers continue from 200.

## Validation protocol

Evaluation uses fresh perturbed supine resets with seeds 101–105, no lift force, full 10-second episodes and pre-reset snapshots. Recovery success is unchanged. An additional, independent posture check requires strict stance and each of the four joint errors <=0.30 rad continuously for at least 0.5 seconds. Reports also measure the worst arm errors and the strict-and-relaxed fraction throughout the final two seconds. This distinguishes brief passage through the target from actually settling there.

The reward's unit check covers standing with relaxed versus raised arms, supine posture and missing foot contact. Simulator evaluation and the rendered GIF are needed to establish physical behavior; unit tests alone do not do so. Nominal simulation success does not establish hardware safety or robustness to untested perturbations.
