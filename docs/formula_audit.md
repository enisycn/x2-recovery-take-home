# Formula audit

This note describes the executable HumanUP-history PPO path. The 19 numerical tests in `isaaclab_ext/test/test_reward_formulas.py` check reset orientation, geometry clearance, schedules, action mapping, reward gates and the strict-success predicate.

## Frames and reset

AgiBot specifies an FLU body frame. Isaac Lab uses scalar-last XYZW quaternions, so the supine root rotation is

\[
q=(0,-\sqrt{1/2},0,\sqrt{1/2}).
\]

It maps the robot's forward/chest axis to world up. Forward kinematics over all 50 official collision elements gives 0.1803007 m from the pelvis to the lowest supine point. The 0.190 m reset and bounded jitter retain at least 6.3 mm floor clearance in 20,000 samples. Root and joint velocities start at zero.

## Observation and network

At 20 Hz, current proprioception is

\[
p_k=[\omega_b(3),\ \mathrm{roll,pitch}(2),\ q-q_0(31),\ \dot q(31),\ a_{k-1}(31)]\in\mathbb{R}^{98}.
\]

The environment supplies `[p_k, e_k, p_{k-9:k}]`, where the 70-value Stage-I extrinsics slot `e_k` is zero because domain randomization is disabled. The critic receives all 1,148 values. The actor normalizes the same vector, keeps current `p_k`, and compresses the ten-state history with HumanUP's released RMA shape:

\[
98\rightarrow30,\quad
\mathrm{Conv1d}(30,20,k=4,s=2),\quad
\mathrm{Conv1d}(20,10,k=2),\quad
30\rightarrow20.
\]

The resulting 118 values pass through an ELU MLP `118→512→256→128→31`. This is a temporal encoder, not an autoencoder: there is no decoder or reconstruction loss. The exported TorchScript graph includes the normalizer and encoder and has a verified `1148→31` contract.

## Timing and action

PhysX runs at 100 Hz and the actor at 20 Hz (`decimation=5`). Let `a_k` be the raw actor output and `[l,u]` the official X2 joint limits contracted once to 98%. Away from the final stance,

\[
q_k^{cmd}=\operatorname{clip}\left(q_k+0.25\tanh(a_k),l,u\right).
\]

The target is computed once and held for five physics steps. Near standing, the X2-specific brake suppresses repeated relative increments:

\[
s(x)=x^2(3-2x),\quad
b=s\!\left(\operatorname{clip}\frac{h-0.55}{0.62-0.55}\right)
\operatorname{clip}\frac{-g_z-0.90}{0.10},
\]

\[
q_k^{cmd}=\operatorname{clip}\left(q_k+0.25\tanh((1-b)a_k),l,u\right).
\]

This deterministic post-processing is active in the selected evaluation. The simulator separately enforces URDF effort and velocity limits.

## Published HumanUP block

`HumanUpStageIRewardsCfg` implements the 24 terms and code-release weights: clipped base/head height `+5/+5`, positive height change `+1`, increased foot force `+1`, two-foot stand `+2.5`, orientation `-1`, body-up exponential `+0.25`, foot height/distance `+2.5/+2`, body/waist action symmetry `-1/-1`, foot orientation/slip `-0.5/-1`, unsafe termination `-500`, default-pose error `-0.03`, base linear/angular speed `-0.1/-0.1`, joint speed `-1e-4`, action rate `-0.1`, torque `-1e-6`, position/torque limit `-5/-0.1`, energy `-1e-4`, and acceleration `-1e-7`.

These terms are copied from the pinned HumanUP release with X2 body names and reachable height caps. The code-versus-paper coefficient audit is in `docs/humanup_paper_review.md`.

## X2 completion block

Let `h` be pelvis height, `g` projected gravity in the pelvis frame, `u=clip(-g_z,0,1)`, `H=clip(h/0.68,0,1)`, `C_L,C_R` the 15 N foot-contact bits, and `N_other` the number of non-foot bodies with at least 15 N floor force. The main dense terms are

\[
r_{rise}=Hu^2,
\]

\[
r_{feet}=I[C_L\land C_R]\operatorname{clip}\left(\frac{h-0.08}{0.60},0,1\right)u^2,
\]

\[
r_{other}=N_{other}\operatorname{clip}\left(\frac{h-0.45}{0.23},0,1\right).
\]

The sagittal leg and remaining-joint alignment terms use a FRASA-style Gaussian around the official default pose, multiplied by height and upright gates. Their weights are `+50` each. The signed upright term is `exp(-(1+g_z)^2/0.25^2)` with weight `+2.5`.

Above `h>0.58`, the X2-adapted HoST completion kernels are

\[
r_\omega=e^{-10\|\omega_{xy}\|^2},\quad
r_v=e^{-20\|v_{xy}\|^2},\quad
r_g=e^{-5\|g_{xy}\|^2},\quad
r_h=e^{-20|h-0.68|}.
\]

The angular/linear kernels are deliberately tighter than HoST's released G1 coefficients `2/5`; the X2 experiment uses `10/20` after measured overshoot at the stance boundary. The height absolute error follows the official HoST code, while its paper table prints a squared scalar norm. The four configured weights are `50, 50, 10, 10`.

For training only, the exact binary target is smoothed as

\[
E=2\left(\frac{\max(0.58-h,0)}{0.10}\right)^2
+\left(\frac{\|g_{xy}\|}{0.15}\right)^2
+\left(\frac{\max(g_z+0.98,0)}{0.05}\right)^2
+0.1\left(\frac{\|v\|}{0.25}\right)^2
+0.1\left(\frac{\|\omega\|}{0.35}\right)^2+2N_{other},
\]

\[
r_{prox}=I[C_L\land C_R]e^{-E}.
\]

It has weight `+100`. The unchanged binary success indicator is also a training reward (`+200` in the rise configuration), but it is never a termination. Isaac Lab multiplies reward rates by the 0.05 s control interval when accumulating return.

## PPO and curriculum

The HumanUP-faithful discovery configuration uses rollout 24, clip 0.2, five epochs, four mini-batches, learning rate `2e-4`, adaptive KL target 0.008, entropy 0.01, gamma 0.99, lambda 0.95 and max gradient norm 1. The focused X2 rise stages retain PPO clipping and use initial/overridden action standard deviation 0.5/0.1–0.15, zero entropy coefficient and learning rate `1e-4`–`2e-4`.

Nine coupled root/joint reference poses span straight standing (`reference_0`) to near-supine (`reference_8`). The chosen checkpoint's last recorded stage sampled only `reference_2`; the subsequent audit therefore evaluates the missing distribution shift explicitly rather than assuming the curriculum reached the floor. All reported final episodes force reference probability to zero.

HumanUP safety endings are timeout, root linear speed above 2.5 m/s, pelvis height outside `[0,1.2]` m, or a non-finite root state. Success is absent from this list so balance must persist.

## Evaluation predicate

Instantaneous success is

\[
h\ge0.58,\quad \|g_{xy}\|\le0.15,\quad g_z\le-0.98,
\]

\[
\|v\|\le0.25\ \mathrm{m/s},\quad
\|\omega\|\le0.35\ \mathrm{rad/s},
\]

with at least 15 N body-to-floor force on each foot and less than 15 N on every other body. These forces come from the contact sensor's `/World/ground` filtered partner matrix, not the net-force buffer that also contains self-collisions. Every condition must hold for ten consecutive 20 Hz decisions (0.5 s).

The independent standing probe held the complete predicate for 0.65 s. The selected learned checkpoint held it for 0.00 s in all five true-supine episodes; `reports/isaac_evaluation_humanup.json` records that unchanged result.
