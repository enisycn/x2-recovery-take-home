# Formula audit

This note states the equations used by the selected checkpoint. The numerical suite in `isaaclab_ext/test/test_reward_formulas.py` checks the reset orientation, collision clearances, schedules, reward gates, action map and strict-success predicate.

## Frames, timing and action

AgiBot documents an FLU body frame: X forward, Y left and Z up. Isaac Lab stores the configured and tensor root quaternions in scalar-last XYZW order. The supine reset

\[
q=(0,-\sqrt{1/2},0,\sqrt{1/2})
\]

rotates the local forward/chest axis onto world up. Applying this transform through all 50 official collision elements gives a 0.1803007 m extent below the pelvis. The 0.190 m reset plus bounded jitter leaves at least 6.3 mm clearance in 20,000 audited samples. The straight standing extent is 0.6749500 m, which motivates the 0.68 m reward target.

PhysX runs at 100 Hz and the policy at 20 Hz (`decimation=5`). Let `q_k` be the measured joint position at policy decision `k`, `a_k` the 31-dimensional raw actor output, and `[l,u]` the official joint range contracted once by the 0.98 soft-limit factor. The command is

\[
\Delta q_k=0.25\tanh(a_k),\qquad
q_k^{cmd}=\operatorname{clip}(q_k+\Delta q_k,l,u).
\]

`q_k^{cmd}` is computed once and held for all five physics steps. This matters: recomputing a relative target at every substep accumulated the same decision five times. `tanh` keeps the map smooth and bounded; zero holds the current pose. The simulator also enforces the effort and speed limits imported from the official URDF.

## Observation

The actor input has 168 scalars in this exact order:

\[
o=[h_b,\ v_b(3),\ \omega_b(3),\ g_b(3),\ q-q_0(31),\ 0.1\dot q(31),\ c_{feet}(2),\ c_{body}(32),\ a_{k-1:k-2}(62)].
\]

The two foot bits and 32 ordered body-contact bits use a 15 N threshold. Whole-body contacts remove the ambiguity between back, pelvis, hand, knee and foot support. Observation corruption is enabled only during training.

## Reward

Let `h` be pelvis height, `g` projected gravity in the pelvis frame, `v` and `omega` root velocities, `I[.]` an indicator and

\[
u=\operatorname{clip}(-g_z,0,1),\qquad
H=\operatorname{clip}(h/0.68,0,1).
\]

The dense recovery term is

\[
r_{rise}=H u^2.
\]

It supplies a non-vanishing height gradient near the floor while assigning no height reward to a sideways or inverted bridge. The signed orientation terms are

\[
r_{HumanUP}=e^{-g_z},\qquad
r_{upright}=\exp\left(-\frac{(1+g_z)^2}{0.25^2}\right).
\]

For contact indicators `C_L`, `C_R` and the number of non-foot supporting bodies `N_other`, define

\[
G_f=\operatorname{clip}\left(\frac{h-0.08}{0.68-0.08},0,1\right),\qquad
G_o=\operatorname{clip}\left(\frac{h-0.45}{0.68-0.45},0,1\right),
\]

\[
r_{feet}=I[C_L\land C_R]G_f u^2,\qquad
r_{other}=N_{other}G_o.
\]

HoST-style completion terms activate only above `h>0.58`:

\[
r_\omega=e^{-2\lVert\omega_{xy}\rVert^2},\quad
r_v=e^{-5\lVert v_{xy}\rVert^2},\quad
r_g=e^{-5\lVert g_{xy}\rVert^2},\quad
r_h=e^{-20(h-0.68)^2}.
\]

The configured reward rate is

\[
40r_{rise}+0.25r_{HumanUP}+2.5r_{upright}+10r_{feet}-2r_{other}
+10(r_\omega+r_v+r_g+r_h)+5r_{still}
\]

\[
-0.01\lVert a\rVert^2-0.02\lVert a-a_{prev}\rVert^2
-10^{-7}\lVert\ddot q\rVert^2-10^{-4}\lVert\dot q\rVert^2
-6\times10^{-7}\lVert\tau\rVert^2
\]

\[
-0.1\lVert\omega\rVert^2-0.1\lVert v\rVert^2-r_{limits}
-0.05r_{symmetry}-0.02r_{non\text{-}sagittal}.
\]

`r_still` is an exponential full-root-speed reward gated by `h>0.60` and `-g_z>0.96`. Isaac Lab multiplies every reward rate by the 0.05 s policy step when accumulating the episode return.

The selected reward deliberately omits raw head height and the positive-vertical-velocity bit. Measured branches exploited both while leaving the pelvis on the floor. The cited HumanUP and HoST terms motivate the reward families; their exact weights are adapted to the X2 and are validated only by the reported experiment.

## Training-only curriculum

At common policy step `k`, the probability of choosing one of nine collision-audited reference poses is

\[
p_{ref}(k)=0.95+(0.35-0.95)\operatorname{clip}(k/3600,0,1).
\]

The reference pelvis heights span 0.09194–0.68000 m and couple the root pose to symmetric hip, knee, ankle, shoulder and elbow angles. The remaining resets are the true supine pose. The schedule uses policy decisions, so changing the number of parallel environments does not shorten it.

HoST's cross-robot guidance scales the exploration pull to 60% of body weight. The pinned X2 mass is 41.966521 kg, so

\[
F_z(k)=247.014943\max(1-k/4000,0)I[-g_z\ge0.80]\ \mathrm{N}.
\]

The force acts at the pelvis only after the torso is nearly vertical. At 12 policy steps per PPO update, reference probability reaches 0.35 after 300 updates and the force reaches zero after about 334 updates. Evaluation sets both reference probability and assistance to zero, disables observation noise and starts every seed from the audited supine distribution.

## Episode ending and success

Training episodes last 8 s, or 160 policy decisions. Timeout is the only termination; ending as soon as the robot becomes upright would not teach it to remain balanced.

Instantaneous evaluation success requires

\[
h\ge0.62,\quad \lVert g_{xy}\rVert\le0.15,\quad g_z\le-0.98,
\]

\[
\lVert v\rVert\le0.20\ \mathrm{m/s},\quad
\lVert\omega\rVert\le0.35\ \mathrm{rad/s},
\]

both foot forces at least 15 N and every other body force below 15 N. All conditions must hold for ten consecutive 20 Hz decisions, or 0.5 s. The signed `g_z` check rejects an upside-down pose even when XY tilt is small.

The independent standing probe satisfied this complete predicate for 0.65 s. The selected learned policy satisfied it for 0.00 s in all five evaluation episodes; the result is reported unchanged.
