# Formula audit

This note makes the recovery equations reviewable independently of a simulator rollout. The numerical suite in `isaaclab_ext/test/test_reward_formulas.py` checks the orientation sign, staged reward, exponential targets, contact predicate and action transform against explicit expected values.

## Frame and action transform

AgiBot documents an FLU body frame: X forward, Y left and Z up. The reset quaternion

\[
q=(w,x,y,z)=(\sqrt{1/2},0,-\sqrt{1/2},0)
\]

is a -90° rotation about Y. It maps the local forward/chest vector `(1,0,0)` to world-up `(0,0,1)`, so the back faces the floor.

For each joint, let the imported hard position limits be contracted by Isaac Lab's `soft_joint_pos_limit_factor=0.90` to `[l,h]`. For raw policy action `a∈[-1,1]`, the command path is

\[
u=\operatorname{clip}(0.85a,-1,1), \qquad
q^*=l+\frac{u+1}{2}(h-l),
\]

followed by the exponential moving average

\[
q_t^{cmd}=0.25q_t^*+0.75q_{t-1}^{cmd}.
\]

Thus the raw action reaches the central 85% of the soft range, not the hard mechanical stops. The simulator then applies the imported effort and velocity limits with the implicit PD gains `Kp=60`, `Kd=4`.

## Task rewards

Let `p_z` be pelvis height, `g_b` projected gravity in the body frame, `u=-g_{b,z}`, and `n=clip(p_z/0.68,0,1)`. Standing upright gives `g_b=(0,0,-1)` and therefore `u=1`.

The staged progress term is

\[
r_{stage}=\begin{cases}
0.65\frac{u+1}{2}+0.35n,&p_z<0.35\\
0.45\frac{u+1}{2}+0.55n,&0.35\le p_z<0.58\\
0.30n+0.70\operatorname{clip}(u,0,1),&p_z\ge0.58.
\end{cases}
\]

The continuous target terms are

\[
r_{upright}=\exp\left(-\frac{(1+g_{b,z})^2}{0.25^2}\right), \qquad
r_{height}=\exp\left(-\frac{(p_z-0.68)^2}{0.12^2}\right).
\]

`r_feet` is one only when both foot forces exceed 15 N. `c_other` counts every non-foot body whose peak force over the three-sample sensor history exceeds 15 N. The near-target stability term is

\[
r_{stable}=I[p_z>0.60]I[-g_{b,z}>0.96]
\exp\left(-\frac{(\lVert v\rVert+\lVert\omega\rVert)^2}{0.25}\right).
\]

The configured reward rate is

\[
4r_{stage}+2r_{upright}+1.5r_{height}+1.5r_{feet}-c_{other}
+2r_{stable}-0.015\lVert\Delta a\rVert^2
-2\times10^{-4}\lVert\dot q\rVert^2
-2\times10^{-6}\lVert\tau\rVert^2
-0.20c_{limits}.
\]

Isaac Lab's reward manager multiplies this rate by the 0.05 s policy step before adding it to the episode return. This makes the accumulated reward approximately invariant to policy frequency when the same continuous behavior is sampled more finely.

## Strict evaluation predicate

Instantaneous success requires all of the following:

\[
p_z\ge0.62,\quad \lVert g_{b,xy}\rVert\le0.15,\quad g_{b,z}\le-0.98,
\]

\[
\lVert v\rVert\le0.20,\quad \lVert\omega\rVert\le0.35,
\]

both foot forces at least 15 N, and every other body force below 15 N. The `g_{b,z}` condition prevents an upside-down pose from passing merely because its XY tilt is small. All conditions must hold for ten consecutive 20 Hz decisions, i.e. 0.5 s.

These thresholds are engineering choices motivated by the cited recovery work and the assignment's definition. They must be recalibrated from the imported X2 standing height, contact noise and hardware limits before deployment.
