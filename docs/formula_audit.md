# Formula audit

This note makes the recovery equations reviewable independently of a simulator rollout. The numerical suite in `isaaclab_ext/test/test_reward_formulas.py` checks the orientation sign, staged reward, exponential targets, contact predicate and action transform against explicit expected values.

## Frame and action transform

AgiBot documents an FLU body frame: X forward, Y left and Z up. The installed Isaac Lab 3.0 `InitialStateCfg` and root-state tensor APIs both store quaternions in scalar-last XYZW order. The reset quaternion

\[
q=(x,y,z,w)=(0,-\sqrt{1/2},0,\sqrt{1/2})
\]

is a -90° rotation about Y. It maps the local forward/chest vector `(1,0,0)` to world-up `(0,0,1)`, so the back faces the floor. Applying the full zero-joint kinematic tree to every official collision mesh gives a 0.1803007 m supine extent below the pelvis and a 0.6749500 m standing extent. The 0.190 m reset and bounded jitter therefore start just above the floor; the 0.68 m standing target puts the feet about 5.05 mm above the mathematical plane before contact compliance.

For each joint, let the imported hard position limits be contracted once by Isaac Lab's `soft_joint_pos_limit_factor=0.98` to `[l,h]`. For raw policy action `a`, the command path follows HoST Equation (1), with its final action bound `β=0.25`:

\[
\Delta q=0.25\operatorname{clip}(a,-1,1), \qquad
q_t^{cmd}=\operatorname{clip}(q_t+\Delta q,l,h).
\]

The zero action therefore holds the current pose, while repeated bounded increments can traverse the full 98%-contracted range. This removes the previous initialization defect where `a=0` mapped every asymmetric joint to the midpoint of its limits. For the official knee range `[0,2.407]` rad, the reachable minimum remains about 0.024 rad; the discarded 90%-then-85% contraction had incorrectly raised it to about 0.282 rad. The simulator applies the imported effort and velocity limits with load-based implicit PD gains: leg/waist `Kp=120, Kd=6`, ankle `80/5`, arm `40/3`, and wrist/head `15/1.5`.

## Task rewards and curriculum

Let `h_b` and `h_h` be pelvis and head height, `g_b` projected gravity in the pelvis frame, `v` and `ω` root linear and angular velocity, and `I[·]` an indicator. Upright gives `g_b=(0,0,-1)`. The dense discovery group follows HumanUP Stage-I Appendix Table II:

\[
r_{base}=\exp(h_b)-1, \qquad
r_{head}=\exp(h_h)-1,
\]

\[
r_{\Delta h}=I[v_z>0], \qquad
r_{upright}=\exp(-g_{b,z}).
\]

HumanUP defines the progress term as `I[h_b(t)>h_b(t-1)]`; because the simulator exposes velocity at every 0.05 s policy instant, `I[v_z>0]` is its continuous-time equivalent. A two-foot term is enabled as the pelvis rises from 0.58 m to the 0.68 m target. Non-foot contacts are allowed during pushing but receive a linearly increasing penalty over the same interval. Both use a 15 N force threshold and a three-sample contact history.

HoST's task-orientation component is represented by the signed upright target

\[
r_{orient}=\exp\!\left(-((1+g_{b,z})/0.10)^2\right).
\]

Its effective weight is 2.5 after HoST's task-group multiplier. This term distinguishes upright `g_{b,z}=-1` from an inverted pose `g_{b,z}=+1`; the later HoST tilt-only post term cannot distinguish those two signs by itself.

Above the X2 final-stage boundary `h_b>0.62`, the HoST Table VI post-task terms become active:

\[
r_{\omega}=\exp(-2\lVert\omega_{xy}\rVert^2), \qquad
r_v=\exp(-5\lVert v_{xy}\rVert^2),
\]

\[
r_g=\exp(-5\lVert g_{b,xy}\rVert^2), \qquad
r_h=\exp(-20(h_b-0.68)^2).
\]

The configured reward rate is

\[
5r_{base}+5r_{head}+r_{\Delta h}+0.25r_{upright}+2.5r_{orient}
+2.5r_{feet}-2c_{other}
+10(r_{\omega}+r_v+r_g+r_h)
\]

\[
-0.1\lVert\Delta a\rVert^2
-10^{-7}\lVert\ddot q\rVert^2
-10^{-4}\lVert\dot q\rVert^2
-6\times10^{-7}\lVert\tau\rVert^2
-0.1\lVert\omega\rVert^2
-0.1\lVert v\rVert^2
-c_{limits}.
\]

The discovery and regularization weights come from HumanUP Table II; the post-task weights and exponents come from HoST Table VI. The one deliberate simplification is that HumanUP's binary torque and position-limit penalties are represented by Isaac Lab's continuous soft-limit terms. Isaac Lab multiplies every rate by the 0.05 s policy step before accumulating return.

Two training-only curricula address sparse contact exploration. The probability of a safe straight standing reset is

\[
p_{stand}(k)=0.5\max(1-k/16000,0),
\]

following HumanUP's Stage-I mixture of standing poses. HoST's official cross-robot guidance scales the pull to about 60% of robot weight and applies it only after the trunk becomes near vertical. The pinned X2 URDF has total mass 41.966521 kg, so the world-up pelvis force is

\[
F_z(k)=0.60(41.966521)(9.81)\max(1-k/24000,0)
I[-g_{b,z}\geq0.80]\;\mathrm{N}.
\]

It begins at 247.015 N. The orientation gate leaves the initial supine ground reaction unchanged and assists only the ground-sitting/rising phase.

Both are exactly zero for the last part of the 1,200-iteration run. The play/evaluation configuration disables them regardless of checkpoint iteration, so they cannot help a reported episode.

The first branches are retained as diagnostic evidence: a 500-iteration policy became upright but stayed low, and a height-dominant branch found an inverted bridge. A later 512-environment audit found that the original single 20 m floor covered only the central part of the approximately 58 m clone grid. The corrected shared floor is 200 m square and covers the configured 2,048-environment grid; a numerical test guards that geometry. Those earlier checkpoints cannot be treated as valid final experiments.

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
