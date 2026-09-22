# Design and evidence

## Robot and physics

The official AgiBot X2 Ultra v1.3.0 URDF is pinned to AgibotTech commit `60c5de582c523cd188f563819e62d34cfdc3d2d0`. Import merges fixed links but preserves the floating articulation, 31 movable joints, 32 rigid bodies, collision meshes, mass and joint limits. The final reset height follows collision-hull measurement rather than visual geometry.

The contact view is recursive because imported rigid bodies are nested below the pelvis prim. Support calculations use the current ground-filtered `force_matrix_w`, not historical net force containing self-collision. The 15 N threshold is an X2 engineering choice.

## Frames

The installed root-state writer uses scalar-last XYZW quaternions. Identity is `[0,0,0,1]`; the nominal supine rotation is `[0,-sqrt(1/2),0,sqrt(1/2)]`. Final orientation observation uses body-frame projected gravity `R(q)^T [0,0,-1]`, avoiding the Euler wrap observed in the historical HumanUP path.

## Literature-to-implementation boundary

- [PPO](https://arxiv.org/abs/1707.06347) supplies the clipped on-policy algorithm.
- [HumanUP](https://www.roboticsproceedings.org/rss21/p063.html) supports contact-rich discovery and staged refinement. The unsuccessful historical RMA experiment was not the final controller.
- [HoST](https://www.roboticsproceedings.org/rss21/p064.html) supports a separate post-task upper-body posture objective. X2 arm targets, gates and weights are local adaptations.
- [FRASA](https://arxiv.org/abs/2410.08655v3) motivates bilateral low-dimensional recovery control. This repository uses PPO and X2-specific absolute targets rather than CrossQ or the FRASA robot.
- [DAPG](https://www.roboticsproceedings.org/rss14/p49.html), [DAgger](https://proceedings.mlr.press/v15/ross11a.html), [GAIL](https://proceedings.neurips.cc/paper/2016/hash/cc7e2b878868cbae992d1fb743995d8f-Abstract.html) and [DeepMimic](https://arxiv.org/abs/1804.02717) were considered but are not used by the selected policy.

No paper establishes the exact X2 reward weights, 15 N contact threshold, 0.58 m success height or 0.30 rad arm tolerance.

See [parameter and evidence provenance](parameter_provenance.md) for the neural-network, PPO, simulation, observation, action, reward, termination and evaluation values classified as paper, framework, robot/task or local decisions.

## Why direct PPO

No trustworthy time-aligned X2 expert state-action trajectory was available. Self-generated rollouts from the failed teacher would clone its failures. Direct PPO was therefore simpler and more defensible for a cheap, parallel simulator with an explicit physical success predicate.

## Main limitations

The five episodes cover nominal flat-floor simulation only. There is no domain randomization, observation noise, hardware state-estimation error, actuator latency, thermal constraint or real-robot validation. Whole-body contact and exact base height may require different sensing on hardware. The symmetric action subspace limits asymmetric recovery.
