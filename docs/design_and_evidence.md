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

## What the cited methods actually use

[HumanUP, Sections III-A and III-B](https://arxiv.org/html/2502.12152v2#S3) trains both stages with PPO. Stage I discovers a recovery motion; Stage II tracks an eight-times slowed version of that discovered trajectory using tracking rewards and stronger control regularization. This is motion imitation through RL, not supervised behavioral cloning of human demonstrations. Its history-based regularized online adaptation is also distinct from an expert state-action dataset.

[HoST](https://www.roboticsproceedings.org/rss21/p064.html) learns standing-up from scratch with reinforcement learning, multiple critics and curricula. It does not require an expert demonstration trajectory.

Our earlier HumanUP-inspired history/RMA variant was an X2 adaptation, not a reproduction of the complete published pipeline. It failed the true-supine evaluation while reference starts omitted the lower recovery transition and several state/contact/control defects were still present. These diagnoses do not isolate the history encoder or prove that imitation learning fails. The selected controller uses PPO, 122 observations and eight symmetric action channels, without a history encoder or imitation loss. The scratch command starts every episode supine. The recorded 500-update scratch experiment scored 0/5: it reached standing height while airborne and never met all stance checks together. This does not isolate reset curriculum as the cause, because initialization and PPO exploration settings also differed from the selected model.

## Main limitations

The five episodes cover nominal flat-floor simulation only. There is no domain randomization, observation noise, hardware state-estimation error, actuator latency, thermal constraint or real-robot validation. Whole-body contact and exact base height may require different sensing on hardware. The symmetric action subspace limits asymmetric recovery.


## Supine-only stabilization experiment

The 500-update supine-only checkpoint reached 0.763-0.783 m in five episodes but maintained strict stance for zero seconds. At its maximum-height snapshots neither foot carried load, and angular speed was 5.36-9.38 rad/s. The required maximum is 0.35 rad/s. Its final mean Gaussian action standard deviation was 2.08045; the selected model450 had 0.07514. Deterministic evaluation removes sampling noise, so noise alone cannot explain the failed deployed motion. These facts support a ballistic local solution to the shaping objective, rather than an isolated URDF/frame defect.

`--stability_refinement` is an optional training reward preset. It keeps the same floating robot, flat floor, supine resets, observation/action contract, actuator limits and strict evaluation thresholds. The ordinary `relaxed_v4` preset and submitted checkpoint remain reproducible.

Let `o = clip((1-g_z)/2, 0, 1)` and `g = clip((z-0.45)/0.13, 0, 1)`. Let `s` mean both feet have at least 15 N ground contact and no other body reaches that threshold. The refinement height term is `clip(z/0.68, 0, 1) * o * (1-g+g*s)`, weight 40. It retains low-height recovery shaping while removing standing-height reward for airborne or other-supported states. The motion cost is `g * o^2 * (||v||^2 + 0.1*||omega||^2)`, weight -2. It discourages ballistic motion near stance without penalizing floor-level recovery. Balance weight increases to 20; a continuous proximity term to the unchanged stance predicate has weight 20. Joint-speed cost is -0.0005 and action-change cost -0.02.

[HumanUP, Appendix A.2](https://arxiv.org/html/2502.12152v2#A2) motivates velocity and control regularization after discovery. [HoST](https://www.roboticsproceedings.org/rss21/p064.html) also constrains smoothness and motion speed. Contact-gated height, gate thresholds, weights and refinement PPO values above are local X2 choices, not constants copied from either paper; neither complete paper pipeline is reproduced.

The first refinement continued only the supine-trained model499 for 100 PPO updates, reset optimizer and critic for the changed reward, started action std at 0.3, and used fixed learning rate 1e-4 and entropy coefficient 0. No upright or squat resets were added. Its final model598 scored **0/5** under the original success checks. Maximum continuous strict stance remained 0 s for every seed. At the maximum-height snapshots both foot forces were still 0 N; angular speeds for seeds 101–105 were 9.13, 5.24, 5.30, 4.93 and 5.01 rad/s. Reduced exploration and these reward changes did not establish recovery within this budget. The preset remains experimental and does not replace the submitted model450 or its recorded 5/5 result.
