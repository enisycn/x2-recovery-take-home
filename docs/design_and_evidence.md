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


## Supine-only training diagnosis

The original 500-update scratch run scored 0/5. It reached 0.763–0.783 m while
both feet were airborne; angular speed at those maximum-height snapshots was
5.36–9.38 rad/s. Strict stance reward was exactly zero throughout training.
Mixed-start pretraining had already observed this reward at iteration 3.
Its reset distribution and random seed differed, so this is evidence of missing
standing experience, not a controlled proof that curriculum is necessary.

The original target mapping can saturate. Its derivative is
`span * (1 - tanh(action)^2)`: at raw action 5 the dimensionless sensitivity is
only 0.000182. In three saved snapshots per evaluation episode, 62/120 action
values from the failed model exceeded magnitude 2.5, versus 9/120 for the historical
model450. These counts describe selected snapshots, not entire trajectories.
Reducing Gaussian std cannot by itself correct a saturated mean action.

## Selected dense-stance reward preset

`--stability_refinement` selects this optional preset for either a fresh run or a
continuation. Robot physics, action mapping, supine resets and strict evaluation
limits remain identical to `relaxed_v4`. The following changes are local X2
engineering choices; they are not paper constants.

| Term | Weight | Definition or change |
| --- | ---: | --- |
| Signed pelvis height | 40 | Original continuous height/orientation shaping retained. |
| Two-foot support | 10 | Original height/upright-gated contact term. |
| Standing still | 20 | Original balance term, increased weight. |
| Stance proximity | 20 | Exponential pose/speed proximity with two-foot contact; training widths: tilt 0.35, linear speed 0.8 m/s, angular speed 2 rad/s. |
| Near-stance motion | -2 | `g * o^2 * (norm(v)^2 + 0.1*norm(omega)^2)`. |
| Leg pose | 20 | `height_gate * upright_gate * exp(-mean(q_leg^2)/0.7^2)` for hip pitch, knee and ankle pitch. Height gate rises from 0.40 to 0.68 m. |
| Action saturation | -0.5 | Squared raw-action excess beyond each group's effective interval. Bounds account for both imported soft-limit clipping and `abs(action)=2.5` tanh saturation. |
| Joint speed | -0.0005 | Sum of squared joint velocities. |
| Action change | -0.02 | Sum of squared successive raw-action differences. |

Here `o=clip((1-g_z)/2,0,1)` and `g=clip((z-0.45)/0.13,0,1)`.
Ankle-pitch soft-limit clipping begins near raw -1.557 and 1.233; knee extension
near -2.275 remains available. All other reward terms retain their original
settings. **The validator still requires tilt <=0.15, linear speed <=0.25 m/s,
angular speed <=0.35 rad/s, correct orientation, height and foot-only support for
0.5 continuous seconds.** Training widths do not relax these checks.

[FRASA, Section IV-D](https://arxiv.org/html/2410.08655v3#S4.SS4) motivates
exponential proximity to the desired pose. Its near-neutral reset sampling is
not used here. [HumanUP, Appendix A.2](https://arxiv.org/html/2502.12152v2#A2)
motivates velocity/control regularization; this repository does not reproduce
its slowed-trajectory tracking stage.

Two earlier refinements of the failed scratch actor scored 0/5 after 100 and 300
updates. They used contact-gated height, which created a reward drop before
hand-to-foot transfer completed. The current preset restores continuous height
shaping and scores contact separately. A new 500-update random-weight run with
the current preset also scored 0/5: seeds 103 and 105 briefly met all criteria
for 0.04 s, below the required 0.5 s. This is partial progress, not recovery.
Continuing the same reward preset for 500 more updates reached **5/5**, with 8.78–9.02 s strict stance at episode end. All training resets in this new lineage were supine. At that 1,000-update stage, the additional neutral-arm criterion was still unmet. This combined reward/exploration change and larger training budget is not a single-factor ablation. The historical mixed-start model450 remains separate.

## Shoulder refinement

The optional posture preset adds
`80 * gate * exp(-(a_shoulder - atanh(0.25))^2 / 9)` while retaining the physical
arm-pose reward. The inverse target follows from `0 = -0.5 + 2*tanh(a_shoulder)`.
The gate multiplies `clip((z-.50)/.15,0,1)`,
`clip((-g_z-.95)/.04,0,1)`, two-foot contact and absence of other-body support.
It therefore leaves the initial ground push unconstrained by this term.

This raw-command objective gives different scores to shoulder actions even
where their tanh-mapped targets are nearly identical. It does not override the
policy at playback, use demonstration trajectories, or couple the shoulder
score to the waist command. The target, width, weight and gate are local choices.
[HoST, Sections IV-C and V-A](https://arxiv.org/html/2502.08378) motivates separate
post-task objectives and motion-quality measurements; its full multi-critic,
L2C2 and curriculum method is not implemented here.

The refinement retains the supine-only actor, critic, optimizer and learned
per-channel exploration standard deviations. Earlier uniform std overrides
also increased noise in already-stable leg channels, so they are not repeated.
Every reset remains supine. This is an explicit reward change during continued
PPO training, not a claim of unchanged rewards throughout the whole lineage.
The evaluator reports final-two-second arm position ranges and joint-velocity
RMS alongside actual pose errors. These quantities are measured at 50 Hz and
are separate from the unchanged recovery predicate.

The optional `--shoulder_command_variance` controls the exponential denominator
(default 9). A final precision experiment uses 1: near-neutral command errors
then receive more distinct scores without changing the target or maximum
reward. Both widths are local choices and are saved in the run configuration.
The original broad reward remains reproducible by omitting this override.

Final precision model2248 passes all five canonical recoveries and the full final-two-second neutral-arm check. The selected ancestry contains 2,254 PPO updates, every reset supine. Exact stages, rejected branches and physical arm measurements are recorded in [development history](development_history.md) and [validation](validation.md).
