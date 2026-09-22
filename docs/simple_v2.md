> Version status: the independent-action v2 experiment was rejected (0/5, inverted-pose exploit). Its reset/contact/measurement fixes and common reward definitions remain in use. The selected successful controller is [symmetric_v3](symmetric_v3.md), with explicitly documented changes.

# Corrected compact baseline (22 September 2026)

This is a new experiment, not a reinterpretation of the failed HumanUP-history checkpoint. The input/control contract is different; use `--environment simple_v2` for evaluation, rendering and the policy server. A successful recovery is not claimed until the recorded five-episode evaluation supports it.

## Fixed defects

- Clear filtered body-to-floor contact history on selected environment resets. The installed PhysX reset kernel omitted this buffer. The local subclass preserves untouched environments; the installed Isaac source is unchanged.
- Test support against the **current** filtered force matrix. A foot touching in an earlier sample cannot satisfy present support. Self-collision forces do not count as floor support.
- Preserve terminal physical state before automatic reset. The evaluator and ROS server use this captured state and stop at the first episode boundary. The older diagnostic reports its last sampled pre-action state explicitly.
- Restrict imitation data to each environment's first episode, including the length boundary. Split behavioral-cloning validation by whole episode; clear stale Adam moments after changing actor weights.
- Apply an explicitly requested learning-rate override after checkpoint loading, which otherwise restores the old optimizer rate.

## Compact learning contract

ManagerBasedRLEnv with the original pinned X2 Ultra model, self-collisions, floating base and imported limits. PhysX at 200 Hz, decisions at 50 Hz. These rates are engineering settings checked by a short 400 Hz comparison; they are not claimed as HumanUP's rates.

Observation width is 168: pelvis height (1), base-frame linear/angular velocity (3+3), projected gravity (3), relative joint positions (31), scaled joint velocities (31), both-foot and whole-body floor-contact bits (2+32), and two previous actions (62). Gravity avoids the roll/pitch discontinuity at the supine Euler singularity. No autoencoder, diffusion model, privileged-zero block or history CNN.

The actor and critic are ELU MLPs `[512,256,128]`. PPO uses clip 0.2, gamma 0.99, GAE lambda 0.95, 5 epochs, 4 minibatches, normalized observations and initial Gaussian standard deviation 1.0. Each update has 3,000 environments × 32 decisions = 96,000 transitions. Initial learning rate is 3e-4 with adaptive KL target 0.01.

For all 31 actuated joints:

`q_target = clip(q_default + 0.75 * action, soft_lower, soft_upper)`.

The absolute target is held during the four PhysX substeps. It retains corrective authority at standing height; no height-dependent command brake exists. Position, velocity and torque limits remain enforced. PPO samples raw Gaussian actions; the environment clips position targets. This is ordinary bounded actuator execution, not a claim of differentiating through the physics.

Every reset selects true back-lying with probability 0.5, otherwise one of nine audited upright-root reference poses. These references are **squats and sitting starts**, not an interpolated supine trajectory. Evaluation probability is zero for references and no external lift force is enabled.

## Research and exact adaptations

Primary references:

1. [HumanUP, RSS 2025](https://www.roboticsproceedings.org/rss21/p063.html), downloaded as `references/humanup_rss2025.pdf`: discovery with weak constraints, absolute position control, and mixed reference starts. Its two-stage method and G1 results do not establish success for X2. This baseline deliberately does not claim full replication.
2. [HoST, RSS 2025](https://www.roboticsproceedings.org/rss21/p064.html) and [official cross-robot guidance](https://github.com/InternRobotics/HoST#-extend-host-to-other-humanoid-robots-tips): robot-specific controller, target posture, collision and curriculum choices matter. HoST's measured-relative torque controller, force curriculum and multi-critic PPO are not implemented by this compact baseline.
3. [Official AgiBot simulation documentation](https://x2-aimdk.agibot.com/en/latest/sim_rl/index.html): distinguishes existing motion-control deployment from RL training, and identifies example gains as simulation recipes. Loading a vendor get-up skill would not validate our own PPO experiment.

RewardManager multiplies the weighted sum below by decision dt. Heights are world Z above the fixed flat floor, gravity `g` is base-frame projected gravity, velocities are root-world velocities. These X2 weights are explicit adaptations, not copied full paper settings.

| Term | Formula/definition | Weight |
| --- | --- | ---: |
| Pelvis height | `exp(clip(z,0,.68))-1`, HumanUP code-release form | 10 |
| Head height | `exp(clip(z_head,0,1.2))-1`, HumanUP code-release form | 5 |
| Upright | `exp(-g_z)`, HumanUP form | 2 |
| Two feet | both floor loads ≥15 N × clipped height ramp .30→.68 × `max(-g_z,0)^2` | 5 |
| Other support | count of non-foot floor contacts ≥15 N × height ramp .50→.68 | -1 |
| Balance | `1(z>.60 and -g_z>.96)*exp(-(norm(v)+norm(omega))^2/.25)` | 10 |
| Strict stance | complete instantaneous predicate below | 20 |
| Action rate | squared norm of action difference | -.005 |
| Torque | squared norm of applied joint torques | -1e-6 |
| Joint limits | sum of violations outside soft position limits | -1 |
| Failure | true safety termination, not timeout | -10 |

No upright-height reward gate blocks low-pose discovery. Non-foot support is allowed during getting up; it is excluded from final success.

A strict sample requires pelvis ≥.58 m, projected-gravity XY norm ≤.15 and Z ≤-.98, linear speed ≤.25 m/s, angular speed ≤.35 rad/s, both feet ≥15 N and every other body <15 N. Evaluation requires 0.5 s consecutive valid samples and additionally reports whether the stance survived to the end of the entire 10 s episode. No success termination is used in training.

## Reproduce

Use the existing interpreter and the preserved HRS CPU set:

```bash
export ISAAC_PYTHON=<USER_HOME>/miniconda3/envs/codex/bin/python
export HRS_CPUSET=<HOST_CPUSET> HRS_NICE=15
./scripts/train_isaac.sh --phase simple_v2 --num_envs 3000 --max_iterations 400 --device cuda:0
./scripts/play_isaac.sh /absolute/path/model.pt --environment simple_v2 --headless --device cuda:0 --output reports/simple_v2_evaluation.json
./scripts/render_isaac_gifs.sh --checkpoint /absolute/path/model.pt --environment simple_v2 --headless --device cuda:0 --output_dir reports/gifs_simple_v2
./scripts/serve_isaac_policy.sh reports/exported_simple_v2/policy.pt --environment simple_v2 --device cuda:0
```

Checkpoint configurations are retained in each run's `params/env.yaml` and `params/agent.yaml`. Do not load an older 1,148-input relative-action checkpoint into this 168-input absolute-action environment.
