# HRS - AgiBot X2 recovery

Isaac Lab / PhysX PPO recovery for the official AgiBot X2 Ultra v1.3.0 model, plus a ROS 2 Humble interface that starts a real simulator episode and publishes measured joint state.

**Selected result: 5/5 recoveries, with neutral arms throughout the final two seconds.** The PPO lineage begins with random weights and every training reset is supine. It uses 2,254 selected updates, without mixed-start pretraining, imitation or lift assistance. Seeds 101–105 all recover and remain standing at episode end for 8.94–9.14 s. Every sample of the final two seconds also meets the separate neutral-arm check: maximum shoulder error 0.233 rad and elbow error 0.099 rad, below the 0.30 rad limits. The largest arm-joint position range is 0.0019 rad and velocity RMS 0.026 rad/s in that window (50 Hz measurements).

## Recovery video

![Supine recovery](reports/gifs_relaxed_v4/x2_final_policy_attempt.gif)

The first two seconds hold the initial supine frame for inspection; the following ten seconds show the recorded recovery episode. On-screen labels are in English. [Download the MP4](reports/videos/x2_recovery_supine_to_standing.mp4).

- [Final GIF](reports/gifs_relaxed_v4/x2_final_policy_attempt.gif)
- [Five-episode evaluation](reports/relaxed_v4_evaluation.json)
- [Checkpoint](reports/checkpoints/x2_supine_model2248.pt)
- [Reward curve](reports/relaxed_v4_training_reward.png)
- [Standing and arm metrics](reports/arm_validation.png)
- [Submission report (PDF)](docs/submission_report.pdf)
- [Requirement map](docs/task_requirements.md)
- [Validation commands and outcomes](docs/validation.md)
- [Setup, training and ROS commands](docs/commands.md)

## Setup and dependencies

Tested on Ubuntu 22.04.5 with one NVIDIA RTX 5080 Laptop GPU (16 GB), Isaac Sim 6.0.1, Isaac Lab 3.0.0-beta2.patch1, RSL-RL 5.0.1, and ROS 2 Humble. Isaac runs in Python 3.12 and ROS in system Python 3.10 as separate processes. The final training stage used 3,000 parallel environments; checkpoint playback needs far less GPU memory.

For a fresh machine, clone this repository, install [Isaac Sim 6.0.1](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/install_python.html), [Isaac Lab 3.0.0-beta2.patch1](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-beta2.patch1), [RSL-RL 5.0.1](https://github.com/leggedrobotics/rsl_rl/releases/tag/v5.0.1) and [ROS 2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html), then follow [the commands in order](docs/commands.md). Set `ISAAC_PYTHON` to the Python executable in the Isaac environment. The guide covers the clone, dependency check, official model fetch/import, training, checkpoint playback, evaluation, ROS build and live request. No user-specific path is required.

The model fetcher takes only official AgiBot source at pinned commit `60c5de582c523cd188f563819e62d34cfdc3d2d0`, without hooks or submodules. Imported assets are local and ignored by Git.

The imported robot is a 41.966521 kg floating articulation with 31 joints, 32 recursively monitored rigid bodies, self-collision, official actuator limits and one 98% soft joint-limit margin. Fixed links are merged during URDF-to-USD conversion. Geometry, mass, inertia, joint axes and actuator limits are not edited. The scene uses a repository-local 200 m x 200 m collision floor.

## Environment

PhysX runs at 200 Hz and policy targets at 50 Hz. Episodes last 10 s. Every final `relaxed_v4` episode starts supine at pelvis height 0.190 m, with zero velocity and small seeded pose perturbations.

The 122 policy inputs are pelvis height; body-frame linear/angular velocity; three-component projected gravity; 31 joint positions and velocities; two foot contacts; 32 whole-body ground-contact bits; and two previous 8-value actions. The network outputs eight numbers, one per joint group. Most groups command the same target for left and right joints; joints outside these groups keep their neutral targets. Each commanded joint receives a position target in radians:

```text
q_target = centre + span * tanh(action)
q_target = clamp(q_target, imported soft joint limits)
```

`centre` is the group's reference angle, `span` its maximum offset, and `tanh` bounds the offset between -1 and 1. For both hip-pitch joints, `centre = -0.9` and `span = 1.4`: an action of 0 targets -0.9 rad; an action of +1 targets about +0.17 rad before the limit clamp. These are joint-position controller targets, not instantaneous joint angles or direct torque commands. The groups cover hip pitch, knee, ankle pitch, shoulder pitch, elbow, waist pitch, ankle roll and hip roll. Actor and critic are separate normalized ELU MLPs with widths `[512, 256, 128]`.

### Rewards

At each 0.02 s policy step, Isaac RewardManager adds `weight × term value × 0.02` across the active terms. Positive weights encourage a behavior; negative weights penalize it. These are reward coefficients, not neural-network weights. Reward formulas are in [mdp.py](isaaclab_ext/x2_recovery_isaac/mdp.py); the final weights and enabled terms are in [simple_cfg.py](isaaclab_ext/x2_recovery_isaac/simple_cfg.py). The exact resolved settings saved with the checkpoint are in [env.yaml](reports/configs/supine_model2248/env.yaml).

| Term | Weight | Definition and purpose |
| --- | ---: | --- |
| Signed pelvis height | 40 | `clip(z/.68,0,1) * clip((1-g_z)/2,0,1)`; inverted high poses receive zero |
| Head height | 5 | Clipped exponential progress toward 1.2 m |
| Upright | 5 | `exp(-g_z)` |
| Both feet | 10 | Current two-foot ground contact, height/upright gated |
| Other support | -1 | Non-foot ground contacts near standing height |
| Balance | 20 | Low root speed near upright standing |
| Strict stance | 20 | Complete recovery predicate below |
| Relaxed arms | 40 | Supported-upright gate and arm-pose Gaussian; shoulder 0, elbow -0.15 rad |
| Shoulder command | 80 | Supported-upright gate × `exp(-(a_shoulder-atanh(.25))^2/v)`; added after 1,000 updates with `v=9`, then narrowed to `v=1` for the final pose |
| Action change | -0.02 | Squared successive action difference |
| Stance proximity | 20 | Smooth distance to upright, supported low-speed stance; training widths 0.35 tilt, 0.8 m/s and 2 rad/s |
| Leg pose | 20 | Neutral hip pitch, knee and ankle pitch above 0.40 m; orientation gated |
| Near-stance motion | -2 | Height/orientation-gated squared base speed plus 0.1 times squared angular speed |
| Action saturation | -0.5 | Squared raw-action excess beyond effective tanh/imported-limit bounds |
| Joint speed | -0.0005 | Squared joint velocity |
| Torque | -1e-6 | Squared joint torque |
| Joint limits | -1 | Soft-limit violation |
| Safety termination | -10 | Non-timeout failure |

HumanUP motivates contact-rich get-up discovery, HoST motivates a separate post-task upper-body objective, and FRASA motivates a symmetric low-dimensional recovery action space. The X2 weights, thresholds, targets and gates are assignment-specific adaptations, not reproductions of those systems. See [design and evidence](docs/design_and_evidence.md).

### Termination and success

Timeout, non-finite state, root linear speed above 2.5 m/s, or pelvis height outside `[0, 1.2]` m ends an episode. Ground contact does not terminate recovery because transitional body support is necessary. Success also does not terminate the episode.

A recovery requires, continuously for at least 0.5 s:

- pelvis height >= 0.58 m;
- projected-gravity XY norm <= 0.15 and Z <= -0.98;
- root linear speed <= 0.25 m/s and angular speed <= 0.35 rad/s;
- each foot ground force >= 15 N;
- every other body ground force < 15 N.

The extra final-pose check requires shoulder-pitch and elbow errors <=0.30 rad while strict stance holds. All five episodes satisfy it throughout the final two seconds. It is reported separately from HRS recovery. Evaluation observes all 500 steps and records the terminal state before automatic reset.

## Train, evaluate and render

PPO uses clip 0.2, gamma 0.99, GAE lambda 0.95, five learning epochs, four minibatches, value-loss coefficient 1, clipped value loss, desired KL 0.01 and gradient clipping 1. All stages use 3,000 environments, 32 steps/environment, seed 47 and entropy coefficient 0. Initial action std is 0.8. The first 1,000 updates use adaptive learning rate starting at 3e-4; shoulder refinement starts at 1e-4, first fixed and finally adaptive.

[Commands](docs/commands.md) records the selected continuation chain. Actor, critic, optimizer and per-channel learned std are retained. `--stability_refinement` selects dense stance shaping. After 1,000 updates, `--posture_refinement` adds the shoulder-command objective; every reset remains supine. There is no reset curriculum, but there is an explicitly documented reward change. A late fixed-rate regression was rejected and training resumed from its earlier stable checkpoint with adaptive KL-based step sizing. RSL-RL repeats the loaded index, so model2248 represents 2,254 selected updates. The final reward combination has not been tested from random weights for 500 updates. This is an empirical recipe, not a minimum training budget or a guarantee of bit-for-bit PhysX retraining.

For graphical checkpoint playback, pass `--viz kit --start-delay 2` to view the supine reset pose for two wall-clock seconds before recovery. The preview does not advance physics or the episode clock. Without a preview, use `--start-delay 0`. Pass `--viz kit` to open the viewer; without a selected visualizer this Isaac Lab version runs headlessly. The guide includes an adjustable seed and a five-seed playback command.

The reward curve covers **all 2,254 updates in the selected ancestry**, including the initial 500-update 0/5 stage. Faint: logged mean return; solid: trailing 20-update mean. Markers identify continuations, the added shoulder objective and its final narrower width. Reward scales differ across those changes. The rejected tail of the fixed-rate trial is a separate branch, documented in development history; it is not inherited by the selected weights. Physical evaluation establishes recovery independently of return.

Every successful `train_isaac.sh` run automatically adds `reward.png`, `reward.csv` and `run_manifest.json` beside its TensorBoard events, parameter snapshots and checkpoints. The experiment-level `LATEST_RUN.txt` points to that directory. See [training outputs and file locations](docs/artifact_locations.md) for the exact tree and commands. Evaluation JSON and GIF rendering remain explicit simulator steps.

Historical model450 used mixed-start pretraining and is superseded by this new lineage. Its files and the earlier failures remain in Git history; see [development history](docs/development_history.md).

## ROS 2

Build and start the Isaac policy server, then launch the recovery and telemetry ROS nodes together. [Commands](docs/commands.md) separates the three terminals; [live validation](docs/ros_live_validation.md) adds the busy rejection and timeout checks.

The launch default is the real `isaac_ipc` backend. A mode-0600 local Unix socket separates the Isaac and ROS Python runtimes. The recovery node returns acceptance before timer-dispatched execution, rejects a second request while running and publishes `IDLE`, `RUNNING`, `SUCCEEDED` or `FAILED` plus 31 simulator joint positions and timestamps. Timeout is configurable. The telemetry node logs status and one joint at 1 Hz.

Re-run the recorded real integration checks with:

```bash
./scripts/validate_ros_isaac_runtime.sh reports/exported_relaxed_v4/policy.pt \
  --environment relaxed_v4 --device cuda:0

PYTHONPATH=isaaclab_ext:src/x2_recovery_ros "$ISAAC_PYTHON" -m pytest -q \
  isaaclab_ext/test/test_reward_formulas.py src/x2_recovery_ros/test
```

See [validation](docs/validation.md) for the fresh build, busy rejection, live telemetry and timeout evidence.

## Results and limits

Seeds 101–105 all recover and remain standing at episode end for 8.94–9.14 s. Every sample of the final two seconds also meets the separate neutral-arm check: maximum shoulder error 0.233 rad and elbow error 0.099 rad, below the 0.30 rad limits. The largest arm-joint position range is 0.0019 rad and velocity RMS 0.026 rad/s in that window (50 Hz measurements). There are no failures among these five nominal tests. Earlier failures and the rejected late refinement remain documented in [development history](docs/development_history.md).

These are nominal simulation results, not hardware or general robustness claims. Observation noise, dynamics randomization, terrain variation, actuator latency and hardware safety validation remain future work. The original 500-update parent and the 1,000-update stance checkpoint remain available for lineage inspection. Configuration snapshots, runtime overrides, source patches and the complete reward curve document the selected lineage. The recipe is reproducible as an experiment, not a guarantee of bit-for-bit PhysX retraining.
