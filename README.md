# HRS - AgiBot X2 recovery

Isaac Lab / PhysX PPO recovery for the official AgiBot X2 Ultra v1.3.0 model, plus a ROS 2 Humble interface that starts a real simulator episode and publishes measured joint state.

**Selected result: 5/5 true-supine recoveries.** The final policy remains in strict unsupported two-foot stance for 8.54-8.94 s of each 10 s episode and satisfies the relaxed-arm criterion for the full final two seconds. The final experiment, evaluation and ROS runtime use no lift force, reference reset, observation noise or domain randomization.

## Recovery video

https://github.com/user-attachments/assets/3cd71353-9c40-4e66-9f9c-ef9d5fe02d2c

The first five seconds hold the initial supine frame for inspection; the following ten seconds show the recorded recovery episode. [Download the MP4](reports/videos/x2_recovery_supine_to_standing.mp4).

- [Final GIF](reports/gifs_relaxed_v4/x2_final_policy_attempt.gif)
- [Five-episode evaluation](reports/relaxed_v4_evaluation.json)
- [Checkpoint](reports/checkpoints/x2_relaxed_v4_model450.pt)
- [Reward curve](reports/relaxed_v4_training_reward.png)
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

The 122 policy inputs are pelvis height; body-frame linear/angular velocity; three-component projected gravity; 31 joint positions and velocities; two foot contacts; 32 whole-body ground-contact bits; and two previous 8-value actions. Eight bilateral commands map to all 31 absolute joint-position targets as:

```text
q_target = centre + span * tanh(action)
q_target = clamp(q_target, imported soft joint limits)
```

Actor and critic are separate normalized ELU MLPs with widths `[512, 256, 128]`. The eight action groups control bilateral hip pitch, knee, ankle pitch, shoulder pitch, elbow, waist pitch, ankle roll and hip roll. Remaining joints stay at neutral targets.

### Rewards

Isaac RewardManager integrates each weighted term with policy timestep 0.02 s. Reward formulas are in [mdp.py](isaaclab_ext/x2_recovery_isaac/mdp.py); the final weights and enabled terms are in [simple_cfg.py](isaaclab_ext/x2_recovery_isaac/simple_cfg.py).

| Term | Weight | Definition and purpose |
| --- | ---: | --- |
| Signed pelvis height | 40 | `clip(z/.68,0,1) * clip((1-g_z)/2,0,1)`; inverted high poses receive zero |
| Head height | 5 | Clipped exponential progress toward 1.2 m |
| Upright | 5 | `exp(-g_z)` |
| Both feet | 5 | Current two-foot ground contact, height/upright gated |
| Other support | -1 | Non-foot ground contacts near standing height |
| Balance | 10 | Low root speed near upright standing |
| Strict stance | 20 | Complete recovery predicate below |
| Relaxed arms | 40 | Supported-upright gate and arm-pose Gaussian; shoulder 0, elbow -0.15 rad |
| Action change | -0.005 | Squared successive action difference |
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

The extra final-pose check requires both shoulder-pitch errors and elbow errors <= 0.30 rad while strict stance holds. Evaluation observes all 500 steps and records the terminal state before automatic reset.

## Train, evaluate and render

PPO uses clip 0.2, gamma 0.99, GAE lambda 0.95, five learning epochs, four minibatches, value-loss coefficient 1, clipped value loss, desired KL 0.01 and gradient clipping 1. The final stage uses 3000 environments, 32 steps/environment, seed 47, fixed learning rate 1e-4, initial action standard deviation 0.10 and entropy 0.001.

The supplied parent checkpoint supports re-running the final 51-update stage. [Commands](docs/commands.md) gives the training, playback and five-episode evaluation sequence.

Every successful `train_isaac.sh` run automatically adds `reward.png`, `reward.csv` and `run_manifest.json` beside its TensorBoard events, parameter snapshots and checkpoints. The experiment-level `LATEST_RUN.txt` points to that directory. See [training outputs and file locations](docs/artifact_locations.md) for the exact tree and commands. Evaluation JSON and GIF rendering remain explicit simulator steps.

Earlier exploratory pretraining used 50% auxiliary upright-root squat/sitting resets. Those states were neither expert trajectories nor a continuous supine trajectory. The submitted final stage, evaluation and ROS runtime use 100% supine resets. See [development history](docs/development_history.md).

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

Seeds 101-105 all succeed and remain standing at episode end. Strict stance lasts 8.54-8.94 s and combined strict/relaxed stance 7.78-8.86 s. There were no failures in the submitted five-episode evaluation; every `failure_reason` field is empty. Historical failures and their fixes are recorded in [development history](docs/development_history.md).

These are nominal simulation results, not hardware or general robustness claims. Observation noise, dynamics randomization, terrain variation, actuator latency and hardware safety validation remain future work. The supplied parent checkpoint reproduces the final 51-update stage; the entire exploratory lineage is documented but is not presented as a bit-for-bit full retraining recipe.
