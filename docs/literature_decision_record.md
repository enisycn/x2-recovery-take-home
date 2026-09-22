> **Final decision, 22 September:** direct PPO succeeded on the nominal X2 task after the corrected measurement/control path, signed height reward and FRASA-inspired symmetric action subspace were combined. No demonstration or imitation training was needed for this selected policy. See [symmetric_v3](symmetric_v3.md) for adaptations and the 5/5 evidence; earlier unsuccessful method decisions below are historical.

# Literature-to-implementation decision record

This record separates published equations, official code behavior, X2 morphology adaptations, and assignment-specific additions. It is intended to make every training choice auditable rather than imply that a mixed implementation is a verbatim reproduction of one paper.

## Primary sources and integrity

| Source | Venue / status | Local artifact | SHA-256 | Code snapshot |
| --- | --- | --- | --- | --- |
| X. He et al., *Learning Getting-Up Policies for Real-World Humanoid Robots* (HumanUP) | RSS 2025 | `references/humanup_rss2025.pdf` | `5a594169d416cf9903bf79224a3461e70b69b20800ad360b2591c4426b7a8cc4` | `RunpeiDong/HumanUP@7516e0f27e6f4d1e7365cf64ea577a78247bd8cb` |
| T. Huang et al., *Learning Humanoid Standing-up Control across Diverse Postures* (HoST) | RSS 2025 | `references/host_rss2025.pdf` | `f3f16aedb3b16ce38a6e65b1d983638b5b68434b6a6de1ab69353edad090286c` | `OpenRobotLab/HoST@70bb580949a336a920833700e4b5dc3bf7fe87ce` |
| C. Gaspard et al., *FRASA: An End-to-End Reinforcement Learning Agent for Fall Recovery and Stand Up of Humanoid Robots* | arXiv:2410.08655v3 | `references/frasa_arxiv2410.08655.pdf` | `119942a95519200c4e95b93d84e02cc4c1a38533fde6c0e480ff0a76f8c16876` | Equations audited from the paper |
| A. Rajeswaran et al., *Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations* (DAPG) | RSS 2018 | `references/dapg_rss2018.pdf` | `be865c6ae5537023c38b86e1c07a6ed342ea659e6aca2d224cb5f95b7b1664fa` | Equations audited from the paper |

All four PDFs were downloaded only from RSS or arXiv over HTTPS and handled as inert documents. Local `pdfinfo`, `pdfdetach` and object-tree inspection found no encryption, JavaScript, embedded files, forms or additional actions. The three files with an `OpenAction` use only a local `GoTo` page destination. A byte-level `/JS` hit in the FRASA file is a font/resource token rather than executable JavaScript. Source repositories were checked out under `tmp/research` with hooks and submodules unused. No downloaded code, attachment or document action was executed.

## What directly applies to this task

### Contact-rich discovery and temporal state: HumanUP

HumanUP identifies accurate collision geometry and sparse exploration as the central difficulties of get-up control. Its Stage I discovers a trajectory with PPO, minimal regularization and 10--20% standing resets. Stage II stretches a successful trajectory and trains a smoother tracking policy. This supports four X2 choices:

1. Use the complete official collision model and whole-body contact sensing.
2. Keep success observable for multiple steps instead of ending the episode on first reaching a stance.
3. Include temporal proprioception because contact transitions are not determined by a single pose.
4. Use imitation only after a usable trajectory exists; HumanUP is not behavioral cloning from human data.

The X2 actor follows HumanUP's released RMA encoder. Current proprioception is 98 values: base angular velocity (3), roll/pitch (2), 31 relative joint positions, 31 joint velocities, and 31 previous actions. Ten historical proprioceptive states are projected `98 -> 30`, passed through temporal convolutions `30 -> 20` (kernel 4, stride 2) and `20 -> 10` (kernel 2), flattened and mapped to a 20-value latent. The actor receives `98 + 20 = 118` values and uses `512 -> 256 -> 128 -> 31`. The critic receives current proprioception, a 70-value Stage-I extrinsics slot, and 980 history values: 1,148 total. This is temporal compression, not an autoencoder: there is no decoder or reconstruction loss.

The faithful discovery PPO configuration copies rollout 24, clip 0.2, five epochs, four minibatches, learning rate `2e-4`, adaptive KL target 0.008, entropy 0.01, gamma 0.99, lambda 0.95, max gradient norm 1.0, and initial standard deviation 1.0. The released 24 Stage-I reward terms and weights are implemented in `HumanUpStageIRewardsCfg`; the paper/code coefficient differences are listed in `humanup_paper_review.md`.

HumanUP's released G1 environment ends an episode on timeout, root linear speed above 2.5 m/s, or pelvis height outside `[0, 1.2]` m. It does not terminate on success. `HumanUpTerminationsCfg` reproduces those broad safety bounds. The exact stance remains a reward/evaluation predicate, so the actor must hold it continuously.

### Staged exploration and post-task stability: HoST

HoST divides standing up into righting, rising and standing stages and separates task, style, regularization and post-task rewards. It uses PPO, multi-critic advantage aggregation, an upward-force/action-bound curriculum, relative joint-position targets, five previous states, and explicit post-task terms to remain standing.

The official action equation is

`q_target = q_current + beta * action`,

with bounded action and a curriculum on `beta`. The X2 teacher uses the same relative-target structure with a fixed conservative `beta=0.25`, smooth `tanh` bounding, one target per 20 Hz policy step, and clipping to imported soft limits. HumanUP's absolute action remains available in the separately named faithful discovery environment.

The official HoST post-task functions used as the baseline are:

- `exp(-2 ||omega_xy||^2)` for base angular velocity;
- `exp(-5 ||v_xy||^2)` for base linear velocity;
- `exp(-5 ||g_xy||^2)` for base orientation;
- `exp(-20 |h-h_target|)` for base height in the released code.

All X2 completion terms activate only after the pelvis exceeds 0.58 m, the morphology-scaled counterpart of HoST's final stage boundary. HoST Table VI prints the height term with a squared norm, while the official implementation at the pinned commit uses absolute error. The X2 code retains `exp(-20|h-h_target|)` and orientation coefficient 5, but tightens angular/linear coefficients from `2/5` to `10/20` and weights them `50/50` after measured overshoot. This is an explicit X2 adaptation, not a verbatim HoST reproduction.

HoST's G1 network uses actor widths `[512, 256, 128]`, critic widths `[512, 256]`, 50 rollout steps, clip 0.2, five epochs, four minibatches, gamma 0.99 and entropy 0.01. Its appendix sets stage heights to 0.45 and 0.65 m and reports an initial 200 N vertical force reduced by 20 N after successful episodes. These values are G1-specific. The X2 bridge heights come from its own collision geometry. Assistance is absent from all evaluation and final imitation rollouts.

### Dense target proximity and stable-duration evaluation: FRASA

FRASA uses an incremental desired-position action, action history, randomized fallen starts and a Gaussian target-state reward

`R_state = exp(-w1 ||psi_t - psi_target||^2)`.

This supports dense proximity rewards and history when a sparse pass/fail event is too rare. FRASA also considers recovery stable only after the state remains within its target tolerance for 0.5 s. The HRS evaluator follows that temporal principle while applying explicit X2 physical conditions: minimum pelvis height, signed upright orientation, low root speeds, both feet loaded, and no other body support for ten consecutive 20 Hz decisions.

FRASA is not copied wholesale. It controls a five-DoF sagittal abstraction with CrossQ, while this task uses all 31 X2 actions, PPO and 3D contacts. Its desired-joint-pose reward would overconstrain X2 to one recovery style, so it is used as evidence for dense goal proximity rather than as a claim of algorithm reproduction.

## Why behavioral cloning is present

Pure PPO learned a successful standing controller in the legacy 168-value observation space, but the HumanUP RMA actor initially lacked a successful X2 trajectory. Successful fixed-seed standing and shallow-rise rollouts were therefore recorded from that locally trained policy and used for supervised action regression. No external or human demonstration data is involved. The resulting RMA actor is then fine-tuned with PPO. This is a compute-efficient policy-transfer bootstrap consistent with the broader imitation-plus-RL literature, and is labelled separately from HumanUP Stage II because it clones actions rather than tracking HumanUP's slowed state trajectory.

The data audit records the distinction:

- `reference0_expert`: 128/128 strict successful standing trajectories;
- `reference1_expert`: 256/256 strict successful shallow-rise trajectories;
- `reference2_partial_expert`: 219 trajectories selected by measured peak height, zero strict successes, explicitly labelled partial.

Behavioral cloning reduced held-out action MSE from 0.2395 to 0.000258 for stages 0--1. The resulting RMA policy passed the strict reference-1 audit for 1.15 s. Adding the partial reference-2 data reproduced its teacher's approximately 0.607 m frontier but did not falsely create a success. PPO is responsible for moving beyond that demonstrated frontier.

DAPG provides the direct methodological precedent for this order of operations: behavior cloning supplies an informed policy initialization, then policy-gradient RL explores beyond the cloned behavior. The X2 implementation does not claim to reproduce DAPG's augmented demonstration-gradient loss; it uses the supported BC-then-RL initialization pattern and relies on environment reward thereafter. This distinction is important because the stage-2 demonstration is imperfect.

## Current diagnostic conclusion

The selected RMA checkpoint is structurally sound: 1,148 observations, a 20-value temporal latent, 31 actions, finite weights, finite action standard deviations and a normalizer count of 54,504,000. From `reference_2` it reaches 0.5972 m, uses only the two feet at its height peak, and holds the complete 0.58 m/0.25 m/s strict predicate for at most 0.25 s. From five true-supine seeds it never exceeds 0.1905 m and scores 0/5. The saved training configuration proves that its final phase sampled only `reference_2`; the major remaining error is therefore coverage of the lower bridge-to-supine transition. Network shape, frame freshness, finite values, target limits, ground-only contact filtering and success termination were checked independently.

The next checkpoint is accepted only if an unassisted live audit and the fixed evaluator show at least 0.5 s of the complete strict predicate. Training reward or pelvis height alone is not success evidence.
