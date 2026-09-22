# Symmetric recovery experiment

The corrected 31-output baseline produced an inverted torso-supported pose: iteration 100 reached ~0.52 m pelvis height but had upright score −0.86 at its peak and ~371 N on its torso. Five actual back-lying episodes failed. Training was stopped after retaining checkpoint 200. Height alone was being exploited; the observation/control plumbing passed but this is not successful recovery.

`symmetric_v3` keeps the v2 reset/measurement fixes and changes two learning choices:

1. Replace the pelvis term with `40 * clip(height/.68,0,1) * clip((1-g_z)/2,0,1)`. At true supine its orientation factor is .5, so there remains a height signal; upside-down high poses lose that signal. Increase `exp(-g_z)` upright weight from 2 to 5. All other reward terms and strict evaluation thresholds are unchanged.
2. Map eight bounded commands into the original 31 motor targets: paired hip pitch, knee, ankle pitch, shoulder pitch, elbow, waist pitch, plus common ankle roll and hip roll for lateral balance. All remaining joints hold their original neutral targets. The official model stays a floating full-body articulation; no body translation/orientation is imposed during motion and no external lift is applied.

Each target is `centre + span*tanh(action)`, followed by the same URDF soft-limit clipping. The exact joint groups/centres/spans are in `synergy_action.py`. Restricting hip pitch to [−2.3,.5] rad avoids the positive extreme used by the inverted baseline. This is a chosen action subspace inside the official limits, not a modification of those limits. The actor input has 122 values (the same 106 current-state values plus 16 previous-action values); output has 8 values. ROS still publishes the 31 actual simulator joint positions.

[FRASA v3](https://arxiv.org/abs/2410.08655v3), Sections III.A–B, uses robot symmetry to observe/control five principal sagittal DoFs. It uses CrossQ and incremental desired-position commands on a smaller Sigmaban. Our 8-command X2 mapping, absolute targets and PPO are explicit adaptations. This reference supports reducing the search space; it does not prove the X2 policy will succeed. HumanUP/HoST still support the getting-up task/reward choices as documented in `simple_v2.md`.

Use `--phase symmetric_v3` to train, and `--environment symmetric_v3` for evaluation, GIF rendering and the ROS policy server. Do not reuse the 31-output checkpoint. The new policy is trained from scratch.
