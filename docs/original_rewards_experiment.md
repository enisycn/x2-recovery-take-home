# Original rewards, supine-only training

This experiment asks whether the original reward set can recover from the
floor with a longer fixed budget, without later reward additions. It starts
from random actor/critic weights and keeps all settings fixed for 2,000 PPO
updates. Every episode starts supine; there are no reference starts, external
forces, demonstrations or pretrained weights.

The 12 reward terms and their parameters match the saved configuration of
the historical mixed-start pretraining run (`posture_from_start`, 22 September)
and the original supine-only scratch run (`supine_from_scratch`, 27 September).
The local reward function bodies also match historical revision `d0f45b0`.
The reset distribution is deliberately supine-only: this tests the old rewards,
not a replay of mixed-start pretraining.

| Original term | Weight |
| --- | ---: |
| Signed pelvis height | 40 |
| Head height | 5 |
| Upright orientation | 5 |
| Two-foot support near standing | 5 |
| Other-body support near standing | -1 |
| Standing still | 10 |
| Strict stance | 20 |
| Action change | -0.005 |
| Joint torque | -0.000001 |
| Joint limits | -1 |
| Failure termination | -10 |
| Supported relaxed arms | 40 |

The later successful supine lineage added stance proximity, near-stance motion
cost, joint-speed cost, leg-posture shaping and action-saturation cost; it also
changed foot, balance and action-change weights. It reached 5/5 stance after
1,000 updates, then received a focused shoulder-command objective and further
training to reach the selected neutral-arm result after 2,254 inherited
updates. Those additions are **not active** in this new run. Neither are the
controlled-rise or load-transfer presets.

## Reproduction

Prepare the Isaac terminal as described in [commands.md](commands.md):

```bash
./scripts/train_isaac.sh --phase relaxed_v4 --num_envs 3000 \
  --max_iterations 2000 --seed 47 --device cuda:0 \
  --run_name original_rewards_supine2000 \
  --action_std_override 1.0 --entropy_coef 0.005 \
  --learning_rate_override 0.0003 --learning_schedule adaptive
```

Do not add a checkpoint or reward-refinement flag. At 32 rollout steps per
environment, 2,000 updates collect 192 million transitions. Initial exploration
std 1.0 and entropy coefficient 0.005 match the original reward experiment,
not the later dense-stance/arm continuation settings.

After completion:

```bash
experiment=logs/rsl_rl/hrs_x2_relaxed_v4
run_dir="$experiment/$(cat "$experiment/LATEST_RUN.txt")"
./scripts/evaluate_isaac.sh "$run_dir/model_1999.pt" \
  --environment relaxed_v4 --device cuda:0 \
  --output "$run_dir/evaluation.json"
```

The evaluation keeps the existing physical success checks and reports
neutral-arm posture separately. Check recovery count, stance at episode end,
the last two seconds of arm posture, and video before choosing a checkpoint.
Reward alone cannot establish recovery. A single training seed can show this
configuration works or fails in that run; it cannot prove later reward terms
are universally necessary or unnecessary.
