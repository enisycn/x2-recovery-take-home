#!/usr/bin/env python3
"""Evaluate an exported X2 TorchScript policy for exactly five fixed episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", type=Path, required=True, help="Exported RSL-RL policy.pt")
parser.add_argument("--output", type=Path, default=Path("reports/isaac_evaluation.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import x2_recovery_isaac  # noqa: E402,F401
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from x2_recovery_isaac import mdp  # noqa: E402
from x2_recovery_isaac.env_cfg import FEET, X2RecoveryPlayEnvCfg  # noqa: E402


SEEDS = [101, 102, 103, 104, 105]
STABLE_STEPS = 10  # 0.5 s at the 20 Hz policy rate, matching FRASA's stability check.


def policy_observation(observation):
    return observation["policy"] if isinstance(observation, dict) else observation


def snapshot(env, feet_cfg: SceneEntityCfg, all_cfg: SceneEntityCfg) -> dict:
    robot = env.scene["robot"]
    contact_sensor = env.scene.sensors["contact_forces"]
    forces = contact_sensor.data.net_forces_w_history.torch.norm(dim=-1).amax(dim=1)[0]
    foot_ids = feet_cfg.body_ids
    unsupported = forces.clone()
    unsupported[foot_ids] = 0.0
    projected_gravity = robot.data.projected_gravity_b.torch[0]
    return {
        "pelvis_height_m": round(float(robot.data.root_pos_w.torch[0, 2].item()), 4),
        "tilt_metric": round(float(projected_gravity[:2].norm().item()), 4),
        "linear_speed_m_s": round(float(robot.data.root_lin_vel_w.torch[0].norm().item()), 4),
        "angular_speed_rad_s": round(float(robot.data.root_ang_vel_w.torch[0].norm().item()), 4),
        "left_foot_force_n": round(float(forces[foot_ids[0]].item()), 2),
        "right_foot_force_n": round(float(forces[foot_ids[1]].item()), 2),
        "max_other_body_force_n": round(float(unsupported.max().item()), 2),
    }


def main() -> None:
    policy_path = args.policy.expanduser().resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"Exported policy not found: {policy_path}")

    config = X2RecoveryPlayEnvCfg()
    config.scene.num_envs = 1
    env = gym.make("HRS-X2-Recovery-v0", cfg=config)
    unwrapped = env.unwrapped
    feet_cfg = SceneEntityCfg("contact_forces", body_names=FEET)
    all_cfg = SceneEntityCfg("contact_forces", body_names=".*")
    feet_cfg.resolve(unwrapped.scene)
    all_cfg.resolve(unwrapped.scene)
    policy = torch.jit.load(str(policy_path), map_location=unwrapped.device).eval()
    max_steps = round(config.episode_length_s / unwrapped.step_dt)
    records: list[dict] = []

    for episode, seed in enumerate(SEEDS, start=1):
        observation, _ = env.reset(seed=seed)
        consecutive_stable = 0
        success = False
        terminal_snapshot: dict = {}
        steps = 0
        for step in range(max_steps):
            steps = step + 1
            with torch.inference_mode():
                instant_success = bool(
                    mdp.strict_success(unwrapped, feet_cfg=feet_cfg, all_bodies_cfg=all_cfg)[0].item()
                )
                consecutive_stable = consecutive_stable + 1 if instant_success else 0
                terminal_snapshot = snapshot(unwrapped, feet_cfg, all_cfg)
                if consecutive_stable >= STABLE_STEPS:
                    success = True
                    break
                action = policy(policy_observation(observation))
                observation, _, _, _, _ = env.step(action)

        record = {
            "episode": episode,
            "seed": seed,
            "success": success,
            "steps": steps,
            "stable_steps": consecutive_stable,
            "failure_reason": "" if success else "timeout before 0.5 s strict stable stance",
            **terminal_snapshot,
        }
        records.append(record)
        print(f"episode={episode} seed={seed} success={success} steps={steps}")

    result = {
        "backend": "Isaac Lab 3.0 / PhysX",
        "task": "HRS-X2-Recovery-v0",
        "policy": str(policy_path),
        "success_definition": {
            "pelvis_height_m_min": 0.62,
            "tilt_metric_max": 0.15,
            "linear_speed_m_s_max": 0.20,
            "angular_speed_rad_s_max": 0.35,
            "each_foot_contact_force_n_min": 15.0,
            "other_body_contact_force_n_max": 15.0,
            "stable_duration_s": 0.5,
        },
        "successful_recoveries": sum(record["success"] for record in records),
        "total_episodes": len(records),
        "episodes": records,
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"successes={result['successful_recoveries']}/5 report={output_path}")
    env.close()


try:
    main()
finally:
    simulation_app.close()
