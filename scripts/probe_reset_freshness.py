#!/usr/bin/env python3
"""Verify that every X2 reset exposes the new supine frame immediately."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("reports/x2_reset_probe.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import x2_recovery_isaac  # noqa: E402,F401
from isaaclab.utils.math import quat_apply  # noqa: E402
from x2_recovery_isaac.env_cfg import X2RecoveryPlayEnvCfg  # noqa: E402


SEEDS = (101, 102, 103, 104, 105)


def main() -> None:
    config = X2RecoveryPlayEnvCfg()
    config.scene.num_envs = 1
    config.sim.device = args.device
    env = gym.make("HRS-X2-Recovery-Play-v0", cfg=config)
    records = []
    try:
        robot = env.unwrapped.scene["robot"]
        for seed in SEEDS:
            observation, _ = env.reset(seed=seed)
            policy_observation = (
                observation["policy"] if isinstance(observation, dict) else observation
            )
            gravity_data = robot.data.projected_gravity_b.torch[0]
            gravity_observation = policy_observation[0, 7:10]
            local_forward = torch.tensor([[1.0, 0.0, 0.0]], device=env.unwrapped.device)
            world_forward = quat_apply(robot.data.root_quat_w.torch[0:1], local_forward)[0]
            record = {
                "seed": seed,
                "pelvis_height_m": round(
                    float(robot.data.root_pos_w.torch[0, 2].item()), 6
                ),
                "root_quaternion_xyzw": [
                    round(float(value), 7)
                    for value in robot.data.root_quat_w.torch[0].tolist()
                ],
                "projected_gravity_data": [
                    round(float(value), 7) for value in gravity_data.tolist()
                ],
                "projected_gravity_observation": [
                    round(float(value), 7) for value in gravity_observation.tolist()
                ],
                "world_forward_axis": [
                    round(float(value), 7) for value in world_forward.tolist()
                ],
                "data_observation_max_error": float(
                    (gravity_data - gravity_observation).abs().max().item()
                ),
                "root_linear_speed_m_s": float(
                    robot.data.root_lin_vel_w.torch[0].norm().item()
                ),
                "root_angular_speed_rad_s": float(
                    robot.data.root_ang_vel_w.torch[0].norm().item()
                ),
            }
            record["passes"] = (
                abs(record["projected_gravity_data"][2]) <= 0.01
                and record["world_forward_axis"][2] >= 0.99
                and record["data_observation_max_error"] <= 1.0e-6
                and record["root_linear_speed_m_s"] <= 1.0e-6
                and record["root_angular_speed_rad_s"] <= 1.0e-6
                and 0.188 <= record["pelvis_height_m"] <= 0.192
            )
            records.append(record)

        result = {
            "backend": "Isaac Lab 3.0 / PhysX",
            "seeds": list(SEEDS),
            "all_resets_fresh_and_supine": all(item["passes"] for item in records),
            "resets": records,
        }
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        if not result["all_resets_fresh_and_supine"]:
            raise RuntimeError("X2 reset freshness probe failed")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        try:
            main()
        except BaseException:
            traceback.print_exc()
            raise
    finally:
        simulation_app.close()
