#!/usr/bin/env python3
"""Verify the supine-to-squat recovery reset curriculum in live PhysX."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("reports/x2_curriculum_probe.json"))
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import x2_recovery_isaac  # noqa: E402,F401
from isaaclab.utils.math import quat_apply  # noqa: E402
from x2_recovery_isaac import mdp  # noqa: E402
from x2_recovery_isaac.env_cfg import (  # noqa: E402
    X2RecoveryEnvCfg,
    all_contact_cfg,
    foot_contact_cfg,
)


def main() -> None:
    config = X2RecoveryEnvCfg()
    config.scene.num_envs = args.num_envs
    config.sim.device = args.device
    # The probe isolates reset geometry from unrelated startup randomization.
    config.events.material = None
    config.events.mass = None
    config.events.pelvis_com = None
    config.events.actuator_gains = None
    env = gym.make("HRS-X2-Recovery-v0", cfg=config)
    try:
        env.reset(seed=42)
        task = env.unwrapped
        robot = task.scene["robot"]
        feet_cfg = foot_contact_cfg()
        all_cfg = all_contact_cfg()
        feet_cfg.resolve(task.scene)
        all_cfg.resolve(task.scene)
        initial_height = robot.data.root_pos_w.torch[:, 2].clone()
        initial_gravity_z = robot.data.projected_gravity_b.torch[:, 2].clone()
        forward = torch.zeros((args.num_envs, 3), device=task.device)
        forward[:, 0] = 1.0
        initial_forward_z = quat_apply(robot.data.root_quat_w.torch, forward)[:, 2]

        reference = initial_gravity_z < -0.95
        supine = initial_gravity_z.abs() < 0.01
        expected_heights = torch.tensor(
            [0.68000, 0.60214, 0.50129, 0.39152, 0.31452, 0.25245, 0.21527, 0.14954, 0.09194],
            device=task.device,
        )
        stage_error, nearest_stage = (initial_height[:, None] - expected_heights[None, :]).abs().min(dim=1)
        reference_geometry_ok = reference & (stage_error < 0.015)
        zero_actions = torch.zeros((args.num_envs, task.action_manager.total_action_dim), device=task.device)
        for _ in range(5):
            env.step(zero_actions)

        final_height = robot.data.root_pos_w.torch[:, 2]
        final_gravity_z = robot.data.projected_gravity_b.torch[:, 2]
        both_feet = mdp._contact_mask(task, feet_cfg, 15.0).all(dim=1)
        retained_reference = reference & (final_height > 0.06) & (final_gravity_z < -0.90)
        loaded_reference = retained_reference & both_feet

        reference_count = int(reference.sum().item())
        supine_count = int(supine.sum().item())
        stage_counts = {
            str(index): int((reference & (nearest_stage == index)).sum().item())
            for index in range(len(expected_heights))
        }
        result = {
            "backend": "Isaac Lab 3.0 / PhysX",
            "seed": 42,
            "num_envs": args.num_envs,
            "policy_steps_observed": 5,
            "reference_initial": reference_count,
            "supine_initial": supine_count,
            "reference_stage_counts": stage_counts,
            "reference_geometry_matches": int(reference_geometry_ok.sum().item()),
            "reference_initial_gravity_z_mean": float(initial_gravity_z[reference].mean().item()),
            "reference_initial_gravity_z_max": float(initial_gravity_z[reference].max().item()),
            "supine_initial_forward_z_min": float(initial_forward_z[supine].min().item()),
            "supine_initial_abs_gravity_z_max": float(initial_gravity_z[supine].abs().max().item()),
            "reference_retained_after_five_steps": int(retained_reference.sum().item()),
            "reference_both_feet_after_five_steps": int(loaded_reference.sum().item()),
        }
        result["passes"] = bool(
            0 < reference_count < args.num_envs
            and reference_count + supine_count == args.num_envs
            and result["reference_geometry_matches"] == reference_count
            and sum(count > 0 for count in stage_counts.values()) >= 7
            and result["reference_initial_gravity_z_max"] <= -0.99
            and result["supine_initial_forward_z_min"] >= 0.99
            and result["supine_initial_abs_gravity_z_max"] <= 0.01
            and result["reference_retained_after_five_steps"] >= 0.80 * reference_count
            and result["reference_both_feet_after_five_steps"] >= 0.80 * reference_count
        )

        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        if not result["passes"]:
            raise RuntimeError("X2 training curriculum probe failed")
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
