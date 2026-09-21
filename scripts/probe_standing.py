#!/usr/bin/env python3
"""Verify that the imported X2 can hold a reachable upright pose in PhysX."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("reports/x2_standing_probe.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import x2_recovery_isaac  # noqa: E402,F401
from x2_recovery_isaac import mdp  # noqa: E402
from x2_recovery_isaac.env_cfg import (  # noqa: E402
    CONTACT_SENSOR_NAME,
    FOOT_CONTACT_BODIES,
    X2RecoveryPlayEnvCfg,
    all_contact_cfg,
    foot_contact_cfg,
)


def contact_forces(task) -> dict[str, float]:
    sensor = task.scene.sensors[CONTACT_SENSOR_NAME]
    forces = sensor.data.net_forces_w_history.torch[0].norm(dim=-1).amax(dim=0)
    return {name: float(force.item()) for name, force in zip(sensor.body_names, forces)}


def main() -> None:
    config = X2RecoveryPlayEnvCfg()
    config.scene.num_envs = 1
    config.sim.device = args.device
    env = gym.make("HRS-X2-Recovery-Play-v0", cfg=config)
    try:
        env.reset(seed=42)
        task = env.unwrapped
        robot = task.scene["robot"]
        feet_cfg = foot_contact_cfg()
        all_bodies_cfg = all_contact_cfg()
        feet_cfg.resolve(task.scene)
        all_bodies_cfg.resolve(task.scene)

        root_pose = robot.data.default_root_pose.torch.clone()
        root_pose[:, :3] = task.scene.env_origins
        # Geometry audit gives 0.67495 m from pelvis origin to the lowest foot
        # hull.  Start 0.05 mm above the floor, without a drop transient.
        root_pose[:, 2] += 0.675
        # Tensorized pose writes and this installed InitialStateCfg both use
        # scalar-last XYZW.
        root_pose[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=task.device)
        robot.write_root_pose_to_sim_index(root_pose=root_pose)
        robot.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((1, 6), device=task.device)
        )

        limits = robot.data.soft_joint_pos_limits.torch[0]
        target = torch.zeros((1, robot.num_joints), device=task.device)
        target = torch.maximum(torch.minimum(target, limits[:, 1]), limits[:, 0])
        robot.write_joint_state_to_sim_index(
            position=target,
            velocity=torch.zeros_like(target),
        )
        # Reset action history after the diagnostic teleport so the probe
        # tests this pose rather than state from the normal supine reset.
        env_ids = torch.tensor([0], dtype=torch.long, device=task.device)
        task.action_manager.reset(env_ids)

        stable_steps = 0
        max_stable_steps = 0
        min_height = float("inf")
        max_tilt = 0.0
        trajectory = []
        # Stop one policy step before the time-limit auto-reset, otherwise the
        # terminal force sample belongs to the next supine episode.
        max_steps = round(config.episode_length_s / task.step_dt) - 1
        for step in range(max_steps):
            with torch.no_grad():
                # The policy action is relative to the measured joint pose.
                # Recompute the inverse smooth map at each 20 Hz decision so
                # the commanded absolute pose remains the audited straight
                # target instead of accumulating a constant increment.
                current = robot.data.joint_pos.torch
                normalized_delta = ((target - current) / 0.25).clamp(-0.999, 0.999)
                action = torch.atanh(normalized_delta)
                _, _, _, _, _ = env.step(action)
                stable = bool(
                    mdp.strict_success(
                        task,
                        feet_cfg=feet_cfg,
                        all_bodies_cfg=all_bodies_cfg,
                    )[0].item()
                )
                stable_steps = stable_steps + 1 if stable else 0
                max_stable_steps = max(max_stable_steps, stable_steps)
                min_height = min(min_height, float(robot.data.root_pos_w.torch[0, 2].item()))
                max_tilt = max(
                    max_tilt,
                    float(robot.data.projected_gravity_b.torch[0, :2].norm().item()),
                )

                if step % 5 == 0 or step == max_steps - 1:
                    current_forces = contact_forces(task)
                    other = [
                        value
                        for name, value in current_forces.items()
                        if name not in FOOT_CONTACT_BODIES
                    ]
                    trajectory.append(
                        {
                            "time_s": round((step + 1) * task.step_dt, 3),
                            "pelvis_height_m": round(
                                float(robot.data.root_pos_w.torch[0, 2].item()), 5
                            ),
                            "projected_gravity": [
                                round(float(value), 5)
                                for value in robot.data.projected_gravity_b.torch[0].tolist()
                            ],
                            "linear_speed_m_s": round(
                                float(robot.data.root_lin_vel_b.torch[0].norm().item()), 5
                            ),
                            "angular_speed_rad_s": round(
                                float(robot.data.root_ang_vel_b.torch[0].norm().item()), 5
                            ),
                            "left_foot_force_n": round(
                                current_forces[FOOT_CONTACT_BODIES[0]], 3
                            ),
                            "right_foot_force_n": round(
                                current_forces[FOOT_CONTACT_BODIES[1]], 3
                            ),
                            "max_other_body_force_n": round(max(other), 3),
                            "strict_stable": stable,
                        }
                    )

        forces = contact_forces(task)
        non_feet = [value for name, value in forces.items() if name not in FOOT_CONTACT_BODIES]
        result = {
            "backend": "Isaac Lab 3.0 / PhysX",
            "duration_s": config.episode_length_s,
            "minimum_pelvis_height_m": round(min_height, 5),
            "maximum_tilt_metric": round(max_tilt, 5),
            "maximum_consecutive_strict_stable_s": round(max_stable_steps * task.step_dt, 3),
            "terminal_left_foot_force_n": round(forces[FOOT_CONTACT_BODIES[0]], 3),
            "terminal_right_foot_force_n": round(forces[FOOT_CONTACT_BODIES[1]], 3),
            "terminal_max_other_body_force_n": round(max(non_feet), 3),
            "reachable_knee_min_rad": round(
                min(
                    float(limits[robot.joint_names.index("left_knee_joint"), 0].item()),
                    float(limits[robot.joint_names.index("right_knee_joint"), 0].item()),
                ),
                6,
            ),
            "trajectory": trajectory,
            "passes": max_stable_steps >= 10,
        }
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        if not result["passes"]:
            raise RuntimeError("X2 standing hold probe failed")
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
