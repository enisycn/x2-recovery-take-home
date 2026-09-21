#!/usr/bin/env python3
"""Audit an X2 checkpoint's network, observations, actions, and live state."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path("reports/x2_policy_io_audit.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import x2_recovery_isaac  # noqa: E402,F401
from x2_recovery_isaac import mdp  # noqa: E402
from x2_recovery_isaac.agents.rsl_rl_ppo_cfg import X2RecoveryPPORunnerCfg  # noqa: E402
from x2_recovery_isaac.env_cfg import (  # noqa: E402
    ALL_CONTACT_BODIES,
    CONTACT_SENSOR_NAME,
    FOOT_CONTACT_BODIES,
    X2RecoveryPlayEnvCfg,
    all_contact_cfg,
    foot_contact_cfg,
)


OBSERVATION_SLICES = {
    "base_height": (0, 1),
    "base_linear_velocity": (1, 4),
    "base_angular_velocity": (4, 7),
    "projected_gravity": (7, 10),
    "joint_position": (10, 41),
    "joint_velocity_scaled_0p1": (41, 72),
    "feet_contact": (72, 74),
    "body_contact": (74, 106),
    "previous_actions_two_steps": (106, 168),
}
AUDIT_SEED = 101


def _finite_range(tensor: torch.Tensor) -> dict:
    tensor = tensor.detach()
    return {
        "finite": bool(torch.isfinite(tensor).all().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "mean": float(tensor.float().mean().item()),
    }


def _checkpoint_audit(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = {}
    for group in ("actor_state_dict", "critic_state_dict"):
        for name, value in checkpoint.get(group, {}).items():
            state[f"{group}.{name}"] = value
    if not state:
        state = checkpoint.get("model_state_dict", checkpoint)
    tensors = {name: value for name, value in state.items() if isinstance(value, torch.Tensor)}
    normalization_count = next(
        (
            float(value.item())
            for name, value in tensors.items()
            if name.endswith("obs_normalizer._count") or name.endswith("obs_normalizer.count")
        ),
        None,
    )
    std_tensor = next((value for name, value in tensors.items() if name.endswith("distribution.std_param")), None)
    return {
        "tensor_count": len(tensors),
        "all_tensors_finite": all(torch.isfinite(value).all().item() for value in tensors.values()),
        "actor_observation_normalizer_count": normalization_count,
        "action_std": _finite_range(std_tensor) if std_tensor is not None else None,
    }


def main() -> None:
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    cfg = X2RecoveryPlayEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    agent_cfg = X2RecoveryPPORunnerCfg()
    agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))

    gym_env = gym.make("HRS-X2-Recovery-Play-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    task = env.unwrapped
    robot = task.scene["robot"]
    feet_cfg = foot_contact_cfg()
    all_cfg = all_contact_cfg()
    feet_cfg.resolve(task.scene)
    all_cfg.resolve(task.scene)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=task.device)
        env.seed(AUDIT_SEED)
        observation, _ = env.reset()
        action_term = task.action_manager.get_term("joint_position")
        joint_names = list(action_term._joint_names)

        observation_ranges = {name: [] for name in OBSERVATION_SLICES}
        action_rows: list[torch.Tensor] = []
        target_delta_rows: list[torch.Tensor] = []
        soft_limit_violations = 0
        nonfinite_observations = 0
        max_height = float(robot.data.root_pos_w.torch[0, 2].item())
        max_upright = float(-robot.data.projected_gravity_b.torch[0, 2].item())
        strict_steps = 0
        max_strict_steps = 0
        maximum_other_contact = 0.0
        maximum_foot_contact = [0.0, 0.0]
        best_upright_joint_position: dict[str, float] = {}
        best_upright_action: dict[str, float] = {}
        final_joint_position: dict[str, float] = {}

        for _ in range(round(cfg.episode_length_s / task.step_dt)):
            policy_observation = observation["policy"]
            nonfinite_observations += int(not torch.isfinite(policy_observation).all().item())
            for name, (start, stop) in OBSERVATION_SLICES.items():
                observation_ranges[name].append(policy_observation[:, start:stop].detach().clone())
            with torch.no_grad():
                action = policy(observation)
            action_rows.append(action.detach().clone())
            current = robot.data.joint_pos.torch[:, action_term._joint_ids].clone()
            processed = torch.tanh(action) * 0.25
            limits = robot.data.soft_joint_pos_limits.torch[:, action_term._joint_ids]
            target = torch.clamp(current + processed, min=limits[..., 0], max=limits[..., 1])
            target_delta_rows.append((target - current).detach().clone())
            soft_limit_violations += int(((target < limits[..., 0]) | (target > limits[..., 1])).sum().item())

            stable = bool(mdp.strict_success(task, feet_cfg=feet_cfg, all_bodies_cfg=all_cfg)[0].item())
            strict_steps = strict_steps + 1 if stable else 0
            max_strict_steps = max(max_strict_steps, strict_steps)
            max_height = max(max_height, float(robot.data.root_pos_w.torch[0, 2].item()))
            current_upright = float(-robot.data.projected_gravity_b.torch[0, 2].item())
            if current_upright > max_upright:
                max_upright = current_upright
                best_upright_joint_position = {
                    name: float(value)
                    for name, value in zip(joint_names, current[0].tolist())
                }
                best_upright_action = {
                    name: float(value)
                    for name, value in zip(joint_names, action[0].tolist())
                }
            forces = task.scene.sensors[CONTACT_SENSOR_NAME].data.net_forces_w_history.torch[0]
            forces = forces.norm(dim=-1).amax(dim=0)
            mapping = dict(zip(ALL_CONTACT_BODIES, forces.tolist()))
            maximum_foot_contact[0] = max(maximum_foot_contact[0], mapping[FOOT_CONTACT_BODIES[0]])
            maximum_foot_contact[1] = max(maximum_foot_contact[1], mapping[FOOT_CONTACT_BODIES[1]])
            maximum_other_contact = max(
                maximum_other_contact,
                max(value for name, value in mapping.items() if name not in FOOT_CONTACT_BODIES),
            )
            final_joint_position = {
                name: float(value)
                for name, value in zip(joint_names, current[0].tolist())
            }
            observation, _, _, _ = env.step(action)

        actions = torch.cat(action_rows, dim=0)
        target_deltas = torch.cat(target_delta_rows, dim=0)
        result = {
            "checkpoint": str(checkpoint),
            "seed": AUDIT_SEED,
            "checkpoint_audit": _checkpoint_audit(checkpoint),
            "network": {
                "observations": int(observation["policy"].shape[1]),
                "actions": int(actions.shape[1]),
                "hidden": [512, 256, 128],
            },
            "observation_order": {name: [start, stop] for name, (start, stop) in OBSERVATION_SLICES.items()},
            "observation_ranges": {
                name: _finite_range(torch.cat(rows, dim=0)) for name, rows in observation_ranges.items()
            },
            "nonfinite_observation_steps": nonfinite_observations,
            "raw_policy_action": {
                **_finite_range(actions),
                "fraction_abs_ge_1": float((actions.abs() >= 1.0).float().mean().item()),
                "maximum_abs_by_joint": {
                    name: float(value) for name, value in zip(joint_names, actions.abs().amax(dim=0).tolist())
                },
            },
            "bounded_target_delta_rad": _finite_range(target_deltas),
            "target_soft_limit_violations": soft_limit_violations,
            "live_episode": {
                "maximum_pelvis_height_m": max_height,
                "maximum_upright_score": max_upright,
                "maximum_strict_stable_s": max_strict_steps * task.step_dt,
                "maximum_left_foot_force_n": maximum_foot_contact[0],
                "maximum_right_foot_force_n": maximum_foot_contact[1],
                "maximum_other_body_force_n": maximum_other_contact,
                "best_upright_joint_position_rad": best_upright_joint_position,
                "best_upright_raw_action": best_upright_action,
                "terminal_joint_position_rad": final_joint_position,
            },
        }
        result["passes_io_integrity"] = bool(
            result["checkpoint_audit"]["all_tensors_finite"]
            and nonfinite_observations == 0
            and observation["policy"].shape[1] == 168
            and actions.shape[1] == 31
            and soft_limit_violations == 0
        )
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
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
