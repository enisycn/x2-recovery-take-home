#!/usr/bin/env python3
"""Collect successful legacy-policy rollouts in HumanUP observation space."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument(
    "--handoff_checkpoint",
    type=Path,
    default=None,
    help="Optional second legacy policy that takes over after a measured height/upright threshold.",
)
parser.add_argument("--handoff_height", type=float, default=0.56)
parser.add_argument("--handoff_upright", type=float, default=0.90)
parser.add_argument("--output", type=Path, default=Path("reports/imitation/reference1_expert.npz"))
parser.add_argument("--stage", type=int, choices=range(9), default=1)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=109)
parser.add_argument(
    "--min_max_height",
    type=float,
    default=None,
    help="If strict success is absent, retain trajectories whose peak pelvis height reaches this threshold.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import x2_recovery_isaac  # noqa: E402,F401
from x2_recovery_isaac import mdp  # noqa: E402
from x2_recovery_isaac.agents.rsl_rl_ppo_cfg import X2RecoveryPPORunnerCfg  # noqa: E402
from x2_recovery_isaac.env_cfg import (  # noqa: E402
    X2ImitationCollectionEnvCfg,
    all_contact_cfg,
    foot_contact_cfg,
)


def main() -> None:
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    cfg = X2ImitationCollectionEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    reset = cfg.events.reset_back_pose.params
    reset["reference_probability_start"] = 1.0
    reset["reference_probability_end"] = 1.0
    reset["reference_min_stage"] = args.stage
    reset["reference_max_stage_start"] = args.stage
    reset["reference_max_stage_end"] = args.stage
    reset["reference_stage_anneal_policy_steps"] = 1

    agent_cfg = X2RecoveryPPORunnerCfg()
    agent_cfg.device = args.device
    agent_cfg.obs_groups = {"actor": ["teacher_policy"], "critic": ["teacher_policy"]}
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))

    gym_env = gym.make("HRS-X2-Recovery-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    task = env.unwrapped
    feet_cfg = foot_contact_cfg()
    all_cfg = all_contact_cfg()
    feet_cfg.resolve(task.scene)
    all_cfg.resolve(task.scene)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=task.device)
        handoff_checkpoint = None
        handoff_policy = None
        handoff_runner = None
        if args.handoff_checkpoint is not None:
            handoff_checkpoint = args.handoff_checkpoint.expanduser().resolve(strict=True)
            handoff_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            handoff_runner.load(str(handoff_checkpoint))
            handoff_policy = handoff_runner.get_inference_policy(device=task.device)
        env.seed(args.seed)
        observation, _ = env.reset()
        observations: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        strict_count = torch.zeros(args.num_envs, dtype=torch.int64, device=task.device)
        first_success = torch.full((args.num_envs,), -1, dtype=torch.int64, device=task.device)
        maximum_height = torch.full((args.num_envs,), -float("inf"), device=task.device)
        peak_step = torch.zeros(args.num_envs, dtype=torch.int64, device=task.device)
        handoff_active = torch.zeros(args.num_envs, dtype=torch.bool, device=task.device)
        first_handoff = torch.full((args.num_envs,), -1, dtype=torch.int64, device=task.device)
        total_steps = max(1, round(cfg.episode_length_s / task.step_dt) - 1)
        active = torch.ones(args.num_envs, dtype=torch.bool, device=task.device)
        episode_lengths = torch.full((args.num_envs,), total_steps, dtype=torch.long, device=task.device)
        for step in range(total_steps):
            with torch.no_grad():
                action = policy(observation)
                if handoff_policy is not None:
                    robot = task.scene["robot"]
                    height = robot.data.root_pos_w.torch[:, 2]
                    upright = -robot.data.projected_gravity_b.torch[:, 2]
                    newly_handed_off = (
                        active & (~handoff_active)
                        & (height >= args.handoff_height)
                        & (upright >= args.handoff_upright)
                    )
                    first_handoff[newly_handed_off] = step
                    handoff_active |= newly_handed_off
                    second_action = handoff_policy(observation)
                    action = torch.where(handoff_active[:, None], second_action, action)
            observations.append(observation["policy"].detach().cpu().to(torch.float16))
            actions.append(action.detach().cpu().to(torch.float16))
            stable = active & mdp.strict_success(task, feet_cfg=feet_cfg, all_bodies_cfg=all_cfg)
            strict_count = torch.where(stable, strict_count + 1, torch.zeros_like(strict_count))
            newly_successful = (strict_count >= 10) & (first_success < 0)
            first_success[newly_successful] = step
            height = task.scene["robot"].data.root_pos_w.torch[:, 2]
            new_peak = active & (height > maximum_height)
            maximum_height = torch.where(new_peak, height, maximum_height)
            peak_step[new_peak] = step
            observation, _, done, _ = env.step(action)
            first_done = active & done.bool()
            episode_lengths[first_done] = step + 1
            active &= ~done.bool()
            if not active.any():
                break

        stacked_obs = torch.stack(observations)
        stacked_actions = torch.stack(actions)
        success_ids = torch.nonzero(first_success >= 0, as_tuple=False).flatten().cpu()
        selected_ids = success_ids
        selection_mode = "strict_success"
        if selected_ids.numel() == 0 and args.min_max_height is not None:
            selected_ids = torch.nonzero(maximum_height >= args.min_max_height, as_tuple=False).flatten().cpu()
            selection_mode = "peak_height"
        if selected_ids.numel() == 0:
            raise RuntimeError("Expert produced no rollout satisfying the selection criterion")

        kept_obs: list[np.ndarray] = []
        kept_actions: list[np.ndarray] = []
        offsets = [0]
        first_success_cpu = first_success.cpu()
        peak_step_cpu = peak_step.cpu()
        for env_id in selected_ids.tolist():
            anchor = (
                int(first_success_cpu[env_id].item())
                if first_success_cpu[env_id] >= 0
                else int(peak_step_cpu[env_id].item())
            )
            end = min(anchor + 21, int(episode_lengths[env_id].item()))
            kept_obs.append(stacked_obs[:end, env_id].numpy())
            kept_actions.append(stacked_actions[:end, env_id].numpy())
            offsets.append(offsets[-1] + end)

        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            observations=np.concatenate(kept_obs),
            actions=np.concatenate(kept_actions),
            episode_offsets=np.asarray(offsets, dtype=np.int64),
            selected_env_ids=selected_ids.numpy(),
        )
        report = {
            "checkpoint": str(checkpoint),
            "handoff_checkpoint": str(handoff_checkpoint) if handoff_checkpoint is not None else None,
            "handoff_height_m": args.handoff_height if handoff_checkpoint is not None else None,
            "handoff_upright_min": args.handoff_upright if handoff_checkpoint is not None else None,
            "handoff_envs": int((first_handoff >= 0).sum().item()),
            "stage": args.stage,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "successful_envs": int(success_ids.numel()),
            "success_rate": float(success_ids.numel() / args.num_envs),
            "selection_mode": selection_mode,
            "episode_boundary_policy": "first episode only; no post-reset samples",
            "selected_envs": int(selected_ids.numel()),
            "min_max_height": args.min_max_height,
            "selected_peak_height_min_m": float(maximum_height[selected_ids.to(task.device)].min().item()),
            "selected_peak_height_max_m": float(maximum_height[selected_ids.to(task.device)].max().item()),
            "samples": int(offsets[-1]),
            "observation_dim": int(stacked_obs.shape[-1]),
            "action_dim": int(stacked_actions.shape[-1]),
            "dataset": str(output),
        }
        report_path = output.with_suffix(".json")
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
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
