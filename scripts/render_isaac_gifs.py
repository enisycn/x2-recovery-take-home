#!/usr/bin/env python3
"""Render honest GIFs for the selected HumanUP policy and reachability probe."""

from __future__ import annotations

import argparse
import importlib.metadata
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--environment", choices=("humanup_rise", "simple_v2", "symmetric_v3"), default="humanup_rise")
parser.add_argument("--output_dir", type=Path, default=Path("reports/gifs"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import x2_recovery_isaac  # noqa: E402,F401
from x2_recovery_isaac.agents.rsl_rl_ppo_cfg import X2HumanUpCurriculumPPORunnerCfg  # noqa: E402
from x2_recovery_isaac.env_cfg import X2HumanUpRiseEnvCfg  # noqa: E402


from x2_recovery_isaac.simple_cfg import X2SimpleRecoveryEnvCfg, X2SimplePPORunnerCfg, X2SymmetricRecoveryEnvCfg, X2SymmetricPPORunnerCfg
from x2_recovery_isaac import mdp
from x2_recovery_isaac.env_cfg import foot_contact_cfg, all_contact_cfg

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _frame(task, title: str, detail: str) -> Image.Image:
    array = task.render(recompute=True)
    if array is None:
        raise RuntimeError("Isaac renderer returned no RGB frame; launch with --enable_cameras")
    array = np.asarray(array)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        scale = 255.0 if array.max() <= 1.0 else 1.0
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    image = Image.fromarray(array).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = ImageFont.truetype(FONT_PATH, 22)
    detail_font = ImageFont.truetype(FONT_PATH, 17)
    draw.rectangle((0, 0, image.width, 64), fill=(0, 0, 0, 185))
    draw.text((14, 8), title, font=title_font, fill=(255, 255, 255, 255))
    draw.text((14, 37), detail, font=detail_font, fill=(235, 235, 235, 255))
    return image


def _save_gif(frames: list[Image.Image], path: Path, duration_ms=50) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )


def main() -> None:
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()

    cfg = {"simple_v2": X2SimpleRecoveryEnvCfg, "symmetric_v3": X2SymmetricRecoveryEnvCfg, "humanup_rise": X2HumanUpRiseEnvCfg}[args.environment]()
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 3.0
    cfg.sim.device = args.device
    cfg.observations.policy.enable_corruption = False
    cfg.events.material = None
    cfg.events.mass = None
    cfg.events.pelvis_com = None
    cfg.events.actuator_gains = None
    cfg.events.lift_assist = None
    cfg.events.reset_back_pose.params["reference_probability_start"] = 0.0
    cfg.events.reset_back_pose.params["reference_probability_end"] = 0.0
    cfg.video_recorder.window_width = 640
    cfg.video_recorder.window_height = 360
    cfg.viewer.eye = (2.4, 2.4, 1.45)
    cfg.viewer.lookat = (0.0, 0.0, 0.45)

    agent_cfg = {"simple_v2": X2SimplePPORunnerCfg, "symmetric_v3": X2SymmetricPPORunnerCfg, "humanup_rise": X2HumanUpCurriculumPPORunnerCfg}[args.environment]()
    agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))

    gym_env = gym.make("HRS-X2-Recovery-Play-v0", cfg=cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    task = env.unwrapped
    robot = task.scene["robot"]
    feet, allb = foot_contact_cfg(), all_contact_cfg()
    feet.resolve(task.scene); allb.resolve(task.scene)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=task.device)

        env.seed(101)
        observation, _ = env.reset()
        # Warm the RTX capture so the first GIF frame is not the renderer's
        # uninitialized black buffer.
        task.render(recompute=True)
        task.render(recompute=True)
        policy_frames: list[Image.Image] = []
        max_steps = round(cfg.episode_length_s / task.step_dt) - 1
        consecutive = 0
        for step in range(max_steps):
            strict = bool(mdp.strict_success(task, feet_cfg=feet, all_bodies_cfg=allb)[0])
            consecutive = consecutive + 1 if strict else 0
            height = float(robot.data.root_pos_w.torch[0, 2].item())
            upright = float(-robot.data.projected_gravity_b.torch[0, 2].item())
            policy_frames.append(
                _frame(
                    task,
                    f"{args.environment} policy - gercek supine deneme",
                    f"t={step * task.step_dt:4.2f}s  pelvis={height:.3f}m  upright={upright:.3f}  kararli={consecutive * task.step_dt:.2f}s",
                )
            )
            with torch.no_grad():
                action = policy(observation)
                observation, _, done, _ = env.step(action)
                if bool(done[0]):
                    break
        policy_path = output_dir / "x2_final_policy_attempt.gif"
        _save_gif(policy_frames, policy_path, round(task.step_dt * 1000))
        if args.environment != "humanup_rise":
            print(policy_path)
            return

        env.seed(42)
        env.reset()
        root_pose = robot.data.default_root_pose.torch.clone()
        root_pose[:, :3] = task.scene.env_origins
        root_pose[:, 2] += 0.675
        root_pose[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=task.device)
        robot.write_root_pose_to_sim_index(root_pose=root_pose)
        robot.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((1, 6), device=task.device)
        )
        limits = robot.data.soft_joint_pos_limits.torch[0]
        target = torch.zeros((1, robot.num_joints), device=task.device)
        target = torch.maximum(torch.minimum(target, limits[:, 1]), limits[:, 0])
        robot.write_joint_state_to_sim_index(position=target, velocity=torch.zeros_like(target))
        task.action_manager.reset(torch.tensor([0], dtype=torch.long, device=task.device))

        standing_frames: list[Image.Image] = []
        for step in range(50):
            with torch.no_grad():
                current = robot.data.joint_pos.torch
                action = torch.atanh(((target - current) / 0.25).clamp(-0.999, 0.999))
                env.step(action)
            height = float(robot.data.root_pos_w.torch[0, 2].item())
            upright = float(-robot.data.projected_gravity_b.torch[0, 2].item())
            standing_frames.append(
                _frame(
                    task,
                    "Reachability probe - ayakta BASLIYOR",
                    f"t={(step + 1) * task.step_dt:4.2f}s  pelvis={height:.3f}m  upright={upright:.3f}  recovery sonucu degil",
                )
            )
        standing_path = output_dir / "x2_standing_reachability_only.gif"
        _save_gif(standing_frames, standing_path)

        print(policy_path)
        print(standing_path)
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
