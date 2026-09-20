"""Task-specific observations, rewards, and success checks for X2 recovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def _contact_mask(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :]
    return forces.norm(dim=-1).amax(dim=1) > threshold


def foot_contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 15.0,
) -> torch.Tensor:
    """Binary left/right foot contact observation."""

    return _contact_mask(env, sensor_cfg, threshold).to(dtype=torch.float32)


def staged_recovery_progress(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense three-stage righting, rising, and standing reward.

    The stage boundaries use pelvis height, following HoST's central idea, while
    keeping one critic and one compact reward for this take-home.
    """

    robot: Articulation = env.scene[asset_cfg.name]
    height = robot.data.root_pos_w.torch[:, 2]
    upright = (-robot.data.projected_gravity_b.torch[:, 2]).clamp(-1.0, 1.0)
    normalized_height = (height / target_height).clamp(0.0, 1.0)
    righting = 0.65 * ((upright + 1.0) * 0.5) + 0.35 * normalized_height
    rising = 0.45 * ((upright + 1.0) * 0.5) + 0.55 * normalized_height
    standing = 0.30 * normalized_height + 0.70 * upright.clamp(0.0, 1.0)
    return torch.where(height < 0.35, righting, torch.where(height < 0.58, rising, standing))


def upright_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    error = 1.0 + robot.data.projected_gravity_b.torch[:, 2]
    return torch.exp(-torch.square(error) / (std * std))


def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    error = robot.data.root_pos_w.torch[:, 2] - target_height
    return torch.exp(-torch.square(error) / (std * std))


def both_feet_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    return _contact_mask(env, sensor_cfg, threshold).all(dim=1).to(dtype=torch.float32)


def unsupported_contacts(
    env: ManagerBasedRLEnv,
    all_bodies_cfg: SceneEntityCfg,
    feet_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Count contacts on every body except the two feet."""

    sensor: ContactSensor = env.scene.sensors[all_bodies_cfg.name]
    forces = sensor.data.net_forces_w_history.torch.norm(dim=-1).amax(dim=1)
    contacts = forces > threshold
    contacts[:, feet_cfg.body_ids] = False
    return contacts.sum(dim=1).to(dtype=torch.float32)


def standing_still(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward low root velocity only near the standing target."""

    robot: Articulation = env.scene[asset_cfg.name]
    height_gate = (robot.data.root_pos_w.torch[:, 2] > target_height - 0.08).to(torch.float32)
    upright_gate = (-robot.data.projected_gravity_b.torch[:, 2] > 0.96).to(torch.float32)
    speed = robot.data.root_lin_vel_w.torch.norm(dim=1) + robot.data.root_ang_vel_w.torch.norm(dim=1)
    return height_gate * upright_gate * torch.exp(-torch.square(speed) / 0.25)


def strict_success(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    all_bodies_cfg: SceneEntityCfg,
    min_height: float = 0.62,
    max_tilt: float = 0.15,
    max_linear_speed: float = 0.20,
    max_angular_speed: float = 0.35,
    contact_threshold: float = 15.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Instantaneous strict success mask used by the five-episode evaluator."""

    robot: Articulation = env.scene[asset_cfg.name]
    projected_gravity = robot.data.projected_gravity_b.torch
    upright = (projected_gravity[:, :2].norm(dim=1) < max_tilt) & (projected_gravity[:, 2] < -0.98)
    height_ok = robot.data.root_pos_w.torch[:, 2] >= min_height
    linear_ok = robot.data.root_lin_vel_w.torch.norm(dim=1) <= max_linear_speed
    angular_ok = robot.data.root_ang_vel_w.torch.norm(dim=1) <= max_angular_speed
    feet_ok = _contact_mask(env, feet_cfg, contact_threshold).all(dim=1)
    unsupported = unsupported_contacts(env, all_bodies_cfg, feet_cfg, contact_threshold) > 0
    return height_ok & upright & linear_ok & angular_ok & feet_ok & ~unsupported


# Re-export standard Isaac Lab MDP terms through one task-local module.
from isaaclab.envs.mdp import *  # noqa: E402,F403
