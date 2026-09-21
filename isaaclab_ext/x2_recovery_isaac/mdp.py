"""Task-specific observations, rewards, and success checks for X2 recovery."""

from __future__ import annotations

from collections.abc import Sequence
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
    return forces.norm(dim=-1).amax(dim=1) >= threshold


def _named_contact_masks(
    env: ManagerBasedRLEnv,
    sensor_names: Sequence[str],
    threshold: float,
) -> torch.Tensor:
    """Return one contact bit per exact-path, single-body sensor."""

    masks = []
    for name in sensor_names:
        sensor: ContactSensor = env.scene.sensors[name]
        forces = sensor.data.net_forces_w_history.torch
        masks.append(forces.norm(dim=-1).amax(dim=(1, 2)) >= threshold)
    return torch.stack(masks, dim=1)


def foot_contacts(
    env: ManagerBasedRLEnv,
    sensor_names: Sequence[str] | None = None,
    threshold: float = 15.0,
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Binary left/right foot contact observation."""

    if sensor_names is not None:
        return _named_contact_masks(env, sensor_names, threshold).to(dtype=torch.float32)
    if sensor_cfg is None:
        raise ValueError("foot_contacts requires sensor_names or sensor_cfg")
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
    # The initial policy became upright but low.  A height-dominant follow-up
    # then exploited the reward by arching upward while inverted.  Keep
    # righting dominant until the torso has rotated the correct way, then
    # reward rising.  The separate height terms are orientation-gated below.
    righting = 0.80 * ((upright + 1.0) * 0.5) + 0.20 * normalized_height
    rising = 0.40 * ((upright + 1.0) * 0.5) + 0.60 * normalized_height
    standing = 0.30 * normalized_height + 0.70 * upright.clamp(0.0, 1.0)
    return torch.where(height < 0.35, righting, torch.where(height < 0.58, rising, standing))


def humanup_base_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """HumanUP Stage-I base-height task reward: ``exp(h_base) - 1``."""

    robot: Articulation = env.scene[asset_cfg.name]
    return torch.exp(robot.data.root_pos_w.torch[:, 2]) - 1.0


def humanup_head_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="head_pitch_link"),
) -> torch.Tensor:
    """HumanUP Stage-I head-height task reward: ``exp(h_head) - 1``."""

    robot: Articulation = env.scene[asset_cfg.name]
    return torch.exp(robot.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]).mean(dim=1) - 1.0


def humanup_height_increase(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Continuous-time form of HumanUP's indicator ``1(h_t > h_{t-1})``."""

    robot: Articulation = env.scene[asset_cfg.name]
    return (robot.data.root_lin_vel_w.torch[:, 2] > 0.0).to(dtype=torch.float32)


def humanup_body_upright(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """HumanUP Stage-I body-upright term ``exp(-g_z^base)``."""

    robot: Articulation = env.scene[asset_cfg.name]
    return torch.exp(-robot.data.projected_gravity_b.torch[:, 2])


def base_linear_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared full base linear speed used by HumanUP Stage I."""

    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.root_lin_vel_w.torch), dim=1)


def base_angular_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared full base angular speed used by HumanUP Stage I."""

    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.root_ang_vel_w.torch), dim=1)


def _post_stand_gate(robot: Articulation, stage_height: float) -> torch.Tensor:
    return (robot.data.root_pos_w.torch[:, 2] > stage_height).to(dtype=torch.float32)


def host_post_base_angular_velocity(
    env: ManagerBasedRLEnv,
    stage_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """HoST post-task base angular-velocity reward, active after stage two."""

    robot: Articulation = env.scene[asset_cfg.name]
    omega_xy = robot.data.root_ang_vel_w.torch[:, :2]
    return torch.exp(-2.0 * torch.sum(torch.square(omega_xy), dim=1)) * _post_stand_gate(robot, stage_height)


def host_post_base_linear_velocity(
    env: ManagerBasedRLEnv,
    stage_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """HoST post-task base linear-velocity reward, active after stage two."""

    robot: Articulation = env.scene[asset_cfg.name]
    velocity_xy = robot.data.root_lin_vel_w.torch[:, :2]
    return torch.exp(-5.0 * torch.sum(torch.square(velocity_xy), dim=1)) * _post_stand_gate(robot, stage_height)


def host_post_base_orientation(
    env: ManagerBasedRLEnv,
    stage_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """HoST post-task projected-gravity orientation reward."""

    robot: Articulation = env.scene[asset_cfg.name]
    tilt_xy = robot.data.projected_gravity_b.torch[:, :2]
    return torch.exp(-5.0 * torch.sum(torch.square(tilt_xy), dim=1)) * _post_stand_gate(robot, stage_height)


def host_post_base_height(
    env: ManagerBasedRLEnv,
    stage_height: float,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """HoST post-task target-height reward."""

    robot: Articulation = env.scene[asset_cfg.name]
    error = robot.data.root_pos_w.torch[:, 2] - target_height
    return torch.exp(-20.0 * torch.square(error)) * _post_stand_gate(robot, stage_height)


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
    upright_gate = (-robot.data.projected_gravity_b.torch[:, 2]).clamp(0.0, 1.0).square()
    return torch.exp(-torch.square(error) / (std * std)) * upright_gate


def base_height_progress(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense height progress that is zero for sideways or inverted exploits."""

    robot: Articulation = env.scene[asset_cfg.name]
    normalized = (robot.data.root_pos_w.torch[:, 2] / target_height).clamp(0.0, 1.0)
    upright_gate = (-robot.data.projected_gravity_b.torch[:, 2]).clamp(0.0, 1.0).square()
    return torch.square(normalized) * upright_gate


def inverted_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize upside-down solutions without penalizing the supine reset."""

    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.projected_gravity_b.torch[:, 2].clamp(0.0, 1.0)


def both_feet_contact(
    env: ManagerBasedRLEnv,
    sensor_names: Sequence[str] | None = None,
    threshold: float = 15.0,
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if sensor_names is not None:
        contacts = _named_contact_masks(env, sensor_names, threshold)
    elif sensor_cfg is not None:
        contacts = _contact_mask(env, sensor_cfg, threshold)
    else:
        raise ValueError("both_feet_contact requires sensor_names or sensor_cfg")
    return contacts.all(dim=1).to(dtype=torch.float32)


def unsupported_contacts(
    env: ManagerBasedRLEnv,
    sensor_names: Sequence[str] | None = None,
    feet_sensor_names: Sequence[str] | None = None,
    threshold: float = 15.0,
    all_bodies_cfg: SceneEntityCfg | None = None,
    feet_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Count contacts on every body except the two feet."""

    if sensor_names is not None and feet_sensor_names is not None:
        non_feet = tuple(name for name in sensor_names if name not in set(feet_sensor_names))
        return _named_contact_masks(env, non_feet, threshold).sum(dim=1).to(dtype=torch.float32)
    if all_bodies_cfg is None or feet_cfg is None:
        raise ValueError("unsupported_contacts requires named sensors or the legacy sensor configs")
    sensor: ContactSensor = env.scene.sensors[all_bodies_cfg.name]
    forces = sensor.data.net_forces_w_history.torch.norm(dim=-1).amax(dim=1)
    contacts = forces >= threshold
    non_feet = torch.ones(contacts.shape[1], dtype=torch.bool, device=contacts.device)
    non_feet[feet_cfg.body_ids] = False
    return contacts[:, non_feet].sum(dim=1).to(dtype=torch.float32)


def both_feet_when_high(
    env: ManagerBasedRLEnv,
    sensor_names: Sequence[str],
    threshold: float,
    gate_start_height: float,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward two-foot support only as the pelvis approaches standing."""

    robot: Articulation = env.scene[asset_cfg.name]
    height_gate = ((robot.data.root_pos_w.torch[:, 2] - gate_start_height) / (target_height - gate_start_height)).clamp(
        0.0, 1.0
    )
    upright_gate = (-robot.data.projected_gravity_b.torch[:, 2]).clamp(0.0, 1.0).square()
    return _named_contact_masks(env, sensor_names, threshold).all(dim=1).to(torch.float32) * height_gate * upright_gate


def unsupported_contacts_when_high(
    env: ManagerBasedRLEnv,
    sensor_names: Sequence[str],
    feet_sensor_names: Sequence[str],
    threshold: float,
    gate_start_height: float,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Allow transitional pushes, then reject non-foot support near standing."""

    robot: Articulation = env.scene[asset_cfg.name]
    gate = ((robot.data.root_pos_w.torch[:, 2] - gate_start_height) / (target_height - gate_start_height)).clamp(
        0.0, 1.0
    )
    contacts = unsupported_contacts(
        env,
        sensor_names=sensor_names,
        feet_sensor_names=feet_sensor_names,
        threshold=threshold,
    )
    return contacts * gate


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
    feet_sensor_names: Sequence[str] | None = None,
    all_sensor_names: Sequence[str] | None = None,
    min_height: float = 0.62,
    max_tilt: float = 0.15,
    max_linear_speed: float = 0.20,
    max_angular_speed: float = 0.35,
    contact_threshold: float = 15.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    feet_cfg: SceneEntityCfg | None = None,
    all_bodies_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Instantaneous strict success mask used by the five-episode evaluator."""

    robot: Articulation = env.scene[asset_cfg.name]
    projected_gravity = robot.data.projected_gravity_b.torch
    upright = (projected_gravity[:, :2].norm(dim=1) <= max_tilt) & (projected_gravity[:, 2] <= -0.98)
    height_ok = robot.data.root_pos_w.torch[:, 2] >= min_height
    linear_ok = robot.data.root_lin_vel_w.torch.norm(dim=1) <= max_linear_speed
    angular_ok = robot.data.root_ang_vel_w.torch.norm(dim=1) <= max_angular_speed
    if feet_sensor_names is not None and all_sensor_names is not None:
        feet_ok = _named_contact_masks(env, feet_sensor_names, contact_threshold).all(dim=1)
        unsupported = (
            unsupported_contacts(
                env,
                sensor_names=all_sensor_names,
                feet_sensor_names=feet_sensor_names,
                threshold=contact_threshold,
            )
            > 0
        )
    elif feet_cfg is not None and all_bodies_cfg is not None:
        feet_ok = _contact_mask(env, feet_cfg, contact_threshold).all(dim=1)
        unsupported = (
            unsupported_contacts(
                env,
                all_bodies_cfg=all_bodies_cfg,
                feet_cfg=feet_cfg,
                threshold=contact_threshold,
            )
            > 0
        )
    else:
        raise ValueError("strict_success requires named sensors or the legacy sensor configs")
    return height_ok & upright & linear_ok & angular_ok & feet_ok & ~unsupported


def _invalidate_root_derived_buffers(robot: Articulation) -> None:
    """Invalidate root-frame quantities after a same-timestamp pose reset.

    Isaac Lab 3.0 beta's PhysX root-pose writer invalidates the root pose but
    not every quantity derived from root orientation.  A reset followed by
    ``sim.forward()`` therefore can expose one stale projected-gravity sample
    because the simulation timestamp has not advanced.  Keep this workaround
    task-local; it can be removed once the installed backend invalidates these
    buffers itself.
    """

    for name in (
        "_projected_gravity_b",
        "_heading_w",
        "_root_link_lin_vel_b",
        "_root_link_ang_vel_b",
        "_root_com_lin_vel_b",
        "_root_com_ang_vel_b",
    ):
        buffer = getattr(robot.data, name, None)
        if buffer is not None:
            buffer.timestamp = -1.0


def linear_anneal(start: float, end: float, step: int, duration_steps: int) -> float:
    """Return a clipped linear schedule value for a non-negative step."""

    fraction = min(max(step / max(duration_steps, 1), 0.0), 1.0)
    return start + fraction * (end - start)


def reset_root_state_uniform_fresh(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset root state and refresh same-timestamp root-frame observations."""

    from isaaclab.envs.mdp import reset_root_state_uniform

    reset_root_state_uniform(
        env,
        env_ids,
        pose_range=pose_range,
        velocity_range=velocity_range,
        asset_cfg=asset_cfg,
    )
    _invalidate_root_derived_buffers(env.scene[asset_cfg.name])


def reset_root_state_recovery_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    supine_pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    standing_probability_start: float,
    standing_probability_end: float,
    standing_probability_anneal_steps: int,
    standing_height_offset: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Mix safe standing resets into early discovery, then anneal to supine.

    HumanUP explicitly mixes standing poses into Stage I to accelerate motion
    discovery.  The final evaluator sets both probabilities to zero, so every
    reported episode still starts from the audited back-lying pose.
    """

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    reset_root_state_uniform_fresh(
        env,
        env_ids,
        pose_range=supine_pose_range,
        velocity_range=velocity_range,
        asset_cfg=asset_cfg,
    )
    step = int(getattr(env, "common_step_counter", 0))
    probability = linear_anneal(
        standing_probability_start,
        standing_probability_end,
        step,
        standing_probability_anneal_steps,
    )
    if probability <= 0.0:
        return
    standing_ids = env_ids[torch.rand(len(env_ids), device=env_ids.device) < probability]
    if len(standing_ids) == 0:
        return
    standing_pose_range = dict(supine_pose_range)
    standing_pose_range["z"] = (standing_height_offset, standing_height_offset)
    standing_pose_range["pitch"] = (0.5 * torch.pi, 0.5 * torch.pi)
    reset_root_state_uniform_fresh(
        env,
        standing_ids,
        pose_range=standing_pose_range,
        velocity_range=velocity_range,
        asset_cfg=asset_cfg,
    )


def apply_vertical_force_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    start_force_n: float,
    end_force_n: float,
    anneal_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="pelvis"),
) -> None:
    """Apply HoST's annealed world-up exploration force at the pelvis."""

    from isaaclab.utils.math import quat_apply_inverse

    robot: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=robot.device)
    step = int(getattr(env, "common_step_counter", 0))
    magnitude = linear_anneal(start_force_n, end_force_n, step, anneal_steps)
    body_ids = asset_cfg.body_ids
    quaternions = robot.data.body_quat_w.torch[env_ids][:, body_ids, :]
    world_force = torch.zeros((*quaternions.shape[:-1], 3), device=robot.device)
    world_force[..., 2] = magnitude
    local_force = quat_apply_inverse(quaternions.reshape(-1, 4), world_force.reshape(-1, 3)).reshape_as(world_force)
    torques = torch.zeros_like(local_force)
    robot.set_external_force_and_torque(
        local_force,
        torques,
        env_ids=env_ids,
        body_ids=body_ids,
    )


# Re-export standard Isaac Lab MDP terms through one task-local module.
from isaaclab.envs.mdp import *  # noqa: E402,F403
