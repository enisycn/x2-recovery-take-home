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


def _ground_force_history(sensor: ContactSensor) -> torch.Tensor:
    """Return contact force against the configured floor partner only.

    ``net_forces_w_history`` includes self-collisions when articulation self
    collision is enabled.  X2 recovery checks floor support, which Isaac Lab
    exposes through the filtered ``force_matrix_w_history`` buffer.
    """

    matrix = sensor.data.force_matrix_w_history
    if matrix is None:
        raise RuntimeError("X2 contact sensor must filter against the ground prim")
    # (environments, history, bodies, filter partners, xyz)
    return matrix.torch.sum(dim=3)


def _ground_forces(sensor: ContactSensor) -> torch.Tensor:
    """Current body-to-floor force; never carry support across time/reset."""
    matrix = sensor.data.force_matrix_w
    if matrix is None:
        raise RuntimeError("X2 contact sensor must filter against the ground prim")
    return matrix.torch.sum(dim=2)


def _contact_mask(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = _ground_forces(sensor)[:, sensor_cfg.body_ids, :]
    return forces.norm(dim=-1) >= threshold


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


def body_contacts(
    env: ManagerBasedRLEnv,
    threshold: float = 15.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_all"),
) -> torch.Tensor:
    """Binary contact state for each selected rigid body."""

    return _contact_mask(env, sensor_cfg, threshold).to(dtype=torch.float32)


def humanup_proprioception(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """HumanUP proprioception adapted from 23-DoF G1 to the 31-DoF X2.

    The order follows the official release exactly: base angular velocity,
    roll/pitch, relative joint position, joint velocity, and previous action.
    For X2 this is ``3 + 2 + 3 * 31 = 98`` values.
    """

    from isaaclab.utils.math import euler_xyz_from_quat

    robot: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w.torch)
    return torch.cat(
        (
            robot.data.root_ang_vel_b.torch,
            torch.stack((roll, pitch), dim=1),
            robot.data.joint_pos.torch - robot.data.default_joint_pos.torch,
            robot.data.joint_vel.torch,
            env.action_manager.action,
        ),
        dim=1,
    )


def humanup_privileged_zeros(env: ManagerBasedRLEnv, dimension: int) -> torch.Tensor:
    """Zero Stage-I extrinsics used by HumanUP when randomization is disabled."""

    return torch.zeros((env.scene.num_envs, dimension), device=env.device)


def bilateral_joint_symmetry_l2(
    env: ManagerBasedRLEnv,
    left_cfg: SceneEntityCfg,
    right_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared mismatch between ordered left/right sagittal joints."""

    robot: Articulation = env.scene[asset_cfg.name]
    left = robot.data.joint_pos.torch[:, left_cfg.joint_ids]
    right = robot.data.joint_pos.torch[:, right_cfg.joint_ids]
    if left.shape != right.shape:
        raise ValueError(f"Bilateral joint groups differ: {left.shape} versus {right.shape}")
    return torch.sum(torch.square(left - right), dim=1)


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


def humanup_base_height_exp_clipped(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """HumanUP code release: ``exp(clamp(h_base, 0, target)) - 1``."""

    robot: Articulation = env.scene[asset_cfg.name]
    height = robot.data.root_pos_w.torch[:, 2].clamp(0.0, target_height)
    return torch.exp(height) - 1.0


def humanup_head_height_exp_clipped(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="head_pitch_link"),
) -> torch.Tensor:
    """HumanUP code release: ``exp(clamp(h_head, 0, target)) - 1``."""

    robot: Articulation = env.scene[asset_cfg.name]
    height = robot.data.body_pos_w.torch[:, asset_cfg.body_ids, 2].mean(dim=1).clamp(0.0, target_height)
    return torch.exp(height) - 1.0


def humanup_feet_contact_force_increase(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """HumanUP indicator for an increase in the combined vertical foot force."""

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    history = _ground_force_history(sensor)[:, :, sensor_cfg.body_ids, 2]
    if history.shape[1] < 2:
        raise RuntimeError("HumanUP foot-force reward requires contact history_length >= 2")
    current = torch.linalg.vector_norm(history[:, 0], dim=1)
    previous = torch.linalg.vector_norm(history[:, 1], dim=1)
    return (current > previous).to(torch.float32)


def humanup_stand_on_feet(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    feet_cfg: SceneEntityCfg,
    force_threshold: float = 2.0,
    height_threshold: float = 0.1,
) -> torch.Tensor:
    """HumanUP two-foot contact and foot-height indicator."""

    robot: Articulation = env.scene[feet_cfg.name]
    contacts = _contact_mask(env, sensor_cfg, force_threshold).all(dim=1)
    low = (robot.data.body_pos_w.torch[:, feet_cfg.body_ids, 2] < height_threshold).all(dim=1)
    return (contacts & low).to(torch.float32)


def humanup_feet_height(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """HumanUP code release: ``exp(-10 * mean(h_feet))``."""

    robot: Articulation = env.scene[feet_cfg.name]
    height = robot.data.body_pos_w.torch[:, feet_cfg.body_ids, 2].mean(dim=1)
    return torch.exp(-10.0 * height)


def humanup_foot_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    feet_cfg: SceneEntityCfg,
    force_threshold: float = 5.0,
) -> torch.Tensor:
    """HumanUP code release's square-root planar foot-speed contact penalty."""

    robot: Articulation = env.scene[feet_cfg.name]
    contact = _contact_mask(env, sensor_cfg, force_threshold).to(torch.float32)
    speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w.torch[:, feet_cfg.body_ids, :2], dim=2)
    return torch.sum(torch.sqrt(speed.clamp_min(0.0)) * contact, dim=1)


def humanup_feet_distance(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    min_distance: float,
    max_distance: float,
) -> torch.Tensor:
    """HumanUP code release's bounded inter-foot distance reward."""

    robot: Articulation = env.scene[feet_cfg.name]
    xy = robot.data.body_pos_w.torch[:, feet_cfg.body_ids, :2]
    distance = torch.linalg.vector_norm(xy[:, 0] - xy[:, 1], dim=1)
    below = (distance - min_distance).clamp(-0.5, 0.0)
    above = (distance - max_distance).clamp(0.0, 1.0)
    return 0.5 * (torch.exp(-100.0 * below.abs()) + torch.exp(-100.0 * above.abs()))


def humanup_feet_orientation(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """HumanUP sum of each foot's projected-gravity XY norm."""

    from isaaclab.utils.math import quat_apply_inverse

    robot: Articulation = env.scene[feet_cfg.name]
    quaternions = robot.data.body_quat_w.torch[:, feet_cfg.body_ids]
    gravity = torch.zeros((*quaternions.shape[:-1], 3), device=robot.device, dtype=quaternions.dtype)
    gravity[..., 2] = -1.0
    projected = quat_apply_inverse(quaternions.reshape(-1, 4), gravity.reshape(-1, 3)).reshape_as(gravity)
    return torch.linalg.vector_norm(projected[..., :2], dim=2).sum(dim=1)


def humanup_energy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """HumanUP code release's norm of absolute joint power."""

    robot: Articulation = env.scene[asset_cfg.name]
    power = torch.abs(
        robot.data.applied_torque.torch[:, asset_cfg.joint_ids]
        * robot.data.joint_vel.torch[:, asset_cfg.joint_ids]
    )
    return torch.linalg.vector_norm(power, dim=1)


def humanup_joint_torque_norm(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """HumanUP code release's L2 torque norm (not squared norm)."""

    robot: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.vector_norm(robot.data.applied_torque.torch[:, asset_cfg.joint_ids], dim=1)


def humanup_joint_torque_limits(
    env: ManagerBasedRLEnv,
    soft_ratio: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """HumanUP normalized torque-limit excess penalty."""

    robot: Articulation = env.scene[asset_cfg.name]
    torque = robot.data.applied_torque.torch[:, asset_cfg.joint_ids].abs()
    limit = robot.data.joint_effort_limits.torch[:, asset_cfg.joint_ids].clamp_min(1.0e-6)
    return ((torque / limit) - soft_ratio).clamp_min(0.0).sum(dim=1)


def humanup_joint_position_error(
    env: ManagerBasedRLEnv,
    head_cfg: SceneEntityCfg,
    standing_head_height: float = 1.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """HumanUP gated squared error from the default joint pose."""

    robot: Articulation = env.scene[asset_cfg.name]
    error = torch.sum(
        torch.square(
            robot.data.joint_pos.torch[:, asset_cfg.joint_ids]
            - robot.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    head_height = robot.data.body_pos_w.torch[:, head_cfg.body_ids, 2].mean(dim=1)
    return error * (head_height > standing_head_height).to(error.dtype)


def humanup_base_linear_velocity_norm(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.vector_norm(robot.data.root_lin_vel_b.torch, dim=1)


def humanup_action_rate_norm(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.linalg.vector_norm(env.action_manager.action - env.action_manager.prev_action, dim=1)


def humanup_action_symmetry(
    env: ManagerBasedRLEnv,
    left_cfg: SceneEntityCfg,
    right_cfg: SceneEntityCfg,
    sign_flip_indices: Sequence[int],
    head_cfg: SceneEntityCfg,
    stop_after_head_height: float = 1.1,
) -> torch.Tensor:
    """HumanUP soft bilateral action symmetry, disabled after standing."""

    actions = env.action_manager.action
    left = actions[:, left_cfg.joint_ids].clone()
    right = actions[:, right_cfg.joint_ids]
    left[:, list(sign_flip_indices)] *= -1.0
    penalty = torch.linalg.vector_norm(left - right, dim=1)
    robot: Articulation = env.scene[head_cfg.name]
    standing = robot.data.body_pos_w.torch[:, head_cfg.body_ids, 2].mean(dim=1) > stop_after_head_height
    return penalty * (~standing).to(penalty.dtype)


def humanup_waist_action_symmetry(
    env: ManagerBasedRLEnv,
    waist_cfg: SceneEntityCfg,
    head_cfg: SceneEntityCfg,
    stop_after_head_height: float = 1.1,
) -> torch.Tensor:
    actions = env.action_manager.action[:, waist_cfg.joint_ids]
    penalty = torch.linalg.vector_norm(actions, dim=1)
    robot: Articulation = env.scene[head_cfg.name]
    standing = robot.data.body_pos_w.torch[:, head_cfg.body_ids, 2].mean(dim=1) > stop_after_head_height
    return penalty * (~standing).to(penalty.dtype)


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
    return torch.exp(-10.0 * torch.sum(torch.square(omega_xy), dim=1)) * _post_stand_gate(robot, stage_height)


def host_post_base_linear_velocity(
    env: ManagerBasedRLEnv,
    stage_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """HoST post-task base linear-velocity reward, active after stage two."""

    robot: Articulation = env.scene[asset_cfg.name]
    velocity_xy = robot.data.root_lin_vel_w.torch[:, :2]
    return torch.exp(-20.0 * torch.sum(torch.square(velocity_xy), dim=1)) * _post_stand_gate(robot, stage_height)


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
    # HoST's released implementation uses exp(-20 * |h - h_target|).
    # Keeping the absolute error matters near the final stance: a squared
    # error becomes almost flat there and did not distinguish 0.620 m from
    # the X2's 0.680 m nominal standing height strongly enough.
    return torch.exp(-20.0 * torch.abs(error)) * _post_stand_gate(robot, stage_height)


def humanup_unsafe_state(
    env: ManagerBasedRLEnv,
    max_linear_speed: float = 2.5,
    min_height: float = 0.0,
    max_height: float = 1.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """HumanUP's non-success safety termination for invalid root states.

    The official G1 discovery task terminates only when root linear speed
    exceeds 2.5 m/s or pelvis height leaves [0, 1.2] m, in addition to the
    normal time limit.  Reaching or holding the target stance is deliberately
    absent: success must remain observable over consecutive control steps.
    """

    robot: Articulation = env.scene[asset_cfg.name]
    height = robot.data.root_pos_w.torch[:, 2]
    speed = torch.linalg.vector_norm(robot.data.root_lin_vel_b.torch, dim=1)
    finite = torch.isfinite(height) & torch.isfinite(speed)
    return (~finite) | (speed > max_linear_speed) | (height > max_height) | (height < min_height)


def humanup_root_speed_too_high(
    env: ManagerBasedRLEnv,
    max_linear_speed: float = 2.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """HumanUP's 2.5 m/s root linear-speed termination."""

    robot: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.vector_norm(robot.data.root_lin_vel_b.torch, dim=1) > max_linear_speed


def humanup_root_height_out_of_bounds(
    env: ManagerBasedRLEnv,
    min_height: float = 0.0,
    max_height: float = 1.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """HumanUP's broad invalid-pelvis-height termination."""

    robot: Articulation = env.scene[asset_cfg.name]
    height = robot.data.root_pos_w.torch[:, 2]
    return (height > max_height) | (height < min_height)


def root_state_nonfinite(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Fail closed if the simulated floating-base state becomes non-finite."""

    robot: Articulation = env.scene[asset_cfg.name]
    return ~torch.isfinite(robot.data.root_state_w.torch).all(dim=1)


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
    # Linear progress keeps a non-vanishing ascent gradient in the low seated
    # state.  Squaring height made the gradient weakest exactly where the
    # first policy became trapped (pelvis at roughly 0.064 m).
    return normalized * upright_gate


def final_leg_pose_exp(
    env: ManagerBasedRLEnv,
    gate_start_height: float,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """FRASA-style final-state proximity for a selected joint group.

    The term is smoothly enabled only after the pelvis has risen and the base
    is upright, so hand/knee use remains available during early recovery.
    """

    robot: Articulation = env.scene[asset_cfg.name]
    error = robot.data.joint_pos.torch[:, asset_cfg.joint_ids] - robot.data.default_joint_pos.torch[
        :, asset_cfg.joint_ids
    ]
    pose_score = torch.exp(-torch.mean(torch.square(error), dim=1) / (std * std))
    height_gate = (
        (robot.data.root_pos_w.torch[:, 2] - gate_start_height)
        / (target_height - gate_start_height)
    ).clamp(0.0, 1.0)
    upright_gate = (-robot.data.projected_gravity_b.torch[:, 2]).clamp(0.0, 1.0).square()
    return pose_score * height_gate * upright_gate


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
    forces = _ground_forces(sensor).norm(dim=-1)
    contacts = forces >= threshold
    non_feet = torch.ones(contacts.shape[1], dtype=torch.bool, device=contacts.device)
    non_feet[feet_cfg.body_ids] = False
    return contacts[:, non_feet].sum(dim=1).to(dtype=torch.float32)


def both_feet_when_high(
    env: ManagerBasedRLEnv,
    threshold: float,
    gate_start_height: float,
    target_height: float,
    sensor_names: Sequence[str] | None = None,
    sensor_cfg: SceneEntityCfg | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward two-foot support only as the pelvis approaches standing."""

    robot: Articulation = env.scene[asset_cfg.name]
    height_gate = ((robot.data.root_pos_w.torch[:, 2] - gate_start_height) / (target_height - gate_start_height)).clamp(
        0.0, 1.0
    )
    upright_gate = (-robot.data.projected_gravity_b.torch[:, 2]).clamp(0.0, 1.0).square()
    if sensor_cfg is not None:
        contacts = _contact_mask(env, sensor_cfg, threshold)
    elif sensor_names is not None:
        contacts = _named_contact_masks(env, sensor_names, threshold)
    else:
        raise ValueError("both_feet_when_high requires sensor_cfg or sensor_names")
    return contacts.all(dim=1).to(torch.float32) * height_gate * upright_gate


def unsupported_contacts_when_high(
    env: ManagerBasedRLEnv,
    threshold: float,
    gate_start_height: float,
    target_height: float,
    sensor_names: Sequence[str] | None = None,
    feet_sensor_names: Sequence[str] | None = None,
    all_bodies_cfg: SceneEntityCfg | None = None,
    feet_cfg: SceneEntityCfg | None = None,
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
        all_bodies_cfg=all_bodies_cfg,
        feet_cfg=feet_cfg,
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
    min_height: float = 0.58,
    max_tilt: float = 0.15,
    max_linear_speed: float = 0.25,
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


def strict_stance_proximity(
    env: ManagerBasedRLEnv,
    feet_sensor_names: Sequence[str] | None = None,
    all_sensor_names: Sequence[str] | None = None,
    min_height: float = 0.58,
    max_tilt: float = 0.15,
    max_linear_speed: float = 0.25,
    max_angular_speed: float = 0.35,
    contact_threshold: float = 15.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    feet_cfg: SceneEntityCfg | None = None,
    all_bodies_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Smooth FRASA-style proximity to the complete HRS stance predicate.

    The binary predicate is intentionally retained for evaluation.  This term
    turns the same physical criteria into a bounded training signal so that a
    rollout just below one threshold is distinguishable from a distant state.
    Non-foot support remains possible during rising, but decays exponentially
    once both feet carry load near the final height.
    """

    robot: Articulation = env.scene[asset_cfg.name]
    height = robot.data.root_pos_w.torch[:, 2]
    projected_gravity = robot.data.projected_gravity_b.torch
    # FRASA uses exp(-w ||state-target||^2). Here the state error is written
    # in units of the assignment tolerances. Height is one-sided: exceeding
    # the minimum is acceptable, while falling short stays dense over a 10 cm
    # margin. The explicit error sum avoids a product of tiny exponentials.
    height_error = torch.relu(min_height - height) / 0.10
    tilt_error = projected_gravity[:, :2].norm(dim=1) / max_tilt
    signed_upright_error = torch.relu(projected_gravity[:, 2] + 0.98) / 0.05
    linear_error = robot.data.root_lin_vel_w.torch.norm(dim=1) / max_linear_speed
    angular_error = robot.data.root_ang_vel_w.torch.norm(dim=1) / max_angular_speed

    if feet_sensor_names is not None and all_sensor_names is not None:
        feet_ok = _named_contact_masks(env, feet_sensor_names, contact_threshold).all(dim=1)
        other_count = unsupported_contacts(
            env,
            sensor_names=all_sensor_names,
            feet_sensor_names=feet_sensor_names,
            threshold=contact_threshold,
        )
    elif feet_cfg is not None and all_bodies_cfg is not None:
        feet_ok = _contact_mask(env, feet_cfg, contact_threshold).all(dim=1)
        other_count = unsupported_contacts(
            env,
            all_bodies_cfg=all_bodies_cfg,
            feet_cfg=feet_cfg,
            threshold=contact_threshold,
        )
    else:
        raise ValueError("strict_stance_proximity requires named sensors or sensor configs")

    squared_error = (
        2.0 * torch.square(height_error)
        + torch.square(tilt_error)
        + torch.square(signed_upright_error)
        # The exact thresholds remain binary validation criteria.  A 0.1
        # coefficient keeps the FRASA exponential informative while the
        # robot is still braking instead of underflowing several threshold
        # widths away from the final stance.
        + 0.1 * torch.square(linear_error)
        + 0.1 * torch.square(angular_error)
        + 2.0 * other_count
    )
    return feet_ok.to(torch.float32) * torch.exp(-squared_error)


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


def scaled_assist_force_n(
    mass_kg: float,
    weight_fraction: float = 0.60,
    gravity_m_s2: float = 9.81,
) -> float:
    """Scale HoST's exploration pull by robot weight."""

    return weight_fraction * mass_kg * gravity_m_s2


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
    reference_probability_start: float,
    reference_probability_end: float,
    reference_probability_anneal_policy_steps: int,
    reference_root_height_offsets: Sequence[float],
    reference_body_angles: Sequence[Sequence[float]],
    reference_min_stage: int = 0,
    reference_max_stage_start: int | None = None,
    reference_max_stage_end: int | None = None,
    reference_stage_anneal_policy_steps: int = 1,
    reference_joint_position_overrides: Sequence[dict[str, float]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Mix collision-audited squat/standing references with supine resets.

    The bridge poses give PPO reachable samples along the rising manifold,
    while the remaining environments always begin supine.  Scheduling uses
    vector-environment policy steps, not the aggregate transition count, so
    changing ``num_envs`` does not collapse the curriculum into a few updates.
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
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.default_joint_pos.torch[env_ids].clone()
    joint_vel = robot.data.default_joint_vel.torch[env_ids].clone()
    robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

    if len(reference_root_height_offsets) != len(reference_body_angles):
        raise ValueError("Each recovery reference needs one root height and one leg pose")
    if any(len(angles) != 5 for angles in reference_body_angles):
        raise ValueError(
            "Each recovery pose must contain hip, knee, ankle, shoulder, and elbow angles"
        )
    if (
        reference_joint_position_overrides is not None
        and len(reference_joint_position_overrides) != len(reference_root_height_offsets)
    ):
        raise ValueError("Each recovery reference needs one joint-position override mapping")

    policy_step = int(getattr(env, "common_step_counter", 0))
    probability = linear_anneal(
        reference_probability_start,
        reference_probability_end,
        policy_step,
        reference_probability_anneal_policy_steps,
    )
    if probability <= 0.0:
        return
    reference_ids = env_ids[torch.rand(len(env_ids), device=env_ids.device) < probability]
    if len(reference_ids) == 0:
        return

    maximum_stage = len(reference_root_height_offsets) - 1
    if reference_max_stage_start is not None or reference_max_stage_end is not None:
        if reference_max_stage_start is None or reference_max_stage_end is None:
            raise ValueError("Both reference maximum-stage endpoints must be specified")
        if not (
            0 <= reference_max_stage_start < len(reference_root_height_offsets)
            and 0 <= reference_max_stage_end < len(reference_root_height_offsets)
        ):
            raise ValueError("Reference maximum stage is outside the configured pose sequence")
        maximum_stage = int(
            linear_anneal(
                float(reference_max_stage_start),
                float(reference_max_stage_end),
                policy_step,
                reference_stage_anneal_policy_steps,
            )
        )
    if not 0 <= reference_min_stage <= maximum_stage:
        raise ValueError("Reference minimum stage is outside the unlocked pose range")
    stages = torch.randint(
        reference_min_stage,
        maximum_stage + 1,
        (len(reference_ids),),
        device=reference_ids.device,
    )
    reference_joint_names = [
        "left_hip_pitch_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_elbow_joint",
        "right_shoulder_pitch_joint",
        "right_elbow_joint",
    ]
    reference_joint_ids, resolved = robot.find_joints(reference_joint_names, preserve_order=True)
    if tuple(resolved) != tuple(reference_joint_names):
        raise RuntimeError(f"Unexpected X2 recovery-joint map: {resolved}")

    for stage, (height_offset, angles) in enumerate(
        zip(reference_root_height_offsets, reference_body_angles)
    ):
        stage_ids = reference_ids[stages == stage]
        if len(stage_ids) == 0:
            continue
        pose_range = dict(supine_pose_range)
        pose_range["z"] = (height_offset, height_offset)
        pose_range["pitch"] = (0.5 * torch.pi, 0.5 * torch.pi)
        reset_root_state_uniform_fresh(
            env,
            stage_ids,
            pose_range=pose_range,
            velocity_range=velocity_range,
            asset_cfg=asset_cfg,
        )
        stage_joint_pos = robot.data.default_joint_pos.torch[stage_ids].clone()
        hip, knee, ankle, shoulder, elbow = angles
        stage_joint_pos[:, reference_joint_ids] = torch.tensor(
            [hip, knee, ankle, hip, knee, ankle, shoulder, elbow, shoulder, elbow],
            device=robot.device,
            dtype=stage_joint_pos.dtype,
        )
        if reference_joint_position_overrides is not None:
            overrides = reference_joint_position_overrides[stage]
            if overrides:
                override_names = list(overrides)
                override_ids, override_resolved = robot.find_joints(override_names, preserve_order=True)
                if tuple(override_resolved) != tuple(override_names):
                    raise RuntimeError(f"Unexpected X2 handoff-joint map: {override_resolved}")
                stage_joint_pos[:, override_ids] = torch.tensor(
                    [overrides[name] for name in override_names],
                    device=robot.device,
                    dtype=stage_joint_pos.dtype,
                )
        stage_joint_vel = robot.data.default_joint_vel.torch[stage_ids].clone()
        robot.write_joint_position_to_sim_index(position=stage_joint_pos, env_ids=stage_ids)
        robot.write_joint_velocity_to_sim_index(velocity=stage_joint_vel, env_ids=stage_ids)


def apply_vertical_force_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    start_force_n: float,
    end_force_n: float,
    anneal_policy_steps: int,
    orientation_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="pelvis"),
) -> None:
    """Apply HoST's annealed world-up exploration force at the pelvis."""

    from isaaclab.utils.math import quat_apply_inverse

    robot: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=robot.device)
    policy_step = int(getattr(env, "common_step_counter", 0))
    magnitude = linear_anneal(start_force_n, end_force_n, policy_step, anneal_policy_steps)
    body_ids = asset_cfg.body_ids
    quaternions = robot.data.body_quat_w.torch[env_ids][:, body_ids, :]
    world_force = torch.zeros((*quaternions.shape[:-1], 3), device=robot.device)
    # HoST enables the pull only after the trunk is near vertical (the
    # ground-sitting/rising phase). Applying it while supine would reduce the
    # useful ground reaction during righting.
    upright_score = -robot.data.projected_gravity_b.torch[env_ids, 2]
    active = (upright_score >= orientation_threshold).to(dtype=world_force.dtype)
    world_force[..., 2] = magnitude * active[:, None]
    local_force = quat_apply_inverse(quaternions.reshape(-1, 4), world_force.reshape(-1, 3)).reshape_as(world_force)
    torques = torch.zeros_like(local_force)
    robot.set_external_force_and_torque(
        local_force,
        torques,
        env_ids=env_ids,
        body_ids=body_ids,
    )


def apply_humanup_height_scaled_force(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    force_n: float,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="head_pitch_link"),
) -> None:
    """HumanUP code-release upward pull, reduced linearly with base height."""

    from isaaclab.utils.math import quat_apply_inverse

    robot: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=robot.device)
    body_ids = asset_cfg.body_ids
    scale = (1.0 - robot.data.root_pos_w.torch[env_ids, 2] / target_height).clamp(0.0, 1.0)
    quaternions = robot.data.body_quat_w.torch[env_ids][:, body_ids, :]
    world_force = torch.zeros((*quaternions.shape[:-1], 3), device=robot.device)
    world_force[..., 2] = force_n * scale[:, None]
    local_force = quat_apply_inverse(quaternions.reshape(-1, 4), world_force.reshape(-1, 3)).reshape_as(world_force)
    robot.set_external_force_and_torque(
        local_force,
        torch.zeros_like(local_force),
        env_ids=env_ids,
        body_ids=body_ids,
    )


# Re-export standard Isaac Lab MDP terms through one task-local module.
from isaaclab.envs.mdp import *  # noqa: E402,F403


def relaxed_arms_when_stable(
    env: ManagerBasedRLEnv,
    shoulder_cfg: SceneEntityCfg,
    elbow_cfg: SceneEntityCfg,
    feet_cfg: SceneEntityCfg,
    all_bodies_cfg: SceneEntityCfg,
    variance: float = 2.0,
) -> torch.Tensor:
    """X2 post-task pose reward; enabled only in supported upright stance.

    HoST motivates separate post-task behavior objectives. The arm targets,
    Gaussian width and weight are our X2 adaptation, not paper constants.
    Zero shoulder pitch hangs the upper arms; elbows retain a 0.15-rad bend.
    """
    robot = env.scene[shoulder_cfg.name]
    shoulders = robot.data.joint_pos.torch[:, shoulder_cfg.joint_ids]
    elbows = robot.data.joint_pos.torch[:, elbow_cfg.joint_ids]
    mse = 0.5 * (shoulders.square().mean(dim=1)
                 + (elbows + 0.15).square().mean(dim=1))
    height_gate = ((robot.data.root_pos_w.torch[:, 2] - .50) / .15).clamp(0., 1.)
    upright_gate = ((-robot.data.projected_gravity_b.torch[:, 2] - .95) / .04).clamp(0., 1.)
    feet = _contact_mask(env, feet_cfg, 15.).all(dim=1)
    no_other = unsupported_contacts(env, all_bodies_cfg=all_bodies_cfg,
                                    feet_cfg=feet_cfg, threshold=15.) == 0
    # Do not switch this objective off for the small velocity needed to lower
    # arms. Existing balance/strict-stance rewards still favor stopping.
    gate = height_gate * upright_gate * feet * no_other
    return gate * torch.exp(-mse / variance)
