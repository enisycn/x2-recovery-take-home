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
parser.add_argument(
    "--handoff_checkpoint",
    type=Path,
    default=None,
    help="Optional same-architecture stabilizer blended in over a height interval.",
)
parser.add_argument("--handoff_start_height", type=float, default=0.50)
parser.add_argument("--handoff_end_height", type=float, default=0.60)
parser.add_argument("--handoff_min_upright", type=float, default=0.85)
parser.add_argument(
    "--handoff_latch",
    action="store_true",
    help="Keep the stabilizer fully active after the end-height is reached once.",
)
parser.add_argument(
    "--hold_after_height",
    type=float,
    default=None,
    help="Latch the current joint pose after crossing this height while upright.",
)
parser.add_argument("--hold_min_upright", type=float, default=0.95)
parser.add_argument(
    "--straighten_after_height",
    type=float,
    default=None,
    help="Latch a rate-limited move to the audited standing joint pose after this height.",
)
parser.add_argument("--brake_start_height", type=float, default=None)
parser.add_argument("--brake_end_height", type=float, default=0.62)
parser.add_argument(
    "--brake_sagittal_only",
    action="store_true",
    help="Brake hip/knee/ankle pitch actions while retaining lateral/yaw balance actions.",
)
parser.add_argument("--align_after_height", type=float, default=None)
parser.add_argument("--align_step_rad", type=float, default=0.01)
parser.add_argument("--align_yaw_only", action="store_true")
parser.add_argument("--disable_action_term_brake", action="store_true")
parser.add_argument("--ankle_balance_kv", type=float, default=0.0)
parser.add_argument("--ankle_balance_kw", type=float, default=0.0)
parser.add_argument("--ankle_roll_kv", type=float, default=0.0)
parser.add_argument("--ankle_roll_kw", type=float, default=0.0)
parser.add_argument("--output", type=Path, default=Path("reports/x2_policy_io_audit.json"))
parser.add_argument("--controller", choices=("policy", "straighten"), default="policy")
parser.add_argument(
    "--straighten_step_rad",
    type=float,
    default=0.02,
    help="Maximum per-policy-step joint change for the reference-pose probe.",
)
parser.add_argument(
    "--environment",
    choices=("recovery", "humanup_discovery", "humanup_rise", "simple_v2"),
    default="recovery",
    help="Evaluate without assistance, or with the paper's Stage-I head force enabled.",
)
parser.add_argument(
    "--initial_pose",
    choices=("supine", "standing", *(f"reference_{index}" for index in range(9))),
    default="supine",
    help="Deterministic reset pose; standing is an alias for reference_0.",
)
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
from x2_recovery_isaac.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    X2HumanUpCurriculumPPORunnerCfg,
    X2HumanUpPPORunnerCfg,
    X2RecoveryPPORunnerCfg,
)
from x2_recovery_isaac.env_cfg import (  # noqa: E402
    ALL_CONTACT_BODIES,
    CONTACT_SENSOR_NAME,
    FOOT_CONTACT_BODIES,
    X2HumanUpDiscoveryEnvCfg,
    X2HumanUpRiseEnvCfg,
    X2RecoveryPlayEnvCfg,
    all_contact_cfg,
    foot_contact_cfg,
)
from x2_recovery_isaac.relative_action import BoundedRelativeJointPositionAction  # noqa: E402


from x2_recovery_isaac.simple_cfg import X2SimpleRecoveryEnvCfg, X2SimplePPORunnerCfg

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
HUMANUP_OBSERVATION_SLICES = {
    "current_proprioception": (0, 98),
    "zero_privileged_extrinsics": (98, 168),
    "proprioception_history_10_steps": (168, 1148),
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


def _resolved_ids(ids: list[int] | slice, count: int) -> list[int]:
    """Expand a SceneEntityCfg's optimized slice for JSON diagnostics."""

    return list(range(count))[ids] if isinstance(ids, slice) else list(ids)


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
    handoff_checkpoint = (
        args.handoff_checkpoint.expanduser().resolve(strict=True)
        if args.handoff_checkpoint is not None
        else None
    )
    if args.handoff_end_height <= args.handoff_start_height:
        raise ValueError("--handoff_end_height must exceed --handoff_start_height")
    cfg_type = {
        "simple_v2": X2SimpleRecoveryEnvCfg,
        "recovery": X2RecoveryPlayEnvCfg,
        "humanup_discovery": X2HumanUpDiscoveryEnvCfg,
        "humanup_rise": X2HumanUpRiseEnvCfg,
    }[args.environment]
    cfg = cfg_type()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.observations.policy.enable_corruption = False
    if args.disable_action_term_brake:
        cfg.actions.joint_position.brake_start_height = None
    cfg.events.material = None
    cfg.events.mass = None
    cfg.events.pelvis_com = None
    cfg.events.actuator_gains = None
    # Supine audits must never be silently replaced by the standing starts
    # that HumanUP uses for 10--20% of the training resets.
    cfg.events.reset_back_pose.params["reference_probability_start"] = 0.0
    cfg.events.reset_back_pose.params["reference_probability_end"] = 0.0
    if args.initial_pose != "supine":
        params = cfg.events.reset_back_pose.params
        reference_index = 0 if args.initial_pose == "standing" else int(args.initial_pose.rsplit("_", 1)[1])
        selected_height = params["reference_root_height_offsets"][reference_index]
        selected_angles = params["reference_body_angles"][reference_index]
        params["reference_probability_start"] = 1.0
        params["reference_probability_end"] = 1.0
        params["reference_root_height_offsets"] = (selected_height,)
        params["reference_body_angles"] = (selected_angles,)
        params["reference_min_stage"] = 0
        params["reference_max_stage_start"] = 0
        params["reference_max_stage_end"] = 0
        params["reference_stage_anneal_policy_steps"] = 1
    agent_cfg = {
        "simple_v2": X2SimplePPORunnerCfg,
        "recovery": X2RecoveryPPORunnerCfg,
        "humanup_discovery": X2HumanUpPPORunnerCfg,
        "humanup_rise": X2HumanUpCurriculumPPORunnerCfg,
    }[args.environment]()
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
    contact_sensor = task.scene.sensors[CONTACT_SENSOR_NAME]
    contact_body_names = list(contact_sensor.body_names)
    if len(contact_body_names) != len(ALL_CONTACT_BODIES):
        raise RuntimeError(
            f"Expected {len(ALL_CONTACT_BODIES)} X2 contact bodies, got {len(contact_body_names)}: "
            f"{contact_body_names}"
        )
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=task.device)
        handoff_policy = None
        handoff_runner = None
        if handoff_checkpoint is not None:
            handoff_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            handoff_runner.load(str(handoff_checkpoint))
            handoff_policy = handoff_runner.get_inference_policy(device=task.device)
        env.seed(AUDIT_SEED)
        observation, _ = env.reset()
        action_term = task.action_manager.get_term("joint_position")
        joint_names = list(action_term._joint_names)
        standing_joint_target = torch.zeros((1, len(joint_names)), device=task.device)
        for joint_name, value in (
            ("left_shoulder_pitch_joint", -1.0),
            ("right_shoulder_pitch_joint", -1.0),
            ("left_elbow_joint", -1.5),
            ("right_elbow_joint", -1.5),
        ):
            standing_joint_target[0, joint_names.index(joint_name)] = value

        observation_slices = (
            OBSERVATION_SLICES if args.environment in ("recovery", "simple_v2") else HUMANUP_OBSERVATION_SLICES
        )
        observation_ranges = {name: [] for name in observation_slices}
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
        terminal_foot_contact = [0.0, 0.0]
        terminal_other_contact = 0.0
        criterion_steps = {
            "height": 0,
            "upright": 0,
            "linear_speed": 0,
            "angular_speed": 0,
            "both_feet": 0,
            "no_other_support": 0,
        }
        best_upright_joint_position: dict[str, float] = {}
        best_upright_action: dict[str, float] = {}
        best_height_state: dict = {}
        final_joint_position: dict[str, float] = {}
        trajectory: list[dict[str, float | bool]] = []

        # Stop one step before the time-limit reset so terminal values describe
        # the policy state rather than a freshly reset pose.
        total_steps = max(1, round(cfg.episode_length_s / task.step_dt) - 1)
        handoff_latched = torch.zeros(cfg.scene.num_envs, dtype=torch.bool, device=task.device)
        hold_latched = torch.zeros(cfg.scene.num_envs, dtype=torch.bool, device=task.device)
        straighten_latched = torch.zeros(cfg.scene.num_envs, dtype=torch.bool, device=task.device)
        hold_joint_target = torch.zeros_like(robot.data.joint_pos.torch[:, action_term._joint_ids])
        episode_done = False
        last_observed_root = None
        for _ in range(total_steps):
            policy_observation = observation["policy"]
            nonfinite_observations += int(not torch.isfinite(policy_observation).all().item())
            for name, (start, stop) in observation_slices.items():
                observation_ranges[name].append(policy_observation[:, start:stop].detach().clone())
            current = robot.data.joint_pos.torch[:, action_term._joint_ids].clone()
            if args.controller == "policy":
                with torch.no_grad():
                    action = policy(observation)
                    if handoff_policy is not None:
                        stabilizer_action = handoff_policy(observation)
                        height_blend = (
                            (robot.data.root_pos_w.torch[:, 2] - args.handoff_start_height)
                            / (args.handoff_end_height - args.handoff_start_height)
                        ).clamp(0.0, 1.0)
                        upright = -robot.data.projected_gravity_b.torch[:, 2]
                        upright_blend = (
                            (upright - args.handoff_min_upright)
                            / (1.0 - args.handoff_min_upright)
                        ).clamp(0.0, 1.0)
                        blend = (height_blend * upright_blend).unsqueeze(1)
                        if args.handoff_latch:
                            handoff_latched |= (
                                (robot.data.root_pos_w.torch[:, 2] >= args.handoff_end_height)
                                & (upright >= args.handoff_min_upright)
                            )
                            blend = torch.where(handoff_latched[:, None], torch.ones_like(blend), blend)
                        # Smoothly reduce the rising policy's momentum before
                        # the stabilizer fully owns the final stance.
                        blend = blend.square() * (3.0 - 2.0 * blend)
                        action = torch.lerp(action, stabilizer_action, blend)
                    if args.hold_after_height is not None:
                        if not isinstance(action_term, BoundedRelativeJointPositionAction):
                            raise ValueError("pose hold requires the bounded relative X2 action")
                        upright = -robot.data.projected_gravity_b.torch[:, 2]
                        newly_latched = (
                            (~hold_latched)
                            & (robot.data.root_pos_w.torch[:, 2] >= args.hold_after_height)
                            & (upright >= args.hold_min_upright)
                        )
                        hold_joint_target[newly_latched] = current[newly_latched]
                        hold_latched |= newly_latched
                        hold_delta = (hold_joint_target - current).clamp(-0.249, 0.249)
                        hold_action = torch.atanh((hold_delta / 0.25).clamp(-0.999, 0.999))
                        action = torch.where(hold_latched[:, None], hold_action, action)
                    if args.straighten_after_height is not None:
                        upright = -robot.data.projected_gravity_b.torch[:, 2]
                        straighten_latched |= (
                            (robot.data.root_pos_w.torch[:, 2] >= args.straighten_after_height)
                            & (upright >= args.hold_min_upright)
                        )
                        desired_delta = torch.clamp(
                            standing_joint_target - current,
                            min=-args.straighten_step_rad,
                            max=args.straighten_step_rad,
                        )
                        straighten_action = torch.atanh((desired_delta / 0.25).clamp(-0.999, 0.999))
                        action = torch.where(straighten_latched[:, None], straighten_action, action)
                    if args.brake_start_height is not None:
                        if args.brake_end_height <= args.brake_start_height:
                            raise ValueError("--brake_end_height must exceed --brake_start_height")
                        upright = -robot.data.projected_gravity_b.torch[:, 2]
                        brake = (
                            (robot.data.root_pos_w.torch[:, 2] - args.brake_start_height)
                            / (args.brake_end_height - args.brake_start_height)
                        ).clamp(0.0, 1.0)
                        brake *= (
                            (upright - args.hold_min_upright)
                            / (1.0 - args.hold_min_upright)
                        ).clamp(0.0, 1.0)
                        brake = brake.square() * (3.0 - 2.0 * brake)
                        if args.brake_sagittal_only:
                            brake_mask = torch.tensor(
                                [
                                    (
                                        "hip_pitch" in name
                                        or "knee" in name
                                        or "ankle_pitch" in name
                                    )
                                    for name in joint_names
                                ],
                                dtype=action.dtype,
                                device=action.device,
                            )
                            action = action * (1.0 - brake[:, None] * brake_mask[None, :])
                        else:
                            action = action * (1.0 - brake[:, None])
                    if args.align_after_height is not None:
                        alignment_active = (
                            (robot.data.root_pos_w.torch[:, 2] >= args.align_after_height)
                            & (-robot.data.projected_gravity_b.torch[:, 2] >= args.hold_min_upright)
                        )
                        align_mask = torch.tensor(
                            [
                                ("yaw" in name)
                                if args.align_yaw_only
                                else not (
                                    "hip_pitch" in name
                                    or "knee" in name
                                    or "ankle_pitch" in name
                                )
                                for name in joint_names
                            ],
                            dtype=torch.bool,
                            device=action.device,
                        )
                        align_delta = torch.clamp(
                            standing_joint_target - current,
                            min=-args.align_step_rad,
                            max=args.align_step_rad,
                        )
                        align_action = torch.atanh((align_delta / 0.25).clamp(-0.999, 0.999))
                        replace = alignment_active[:, None] & align_mask[None, :]
                        action = torch.where(replace, align_action, action)
                    if args.ankle_balance_kv or args.ankle_balance_kw:
                        balance_active = (
                            (robot.data.root_pos_w.torch[:, 2] >= 0.55)
                            & (-robot.data.projected_gravity_b.torch[:, 2] >= args.hold_min_upright)
                        )
                        correction = (
                            args.ankle_balance_kv * robot.data.root_lin_vel_w.torch[:, 0]
                            + args.ankle_balance_kw * robot.data.root_ang_vel_w.torch[:, 1]
                        ).clamp(-0.06, 0.06)
                        ankle_mask = torch.tensor(
                            ["ankle_pitch" in name for name in joint_names],
                            dtype=torch.bool,
                            device=action.device,
                        )
                        physical_delta = 0.25 * torch.tanh(action)
                        physical_delta[:, ankle_mask] += (
                            correction * balance_active.to(correction.dtype)
                        )[:, None]
                        corrected_action = torch.atanh((physical_delta / 0.25).clamp(-0.999, 0.999))
                        action = torch.where(ankle_mask[None, :], corrected_action, action)
                    if args.ankle_roll_kv or args.ankle_roll_kw:
                        balance_active = (
                            (robot.data.root_pos_w.torch[:, 2] >= 0.55)
                            & (-robot.data.projected_gravity_b.torch[:, 2] >= args.hold_min_upright)
                        )
                        correction = (
                            -args.ankle_roll_kv * robot.data.root_lin_vel_w.torch[:, 1]
                            + args.ankle_roll_kw * robot.data.root_ang_vel_w.torch[:, 0]
                        ).clamp(-0.04, 0.04)
                        ankle_mask = torch.tensor(
                            ["ankle_roll" in name for name in joint_names],
                            dtype=torch.bool,
                            device=action.device,
                        )
                        physical_delta = 0.25 * torch.tanh(action)
                        physical_delta[:, ankle_mask] += (
                            correction * balance_active.to(correction.dtype)
                        )[:, None]
                        corrected_action = torch.atanh((physical_delta / 0.25).clamp(-0.999, 0.999))
                        action = torch.where(ankle_mask[None, :], corrected_action, action)
            else:
                if not isinstance(action_term, BoundedRelativeJointPositionAction):
                    raise ValueError("straighten controller requires the bounded relative X2 action")
                desired_delta = torch.clamp(
                    standing_joint_target - current,
                    min=-args.straighten_step_rad,
                    max=args.straighten_step_rad,
                )
                action = torch.atanh((desired_delta / 0.25).clamp(-0.999, 0.999))
            action_rows.append(action.detach().clone())
            limits = robot.data.soft_joint_pos_limits.torch[:, action_term._joint_ids]
            action_term.process_actions(action)
            target = (action_term.joint_position_targets if isinstance(action_term, BoundedRelativeJointPositionAction)
                      else action_term.processed_actions)
            target_delta_rows.append((target - current).detach().clone())
            soft_limit_violations += int(((target < limits[..., 0]) | (target > limits[..., 1])).sum().item())

            stable = bool(mdp.strict_success(task, feet_cfg=feet_cfg, all_bodies_cfg=all_cfg)[0].item())
            strict_steps = strict_steps + 1 if stable else 0
            max_strict_steps = max(max_strict_steps, strict_steps)
            current_height_value = float(robot.data.root_pos_w.torch[0, 2].item())
            is_new_height_peak = current_height_value > max_height
            max_height = max(max_height, current_height_value)
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
            forces = mdp._ground_forces(contact_sensor)[0].norm(dim=-1)
            # PhysX owns the tensor's body ordering.  It is not guaranteed to
            # match the URDF/tree order used by ALL_CONTACT_BODIES, so label
            # force columns with the sensor's resolved names.
            mapping = dict(zip(contact_body_names, forces.tolist(), strict=True))
            maximum_foot_contact[0] = max(maximum_foot_contact[0], mapping[FOOT_CONTACT_BODIES[0]])
            maximum_foot_contact[1] = max(maximum_foot_contact[1], mapping[FOOT_CONTACT_BODIES[1]])
            current_other_contact = max(
                value for name, value in mapping.items() if name not in FOOT_CONTACT_BODIES
            )
            maximum_other_contact = max(maximum_other_contact, current_other_contact)
            terminal_foot_contact = [mapping[name] for name in FOOT_CONTACT_BODIES]
            terminal_other_contact = current_other_contact
            gravity = robot.data.projected_gravity_b.torch[0]
            criteria = {
                "height": bool(robot.data.root_pos_w.torch[0, 2].item() >= 0.58),
                "upright": bool(gravity[:2].norm().item() <= 0.15 and gravity[2].item() <= -0.98),
                "linear_speed": bool(robot.data.root_lin_vel_w.torch[0].norm().item() <= 0.25),
                "angular_speed": bool(robot.data.root_ang_vel_w.torch[0].norm().item() <= 0.35),
                "both_feet": bool(all(force >= 15.0 for force in terminal_foot_contact)),
                "no_other_support": bool(current_other_contact < 15.0),
            }
            for name, passed in criteria.items():
                criterion_steps[name] += int(passed)
            trajectory.append(
                {
                    "time_s": len(trajectory) * task.step_dt,
                    "pelvis_height_m": current_height_value,
                    "upright_score": current_upright,
                    "tilt_norm": float(gravity[:2].norm().item()),
                    "linear_speed_m_s": float(robot.data.root_lin_vel_w.torch[0].norm().item()),
                    "angular_speed_rad_s": float(robot.data.root_ang_vel_w.torch[0].norm().item()),
                    "linear_velocity_world_m_s": [
                        float(value) for value in robot.data.root_lin_vel_w.torch[0].tolist()
                    ],
                    "angular_velocity_world_rad_s": [
                        float(value) for value in robot.data.root_ang_vel_w.torch[0].tolist()
                    ],
                    "projected_gravity_body": [float(value) for value in gravity.tolist()],
                    "left_foot_force_n": terminal_foot_contact[0],
                    "right_foot_force_n": terminal_foot_contact[1],
                    "other_body_force_n": current_other_contact,
                    "strict": stable,
                }
            )
            if is_new_height_peak:
                best_height_state = {
                    "pelvis_height_m": current_height_value,
                    "upright_score": current_upright,
                    "linear_speed_m_s": float(robot.data.root_lin_vel_w.torch[0].norm().item()),
                    "angular_speed_rad_s": float(robot.data.root_ang_vel_w.torch[0].norm().item()),
                    "linear_velocity_world_m_s": [
                        float(value) for value in robot.data.root_lin_vel_w.torch[0].tolist()
                    ],
                    "angular_velocity_world_rad_s": [
                        float(value) for value in robot.data.root_ang_vel_w.torch[0].tolist()
                    ],
                    "left_foot_force_n": terminal_foot_contact[0],
                    "right_foot_force_n": terminal_foot_contact[1],
                    "maximum_other_body_force_n": current_other_contact,
                    "supporting_other_bodies_n": {
                        name: float(value)
                        for name, value in mapping.items()
                        if name not in FOOT_CONTACT_BODIES and value >= 15.0
                    },
                    "criteria": criteria,
                    "joint_position_rad": {
                        name: float(value)
                        for name, value in zip(joint_names, current[0].tolist())
                    },
                    "raw_action": {
                        name: float(value)
                        for name, value in zip(joint_names, action[0].tolist())
                    },
                }
            final_joint_position = {
                name: float(value)
                for name, value in zip(joint_names, current[0].tolist())
            }
            last_observed_root = {
                "height": float(robot.data.root_pos_w.torch[0, 2]),
                "gravity": robot.data.projected_gravity_b.torch[0].clone(),
                "linear": robot.data.root_lin_vel_w.torch[0].clone(),
                "angular": robot.data.root_ang_vel_w.torch[0].clone(),
            }
            observation, _, done, _ = env.step(action)
            if bool(done[0].item()):
                episode_done = True
                break

        actions = torch.cat(action_rows, dim=0)
        target_deltas = torch.cat(target_delta_rows, dim=0)
        result = {
            "checkpoint": str(checkpoint),
            "handoff_checkpoint": str(handoff_checkpoint) if handoff_checkpoint is not None else None,
            "handoff": {
                "start_height_m": args.handoff_start_height,
                "end_height_m": args.handoff_end_height,
                "minimum_upright": args.handoff_min_upright,
                "latched": args.handoff_latch,
            }
            if handoff_checkpoint is not None
            else None,
            "controller": args.controller,
            "pose_hold": {
                "height_m": args.hold_after_height,
                "minimum_upright": args.hold_min_upright,
            }
            if args.hold_after_height is not None
            else None,
            "straighten_after_height_m": args.straighten_after_height,
            "action_brake": {
                "start_height_m": args.brake_start_height,
                "end_height_m": args.brake_end_height,
                "sagittal_only": args.brake_sagittal_only,
            }
            if args.brake_start_height is not None
            else None,
            "non_sagittal_alignment": {
                "start_height_m": args.align_after_height,
                "step_rad": args.align_step_rad,
                "yaw_only": args.align_yaw_only,
            }
            if args.align_after_height is not None
            else None,
            "ankle_balance": {
                "linear_velocity_gain": args.ankle_balance_kv,
                "pitch_rate_gain": args.ankle_balance_kw,
                "lateral_velocity_gain": args.ankle_roll_kv,
                "roll_rate_gain": args.ankle_roll_kw,
            },
            "seed": AUDIT_SEED,
            "environment": args.environment,
            "lift_assist_enabled": cfg.events.lift_assist is not None,
            "initial_pose": args.initial_pose,
            "checkpoint_audit": _checkpoint_audit(checkpoint),
            "network": {
                "observations": int(observation["policy"].shape[1]),
                "actions": int(actions.shape[1]),
                "hidden": [512, 256, 128],
                "history_encoder": [98, 30, "conv30x20", "conv20x10", 20]
                if args.environment not in ("recovery", "simple_v2")
                else None,
            },
            "observation_order": {name: [start, stop] for name, (start, stop) in observation_slices.items()},
            "contact_sensor_order": contact_body_names,
            "resolved_contact_ids": {
                "feet": _resolved_ids(feet_cfg.body_ids, contact_sensor.num_sensors),
                "all_bodies": _resolved_ids(all_cfg.body_ids, contact_sensor.num_sensors),
            },
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
                "stopped_on_first_done": episode_done,
                "sampled_policy_steps": len(action_rows),
                "terminal_snapshot_scope": "last sampled state before action, before any reset",
                "maximum_pelvis_height_m": max_height,
                "maximum_upright_score": max_upright,
                "maximum_strict_stable_s": max_strict_steps * task.step_dt,
                "maximum_left_foot_force_n": maximum_foot_contact[0],
                "maximum_right_foot_force_n": maximum_foot_contact[1],
                "maximum_other_body_force_n": maximum_other_contact,
                "criterion_fraction": {
                    name: count / len(action_rows) for name, count in criterion_steps.items()
                },
                "terminal_left_foot_force_n": terminal_foot_contact[0],
                "terminal_right_foot_force_n": terminal_foot_contact[1],
                "terminal_other_body_force_n": terminal_other_contact,
                "terminal_pelvis_height_m": last_observed_root["height"],
                "terminal_upright_score": float(-last_observed_root["gravity"][2]),
                "terminal_linear_speed_m_s": float(last_observed_root["linear"].norm()),
                "terminal_angular_speed_rad_s": float(last_observed_root["angular"].norm()),
                "terminal_linear_velocity_world_m_s": [
                    float(value) for value in last_observed_root["linear"].tolist()
                ],
                "terminal_angular_velocity_world_rad_s": [
                    float(value) for value in last_observed_root["angular"].tolist()
                ],
                "best_height_state": best_height_state,
                "best_upright_joint_position_rad": best_upright_joint_position,
                "best_upright_raw_action": best_upright_action,
                "terminal_joint_position_rad": final_joint_position,
                "trajectory": trajectory,
            },
        }
        result["passes_io_integrity"] = bool(
            result["checkpoint_audit"]["all_tensors_finite"]
            and nonfinite_observations == 0
            and observation["policy"].shape[1] == (168 if args.environment in ("recovery", "simple_v2") else 1148)
            and actions.shape[1] == 31
            and soft_limit_violations == 0
        )
        result["passes_strict_stance"] = bool(max_strict_steps * task.step_dt >= 0.5)
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
