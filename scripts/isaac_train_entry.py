"""Offline Isaac Lab/RSL-RL training entry for the HRS X2 task.

The SimulationApp is created before task modules are imported.  Keeping that
ordering here avoids loading USD/PhysX plug-ins before Kit owns the process.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import traceback
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train the HRS X2 recovery policy with RSL-RL PPO.")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=1500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", default="")
parser.add_argument("--learning_schedule", choices=("fixed", "adaptive"), default=None)
parser.add_argument("--entropy_coef", type=float, default=None)
parser.add_argument("--reference_reset_probability", type=float, default=None,
                    help="Explicit auxiliary-pretraining override; submitted relaxed_v4 defaults to supine only.")
parser.add_argument("--checkpoint", default=None, help="Optional RSL-RL checkpoint to resume from.")
parser.add_argument(
    "--action_std_override",
    type=float,
    default=None,
    help="Optional exploration standard deviation applied after loading a checkpoint.",
)
parser.add_argument(
    "--reset_optimizer",
    action="store_true",
    help="Load actor/critic weights but initialize a fresh PPO optimizer.",
)
parser.add_argument(
    "--learning_rate_override",
    type=float,
    default=None,
    help="Optional PPO learning rate for conservative fine-tuning.",
)
parser.add_argument(
    "--phase",
    choices=("recovery", "standing", "rise", "humanup_discovery", "humanup_standing", "humanup_rise", "simple_v2", "symmetric_v3", "relaxed_v4"),
    default="recovery",
    help="Training curriculum phase.",
)
parser.add_argument(
    "--reference_max_stage",
    type=int,
    choices=range(9),
    default=None,
    help="For rise training, hold the unlocked reference range at stages 0..N.",
)
parser.add_argument(
    "--reference_min_stage",
    type=int,
    choices=range(9),
    default=0,
    help="For fixed rise training, sample reference stages N..max instead of 0..max.",
)
parser.add_argument(
    "--root_linear_velocity_range",
    type=float,
    default=0.0,
    help="Symmetric x/y reset-speed range; z uses 37.5%% of this value.",
)
parser.add_argument(
    "--root_angular_velocity_range",
    type=float,
    default=0.0,
    help="Symmetric roll/pitch/yaw reset angular-speed range in rad/s.",
)
parser.add_argument(
    "--handoff_state_report",
    type=Path,
    default=None,
    help="Train from the measured peak state stored by diagnose_isaac_policy.py.",
)
parser.add_argument(
    "--handoff_velocity_scale",
    type=float,
    default=1.0,
    help="Scale the measured handoff root velocities for a catch curriculum.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac/Kit-dependent imports must remain below AppLauncher construction.
import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import x2_recovery_isaac  # noqa: E402,F401
from x2_recovery_isaac.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    X2HumanUpCurriculumPPORunnerCfg,
    X2HumanUpPPORunnerCfg,
    X2RecoveryPPORunnerCfg,
)
from x2_recovery_isaac.env_cfg import (  # noqa: E402
    ALL_CONTACT_BODIES,
    CONTACT_SENSOR_NAME,
    X2RecoveryEnvCfg,
    X2HumanUpDiscoveryEnvCfg,
    X2HumanUpRiseEnvCfg,
    X2HumanUpStandingEnvCfg,
    X2RiseEnvCfg,
    X2StandingEnvCfg,
)


from x2_recovery_isaac.simple_cfg import X2RelaxedRecoveryEnvCfg, X2RelaxedPPORunnerCfg, X2SimpleRecoveryEnvCfg, X2SimplePPORunnerCfg, X2SymmetricRecoveryEnvCfg, X2SymmetricPPORunnerCfg


def main() -> Path:
    env_cfg_type = {
        "simple_v2": X2SimpleRecoveryEnvCfg,
        "symmetric_v3": X2SymmetricRecoveryEnvCfg, "relaxed_v4": X2RelaxedRecoveryEnvCfg,
        "recovery": X2RecoveryEnvCfg,
        "standing": X2StandingEnvCfg,
        "rise": X2RiseEnvCfg,
        "humanup_discovery": X2HumanUpDiscoveryEnvCfg,
        "humanup_standing": X2HumanUpStandingEnvCfg,
        "humanup_rise": X2HumanUpRiseEnvCfg,
    }[args.phase]
    env_cfg = env_cfg_type()
    if args.reference_reset_probability is not None:
        if not 0.0 <= args.reference_reset_probability <= 1.0:
            raise ValueError("--reference_reset_probability must be in [0, 1]")
        env_cfg.events.reset_back_pose.params["reference_probability_start"] = args.reference_reset_probability
        env_cfg.events.reset_back_pose.params["reference_probability_end"] = args.reference_reset_probability
    if args.handoff_state_report is not None:
        if not 0.0 <= args.handoff_velocity_scale <= 1.5:
            raise ValueError("--handoff_velocity_scale must be in [0, 1.5]")
        report_path = args.handoff_state_report.expanduser().resolve(strict=True)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        state = report["live_episode"]["best_height_state"]
        if not state:
            raise ValueError(f"Handoff report has no best-height state: {report_path}")
        reset_params = env_cfg.events.reset_back_pose.params
        reset_params["reference_probability_start"] = 1.0
        reset_params["reference_probability_end"] = 1.0
        reset_params["reference_probability_anneal_policy_steps"] = 1
        reset_params["reference_root_height_offsets"] = (float(state["pelvis_height_m"]) - 0.190,)
        reset_params["reference_body_angles"] = ((0.0, 0.0, 0.0, 0.0, 0.0),)
        reset_params["reference_min_stage"] = 0
        reset_params["reference_max_stage_start"] = 0
        reset_params["reference_max_stage_end"] = 0
        reset_params["reference_stage_anneal_policy_steps"] = 1
        reset_params["reference_joint_position_overrides"] = (state["joint_position_rad"],)
        linear_velocity = [
            args.handoff_velocity_scale * value for value in state["linear_velocity_world_m_s"]
        ]
        angular_velocity = [
            args.handoff_velocity_scale * value for value in state["angular_velocity_world_rad_s"]
        ]
        linear_jitter = 0.02 + 0.03 * args.handoff_velocity_scale
        angular_jitter = 0.05 + 0.05 * args.handoff_velocity_scale
        reset_params["velocity_range"] = {
            "x": (linear_velocity[0] - linear_jitter, linear_velocity[0] + linear_jitter),
            "y": (linear_velocity[1] - linear_jitter, linear_velocity[1] + linear_jitter),
            "z": (linear_velocity[2] - linear_jitter, linear_velocity[2] + linear_jitter),
            "roll": (angular_velocity[0] - angular_jitter, angular_velocity[0] + angular_jitter),
            "pitch": (angular_velocity[1] - angular_jitter, angular_velocity[1] + angular_jitter),
            "yaw": (angular_velocity[2] - angular_jitter, angular_velocity[2] + angular_jitter),
        }
    if args.reference_max_stage is not None:
        if args.handoff_state_report is not None:
            raise ValueError("Use either --handoff_state_report or reference-stage overrides")
        if args.phase not in ("rise", "humanup_rise"):
            raise ValueError("--reference_max_stage is only valid with a rise phase")
        reset_params = env_cfg.events.reset_back_pose.params
        if args.reference_min_stage > args.reference_max_stage:
            raise ValueError("--reference_min_stage must not exceed --reference_max_stage")
        reset_params["reference_min_stage"] = args.reference_min_stage
        reset_params["reference_max_stage_start"] = args.reference_max_stage
        reset_params["reference_max_stage_end"] = args.reference_max_stage
        reset_params["reference_stage_anneal_policy_steps"] = 1
    if args.root_linear_velocity_range < 0.0 or args.root_angular_velocity_range < 0.0:
        raise ValueError("Root velocity ranges must be non-negative")
    if args.root_linear_velocity_range or args.root_angular_velocity_range:
        reset_params = env_cfg.events.reset_back_pose.params
        linear = args.root_linear_velocity_range
        angular = args.root_angular_velocity_range
        reset_params["velocity_range"] = {
            "x": (-linear, linear),
            "y": (-linear, linear),
            "z": (-0.375 * linear, 0.375 * linear),
            "roll": (-angular, angular),
            "pitch": (-angular, angular),
            "yaw": (-angular, angular),
        }
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed

    if args.phase == "relaxed_v4":
        agent_cfg = X2RelaxedPPORunnerCfg()
    elif args.phase == "symmetric_v3":
        agent_cfg = X2SymmetricPPORunnerCfg()
    elif args.phase == "simple_v2":
        agent_cfg = X2SimplePPORunnerCfg()
    elif args.phase == "humanup_discovery":
        agent_cfg = X2HumanUpPPORunnerCfg()
    elif args.phase.startswith("humanup_"):
        agent_cfg = X2HumanUpCurriculumPPORunnerCfg()
    else:
        agent_cfg = X2RecoveryPPORunnerCfg()
    if args.learning_schedule is not None:
        agent_cfg.algorithm.schedule = args.learning_schedule
    if args.entropy_coef is not None:
        if args.entropy_coef < 0.0:
            raise ValueError("--entropy_coef must be non-negative")
        agent_cfg.algorithm.entropy_coef = args.entropy_coef
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    if args.learning_rate_override is not None:
        if args.learning_rate_override <= 0.0:
            raise ValueError("--learning_rate_override must be positive")
        agent_cfg.algorithm.learning_rate = args.learning_rate_override
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg,
        importlib.metadata.version("rsl-rl-lib"),
    )

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_name:
        run_id += f"_{args.run_name}"
    log_dir = Path("logs") / "rsl_rl" / agent_cfg.experiment_name / run_id
    log_dir = log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=False)
    env_cfg.log_dir = str(log_dir)

    print(f"[HRS] Offline training log: {log_dir}", flush=True)
    print(
        f"[HRS] phase={args.phase} reference_stage_range="
        f"{args.reference_min_stage}..{args.reference_max_stage} "
        f"device={args.device} envs={args.num_envs} "
        f"iterations={args.max_iterations} seed={args.seed}",
        flush=True,
    )
    print(
        f"[HRS] reset_velocity_ranges=linear_xy±{args.root_linear_velocity_range:.3f}m/s "
        f"angular±{args.root_angular_velocity_range:.3f}rad/s",
        flush=True,
    )
    if args.handoff_state_report is not None:
        print(
            f"[HRS] measured_handoff_state={report_path} "
            f"velocity_scale={args.handoff_velocity_scale:.2f}",
            flush=True,
        )

    env = gym.make("HRS-X2-Recovery-v0", cfg=env_cfg)
    resolved_contact_bodies = tuple(env.unwrapped.scene.sensors[CONTACT_SENSOR_NAME].body_names)
    if resolved_contact_bodies != ALL_CONTACT_BODIES:
        raise RuntimeError(
            "Invalid recursive X2 contact mapping:\n"
            f"expected={ALL_CONTACT_BODIES}\nresolved={resolved_contact_bodies}"
        )
    print(
        f"[HRS] Verified one recursive view with {len(resolved_contact_bodies)} X2 contact bodies.",
        flush=True,
    )
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
        if args.checkpoint:
            checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
            load_cfg = None
            if args.reset_optimizer:
                load_cfg = {"actor": True, "critic": True, "optimizer": False, "iteration": True, "rnd": True}
            runner.load(str(checkpoint), load_cfg=load_cfg)
            print(f"[HRS] Resumed from {checkpoint}", flush=True)
        if args.learning_rate_override is not None:
            runner.alg.learning_rate = args.learning_rate_override
            for group in runner.alg.optimizer.param_groups:
                group["lr"] = args.learning_rate_override
        if args.action_std_override is not None:
            if args.action_std_override <= 0.0:
                raise ValueError("--action_std_override must be positive")
            distribution = runner.alg.get_policy().distribution
            distribution.std_param.data.fill_(args.action_std_override)
            print(f"[HRS] Reset action standard deviation to {args.action_std_override:.4f}", flush=True)

        dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
        dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    finally:
        wrapped_env.close()
    return log_dir


if __name__ == "__main__":
    try:
        try:
            main()
        except BaseException:
            traceback.print_exc()
            raise
    finally:
        simulation_app.close()
