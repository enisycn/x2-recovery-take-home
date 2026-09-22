#!/usr/bin/env python3
"""Evaluate the selected HumanUP-history X2 policy in five supine Isaac episodes."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True, help="RSL-RL model checkpoint")
parser.add_argument("--seeds", type=int, nargs="+", default=[101,102,103,104,105])
parser.add_argument("--environment", choices=("humanup_rise", "simple_v2", "symmetric_v3"), default="humanup_rise")
parser.add_argument("--output", type=Path, default=Path("reports/isaac_evaluation.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac/Kit-dependent imports must remain below AppLauncher construction.
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import x2_recovery_isaac  # noqa: E402,F401
from x2_recovery_isaac import mdp  # noqa: E402
from x2_recovery_isaac.agents.rsl_rl_ppo_cfg import X2HumanUpCurriculumPPORunnerCfg  # noqa: E402
from x2_recovery_isaac.agents.humanup_history_model import HumanUpInferenceModule  # noqa: E402
from x2_recovery_isaac.env_cfg import (  # noqa: E402
    CONTACT_SENSOR_NAME,
    FOOT_CONTACT_BODIES,
    X2HumanUpRiseEnvCfg,
    all_contact_cfg,
    foot_contact_cfg,
)

from x2_recovery_isaac.simple_cfg import X2SimpleRecoveryEnvCfg, X2SimplePPORunnerCfg, X2SymmetricRecoveryEnvCfg, X2SymmetricPPORunnerCfg

SEEDS = tuple(args.seeds)
STABLE_STEPS = 10  # 0.5 s at the 20 Hz policy rate, as in the FRASA check.
CONTACT_THRESHOLD_N = 15.0
SUCCESS_LIMITS = {
    "pelvis_height_m_min": 0.58,
    "tilt_metric_max": 0.15,
    "projected_gravity_z_max": -0.98,
    "linear_speed_m_s_max": 0.25,
    "angular_speed_rad_s_max": 0.35,
    "each_foot_contact_force_n_min": CONTACT_THRESHOLD_N,
    "other_body_contact_force_n_max": CONTACT_THRESHOLD_N,
}


def _portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _ground_contact_forces(env) -> dict[str, float]:
    """Return only body-to-floor force, excluding X2 self-collisions."""

    sensor = env.scene.sensors[CONTACT_SENSOR_NAME]
    forces = mdp._ground_forces(sensor)[0].norm(dim=-1)
    return {
        name: float(force.item())
        for name, force in zip(sensor.body_names, forces, strict=True)
    }


def snapshot(env) -> dict:
    robot = env.scene["robot"]
    forces = _ground_contact_forces(env)
    other_forces = [value for name, value in forces.items() if name not in FOOT_CONTACT_BODIES]
    gravity = robot.data.projected_gravity_b.torch[0]
    linear_speed = float(robot.data.root_lin_vel_w.torch[0].norm().item())
    angular_speed = float(robot.data.root_ang_vel_w.torch[0].norm().item())
    pelvis_height = float(robot.data.root_pos_w.torch[0, 2].item())
    tilt = float(gravity[:2].norm().item())
    gravity_z = float(gravity[2].item())
    left_force = forces[FOOT_CONTACT_BODIES[0]]
    right_force = forces[FOOT_CONTACT_BODIES[1]]
    other_force = max(other_forces)
    criteria = {
        "height": pelvis_height >= SUCCESS_LIMITS["pelvis_height_m_min"],
        "upright": (
            tilt <= SUCCESS_LIMITS["tilt_metric_max"]
            and gravity_z <= SUCCESS_LIMITS["projected_gravity_z_max"]
        ),
        "linear_speed": linear_speed <= SUCCESS_LIMITS["linear_speed_m_s_max"],
        "angular_speed": angular_speed <= SUCCESS_LIMITS["angular_speed_rad_s_max"],
        "both_feet": left_force >= CONTACT_THRESHOLD_N and right_force >= CONTACT_THRESHOLD_N,
        "no_other_support": other_force < CONTACT_THRESHOLD_N,
    }
    return {
        "pelvis_height_m": round(pelvis_height, 4),
        "tilt_metric": round(tilt, 4),
        "projected_gravity_z": round(gravity_z, 4),
        "linear_speed_m_s": round(linear_speed, 4),
        "angular_speed_rad_s": round(angular_speed, 4),
        "left_foot_force_n": round(left_force, 2),
        "right_foot_force_n": round(right_force, 2),
        "max_other_body_force_n": round(other_force, 2),
        "strict": all(criteria.values()),
        "criteria": criteria,
        "joint_position_rad": dict(zip(robot.joint_names, robot.data.joint_pos.torch[0].tolist())),
    }


def _failure_mode(max_height: float, max_stable_s: float, ended_early: bool) -> str:
    if max_stable_s > 0.0:
        return "entered strict stance but did not hold it for 0.5 s"
    if max_height < SUCCESS_LIMITS["pelvis_height_m_min"]:
        return "never reached the minimum pelvis height"
    if ended_early:
        return "safety termination before a strict stable stance"
    return "strict height/orientation/speed/contact criteria never overlapped"


def main() -> None:
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
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
    # Training uses a reference curriculum. Evaluation always begins from a
    # perturbed true back-lying state and gives the policy no assistance.
    reset_params = cfg.events.reset_back_pose.params
    reset_params["reference_probability_start"] = 0.0
    reset_params["reference_probability_end"] = 0.0

    agent_cfg = {"simple_v2": X2SimplePPORunnerCfg, "symmetric_v3": X2SymmetricPPORunnerCfg, "humanup_rise": X2HumanUpCurriculumPPORunnerCfg}[args.environment]()
    agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg, importlib.metadata.version("rsl-rl-lib")
    )
    gym_env = gym.make("HRS-X2-Recovery-Play-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    task = env.unwrapped
    stable_steps = round(0.5 / task.step_dt)
    observation_width = {"simple_v2": 168, "symmetric_v3": 122, "humanup_rise": 1148}[args.environment]
    action_width = task.action_manager.total_action_dim
    feet_cfg = foot_contact_cfg()
    all_bodies_cfg = all_contact_cfg()
    feet_cfg.resolve(task.scene)
    all_bodies_cfg.resolve(task.scene)
    records: list[dict] = []
    export_status: dict = {"succeeded": False}
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=task.device)

        export_dir = args.output.expanduser().resolve().parent / ("exported_humanup" if args.environment == "humanup_rise" else f"exported_{args.environment}")
        export_dir.mkdir(parents=True, exist_ok=True)
        try:
            actor = copy.deepcopy(runner.alg.get_policy()).to("cpu").eval()
            deployment = (torch.nn.Sequential(actor.obs_normalizer, actor.mlp)
                          if args.environment != "humanup_rise" else HumanUpInferenceModule(actor)).eval()
            example = torch.zeros((1, observation_width), dtype=torch.float32)
            traced = torch.jit.trace(deployment, example)
            traced.save(str(export_dir / "policy.pt"))
            # Verify the serialized module against the source deployment graph.
            probe = torch.linspace(-1.0, 1.0, observation_width).reshape(1, -1)
            with torch.no_grad():
                loaded_output = torch.jit.load(str(export_dir / "policy.pt"))(probe)
                maximum_error = float((loaded_output - deployment(probe)).abs().max())
            export_status = {
                "succeeded": maximum_error <= 1.0e-6,
                "torchscript": _portable_path(export_dir / "policy.pt"),
                "maximum_verification_error": maximum_error,
                "input_width": observation_width,
                "output_width": action_width,
            }
        except Exception as error:  # Export failure must not invalidate the live evaluation.
            export_status = {"succeeded": False, "error": f"{type(error).__name__}: {error}"}

        task.terminal_observer = lambda task, env_ids: snapshot(task)
        max_steps = round(cfg.episode_length_s / task.step_dt)
        for episode, seed in enumerate(SEEDS, start=1):
            env.seed(seed)
            observation, _ = env.reset()
            initial_snapshot = snapshot(task)
            if not (
                initial_snapshot["pelvis_height_m"] < 0.25
                and initial_snapshot["tilt_metric"] > 0.95
                and initial_snapshot["linear_speed_m_s"] < 1.0e-4
                and initial_snapshot["angular_speed_rad_s"] < 1.0e-4
            ):
                raise RuntimeError(f"Seed {seed} did not produce a fresh supine reset: {initial_snapshot}")

            consecutive_strict = maximum_consecutive_strict = 0
            consecutive_two_feet = maximum_consecutive_two_feet = 0
            criterion_counts = {name: 0 for name in initial_snapshot["criteria"]}
            success = ended_early = False
            steps = 0
            maximum_height = maximum_upright = float("-inf")
            best_height_snapshot: dict = {}
            best_upright_snapshot: dict = {}
            terminal_snapshot = initial_snapshot

            for step in range(max_steps):
                steps = step + 1
                with torch.no_grad():
                    action = policy(observation)
                    observation, _, done, info = env.step(action)
                episode_done = bool(done[0].item())
                terminal_snapshot = task.terminal_snapshot if episode_done else snapshot(task)
                strict = bool(terminal_snapshot["strict"])
                consecutive_strict = consecutive_strict + 1 if strict else 0
                maximum_consecutive_strict = max(maximum_consecutive_strict, consecutive_strict)
                two_feet = bool(terminal_snapshot["criteria"]["both_feet"])
                consecutive_two_feet = consecutive_two_feet + 1 if two_feet else 0
                maximum_consecutive_two_feet = max(maximum_consecutive_two_feet, consecutive_two_feet)
                for name, passed in terminal_snapshot["criteria"].items():
                    criterion_counts[name] += int(passed)
                height = float(terminal_snapshot["pelvis_height_m"])
                upright = -float(terminal_snapshot["projected_gravity_z"])
                if height > maximum_height:
                    maximum_height, best_height_snapshot = height, dict(terminal_snapshot)
                if upright > maximum_upright:
                    maximum_upright, best_upright_snapshot = upright, dict(terminal_snapshot)
                if consecutive_strict >= stable_steps:
                    success = True

                if episode_done:
                    ended_early = bool(task.reset_terminated[0].item())
                    break

            max_stable_s = maximum_consecutive_strict * task.step_dt
            record = {
                "episode": episode,
                "seed": seed,
                "success": success,
                "steps": steps,
                "ended_by_safety_termination": ended_early,
                "initial_snapshot": initial_snapshot,
                "maximum_strict_stable_s": round(max_stable_s, 3),
                "final_strict_stable_s": round(consecutive_strict * task.step_dt, 3),
                "standing_at_episode_end": consecutive_strict >= stable_steps,
                "maximum_pelvis_height_m": round(maximum_height, 4),
                "maximum_upright_score": round(maximum_upright, 4),
                "maximum_two_foot_contact_s": round(maximum_consecutive_two_feet * task.step_dt, 3),
                "criterion_fraction": {
                    name: round(count / steps, 4) for name, count in criterion_counts.items()
                },
                "best_height_snapshot": best_height_snapshot,
                "best_upright_snapshot": best_upright_snapshot,
                "terminal_snapshot_before_reset": terminal_snapshot,
                "failure_reason": "" if success else _failure_mode(maximum_height, max_stable_s, ended_early),
            }
            records.append(record)
            print(
                f"episode={episode} seed={seed} success={success} steps={steps} "
                f"max_height={maximum_height:.3f} strict={max_stable_s:.2f}s",
                flush=True,
            )

        result = {
            "backend": "Isaac Lab 3.0 / PhysX",
            "task": f"{type(cfg).__name__} evaluated from true supine resets",
            "checkpoint": _portable_path(checkpoint),
            "policy_architecture": {
                "family": args.environment,
                "observation_width": observation_width,
                "current_proprioception": 98 if observation_width == 1148 else None,
                "zero_privileged_extrinsics": 70 if observation_width == 1148 else 0,
                "history": "10 x 98" if observation_width == 1148 else "2 previous actions",
                "history_latent": 20 if observation_width == 1148 else None,
                "actor_hidden": [512, 256, 128],
                "actions": action_width,
            },
            "export": export_status,
            "seeds": list(SEEDS),
            "evaluation_controls": {
                "reference_start_probability": 0.0,
                "lift_assistance": False,
                "observation_noise": False,
                "domain_randomization": False,
                "policy_rate_hz": round(1.0 / task.step_dt),
                "episode_limit_s": cfg.episode_length_s,
            },
            "success_definition": {
                **SUCCESS_LIMITS,
                "stable_duration_s": stable_steps * task.step_dt,
                "contact_source": "filtered body-to-/World/ground force matrix",
            },
            "successful_recoveries": sum(record["success"] for record in records),
            "total_episodes": len(records),
            "episodes": records,
        }
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            f"successes={result['successful_recoveries']}/{result['total_episodes']} report={output_path}",
            flush=True,
        )
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
