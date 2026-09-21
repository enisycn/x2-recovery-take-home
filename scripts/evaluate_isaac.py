#!/usr/bin/env python3
"""Evaluate an X2 RSL-RL checkpoint for exactly five fixed Isaac episodes."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True, help="RSL-RL model checkpoint")
parser.add_argument("--output", type=Path, default=Path("reports/isaac_evaluation.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac/Kit-dependent imports must remain below AppLauncher construction.
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

import x2_recovery_isaac  # noqa: E402,F401
from x2_recovery_isaac import mdp  # noqa: E402
from x2_recovery_isaac.agents.rsl_rl_ppo_cfg import X2RecoveryPPORunnerCfg  # noqa: E402
from x2_recovery_isaac.env_cfg import (  # noqa: E402
    ALL_CONTACT_SENSORS,
    FOOT_CONTACT_SENSORS,
    X2RecoveryPlayEnvCfg,
)


SEEDS = [101, 102, 103, 104, 105]
STABLE_STEPS = 10  # 0.5 s at the 20 Hz policy rate, matching FRASA's stability check.
CONTACT_THRESHOLD_N = 15.0


def _portable_path(path: Path) -> str:
    """Prefer a repository-relative report path when possible."""

    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _contact_forces(env) -> dict[str, float]:
    values = {}
    for name in ALL_CONTACT_SENSORS:
        sensor = env.scene.sensors[name]
        force = sensor.data.net_forces_w_history.torch[0].norm(dim=-1).amax()
        values[name] = float(force.item())
    return values


def snapshot(env) -> dict:
    robot = env.scene["robot"]
    forces = _contact_forces(env)
    other_forces = [value for name, value in forces.items() if name not in FOOT_CONTACT_SENSORS]
    projected_gravity = robot.data.projected_gravity_b.torch[0]
    return {
        "pelvis_height_m": round(float(robot.data.root_pos_w.torch[0, 2].item()), 4),
        "tilt_metric": round(float(projected_gravity[:2].norm().item()), 4),
        "projected_gravity_z": round(float(projected_gravity[2].item()), 4),
        "linear_speed_m_s": round(float(robot.data.root_lin_vel_w.torch[0].norm().item()), 4),
        "angular_speed_rad_s": round(float(robot.data.root_ang_vel_w.torch[0].norm().item()), 4),
        "left_foot_force_n": round(forces[FOOT_CONTACT_SENSORS[0]], 2),
        "right_foot_force_n": round(forces[FOOT_CONTACT_SENSORS[1]], 2),
        "max_other_body_force_n": round(max(other_forces), 2),
    }


def main() -> None:
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    config = X2RecoveryPlayEnvCfg()
    config.scene.num_envs = 1
    config.sim.device = args.device

    agent_cfg = X2RecoveryPPORunnerCfg()
    agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg,
        importlib.metadata.version("rsl-rl-lib"),
    )

    gym_env = gym.make("HRS-X2-Recovery-Play-v0", cfg=config)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    unwrapped = env.unwrapped
    records: list[dict] = []
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=unwrapped.device)

        export_dir = args.output.expanduser().resolve().parent / "exported"
        export_dir.mkdir(parents=True, exist_ok=True)
        runner.export_policy_to_jit(path=str(export_dir), filename="policy.pt")
        runner.export_policy_to_onnx(path=str(export_dir), filename="policy.onnx")

        max_steps = round(config.episode_length_s / unwrapped.step_dt)
        for episode, seed in enumerate(SEEDS, start=1):
            env.seed(seed)
            observation, _ = env.reset()
            initial_snapshot = snapshot(unwrapped)
            consecutive_stable = 0
            maximum_consecutive_stable = 0
            success = False
            steps = 0
            maximum_height = float("-inf")
            maximum_upright = float("-inf")
            maximum_two_foot_steps = 0
            consecutive_two_feet = 0
            terminal_snapshot: dict = {}
            best_height_snapshot: dict = {}
            best_upright_snapshot: dict = {}

            for step in range(max_steps):
                steps = step + 1
                # Isaac keeps action/history tensors across resets; no_grad
                # avoids autograd without turning those buffers into immutable
                # inference tensors.
                with torch.no_grad():
                    instant_success = bool(
                        mdp.strict_success(
                            unwrapped,
                            feet_sensor_names=FOOT_CONTACT_SENSORS,
                            all_sensor_names=ALL_CONTACT_SENSORS,
                        )[0].item()
                    )
                    consecutive_stable = consecutive_stable + 1 if instant_success else 0
                    maximum_consecutive_stable = max(
                        maximum_consecutive_stable, consecutive_stable
                    )
                    terminal_snapshot = snapshot(unwrapped)
                    current_height = terminal_snapshot["pelvis_height_m"]
                    current_upright = -terminal_snapshot["projected_gravity_z"]
                    if current_height > maximum_height:
                        maximum_height = current_height
                        best_height_snapshot = dict(terminal_snapshot)
                    if current_upright > maximum_upright:
                        maximum_upright = current_upright
                        best_upright_snapshot = dict(terminal_snapshot)
                    two_feet = (
                        terminal_snapshot["left_foot_force_n"] >= CONTACT_THRESHOLD_N
                        and terminal_snapshot["right_foot_force_n"] >= CONTACT_THRESHOLD_N
                    )
                    consecutive_two_feet = consecutive_two_feet + 1 if two_feet else 0
                    maximum_two_foot_steps = max(maximum_two_foot_steps, consecutive_two_feet)
                    if consecutive_stable >= STABLE_STEPS:
                        success = True
                        break
                    action = policy(observation)
                    observation, _, _, _ = env.step(action)

            record = {
                "episode": episode,
                "seed": seed,
                "success": success,
                "steps": steps,
                "initial_snapshot": initial_snapshot,
                "maximum_strict_stable_s": round(
                    maximum_consecutive_stable * unwrapped.step_dt, 3
                ),
                "maximum_pelvis_height_m": round(maximum_height, 4),
                "maximum_upright_score": round(maximum_upright, 4),
                "maximum_two_foot_contact_s": round(maximum_two_foot_steps * unwrapped.step_dt, 3),
                "best_height_snapshot": best_height_snapshot,
                "best_upright_snapshot": best_upright_snapshot,
                "failure_reason": "" if success else "timeout before 0.5 s strict stable stance",
                **terminal_snapshot,
            }
            records.append(record)
            print(
                f"episode={episode} seed={seed} success={success} steps={steps} "
                f"max_height={maximum_height:.3f}",
                flush=True,
            )

        result = {
            "backend": "Isaac Lab 3.0 / PhysX",
            "task": "HRS-X2-Recovery-Play-v0",
            "checkpoint": _portable_path(checkpoint),
            "exported_policy": _portable_path(export_dir / "policy.pt"),
            "seeds": SEEDS,
            "success_definition": {
                "pelvis_height_m_min": 0.62,
                "tilt_metric_max": 0.15,
                "projected_gravity_z_max": -0.98,
                "linear_speed_m_s_max": 0.20,
                "angular_speed_rad_s_max": 0.35,
                "each_foot_contact_force_n_min": CONTACT_THRESHOLD_N,
                "other_body_contact_force_n_max": CONTACT_THRESHOLD_N,
                "stable_duration_s": STABLE_STEPS * unwrapped.step_dt,
            },
            "successful_recoveries": sum(record["success"] for record in records),
            "total_episodes": len(records),
            "episodes": records,
        }
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"successes={result['successful_recoveries']}/5 report={output_path}", flush=True)
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
