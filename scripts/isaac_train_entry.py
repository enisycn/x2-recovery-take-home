"""Offline Isaac Lab/RSL-RL training entry for the HRS X2 task.

The SimulationApp is created before task modules are imported.  Keeping that
ordering here avoids loading USD/PhysX plug-ins before Kit owns the process.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
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
parser.add_argument("--checkpoint", default=None, help="Optional RSL-RL checkpoint to resume from.")
parser.add_argument("--stability_refinement", action="store_true",
                    help="Use dense stance shaping and action regularization; keep every reset supine.")
parser.add_argument("--posture_refinement", action="store_true",
                    help="Add positive supported neutral-command shaping to the stability preset.")
parser.add_argument(
    "--action_std_override",
    type=float,
    default=None,
    help="Optional exploration standard deviation applied after loading a checkpoint.",
)
parser.add_argument(
    "--reset_optimizer",
    action="store_true",
    help="Initialize a fresh optimizer; stability refinement also resets the critic for the changed reward.",
)
parser.add_argument(
    "--learning_rate_override",
    type=float,
    default=None,
    help="Optional PPO learning rate for conservative fine-tuning.",
)
parser.add_argument(
    "--phase", choices=("relaxed_v4",), default="relaxed_v4",
    help="Supine-only recovery environment.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.posture_refinement and not args.stability_refinement:
    parser.error("--posture_refinement requires --stability_refinement")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac/Kit-dependent imports must remain below AppLauncher construction.
import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import x2_recovery_isaac  # noqa: E402,F401
from x2_recovery_isaac.env_cfg import ALL_CONTACT_BODIES, CONTACT_SENSOR_NAME  # noqa: E402
from x2_recovery_isaac.simple_cfg import (  # noqa: E402
    X2RelaxedRecoveryEnvCfg, X2RelaxedPPORunnerCfg, X2StabilityRefinementEnvCfg, X2PostureRefinementEnvCfg,
)


def main() -> Path:
    env_cfg = (X2PostureRefinementEnvCfg() if args.posture_refinement else
               X2StabilityRefinementEnvCfg() if args.stability_refinement else X2RelaxedRecoveryEnvCfg())
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed

    agent_cfg = X2RelaxedPPORunnerCfg()
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
    print(f"[HRS] phase={args.phase} device={args.device} envs={args.num_envs} "
          f"iterations={args.max_iterations} seed={args.seed}", flush=True)
    print("[HRS] reset=supine_only assistance=off", flush=True)
    print(f"[HRS] stability_refinement={args.stability_refinement}", flush=True)
    print(f"[HRS] initialization={'checkpoint' if args.checkpoint else 'random_weights'}", flush=True)
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
                load_cfg = {"actor": True, "critic": not args.stability_refinement,
                            "optimizer": False, "iteration": True, "rnd": True}
            runner.load(str(checkpoint), load_cfg=load_cfg)
            print(f"[HRS] Resumed from {checkpoint}", flush=True)
            if args.reset_optimizer and args.stability_refinement:
                print("[HRS] critic_initialization=random_weights (reward changed)", flush=True)
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
        runtime_settings = {
            "parent_checkpoint": args.checkpoint,
            "stability_refinement": args.stability_refinement,
            "posture_refinement": args.posture_refinement,
            "reset_optimizer": args.reset_optimizer,
            "critic_initialization": "random_weights" if not args.checkpoint or
                (args.reset_optimizer and args.stability_refinement) else "checkpoint",
            "action_std_override": args.action_std_override,
            "learning_rate": runner.alg.learning_rate,
            "learning_schedule": agent_cfg.algorithm.schedule,
            "entropy_coef": agent_cfg.algorithm.entropy_coef,
            "reset": "supine_only",
            "external_assistance": False,
        }
        # Preserve the implementation used by this run even before a commit.
        repository = Path(__file__).resolve().parents[1]
        try:
            runtime_settings["source_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repository, text=True,
            ).strip()
            patch = subprocess.check_output(
                ["git", "diff", "HEAD", "--", "isaaclab_ext", "scripts"],
                cwd=repository, text=True,
            )
            if patch:
                (log_dir / "params" / "source_changes.patch").write_text(patch, encoding="utf-8")
                runtime_settings["source_patch"] = "source_changes.patch"
        except (OSError, subprocess.CalledProcessError) as error:
            runtime_settings["source_capture_error"] = str(error)
        (log_dir / "params" / "runtime_settings.json").write_text(
            json.dumps(runtime_settings, indent=2) + "\n", encoding="utf-8"
        )
        runner.learn(num_learning_iterations=agent_cfg.max_iterations,
                     init_at_random_ep_len=bool(args.checkpoint))
    finally:
        wrapped_env.close()
    return log_dir


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException as error:
        exit_code = 130 if isinstance(error, KeyboardInterrupt) else 1
        traceback.print_exc()
    finally:
        # Kit's fast shutdown exits the process; preserve failures for the wrapper.
        simulation_app.close(exit_code=exit_code)
    raise SystemExit(exit_code)
