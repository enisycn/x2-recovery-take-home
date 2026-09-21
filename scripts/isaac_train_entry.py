"""Offline Isaac Lab/RSL-RL training entry for the HRS X2 task.

The SimulationApp is created before task modules are imported.  Keeping that
ordering here avoids loading USD/PhysX plug-ins before Kit owns the process.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import traceback
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train the HRS X2 recovery policy with RSL-RL PPO.")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=1500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", default="")
parser.add_argument("--checkpoint", default=None, help="Optional RSL-RL checkpoint to resume from.")
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
from x2_recovery_isaac.agents.rsl_rl_ppo_cfg import X2RecoveryPPORunnerCfg  # noqa: E402
from x2_recovery_isaac.env_cfg import (  # noqa: E402
    ALL_CONTACT_BODIES,
    CONTACT_SENSOR_NAME,
    X2RecoveryEnvCfg,
)


def main() -> Path:
    env_cfg = X2RecoveryEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed

    agent_cfg = X2RecoveryPPORunnerCfg()
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
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
        f"[HRS] device={args.device} envs={args.num_envs} "
        f"iterations={args.max_iterations} seed={args.seed}",
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
            runner.load(str(checkpoint))
            print(f"[HRS] Resumed from {checkpoint}", flush=True)

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
