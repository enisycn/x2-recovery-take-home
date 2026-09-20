"""Gym registration for the external HRS X2 recovery task."""

from __future__ import annotations

import sys

import gymnasium as gym


gym.register(
    id="HRS-X2-Recovery-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "x2_recovery_isaac.env_cfg:X2RecoveryEnvCfg",
        "rsl_rl_cfg_entry_point": "x2_recovery_isaac.agents.rsl_rl_ppo_cfg:X2RecoveryPPORunnerCfg",
    },
)

gym.register(
    id="HRS-X2-Recovery-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "x2_recovery_isaac.env_cfg:X2RecoveryPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "x2_recovery_isaac.agents.rsl_rl_ppo_cfg:X2RecoveryPPORunnerCfg",
    },
)


def register() -> list[str]:
    """Registration callback used by Isaac Lab's train/play CLIs."""

    return sys.argv[1:]
