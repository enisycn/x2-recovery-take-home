"""Simulator-independent recovery episode orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Callable

import numpy as np

from .mujoco_env import MujocoRecoveryEnv
from .policy import SCRIPTED_ACTIONS, PhasePolicy
from .reduced_env import ReducedOrderRecoveryEnv


class AttemptGate:
    """Atomic gate shared by pending and running recovery attempts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._busy = False

    def try_acquire(self) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True
            return True

    def release(self) -> None:
        with self._lock:
            self._busy = False

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy


@dataclass(frozen=True)
class SessionResult:
    success: bool
    status: str
    steps: int
    failure_reason: str


class RecoverySession:
    def __init__(
        self,
        *,
        backend: str,
        checkpoint: str | Path,
        policy_mode: str,
        seed: int,
        timeout_sec: float,
        model_path: str | Path | None = None,
        real_time: bool = True,
    ) -> None:
        self.backend = backend
        self.checkpoint = Path(checkpoint)
        self.policy_mode = policy_mode
        self.seed = int(seed)
        self.timeout_sec = float(timeout_sec)
        self.model_path = Path(model_path) if model_path else None
        self.real_time = bool(real_time)

    def _environment(self):
        if self.backend == "reduced":
            return ReducedOrderRecoveryEnv(seed=self.seed, timeout_sec=self.timeout_sec)
        if self.backend == "mujoco":
            if self.model_path is None:
                raise ValueError("model_path is required for backend=mujoco")
            return MujocoRecoveryEnv(
                model_path=self.model_path,
                seed=self.seed,
                timeout_sec=self.timeout_sec,
            )
        raise ValueError(f"unknown backend: {self.backend}")

    def _policy(self):
        if self.policy_mode == "checkpoint":
            return PhasePolicy.load(self.checkpoint).act
        if self.policy_mode == "scripted":
            return PhasePolicy(SCRIPTED_ACTIONS).act
        if self.policy_mode == "zero":
            return lambda observation: np.zeros(4, dtype=np.float64)
        raise ValueError(f"unknown policy_mode: {self.policy_mode}")

    def run(self, on_step: Callable[[dict], None] | None = None) -> SessionResult:
        env = self._environment()
        action_fn = self._policy()
        observation, info = env.reset(seed=self.seed)
        while True:
            started = time.monotonic()
            observation, _, terminated, truncated, info = env.step(action_fn(observation))
            if on_step is not None:
                on_step(info)
            if terminated or truncated:
                break
            if self.real_time:
                time.sleep(max(0.0, env.dt - (time.monotonic() - started)))
        success = bool(info["success"])
        return SessionResult(
            success=success,
            status="SUCCEEDED" if success else "FAILED",
            steps=int(info["step"]),
            failure_reason=str(info["failure_reason"]),
        )

