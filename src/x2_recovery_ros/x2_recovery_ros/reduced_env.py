"""Small deterministic recovery environment used as the runnable CPU baseline.

This is a reduced-order state machine with continuous, noisy dynamics. It is
useful for testing policy search, success checks and the ROS control path. It is
deliberately not represented as a rigid-body or hardware result.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .constants import JOINT_LIMITS, JOINT_NAMES


@dataclass(frozen=True)
class RecoveryMetrics:
    height: float
    pitch: float
    pitch_rate: float
    lateral_tilt: float
    both_feet_contact: bool
    other_body_support: bool
    stable_time: float


class ReducedOrderRecoveryEnv:
    """Continuous X2 recovery baseline with Gym-like ``reset``/``step`` calls."""

    dt = 0.05
    action_size = 4
    observation_size = 22
    max_episode_steps = 120
    stable_duration = 0.40

    def __init__(self, seed: int = 0, timeout_sec: float = 6.0) -> None:
        self.rng = np.random.default_rng(seed)
        self.timeout_sec = float(timeout_sec)
        self.max_episode_steps = max(1, int(round(self.timeout_sec / self.dt)))
        self.joint_names = JOINT_NAMES
        self._seed = seed
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._seed = int(seed)
            self.rng = np.random.default_rng(self._seed)

        self.steps = 0
        self.progress = float(self.rng.uniform(0.0, 0.015))
        self.progress_speed = 0.0
        self.lateral_tilt = float(self.rng.normal(0.0, 0.025))
        self.lateral_rate = 0.0
        self.joint_positions = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self.joint_velocities = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self.stable_time = 0.0
        self._last_action = np.zeros(self.action_size, dtype=np.float64)
        return self._observation(), self._info(success=False)

    @staticmethod
    def _phase_target(progress: float) -> np.ndarray:
        if progress < 0.32:
            return np.asarray([1.0, 0.85, -0.55, 0.0])
        if progress < 0.72:
            return np.asarray([0.25, 0.55, 1.0, 0.30])
        return np.asarray([0.0, 0.0, 0.45, 1.0])

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_size,):
            raise ValueError(f"expected action shape {(self.action_size,)}, got {action.shape}")
        action = np.clip(action, -1.0, 1.0)
        self._last_action = action
        previous_progress = self.progress

        target = self._phase_target(self.progress)
        skill = math.exp(-2.5 * float(np.mean(np.square(action - target))))
        recovery_drive = max(0.0, skill - 0.45) * 2.3
        damping = 1.8 * self.progress_speed
        tilt_cost = 0.45 * abs(self.lateral_tilt)
        noise = float(self.rng.normal(0.0, 0.012))
        self.progress_speed += self.dt * (recovery_drive - damping - tilt_cost + noise)
        self.progress_speed = float(np.clip(self.progress_speed, -0.08, 0.90))
        self.progress = float(np.clip(self.progress + self.dt * self.progress_speed, 0.0, 1.0))
        if self.progress >= 1.0:
            self.progress_speed *= 0.55

        stabilizer = 1.0 + 1.8 * max(0.0, action[3])
        lateral_forcing = 0.32 * (action[0] - action[1]) * (1.0 - self.progress)
        tilt_accel = (
            lateral_forcing
            - stabilizer * self.lateral_tilt
            - 0.85 * self.lateral_rate
            + float(self.rng.normal(0.0, 0.008))
        )
        self.lateral_rate += self.dt * tilt_accel
        self.lateral_tilt += self.dt * self.lateral_rate

        targets = self._joint_targets(action)
        old_positions = self.joint_positions.copy()
        self.joint_positions += 0.22 * (targets - self.joint_positions)
        self.joint_positions = np.clip(
            self.joint_positions, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]
        )
        self.joint_velocities = (self.joint_positions - old_positions) / self.dt

        metrics = self.metrics()
        stable_now = self._stable_now(metrics)
        self.stable_time = self.stable_time + self.dt if stable_now else 0.0
        success = self.stable_time + 1e-9 >= self.stable_duration
        self.steps += 1
        terminated = bool(success or abs(self.lateral_tilt) > 0.85)
        truncated = bool(self.steps >= self.max_episode_steps and not terminated)

        upright = math.cos(metrics.pitch)
        progress_rate = (self.progress - previous_progress) / self.dt
        reward = self.dt * (
            1.5 * metrics.height
            + 1.5 * upright
            + 1.2 * progress_rate
            - 0.04 * float(np.mean(np.square(action)))
            - 0.25 * self.lateral_tilt**2
        )
        reward += 50.0 if success else 0.0
        reward -= 5.0 if truncated else 0.0
        return self._observation(), float(reward), terminated, truncated, self._info(success)

    def _joint_targets(self, action: np.ndarray) -> np.ndarray:
        tuck, _, extend, balance = action
        hip = 1.25 * max(tuck, 0.0) - 0.75 * max(extend, 0.0)
        knee = 1.90 * max(tuck, 0.0) - 1.25 * max(extend, 0.0)
        ankle = -0.28 * max(tuck, 0.0) + 0.30 * max(balance, 0.0)
        return np.asarray([hip, knee, ankle, hip, knee, ankle], dtype=np.float64)

    def metrics(self) -> RecoveryMetrics:
        height = 0.22 + 0.48 * self.progress
        pitch = -0.5 * math.pi * (1.0 - self.progress)
        pitch_rate = 0.5 * math.pi * self.progress_speed
        both_feet = self.progress > 0.73 and abs(self.lateral_tilt) < 0.30
        other_support = self.progress < 0.90
        return RecoveryMetrics(
            height=height,
            pitch=pitch,
            pitch_rate=pitch_rate,
            lateral_tilt=self.lateral_tilt,
            both_feet_contact=both_feet,
            other_body_support=other_support,
            stable_time=self.stable_time,
        )

    @staticmethod
    def _stable_now(metrics: RecoveryMetrics) -> bool:
        return bool(
            metrics.height >= 0.66
            and abs(metrics.pitch) <= 0.14
            and abs(metrics.pitch_rate) <= 0.28
            and abs(metrics.lateral_tilt) <= 0.12
            and metrics.both_feet_contact
            and not metrics.other_body_support
        )

    def _observation(self) -> np.ndarray:
        metrics = self.metrics()
        middle = JOINT_LIMITS.mean(axis=1)
        half_range = 0.5 * np.ptp(JOINT_LIMITS, axis=1)
        normalized_q = (self.joint_positions - middle) / half_range
        normalized_qd = np.clip(self.joint_velocities / 12.0, -1.0, 1.0)
        return np.concatenate(
            (
                np.asarray(
                    [
                        metrics.height,
                        math.sin(metrics.pitch),
                        math.cos(metrics.pitch),
                        metrics.pitch_rate,
                        self.progress,
                        self.progress_speed,
                        self.lateral_tilt,
                        self.lateral_rate,
                        float(metrics.both_feet_contact),
                        float(metrics.other_body_support),
                    ]
                ),
                normalized_q,
                normalized_qd,
            )
        ).astype(np.float32)

    def _info(self, success: bool) -> dict[str, Any]:
        metrics = self.metrics()
        return {
            "success": bool(success),
            "step": self.steps,
            "progress": self.progress,
            "height": metrics.height,
            "pitch": metrics.pitch,
            "pitch_rate": metrics.pitch_rate,
            "lateral_tilt": metrics.lateral_tilt,
            "both_feet_contact": metrics.both_feet_contact,
            "other_body_support": metrics.other_body_support,
            "stable_time": self.stable_time,
            "joint_names": list(self.joint_names),
            "joint_positions": self.joint_positions.tolist(),
            "failure_reason": "" if success else self.failure_reason(),
        }

    def failure_reason(self) -> str:
        metrics = self.metrics()
        if abs(metrics.lateral_tilt) > 0.85:
            return "fell laterally"
        if self.steps >= self.max_episode_steps:
            return "timeout before stable two-foot stance"
        return "episode running"
