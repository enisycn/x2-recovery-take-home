"""Optional MuJoCo backend for the official AgiBot X2 Ultra v1.3.0 scene."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .constants import JOINT_LIMITS, JOINT_NAMES
from .reduced_env import RecoveryMetrics


class MujocoUnavailableError(RuntimeError):
    pass


class MujocoRecoveryEnv:
    """Minimal torque-synergy wrapper around AgiBot's official MuJoCo model.

    The class intentionally exposes the same 22-value observation and four
    recovery synergies as ``ReducedOrderRecoveryEnv``. A policy still needs to
    be trained and validated on this backend before it can be called a MuJoCo
    result.
    """

    dt = 0.02
    action_size = 4
    observation_size = 22
    stable_duration = 0.40

    def __init__(self, model_path: str | Path, seed: int = 0, timeout_sec: float = 6.0):
        try:
            import mujoco
        except ImportError as exc:
            raise MujocoUnavailableError(
                "MuJoCo is not installed; install requirements.txt or use backend=reduced"
            ) from exc

        self.mujoco = mujoco
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"AgiBot scene not found at {self.model_path}; run scripts/fetch_agibot_model.sh"
            )
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.timeout_sec = float(timeout_sec)
        self.max_episode_steps = max(1, int(round(self.timeout_sec / self.dt)))
        self.frame_skip = max(1, int(round(self.dt / self.model.opt.timestep)))
        self.joint_names = JOINT_NAMES
        self._joint_ids = [self._name_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
        self._qpos_addresses = [int(self.model.jnt_qposadr[item]) for item in self._joint_ids]
        self._qvel_addresses = [int(self.model.jnt_dofadr[item]) for item in self._joint_ids]
        self._pelvis_body = self._name_id(mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self._floor_geom = self._name_id(mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._actuator_ids = {
            name: self._name_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_" + name)
            for name in self._all_controlled_joint_names()
        }
        self.reset(seed=seed)

    @staticmethod
    def _all_controlled_joint_names() -> tuple[str, ...]:
        return JOINT_NAMES + ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint")

    def _name_id(self, object_type: Any, name: str) -> int:
        identifier = int(self.mujoco.mj_name2id(self.model, object_type, name))
        if identifier < 0:
            raise ValueError(f"official model does not contain required name: {name}")
        return identifier

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.mujoco.mj_resetData(self.model, self.data)
        # Free joint: xyz followed by wxyz. +90 degrees about Y places the back
        # parallel to the floor. The initial height is deliberately conservative.
        self.data.qpos[:3] = [0.0, 0.0, 0.46]
        half_angle = 0.25 * math.pi
        self.data.qpos[3:7] = [math.cos(half_angle), 0.0, math.sin(half_angle), 0.0]
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        if self._has_penetrating_contact():
            raise RuntimeError("X2 reset intersects the floor; raise the reset base height")

        # Let the robot settle onto its back before the controlled episode.
        settling_steps = int(round(0.75 / self.model.opt.timestep))
        for _ in range(settling_steps):
            self.mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.stable_time = 0.0
        self._last_progress = self._progress()
        return self._observation(), self._info(False)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_size,):
            raise ValueError(f"expected action shape {(self.action_size,)}, got {action.shape}")
        action = np.clip(action, -1.0, 1.0)
        self._apply_synergies(action)
        for _ in range(self.frame_skip):
            self.mujoco.mj_step(self.model, self.data)

        metrics = self.metrics()
        stable_now = self._stable_now(metrics)
        self.stable_time = self.stable_time + self.dt if stable_now else 0.0
        success = self.stable_time + 1e-9 >= self.stable_duration
        self.steps += 1
        terminated = bool(success or self.data.qpos[2] < 0.08)
        truncated = bool(self.steps >= self.max_episode_steps and not terminated)
        progress = self._progress()
        progress_rate = (progress - self._last_progress) / self.dt
        self._last_progress = progress
        upright = math.cos(metrics.pitch)
        reward = self.dt * (
            1.5 * metrics.height
            + 1.5 * upright
            + 1.2 * progress_rate
            - 0.04 * float(np.mean(np.square(action)))
        )
        reward += 50.0 if success else 0.0
        reward -= 5.0 if truncated else 0.0
        return self._observation(), float(reward), terminated, truncated, self._info(success)

    def _apply_synergies(self, action: np.ndarray) -> None:
        tuck, arm_push, extend, balance = action
        torques = {
            "left_hip_pitch_joint": 42.0 * (tuck - extend),
            "right_hip_pitch_joint": 42.0 * (tuck - extend),
            "left_knee_joint": 58.0 * (tuck - extend),
            "right_knee_joint": 58.0 * (tuck - extend),
            "left_ankle_pitch_joint": 14.0 * balance,
            "right_ankle_pitch_joint": 14.0 * balance,
            "left_shoulder_pitch_joint": -28.0 * arm_push,
            "right_shoulder_pitch_joint": -28.0 * arm_push,
        }
        self.data.ctrl[:] = 0.0
        for joint_name, torque in torques.items():
            actuator_id = self._actuator_ids[joint_name]
            low, high = self.model.actuator_ctrlrange[actuator_id]
            self.data.ctrl[actuator_id] = np.clip(torque, low, high)

    def _contacts(self) -> tuple[bool, bool, bool]:
        left_foot = False
        right_foot = False
        other_support = False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if contact.geom1 != self._floor_geom and contact.geom2 != self._floor_geom:
                continue
            robot_geom = contact.geom2 if contact.geom1 == self._floor_geom else contact.geom1
            body_id = int(self.model.geom_bodyid[robot_geom])
            body_name = self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_id
            ) or ""
            if body_name == "left_ankle_roll_link":
                left_foot = True
            elif body_name == "right_ankle_roll_link":
                right_foot = True
            else:
                other_support = True
        return left_foot, right_foot, other_support

    def _has_penetrating_contact(self) -> bool:
        return any(self.data.contact[index].dist < -1e-4 for index in range(self.data.ncon))

    def metrics(self) -> RecoveryMetrics:
        rotation = self.data.xmat[self._pelvis_body].reshape(3, 3)
        pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[2, 1], rotation[2, 2]))
        left, right, other = self._contacts()
        return RecoveryMetrics(
            height=float(self.data.xpos[self._pelvis_body, 2]),
            pitch=pitch,
            pitch_rate=float(self.data.qvel[4]),
            lateral_tilt=math.atan2(rotation[2, 1], rotation[2, 2]),
            both_feet_contact=left and right,
            other_body_support=other,
            stable_time=self.stable_time,
        )

    @staticmethod
    def _stable_now(metrics: RecoveryMetrics) -> bool:
        return bool(
            metrics.height >= 0.60
            and abs(metrics.pitch) <= 0.18
            and abs(metrics.pitch_rate) <= 0.35
            and abs(metrics.lateral_tilt) <= 0.15
            and metrics.both_feet_contact
            and not metrics.other_body_support
        )

    def _progress(self) -> float:
        metrics = self.metrics()
        orientation_progress = float(np.clip(math.cos(metrics.pitch), 0.0, 1.0))
        height_progress = float(np.clip((metrics.height - 0.20) / 0.48, 0.0, 1.0))
        return 0.65 * orientation_progress + 0.35 * height_progress

    def _observation(self) -> np.ndarray:
        metrics = self.metrics()
        qpos = np.asarray([self.data.qpos[item] for item in self._qpos_addresses])
        qvel = np.asarray([self.data.qvel[item] for item in self._qvel_addresses])
        middle = JOINT_LIMITS.mean(axis=1)
        half_range = 0.5 * np.ptp(JOINT_LIMITS, axis=1)
        normalized_q = (qpos - middle) / half_range
        normalized_qd = np.clip(qvel / 12.0, -1.0, 1.0)
        return np.concatenate(
            (
                np.asarray(
                    [
                        metrics.height,
                        math.sin(metrics.pitch),
                        math.cos(metrics.pitch),
                        metrics.pitch_rate,
                        self._progress(),
                        0.0,
                        metrics.lateral_tilt,
                        float(self.data.qvel[3]),
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
        qpos = [float(self.data.qpos[item]) for item in self._qpos_addresses]
        return {
            "success": bool(success),
            "step": self.steps,
            "progress": self._progress(),
            "height": metrics.height,
            "pitch": metrics.pitch,
            "pitch_rate": metrics.pitch_rate,
            "lateral_tilt": metrics.lateral_tilt,
            "both_feet_contact": metrics.both_feet_contact,
            "other_body_support": metrics.other_body_support,
            "stable_time": self.stable_time,
            "joint_names": list(self.joint_names),
            "joint_positions": qpos,
            "failure_reason": "" if success else self.failure_reason(),
        }

    def failure_reason(self) -> str:
        if self.steps >= self.max_episode_steps:
            return "timeout before stable two-foot stance"
        if self.data.qpos[2] < 0.08:
            return "base dropped below safety height"
        return "episode running"

