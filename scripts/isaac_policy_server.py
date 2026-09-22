#!/usr/bin/env python3
"""Serve one X2 Isaac Lab policy over a local Unix socket for the ROS node."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", type=Path, required=True, help="Exported RSL-RL policy.pt")
parser.add_argument("--socket", default="/tmp/hrs_x2_recovery.sock")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import x2_recovery_isaac  # noqa: E402,F401
from x2_recovery_isaac import mdp  # noqa: E402
from x2_recovery_isaac.env_cfg import (  # noqa: E402
    X2HumanUpRiseEnvCfg,
    all_contact_cfg,
    foot_contact_cfg,
)


STABLE_STEPS = 10


def _write(stream, event: dict) -> None:
    stream.write((json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8"))
    stream.flush()


def _policy_observation(observation):
    return observation["policy"] if isinstance(observation, dict) else observation


def serve_attempt(stream, env, policy, request: dict, feet_cfg, all_bodies_cfg) -> None:
    if request.get("command") != "start":
        raise ValueError("expected command=start")
    seed = int(request["seed"])
    timeout_sec = float(request["timeout_sec"])
    real_time = bool(request.get("real_time", True))
    observation, _ = env.reset(seed=seed)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    max_steps = max(1, round(timeout_sec / unwrapped.step_dt))
    consecutive_stable = 0

    for step in range(1, max_steps + 1):
        started = time.monotonic()
        # no_grad keeps Isaac's action/history buffers mutable across resets.
        with torch.no_grad():
            instant = bool(
                mdp.strict_success(
                    unwrapped,
                    feet_cfg=feet_cfg,
                    all_bodies_cfg=all_bodies_cfg,
                )[0].item()
            )
            consecutive_stable = consecutive_stable + 1 if instant else 0
            _write(
                stream,
                {
                    "type": "step",
                    "step": step,
                    "joint_names": list(robot.joint_names),
                    "joint_positions": robot.data.joint_pos.torch[0].tolist(),
                },
            )
            if consecutive_stable >= STABLE_STEPS:
                _write(
                    stream,
                    {"type": "result", "success": True, "steps": step, "failure_reason": ""},
                )
                return
            action = policy(_policy_observation(observation))
            observation, _, _, _, _ = env.step(action)
        if real_time:
            time.sleep(max(0.0, unwrapped.step_dt - (time.monotonic() - started)))

    _write(
        stream,
        {
            "type": "result",
            "success": False,
            "steps": max_steps,
            "failure_reason": "timeout before 0.5 s strict stable stance",
        },
    )


def main() -> None:
    policy_path = args.policy.expanduser().resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"Exported policy not found: {policy_path}")
    socket_path = Path(args.socket).expanduser().resolve()
    socket_path.unlink(missing_ok=True)

    # Deployment uses the same 1,148-value HumanUP observation and bounded
    # relative action contract as training/evaluation.  Keep one deterministic
    # true-supine environment: the ROS bridge must report the learned result,
    # never a curriculum-reference start or an assisted attempt.
    config = X2HumanUpRiseEnvCfg()
    config.scene.num_envs = 1
    config.scene.env_spacing = 3.0
    config.observations.policy.enable_corruption = False
    config.events.material = None
    config.events.mass = None
    config.events.pelvis_com = None
    config.events.actuator_gains = None
    config.events.lift_assist = None
    config.events.reset_back_pose.params["reference_probability_start"] = 0.0
    config.events.reset_back_pose.params["reference_probability_end"] = 0.0
    env = gym.make("HRS-X2-Recovery-Play-v0", cfg=config)
    feet_cfg = foot_contact_cfg()
    all_bodies_cfg = all_contact_cfg()
    feet_cfg.resolve(env.unwrapped.scene)
    all_bodies_cfg.resolve(env.unwrapped.scene)
    policy = torch.jit.load(str(policy_path), map_location=env.unwrapped.device).eval()

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        server.listen(1)
        print(f"Isaac policy server ready: {socket_path}")
        try:
            while simulation_app.is_running():
                connection, _ = server.accept()
                with connection, connection.makefile("rwb") as stream:
                    try:
                        raw_request = stream.readline()
                        if not raw_request:
                            continue
                        serve_attempt(
                            stream,
                            env,
                            policy,
                            json.loads(raw_request.decode("utf-8")),
                            feet_cfg,
                            all_bodies_cfg,
                        )
                    except Exception as exc:
                        _write(
                            stream,
                            {
                                "type": "result",
                                "success": False,
                                "steps": 0,
                                "failure_reason": f"Isaac server error: {exc}",
                            },
                        )
        finally:
            env.close()
            socket_path.unlink(missing_ok=True)


try:
    main()
finally:
    simulation_app.close()
