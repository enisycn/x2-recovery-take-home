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
parser.add_argument("--environment", choices=("humanup_rise", "simple_v2", "symmetric_v3", "relaxed_v4"), default="humanup_rise")
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


from x2_recovery_isaac.simple_cfg import X2RelaxedRecoveryEnvCfg, X2SimpleRecoveryEnvCfg, X2SymmetricRecoveryEnvCfg



def _write(stream, event: dict) -> None:
    stream.write((json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8"))
    stream.flush()


def _policy_observation(observation):
    return observation if isinstance(observation, torch.Tensor) else observation["policy"]


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

    stable_steps = round(0.5 / unwrapped.step_dt)

    def measured_state(task, env_ids=None):
        return {
            "strict": bool(mdp.strict_success(task, feet_cfg=feet_cfg, all_bodies_cfg=all_bodies_cfg)[0]),
            "joint_positions": robot.data.joint_pos.torch[0].tolist(),
        }

    unwrapped.terminal_observer = measured_state
    for step in range(1, max_steps + 1):
        started = time.monotonic()
        with torch.no_grad():
            action = policy(_policy_observation(observation))
            observation, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated[0] or truncated[0])
            state = unwrapped.terminal_snapshot if done else measured_state(unwrapped)
            ready = state["strict"]
            if args.environment == "relaxed_v4":
                joints = dict(zip(robot.joint_names, state["joint_positions"], strict=True))
                ready = ready and all(
                    abs(joints[f"{side}_shoulder_pitch_joint"]) <= .30
                    and abs(joints[f"{side}_elbow_joint"] + .15) <= .30
                    for side in ("left", "right"))
            consecutive_stable = consecutive_stable + 1 if ready else 0
            _write(stream, {"type": "step", "step": step,
                "joint_names": list(robot.joint_names), "joint_positions": state["joint_positions"]})
            if consecutive_stable >= stable_steps:
                _write(stream, {"type": "result", "success": True, "steps": step, "failure_reason": ""})
                return
            if done:
                _write(stream, {"type": "result", "success": False, "steps": step,
                    "failure_reason": "safety termination" if bool(terminated[0]) else "episode timeout"})
                return
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

    # Keep the selected observation/action contract and a true supine reset.
    # v4 additionally waits for relaxed arm posture before reporting success.
    config = {"simple_v2": X2SimpleRecoveryEnvCfg, "symmetric_v3": X2SymmetricRecoveryEnvCfg, "relaxed_v4": X2RelaxedRecoveryEnvCfg, "humanup_rise": X2HumanUpRiseEnvCfg}[args.environment]()
    config.sim.device = args.device
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
