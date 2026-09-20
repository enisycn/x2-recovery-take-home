"""Five-episode evaluation with machine-readable output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .policy import PhasePolicy
from .reduced_env import ReducedOrderRecoveryEnv


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def evaluate(policy: PhasePolicy, seeds: list[int]) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for episode_index, seed in enumerate(seeds, start=1):
        env = ReducedOrderRecoveryEnv(seed=seed)
        observation, _ = env.reset(seed=seed)
        total_reward = 0.0
        while True:
            observation, reward, terminated, truncated, info = env.step(
                policy.act(observation)
            )
            total_reward += reward
            if terminated or truncated:
                break
        record = {
            "episode": episode_index,
            "seed": seed,
            "success": bool(info["success"]),
            "steps": int(info["step"]),
            "return": round(total_reward, 4),
            "final_height_m": round(float(info["height"]), 4),
            "final_pitch_rad": round(float(info["pitch"]), 4),
            "both_feet_contact": bool(info["both_feet_contact"]),
            "other_body_support": bool(info["other_body_support"]),
            "failure_reason": str(info["failure_reason"]),
        }
        episodes.append(record)
        print(
            f"episode={episode_index} seed={seed} success={record['success']} "
            f"steps={record['steps']} return={record['return']:.2f}"
        )
    return {
        "backend": "reduced_order_v1",
        "policy": "cross_entropy_phase_policy",
        "success_definition": {
            "pelvis_height_m_min": 0.66,
            "abs_pitch_rad_max": 0.14,
            "abs_pitch_rate_rad_s_max": 0.28,
            "abs_lateral_tilt_rad_max": 0.12,
            "both_feet_contact": True,
            "other_body_support": False,
            "stable_duration_sec": 0.4,
        },
        "successful_recoveries": sum(item["success"] for item in episodes),
        "total_episodes": len(episodes),
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    root = project_root()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "src/x2_recovery_ros/artifacts/recovery_policy.npz",
    )
    parser.add_argument("--output", type=Path, default=root / "reports/evaluation.json")
    args = parser.parse_args()

    result = evaluate(PhasePolicy.load(args.checkpoint), [101, 102, 103, 104, 105])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"successes={result['successful_recoveries']}/{result['total_episodes']} "
        f"report={args.output}"
    )


if __name__ == "__main__":
    main()
