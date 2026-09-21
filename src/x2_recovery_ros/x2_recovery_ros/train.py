"""Seeded cross-entropy policy search for the reduced-order environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .policy import SCRIPTED_ACTIONS, PhasePolicy
from .reduced_env import ReducedOrderRecoveryEnv


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def rollout(actions: np.ndarray, seed: int) -> tuple[float, bool]:
    env = ReducedOrderRecoveryEnv(seed=seed)
    policy = PhasePolicy(actions)
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    success = False
    while True:
        observation, reward, terminated, truncated, info = env.step(policy.act(observation))
        total_reward += reward
        if terminated or truncated:
            success = bool(info["success"])
            break
    return total_reward, success


def train(
    seed: int = 7,
    iterations: int = 60,
    population: int = 80,
    elite_count: int = 8,
) -> tuple[PhasePolicy, list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    mean = SCRIPTED_ACTIONS * 0.65
    std = np.full((3, 4), 0.45, dtype=np.float64)
    history: list[dict[str, float]] = []
    best_actions = mean.copy()
    best_score = float("-inf")

    for iteration in range(iterations):
        candidates = np.clip(
            rng.normal(mean, std, size=(population, 3, 4)), -1.0, 1.0
        )
        scores = np.empty(population, dtype=np.float64)
        successes = np.zeros(population, dtype=np.float64)
        for candidate_index, candidate in enumerate(candidates):
            outcomes = [
                rollout(candidate, seed=10_000 + iteration * 1_000 + candidate_index * 10 + offset)
                for offset in range(2)
            ]
            scores[candidate_index] = np.mean([outcome[0] for outcome in outcomes])
            successes[candidate_index] = np.mean([outcome[1] for outcome in outcomes])

        elite_indices = np.argsort(scores)[-elite_count:]
        elites = candidates[elite_indices]
        iteration_best = int(np.argmax(scores))
        if scores[iteration_best] > best_score:
            best_score = float(scores[iteration_best])
            best_actions = candidates[iteration_best].copy()
        mean = 0.25 * mean + 0.75 * elites.mean(axis=0)
        std = np.maximum(0.05, 0.30 * std + 0.70 * elites.std(axis=0))
        record = {
            "iteration": float(iteration),
            "mean_reward": float(scores.mean()),
            "best_reward": float(scores.max()),
            "population_success_rate": float(successes.mean()),
        }
        history.append(record)
        print(
            f"iteration={iteration:02d} mean={record['mean_reward']:.2f} "
            f"best={record['best_reward']:.2f} success={record['population_success_rate']:.2f}"
        )

    candidate_pool = [mean, best_actions]
    validation_seeds = [90_001, 90_002, 90_003, 90_004, 90_005]
    validation_scores = []
    for candidate in candidate_pool:
        outcomes = [rollout(candidate, seed=item) for item in validation_seeds]
        success_rate = np.mean([outcome[1] for outcome in outcomes])
        mean_return = np.mean([outcome[0] for outcome in outcomes])
        validation_scores.append(1_000.0 * success_rate + mean_return)
    selected = int(np.argmax(validation_scores))
    print(f"selected_candidate={selected} validation_score={validation_scores[selected]:.2f}")
    return PhasePolicy(candidate_pool[selected]), history


def save_plot(history: list[dict[str, float]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    iterations = [int(item["iteration"]) for item in history]
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(iterations, [item["mean_reward"] for item in history], label="population mean")
    plt.plot(iterations, [item["best_reward"] for item in history], label="iteration best")
    plt.xlabel("CEM iteration")
    plt.ylabel("episode return")
    plt.title("X2 reduced-order recovery policy search")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--population", type=int, default=80)
    args = parser.parse_args()

    root = project_root()
    policy, history = train(
        seed=args.seed, iterations=args.iterations, population=args.population
    )
    checkpoint = root / "src/x2_recovery_ros/artifacts/recovery_policy.npz"
    policy.save(
        checkpoint,
        algorithm=np.asarray("cross_entropy_method"),
        seed=np.asarray(args.seed),
        iterations=np.asarray(args.iterations),
        population=np.asarray(args.population),
        elite_count=np.asarray(8),
        rollouts_per_candidate=np.asarray(2),
        backend=np.asarray("reduced_order_v1"),
    )
    save_plot(history, root / "reports/training_reward.png")
    print(f"checkpoint={checkpoint}")
    print(f"plot={root / 'reports/training_reward.png'}")


if __name__ == "__main__":
    main()
