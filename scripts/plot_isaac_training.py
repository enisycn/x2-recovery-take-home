#!/usr/bin/env python3
"""Plot measured RSL-RL reward from the final X2 run and optional diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def scalars(run: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    events = EventAccumulator(str(run), size_guidance={"scalars": 0})
    events.Reload()
    values = events.Scalars(tag)
    if not values:
        raise RuntimeError(f"missing TensorBoard tag {tag!r} in {run}")
    return (
        np.asarray([value.step for value in values], dtype=int),
        np.asarray([value.value for value in values], dtype=float),
    )


def moving_average(values: np.ndarray, window: int = 20) -> np.ndarray:
    if values.size < window:
        return values.copy()
    weights = np.ones(window) / window
    valid = np.convolve(values, weights, mode="valid")
    return np.concatenate((np.full(window - 1, np.nan), valid))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--corrected", type=Path)
    parser.add_argument("--orientation-gated", type=Path)
    parser.add_argument("--max-iteration", type=int, default=None)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/isaac_training_reward.png"))
    parser.add_argument("--csv", type=Path, default=Path("reports/isaac_training_reward.csv"))
    args = parser.parse_args()

    runs = []
    if args.initial is not None:
        runs.append(("initial", args.initial, "#1d4ed8"))
    if args.corrected is not None:
        runs.append(("height_only_branch", args.corrected, "#c2410c"))
    if args.orientation_gated is not None:
        runs.append(("orientation_gated_diagnostic", args.orientation_gated, "#7c3aed"))
    runs.append(("final_audited", args.final, "#15803d"))
    series = [(label, *scalars(path, "Train/mean_reward"), color) for label, path, color in runs]

    if args.max_iteration is not None:
        series = [(label, steps[steps <= args.max_iteration], rewards[steps <= args.max_iteration], color)
                  for label, steps, rewards, color in series]

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as stream:
        # Keep the committed artifact platform-neutral and friendly to Git's
        # whitespace checks instead of csv.writer's default CRLF dialect.
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("phase", "iteration", "mean_reward"))
        for label, steps, rewards, _ in series:
            writer.writerows((label, int(step), float(value)) for step, value in zip(steps, rewards))

    figure, axis = plt.subplots(figsize=(10.5, 5.6), constrained_layout=True)
    labels = {
        "initial": "Initial reward",
        "height_only_branch": "Height-only branch (discarded)",
        "orientation_gated_diagnostic": "Orientation-gated diagnostic",
        "final_audited": "Final audited run",
    }
    for label, steps, rewards, color in series:
        axis.plot(steps, rewards, color=color, alpha=0.16, linewidth=0.8)
        axis.plot(steps, moving_average(rewards), color=color, linewidth=2.2, label=f"{labels[label]} (20-iteration mean)")
    axis.set(title="AgiBot X2 recovery PPO training", xlabel="PPO iteration", ylabel="Mean episode reward")
    axis.grid(alpha=0.20)
    axis.legend(frameon=False, loc="best")
    figure.savefig(args.output, dpi=180)
    print(f"wrote {args.output} and {args.csv}")


if __name__ == "__main__":
    main()
