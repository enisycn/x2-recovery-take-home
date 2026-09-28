#!/usr/bin/env python3
"""Replot the committed reward trace for the selected all-supine policy."""

from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("reports/selected_supine_training_reward.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/selected_supine_training_reward.png"))
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    steps = [int(row["completed_updates"]) for row in rows]
    rewards = [float(row["mean_episode_return"]) for row in rows]
    stages = [row["stage"] for row in rows]
    if steps != list(range(1, 3861)):
        raise ValueError("The selected trace must contain exactly 3,860 ordered updates")

    smooth = []
    window: deque[float] = deque(maxlen=20)
    previous_stage = None
    boundaries = []
    for step, reward, stage in zip(steps, rewards, stages):
        if stage != previous_stage:
            window.clear()
            if previous_stage is not None:
                boundaries.append(step - 0.5)
        window.append(reward)
        smooth.append(sum(window) / len(window) if len(window) == 20 else float("nan"))
        previous_stage = stage

    fig, ax = plt.subplots(figsize=(12, 5.3))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.88, bottom=0.18)
    ax.plot(steps, rewards, color="#86b9a2", alpha=0.32, linewidth=0.65, label="Logged episode reward")
    ax.plot(steps, smooth, color="#137a47", linewidth=1.8, label="20-update mean within each stage")
    for boundary in boundaries:
        ax.axvline(boundary, color="#6b7280", linewidth=0.6, alpha=0.45)
    ax.axvline(2404.5, color="#1d4ed8", linewidth=1.25, alpha=0.85,
               label="Forward-arm continuation begins")
    ax.set(xlabel="Selected PPO updates from random initialization", ylabel="Mean episode reward",
           title="AgiBot X2: selected supine-only PPO lineage (3,860 updates)")
    ax.grid(alpha=0.15)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.text(0.5, 0.045,
             "Vertical lines mark resumes. Logger buffers reset and reward coefficients change; compare physical evaluations for success.",
             ha="center", fontsize=8, color="#475569")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
