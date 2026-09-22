#!/usr/bin/env python3
"""Generate the standard graph, CSV and file index for one completed run."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


CHECKPOINT_PATTERN = re.compile(r"model_(\d+)\.pt$")


def checkpoint_iteration(path: Path) -> int:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else -1


def relative_files(run_dir: Path, pattern: str) -> list[str]:
    return [str(path.relative_to(run_dir)) for path in sorted(run_dir.glob(pattern))]


def write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve(strict=True)
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)

    plot_path = run_dir / "reward.png"
    csv_path = run_dir / "reward.csv"
    plot_script = Path(__file__).with_name("plot_isaac_training.py")
    subprocess.run(
        [
            sys.executable,
            str(plot_script),
            "--final",
            str(run_dir),
            "--output",
            str(plot_path),
            "--csv",
            str(csv_path),
            "--title",
            f"AgiBot X2 PPO training: {run_dir.name}",
            "--label",
            "This run",
        ],
        check=True,
    )

    checkpoints = sorted(run_dir.glob("model_*.pt"), key=checkpoint_iteration)
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file found in {run_dir}")

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": run_dir.parent.name,
        "run": run_dir.name,
        "reward_plot": plot_path.name,
        "reward_csv": csv_path.name,
        "tensorboard_events": [path.name for path in event_files],
        "checkpoints": [
            {
                "file": path.name,
                "iteration": checkpoint_iteration(path),
                "size_bytes": path.stat().st_size,
            }
            for path in checkpoints
        ],
        "latest_checkpoint": checkpoints[-1].name if checkpoints else None,
        "parameters": relative_files(run_dir, "params/*"),
    }
    manifest_path = run_dir / "run_manifest.json"
    write_text_atomic(manifest_path, json.dumps(manifest, indent=2) + "\n")
    write_text_atomic(run_dir.parent / "LATEST_RUN.txt", run_dir.name + "\n")

    print(f"[HRS] Per-run reward plot: {plot_path}")
    print(f"[HRS] Per-run reward CSV: {csv_path}")
    print(f"[HRS] Run manifest: {manifest_path}")
    print(f"[HRS] Latest-run pointer: {run_dir.parent / 'LATEST_RUN.txt'}")


if __name__ == "__main__":
    main()
