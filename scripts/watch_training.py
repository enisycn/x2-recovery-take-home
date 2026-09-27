"""Show the latest complete PPO iteration from a live training console log."""

import argparse
from datetime import datetime
from pathlib import Path
import re
import sys
import tempfile
import time


FIELDS = (
    "Total steps", "Steps per second", "Mean reward", "Mean episode length",
    "Episode_Reward/feet", "Episode_Reward/balance", "Episode_Reward/strict_stance",
    "Collection time", "Learning time", "Iteration time", "Time elapsed", "ETA",
)


def latest_iteration(path: Path) -> str:
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        text = stream.read().decode(errors="replace")
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    blocks = re.split(r"(?m)^\s*Learning iteration ", text)[1:]
    complete = [block for block in blocks if re.search(r"(?m)^\s*ETA:", block)]
    if not complete:
        return "Waiting for the first complete PPO iteration..."
    lines = complete[-1].splitlines()
    metrics = {}
    for line in lines[1:]:
        key, separator, value = line.strip().partition(":")
        if separator and key in FIELDS:
            metrics[key] = value.strip()
    return "Learning iteration " + lines[0].strip() + "\n\n" + "\n".join(
        f"{key}: {metrics[key]}" for key in FIELDS if key in metrics
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", type=Path, help="Exact temporary console log; defaults to newest.")
    args = parser.parse_args()
    logs = list(Path(tempfile.gettempdir()).glob("hrs_x2_train.*.log"))
    path = args.log or (max(logs, key=lambda item: item.stat().st_mtime) if logs else None)
    if path is None:
        parser.error("No active training console log found. Start training first.")
    try:
        while True:
            try:
                summary = latest_iteration(path)
            except FileNotFoundError:
                print("Training console log removed; monitoring stopped. Check the training terminal for its outcome.")
                return
            if sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(f"PPO progress | {datetime.now():%H:%M:%S} | refresh: 5 s\n{path}\n\n{summary}", flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nMonitor stopped; training continues independently.")


if __name__ == "__main__":
    main()
