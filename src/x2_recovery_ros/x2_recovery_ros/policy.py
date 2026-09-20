"""Compact three-phase recovery policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class PhasePolicy:
    """Selects one learned four-synergy action for each recovery phase."""

    def __init__(self, actions: np.ndarray) -> None:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (3, 4):
            raise ValueError(f"expected a (3, 4) action table, got {actions.shape}")
        self.actions = np.clip(actions, -1.0, 1.0)

    def act(self, observation: np.ndarray) -> np.ndarray:
        progress = float(observation[4])
        phase = 0 if progress < 0.32 else 1 if progress < 0.72 else 2
        return self.actions[phase].copy()

    def save(self, path: str | Path, **metadata: object) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, actions=self.actions, **metadata)

    @classmethod
    def load(cls, path: str | Path) -> "PhasePolicy":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(data["actions"])


SCRIPTED_ACTIONS = np.asarray(
    [
        [0.90, 0.75, -0.40, 0.05],
        [0.30, 0.50, 0.85, 0.25],
        [0.00, 0.00, 0.40, 0.85],
    ],
    dtype=np.float64,
)

