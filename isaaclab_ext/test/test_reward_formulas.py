"""Numerical checks for the Isaac recovery reward and success equations."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from isaaclab.utils import math as math_utils
from x2_recovery_isaac import mdp


class TorchField:
    def __init__(self, value: torch.Tensor) -> None:
        self.torch = value


class FakeScene(dict):
    def __init__(self, robot, contact_sensor=None) -> None:
        super().__init__(robot=robot)
        self.sensors = {}
        if contact_sensor is not None:
            self.sensors["contact_forces"] = contact_sensor


def fake_robot(
    heights: torch.Tensor,
    projected_gravity: torch.Tensor,
    linear_velocity: torch.Tensor | None = None,
    angular_velocity: torch.Tensor | None = None,
):
    count = heights.shape[0]
    root_position = torch.zeros((count, 3), dtype=torch.float32)
    root_position[:, 2] = heights
    zeros = torch.zeros((count, 3), dtype=torch.float32)
    return SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=TorchField(root_position),
            projected_gravity_b=TorchField(projected_gravity),
            root_lin_vel_w=TorchField(zeros if linear_velocity is None else linear_velocity),
            root_ang_vel_w=TorchField(zeros if angular_velocity is None else angular_velocity),
        )
    )


def test_minus_ninety_degree_pitch_is_supine() -> None:
    """The configured quaternion maps X-forward/chest to world-up."""

    w, x, y, z = 2.0**-0.5, 0.0, -(2.0**-0.5), 0.0
    rotation = torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    world_forward = rotation @ torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(world_forward, torch.tensor([0.0, 0.0, 1.0]), atol=1.0e-6)


def test_staged_reward_matches_righting_rising_standing_equations() -> None:
    heights = torch.tensor([0.28, 0.50, 0.68])
    gravity = torch.tensor([[1.0, 0.0, 0.0], [0.8660254, 0.0, -0.5], [0.0, 0.0, -1.0]])
    env = SimpleNamespace(scene=FakeScene(fake_robot(heights, gravity)))

    actual = mdp.staged_recovery_progress(env, target_height=0.68)
    expected = torch.tensor(
        [
            0.65 * 0.5 + 0.35 * (0.28 / 0.68),
            0.45 * 0.75 + 0.55 * (0.50 / 0.68),
            0.30 + 0.70,
        ]
    )
    assert torch.allclose(actual, expected, atol=1.0e-6)
    assert actual[0] < actual[1] < actual[2]


def test_upright_and_height_exponentials_peak_at_the_target() -> None:
    heights = torch.tensor([0.68, 0.50])
    gravity = torch.tensor([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    env = SimpleNamespace(scene=FakeScene(fake_robot(heights, gravity)))

    upright = mdp.upright_exp(env, std=0.25)
    height = mdp.base_height_exp(env, target_height=0.68, std=0.12)
    assert torch.isclose(upright[0], torch.tensor(1.0))
    assert upright[1] < 1.0e-6
    assert torch.isclose(height[0], torch.tensor(1.0))
    assert height[1] < height[0]


def test_strict_success_rejects_inversion_missing_foot_and_other_support() -> None:
    count, history, bodies = 4, 3, 3
    heights = torch.full((count,), 0.68)
    gravity = torch.tensor(
        [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]
    )
    forces = torch.zeros((count, history, bodies, 3))
    # Equality is intentional: feet at the threshold count as contact, while
    # a non-foot body at the same threshold invalidates success.
    forces[:, :, 0, 2] = 15.0
    forces[:, :, 1, 2] = 15.0
    forces[2, :, 2, 2] = 15.0
    forces[3, :, 1, 2] = 0.0
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w_history=TorchField(forces)))
    env = SimpleNamespace(scene=FakeScene(fake_robot(heights, gravity), sensor))
    feet = SimpleNamespace(name="contact_forces", body_ids=[0, 1])
    all_bodies = SimpleNamespace(name="contact_forces", body_ids=[0, 1, 2])

    result = mdp.strict_success(env, feet_cfg=feet, all_bodies_cfg=all_bodies)
    assert result.tolist() == [True, False, False, False]


def test_action_scale_limit_mapping_and_ema_are_bounded() -> None:
    """Check raw action -> central 85% of soft limits -> alpha=.25 EMA."""

    raw = torch.tensor([[-1.0, 1.0]])
    lower = torch.tensor([[-2.0, -2.0]])
    upper = torch.tensor([[2.0, 2.0]])
    target = math_utils.unscale_transform((0.85 * raw).clamp(-1.0, 1.0), lower, upper)
    filtered = 0.25 * target + 0.75 * torch.zeros_like(target)

    assert torch.allclose(target, torch.tensor([[-1.7, 1.7]]), atol=1.0e-6)
    assert torch.allclose(filtered, torch.tensor([[-0.425, 0.425]]), atol=1.0e-6)
    assert torch.all(target >= lower) and torch.all(target <= upper)
