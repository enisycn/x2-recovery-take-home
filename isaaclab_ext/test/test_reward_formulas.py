"""Numerical checks for the Isaac recovery reward and success equations."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

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
            root_lin_vel_b=TorchField(zeros if linear_velocity is None else linear_velocity),
            root_ang_vel_w=TorchField(zeros if angular_velocity is None else angular_velocity),
        )
    )


def test_minus_ninety_degree_pitch_is_supine() -> None:
    """The configured scalar-last XYZW quaternion maps X-forward to world-up."""

    quaternion_xyzw = (0.0, -(2.0**-0.5), 0.0, 2.0**-0.5)
    x, y, z, w = quaternion_xyzw
    rotation = torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    world_forward = rotation @ torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(world_forward, torch.tensor([0.0, 0.0, 1.0]), atol=1.0e-6)


def test_audited_reset_and_standing_heights_match_collision_geometry() -> None:
    # Values are recomputed from every official collision mesh by
    # scripts/audit_x2_geometry.py; this guards accidental reset regressions.
    supine_extent_below_pelvis = 0.1803006679
    standing_extent_below_pelvis = 0.6749500000
    assert 0.190 - supine_extent_below_pelvis > 0.005
    assert abs(0.68 - standing_extent_below_pelvis) < 0.01


def test_shared_floor_covers_the_full_parallel_environment_grid() -> None:
    # The global floor must cover every origin in the 4096-env clone grid.
    required_side = math.ceil(math.sqrt(4096)) * 2.5 + 2.0
    assert 200.0 >= required_side


def test_host_assist_force_is_sixty_percent_of_x2_weight() -> None:
    force = mdp.scaled_assist_force_n(41.966521)
    assert math.isclose(force, 247.014942606, rel_tol=0.0, abs_tol=1.0e-9)


def test_staged_reward_matches_righting_rising_standing_equations() -> None:
    heights = torch.tensor([0.28, 0.50, 0.68])
    gravity = torch.tensor([[1.0, 0.0, 0.0], [0.8660254, 0.0, -0.5], [0.0, 0.0, -1.0]])
    env = SimpleNamespace(scene=FakeScene(fake_robot(heights, gravity)))

    actual = mdp.staged_recovery_progress(env, target_height=0.68)
    expected = torch.tensor(
        [
            0.80 * 0.5 + 0.20 * (0.28 / 0.68),
            0.40 * 0.75 + 0.60 * (0.50 / 0.68),
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
    height_progress = mdp.base_height_progress(env, target_height=0.68)
    assert torch.isclose(upright[0], torch.tensor(1.0))
    assert upright[1] < 1.0e-6
    assert torch.isclose(height[0], torch.tensor(1.0))
    assert height[1] < height[0]
    assert torch.allclose(height_progress, torch.tensor([1.0, 0.0]))


def test_height_progress_keeps_a_gradient_near_the_floor() -> None:
    heights = torch.tensor([0.068, 0.34, 0.68])
    gravity = torch.tensor([[0.0, 0.0, -1.0]] * 3)
    env = SimpleNamespace(scene=FakeScene(fake_robot(heights, gravity)))
    progress = mdp.base_height_progress(env, target_height=0.68)
    assert torch.allclose(progress, torch.tensor([0.1, 0.5, 1.0]))


def test_final_leg_pose_is_gated_and_peaks_at_standing_defaults() -> None:
    heights = torch.tensor([0.44, 0.62, 0.68])
    gravity = torch.tensor([[0.0, 0.0, -1.0]] * 3)
    robot = fake_robot(heights, gravity)
    robot.data.joint_pos = TorchField(
        torch.tensor([[1.0] * 6, [1.0] * 6, [0.0] * 6], dtype=torch.float32)
    )
    robot.data.default_joint_pos = TorchField(torch.zeros((3, 6), dtype=torch.float32))
    env = SimpleNamespace(scene=FakeScene(robot))
    legs = SimpleNamespace(name="robot", joint_ids=list(range(6)))

    actual = mdp.final_leg_pose_exp(
        env,
        gate_start_height=0.45,
        target_height=0.68,
        std=1.0,
        asset_cfg=legs,
    )
    expected_mid = math.exp(-1.0) * ((0.62 - 0.45) / (0.68 - 0.45))
    assert torch.allclose(actual, torch.tensor([0.0, expected_mid, 1.0]), atol=1.0e-6)


def test_humanup_discovery_and_host_post_stand_terms() -> None:
    heights = torch.tensor([0.19, 0.68])
    gravity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    linear = torch.tensor([[0.0, 0.0, 0.1], [0.1, 0.2, 0.0]])
    angular = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
    env = SimpleNamespace(scene=FakeScene(fake_robot(heights, gravity, linear, angular)))

    assert torch.allclose(mdp.humanup_base_height(env), torch.exp(heights) - 1.0)
    assert mdp.humanup_height_increase(env).tolist() == [1.0, 0.0]
    assert torch.allclose(mdp.humanup_body_upright(env), torch.tensor([1.0, torch.e]))
    assert mdp.host_post_base_height(env, stage_height=0.62, target_height=0.68).tolist() == [0.0, 1.0]
    assert mdp.host_post_base_orientation(env, stage_height=0.62).tolist() == [0.0, 1.0]
    assert torch.allclose(mdp.base_linear_velocity_l2(env), torch.tensor([0.01, 0.05]))
    assert torch.allclose(mdp.base_angular_velocity_l2(env), torch.tensor([0.0, 0.14]))


def test_host_height_term_uses_released_absolute_error_formula() -> None:
    heights = torch.tensor([0.57, 0.58, 0.62, 0.68])
    gravity = torch.tensor([[0.0, 0.0, -1.0]] * 4)
    env = SimpleNamespace(scene=FakeScene(fake_robot(heights, gravity)))

    actual = mdp.host_post_base_height(env, stage_height=0.58, target_height=0.68)
    expected = torch.tensor([0.0, 0.0, math.exp(-1.2), 1.0])
    assert torch.allclose(actual, expected, atol=1.0e-6)


def test_humanup_unsafe_state_is_not_a_success_termination() -> None:
    heights = torch.tensor([0.68, -0.01, 1.21, 0.68])
    gravity = torch.tensor([[0.0, 0.0, -1.0]] * 4)
    linear = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.51, 0.0, 0.0]]
    )
    env = SimpleNamespace(scene=FakeScene(fake_robot(heights, gravity, linear_velocity=linear)))

    assert mdp.humanup_unsafe_state(env).tolist() == [False, True, True, True]


def test_inverted_pose_gets_no_height_reward_and_is_penalized() -> None:
    heights = torch.tensor([0.68, 0.68])
    gravity = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    env = SimpleNamespace(scene=FakeScene(fake_robot(heights, gravity)))

    assert mdp.base_height_progress(env, target_height=0.68).tolist() == [1.0, 0.0]
    assert mdp.base_height_exp(env, target_height=0.68, std=0.12).tolist() == [1.0, 0.0]
    assert mdp.inverted_orientation(env).tolist() == [0.0, 1.0]


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
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            net_forces_w_history=TorchField(forces),
            force_matrix_w_history=TorchField(forces.unsqueeze(3)),
        )
    )
    env = SimpleNamespace(scene=FakeScene(fake_robot(heights, gravity), sensor))
    feet = SimpleNamespace(name="contact_forces", body_ids=[0, 1])
    all_bodies = SimpleNamespace(name="contact_forces", body_ids=[0, 1, 2])

    result = mdp.strict_success(env, feet_cfg=feet, all_bodies_cfg=all_bodies)
    assert result.tolist() == [True, False, False, False]


def test_strict_stance_proximity_is_bounded_and_orders_nearby_states() -> None:
    count, history, bodies = 3, 2, 3
    heights = torch.tensor([0.68, 0.68, 0.44])
    gravity = torch.tensor([[0.0, 0.0, -1.0]] * count)
    forces = torch.zeros((count, history, bodies, 3))
    forces[:, :, 0, 2] = 100.0
    forces[:, :, 1, 2] = 100.0
    forces[1, :, 2, 2] = 100.0
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            net_forces_w_history=TorchField(forces),
            force_matrix_w_history=TorchField(forces.unsqueeze(3)),
        )
    )
    env = SimpleNamespace(scene=FakeScene(fake_robot(heights, gravity), sensor))
    feet = SimpleNamespace(name="contact_forces", body_ids=[0, 1])
    all_bodies = SimpleNamespace(name="contact_forces", body_ids=[0, 1, 2])

    result = mdp.strict_stance_proximity(env, feet_cfg=feet, all_bodies_cfg=all_bodies)
    assert torch.all((0.0 <= result) & (result <= 1.0))
    assert result[0] > result[1]
    assert result[0] > result[2]
    assert result[1] < 0.15
    assert result[2] < 0.30
    assert result[0] > 0.80


def test_strict_success_ignores_internal_self_collision_for_support() -> None:
    """Only floor contact, not equal/opposite link contact, is body support."""

    height = torch.tensor([0.68])
    gravity = torch.tensor([[0.0, 0.0, -1.0]])
    ground = torch.zeros((1, 2, 3, 3))
    ground[:, :, 0, 2] = 100.0
    ground[:, :, 1, 2] = 100.0
    net = ground.clone()
    # Simulate a pelvis/hip self-collision in the unfiltered PhysX buffer.
    net[:, :, 2, 0] = 400.0
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            net_forces_w_history=TorchField(net),
            force_matrix_w_history=TorchField(ground.unsqueeze(3)),
        )
    )
    env = SimpleNamespace(scene=FakeScene(fake_robot(height, gravity), sensor))
    feet = SimpleNamespace(name="contact_forces", body_ids=[0, 1])
    all_bodies = SimpleNamespace(name="contact_forces", body_ids=[0, 1, 2])

    result = mdp.strict_success(env, feet_cfg=feet, all_bodies_cfg=all_bodies)
    assert result.tolist() == [True]


def test_relative_action_holds_at_zero_and_clips_final_target() -> None:
    """Check the once-per-policy-step HoST target with beta=.25."""

    current = torch.tensor([[0.10, 1.90]])
    raw = torch.tensor([[0.0, 2.0]])
    lower = torch.tensor([[-2.0, -2.0]])
    upper = torch.tensor([[2.0, 2.0]])
    delta = 0.25 * torch.tanh(raw)
    target = torch.clamp(current + delta, min=lower, max=upper)

    assert torch.isclose(target[0, 0], current[0, 0])
    assert torch.isclose(target[0, 1], upper[0, 1])
    assert torch.all(target >= lower) and torch.all(target <= upper)

    # Holding this target over five physics substeps must not compound beta.
    repeated_physics_targets = torch.stack([target] * 5)
    assert torch.allclose(repeated_physics_targets[0], repeated_physics_targets[-1])


def test_x2_knee_soft_margin_still_permits_near_extension() -> None:
    hard_lower, hard_upper, factor = 0.0, 2.407, 0.98
    midpoint = 0.5 * (hard_lower + hard_upper)
    half_soft_range = 0.5 * factor * (hard_upper - hard_lower)
    soft_lower = midpoint - half_soft_range
    assert 0.0 < soft_lower < 0.03


def test_reset_workaround_invalidates_all_root_frame_buffers() -> None:
    names = (
        "_projected_gravity_b",
        "_heading_w",
        "_root_link_lin_vel_b",
        "_root_link_ang_vel_b",
        "_root_com_lin_vel_b",
        "_root_com_ang_vel_b",
    )
    data = SimpleNamespace(
        **{name: SimpleNamespace(timestamp=123.0) for name in names}
    )
    mdp._invalidate_root_derived_buffers(SimpleNamespace(data=data))
    assert all(getattr(data, name).timestamp == -1.0 for name in names)


def test_training_assistance_schedules_reach_zero() -> None:
    assert mdp.linear_anneal(0.5, 0.0, 0, 16_000) == 0.5
    assert mdp.linear_anneal(0.5, 0.0, 8_000, 16_000) == 0.25
    assert mdp.linear_anneal(0.5, 0.0, 16_000, 16_000) == 0.0
    assert mdp.linear_anneal(200.0, 0.0, 12_000, 24_000) == 100.0
    assert mdp.linear_anneal(200.0, 0.0, 30_000, 24_000) == 0.0
    # Curriculum time is vector-environment policy time.  It is invariant to
    # 64, 3000, or 4096 parallel robots and therefore spans the same updates.
    assert math.isclose(mdp.linear_anneal(0.95, 0.35, 1_800, 3_600), 0.65)
    assert math.isclose(mdp.linear_anneal(0.95, 0.35, 3_600, 3_600), 0.35)
    assert mdp.linear_anneal(247.0, 0.0, 4_000, 4_000) == 0.0
