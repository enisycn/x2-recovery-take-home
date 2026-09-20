import numpy as np

from x2_recovery_ros.policy import SCRIPTED_ACTIONS, PhasePolicy
from x2_recovery_ros.reduced_env import ReducedOrderRecoveryEnv


def test_reset_is_on_back_and_clear_of_floor():
    env = ReducedOrderRecoveryEnv(seed=3)
    observation, info = env.reset(seed=3)
    assert observation.shape == (env.observation_size,)
    assert info["height"] >= 0.22
    assert abs(info["pitch"] + 0.5 * np.pi) < 0.05
    assert not info["both_feet_contact"]
    assert info["other_body_support"]


def test_actions_and_joints_are_clipped_to_limits():
    env = ReducedOrderRecoveryEnv(seed=4)
    env.step(np.full(4, 100.0))
    from x2_recovery_ros.constants import JOINT_LIMITS

    assert np.all(env.joint_positions >= JOINT_LIMITS[:, 0])
    assert np.all(env.joint_positions <= JOINT_LIMITS[:, 1])


def test_scripted_baseline_can_meet_full_success_predicate():
    env = ReducedOrderRecoveryEnv(seed=5)
    policy = PhasePolicy(SCRIPTED_ACTIONS)
    observation, _ = env.reset(seed=5)
    for _ in range(env.max_episode_steps):
        observation, _, terminated, truncated, info = env.step(policy.act(observation))
        if terminated or truncated:
            break
    assert info["success"]
    assert info["both_feet_contact"]
    assert not info["other_body_support"]


def test_zero_policy_times_out_without_false_success():
    env = ReducedOrderRecoveryEnv(seed=6, timeout_sec=1.0)
    env.reset(seed=6)
    for _ in range(env.max_episode_steps):
        _, _, terminated, truncated, info = env.step(np.zeros(4))
        if terminated or truncated:
            break
    assert not info["success"]
    assert info["failure_reason"] == "timeout before stable two-foot stance"

