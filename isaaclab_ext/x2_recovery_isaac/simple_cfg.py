"""Version 2: continuous observations and absolute, unsuppressed joint control.

HumanUP RSS 2025 motivates weakly regularized discovery and absolute targets.
The compact reward set and X2-specific scales below are adaptations, not a
claim to reproduce the complete HumanUP method.
"""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from . import mdp
from .env_cfg import (X2RecoveryEnvCfg, HumanUpActionsCfg, HumanUpTerminationsCfg,
                      foot_contact_cfg, all_contact_cfg, _humanup_head_cfg)
from .agents.rsl_rl_ppo_cfg import X2RecoveryPPORunnerCfg


@configclass
class SimpleRewardsCfg:
    pelvis_height = RewTerm(func=mdp.humanup_base_height_exp_clipped, weight=10.0,
                            params={"target_height": 0.68})
    head_height = RewTerm(func=mdp.humanup_head_height_exp_clipped, weight=5.0,
                          params={"target_height": 1.2, "asset_cfg": _humanup_head_cfg()})
    upright = RewTerm(func=mdp.humanup_body_upright, weight=2.0)
    feet = RewTerm(func=mdp.both_feet_when_high, weight=5.0,
        params={"sensor_cfg": foot_contact_cfg(), "threshold": 15.0,
                "gate_start_height": 0.30, "target_height": 0.68})
    other_support = RewTerm(func=mdp.unsupported_contacts_when_high, weight=-1.0,
        params={"all_bodies_cfg": all_contact_cfg(), "feet_cfg": foot_contact_cfg(),
                "threshold": 15.0, "gate_start_height": 0.50, "target_height": 0.68})
    balance = RewTerm(func=mdp.standing_still, weight=10.0, params={"target_height": 0.68})
    strict_stance = RewTerm(func=mdp.strict_success, weight=20.0,
        params={"feet_cfg": foot_contact_cfg(), "all_bodies_cfg": all_contact_cfg()})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    failure = RewTerm(func=mdp.is_terminated, weight=-10.0)


@configclass
class X2SimpleRecoveryEnvCfg(X2RecoveryEnvCfg):
    actions: HumanUpActionsCfg = HumanUpActionsCfg()
    rewards: SimpleRewardsCfg = SimpleRewardsCfg()
    terminations: HumanUpTerminationsCfg = HumanUpTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_position.scale = 0.75
        # 200 Hz PhysX and 50 Hz policy: no state-dependent action brake.
        self.sim.dt = 0.005
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0
        self.observations.policy.enable_corruption = False
        self.events.lift_assist = None
        p = self.events.reset_back_pose.params
        # Every training stage includes actual back-lying starts. Upright-root
        # squat references are explicitly auxiliary starts, not back recovery.
        p['reference_probability_start'] = 0.5
        p['reference_probability_end'] = 0.5
        p['reference_probability_anneal_policy_steps'] = 1
        p['reference_min_stage'] = 0
        p['reference_max_stage_start'] = 8
        p['reference_max_stage_end'] = 8
        p['reference_stage_anneal_policy_steps'] = 1


@configclass
class X2SimplePPORunnerCfg(X2RecoveryPPORunnerCfg):
    experiment_name = 'hrs_x2_simple_v2'
    num_steps_per_env = 32
    save_interval = 100

    def __post_init__(self):
        self.actor.distribution_cfg.init_std = 1.0
        self.algorithm.entropy_coef = 0.005
        self.algorithm.learning_rate = 3.0e-4


from .synergy_action import SymmetricRecoveryActionCfg


def signed_height_progress(env):
    robot = env.scene['robot']
    height = (robot.data.root_pos_w.torch[:, 2] / .68).clamp(0., 1.)
    # Supine retains a height gradient (gate=.5). Inverted height is worthless.
    orientation = ((1. - robot.data.projected_gravity_b.torch[:, 2]) * .5).clamp(0., 1.)
    return height * orientation


@configclass
class SymmetricActionsCfg:
    joint_position: SymmetricRecoveryActionCfg = SymmetricRecoveryActionCfg()


@configclass
class X2SymmetricRecoveryEnvCfg(X2SimpleRecoveryEnvCfg):
    actions: SymmetricActionsCfg = SymmetricActionsCfg()

    def __post_init__(self):
        # Use the same simulator/reset setup without the absolute-action scale.
        X2RecoveryEnvCfg.__post_init__(self)
        self.sim.dt=.005; self.decimation=4; self.sim.render_interval=4
        self.episode_length_s=10.
        self.observations.policy.enable_corruption=False
        self.events.lift_assist=None
        p=self.events.reset_back_pose.params
        p.update(reference_probability_start=.5, reference_probability_end=.5,
                 reference_probability_anneal_policy_steps=1, reference_min_stage=0,
                 reference_max_stage_start=8, reference_max_stage_end=8,
                 reference_stage_anneal_policy_steps=1)
        self.rewards.pelvis_height=RewTerm(func=signed_height_progress,weight=40.)
        self.rewards.upright.weight=5.


@configclass
class X2SymmetricPPORunnerCfg(X2SimplePPORunnerCfg):
    experiment_name='hrs_x2_symmetric_v3'


@configclass
class X2RelaxedRecoveryEnvCfg(X2SymmetricRecoveryEnvCfg):
    """Preserve v3 recovery, refine only stable final arm posture."""

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import SceneEntityCfg
        # The submitted task starts EVERY episode supine (PDF requirement).
        self.events.reset_back_pose.params["reference_probability_start"] = 0.0
        self.events.reset_back_pose.params["reference_probability_end"] = 0.0
        self.rewards.relaxed_arms = RewTerm(
            func=mdp.relaxed_arms_when_stable, weight=40.0,
            params={
                "shoulder_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_pitch_joint"]),
                "elbow_cfg": SceneEntityCfg("robot", joint_names=[".*_elbow_joint"]),
                "feet_cfg": foot_contact_cfg(), "all_bodies_cfg": all_contact_cfg(),
                "variance": 2.0,
            })


@configclass
class X2RelaxedPPORunnerCfg(X2SymmetricPPORunnerCfg):
    experiment_name = "hrs_x2_relaxed_v4"
    save_interval = 50

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.schedule = "adaptive"
        self.algorithm.entropy_coef = 0.005
