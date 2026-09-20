"""Manager-based Isaac Lab environment for X2 stand-up from its back."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_physx.sensors import ContactSensorCfg

from . import mdp
from .x2_robot_cfg import X2_CFG


FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]


@configclass
class X2RecoverySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(20.0, 20.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
        ),
    )
    robot: ArticulationCfg = X2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=750.0),
    )


@configclass
class ActionsCfg:
    # Bounded desired positions plus EMA give a compact hardware-conscious action space.
    joint_position = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.85,
        rescale_to_limits=True,
        alpha=0.25,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_height = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-0.01, n_max=0.01))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.10, n_max=0.10))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.02, n_max=0.02))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.10, noise=Unoise(n_min=-0.10, n_max=0.10))
        feet_contact = ObsTerm(
            func=mdp.foot_contacts,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)},
        )
        previous_actions = ObsTerm(func=mdp.last_action, history_length=2)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    # CPU property randomization runs once at scene creation, as Isaac Lab recommends.
    material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.70, 1.10),
            "dynamic_friction_range": (0.60, 1.00),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 32,
            "make_consistent": True,
        },
    )
    mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )
    pelvis_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "com_range": {"x": (-0.015, 0.015), "y": (-0.015, 0.015), "z": (-0.010, 0.010)},
        },
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.90, 1.10),
            "damping_distribution_params": (0.90, 1.10),
            "operation": "scale",
        },
    )
    reset_back_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.04, 0.04),
                "y": (-0.04, 0.04),
                "z": (-0.01, 0.01),
                "roll": (-0.06, 0.06),
                "pitch": (-0.08, 0.08),
                "yaw": (-0.10, 0.10),
            },
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.02, 0.02),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.04, 0.04), "velocity_range": (-0.05, 0.05)},
    )


@configclass
class RewardsCfg:
    stage_progress = RewTerm(func=mdp.staged_recovery_progress, weight=4.0, params={"target_height": 0.68})
    upright = RewTerm(func=mdp.upright_exp, weight=2.0, params={"std": 0.25})
    height = RewTerm(func=mdp.base_height_exp, weight=1.5, params={"target_height": 0.68, "std": 0.12})
    feet = RewTerm(
        func=mdp.both_feet_contact,
        weight=1.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET), "threshold": 15.0},
    )
    unsupported_support = RewTerm(
        func=mdp.unsupported_contacts,
        weight=-1.0,
        params={
            "all_bodies_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
            "feet_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "threshold": 15.0,
        },
    )
    stable = RewTerm(func=mdp.standing_still, weight=2.0, params={"target_height": 0.68})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.015)
    joint_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-2.0e-4)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-6)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.20)


@configclass
class TerminationsCfg:
    # Do not end when first upright: the policy must learn to remain stable.
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class X2RecoveryEnvCfg(ManagerBasedRLEnvCfg):
    scene: X2RecoverySceneCfg = X2RecoverySceneCfg(num_envs=2048, env_spacing=2.5, clone_in_fabric=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        # FRASA uses 20 Hz decisions; ten 200 Hz physics steps preserve contact fidelity.
        self.decimation = 10
        self.episode_length_s = 8.0
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 0.9
        self.sim.physics_material.dynamic_friction = 0.8
        self.sim.physics_material.restitution = 0.0
        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.6)


@configclass
class X2RecoveryPlayEnvCfg(X2RecoveryEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        # Evaluation is fixed-seed and deterministic apart from the physics solver.
        self.events.material = None
        self.events.mass = None
        self.events.pelvis_com = None
        self.events.actuator_gains = None
        self.events.reset_back_pose.params["pose_range"] = {
            key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_back_pose.params["velocity_range"] = {
            key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_joints.params["velocity_range"] = (0.0, 0.0)
