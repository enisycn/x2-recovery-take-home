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
from .exact_contact_sensor import ExactPathContactSensor
from .relative_action import BoundedRelativeJointPositionActionCfg
from .x2_robot_cfg import X2_CFG


MAX_CONFIGURED_ENVS = 2048
ENV_SPACING_M = 2.5
GROUND_SIDE_M = 200.0
_required_ground_side = math.ceil(math.sqrt(MAX_CONFIGURED_ENVS)) * ENV_SPACING_M + 2.0
if GROUND_SIDE_M < _required_ground_side:
    raise ValueError(
        f"ground side {GROUND_SIDE_M} m does not cover the configured environment grid "
        f"({_required_ground_side:.1f} m required)"
    )


FOOT_CONTACT_SENSORS = (
    "contact_left_ankle_roll_link",
    "contact_right_ankle_roll_link",
)
ALL_CONTACT_SENSORS = (
    "contact_pelvis",
    "contact_left_hip_pitch_link",
    "contact_left_hip_roll_link",
    "contact_left_hip_yaw_link",
    "contact_left_knee_link",
    "contact_left_ankle_pitch_link",
    "contact_left_ankle_roll_link",
    "contact_right_hip_pitch_link",
    "contact_right_hip_roll_link",
    "contact_right_hip_yaw_link",
    "contact_right_knee_link",
    "contact_right_ankle_pitch_link",
    "contact_right_ankle_roll_link",
    "contact_waist_yaw_link",
    "contact_waist_pitch_link",
    "contact_torso_link",
    "contact_left_shoulder_pitch_link",
    "contact_left_shoulder_roll_link",
    "contact_left_shoulder_yaw_link",
    "contact_left_elbow_link",
    "contact_left_wrist_yaw_link",
    "contact_left_wrist_pitch_link",
    "contact_left_wrist_roll_link",
    "contact_right_shoulder_pitch_link",
    "contact_right_shoulder_roll_link",
    "contact_right_shoulder_yaw_link",
    "contact_right_elbow_link",
    "contact_right_wrist_yaw_link",
    "contact_right_wrist_pitch_link",
    "contact_right_wrist_roll_link",
    "contact_head_yaw_link",
    "contact_head_pitch_link",
)

# Sum of the masses in the pinned official X2 Ultra v1.3.0 URDF. HoST's
# official cross-robot guidance recommends a pull near 60% of robot weight.
X2_TOTAL_MASS_KG = 41.966521
HOST_ASSIST_FORCE_N = mdp.scaled_assist_force_n(X2_TOTAL_MASS_KG)


def _contact_sensor(relative_body_path: str) -> ContactSensorCfg:
    """Create an exact-path sensor for one link in X2's hierarchical USD."""

    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot{relative_body_path}",
        class_type=ExactPathContactSensor,
        history_length=3,
        track_air_time=True,
    )


@configclass
class X2RecoverySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        # GroundPlaneCfg references NVIDIA's remote Nucleus asset.  A thin,
        # local kinematic cuboid keeps training fully offline and equivalent
        # for the robot's contact patch.
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
        spawn=sim_utils.CuboidCfg(
            # This prim is shared by every cloned environment.  Cover the
            # entire 2048-env grid, not just the central 20 m patch.
            size=(GROUND_SIDE_M, GROUND_SIDE_M, 0.10),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
        ),
    )
    robot: ArticulationCfg = X2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_pelvis = _contact_sensor("/Geometry/pelvis")
    contact_left_hip_pitch_link = _contact_sensor("/Geometry/pelvis/left_hip_pitch_link")
    contact_left_hip_roll_link = _contact_sensor("/Geometry/pelvis/left_hip_pitch_link/left_hip_roll_link")
    contact_left_hip_yaw_link = _contact_sensor("/Geometry/pelvis/left_hip_pitch_link/left_hip_roll_link/left_hip_yaw_link")
    contact_left_knee_link = _contact_sensor("/Geometry/pelvis/left_hip_pitch_link/left_hip_roll_link/left_hip_yaw_link/left_knee_link")
    contact_left_ankle_pitch_link = _contact_sensor("/Geometry/pelvis/left_hip_pitch_link/left_hip_roll_link/left_hip_yaw_link/left_knee_link/left_ankle_pitch_link")
    contact_left_ankle_roll_link = _contact_sensor("/Geometry/pelvis/left_hip_pitch_link/left_hip_roll_link/left_hip_yaw_link/left_knee_link/left_ankle_pitch_link/left_ankle_roll_link")
    contact_right_hip_pitch_link = _contact_sensor("/Geometry/pelvis/right_hip_pitch_link")
    contact_right_hip_roll_link = _contact_sensor("/Geometry/pelvis/right_hip_pitch_link/right_hip_roll_link")
    contact_right_hip_yaw_link = _contact_sensor("/Geometry/pelvis/right_hip_pitch_link/right_hip_roll_link/right_hip_yaw_link")
    contact_right_knee_link = _contact_sensor("/Geometry/pelvis/right_hip_pitch_link/right_hip_roll_link/right_hip_yaw_link/right_knee_link")
    contact_right_ankle_pitch_link = _contact_sensor("/Geometry/pelvis/right_hip_pitch_link/right_hip_roll_link/right_hip_yaw_link/right_knee_link/right_ankle_pitch_link")
    contact_right_ankle_roll_link = _contact_sensor("/Geometry/pelvis/right_hip_pitch_link/right_hip_roll_link/right_hip_yaw_link/right_knee_link/right_ankle_pitch_link/right_ankle_roll_link")
    contact_waist_yaw_link = _contact_sensor("/Geometry/pelvis/waist_yaw_link")
    contact_waist_pitch_link = _contact_sensor("/Geometry/pelvis/waist_yaw_link/waist_pitch_link")
    contact_torso_link = _contact_sensor("/Geometry/pelvis/waist_yaw_link/waist_pitch_link/torso_link")
    contact_left_shoulder_pitch_link = _contact_sensor("/Geometry/pelvis/waist_yaw_link/waist_pitch_link/torso_link/left_shoulder_pitch_link")
    contact_left_shoulder_roll_link = _contact_sensor("/Geometry/pelvis/waist_yaw_link/waist_pitch_link/torso_link/left_shoulder_pitch_link/left_shoulder_roll_link")
    contact_left_shoulder_yaw_link = _contact_sensor("/Geometry/pelvis/waist_yaw_link/waist_pitch_link/torso_link/left_shoulder_pitch_link/left_shoulder_roll_link/left_shoulder_yaw_link")
    contact_left_elbow_link = _contact_sensor("/Geometry/pelvis/waist_yaw_link/waist_pitch_link/torso_link/left_shoulder_pitch_link/left_shoulder_roll_link/left_shoulder_yaw_link/left_elbow_link")
    contact_left_wrist_yaw_link = _contact_sensor("/Geometry/pelvis/waist_yaw_link/waist_pitch_link/torso_link/left_shoulder_pitch_link/left_shoulder_roll_link/left_shoulder_yaw_link/left_elbow_link/left_wrist_yaw_link")
    contact_left_wrist_pitch_link = _contact_sensor("/Geometry/pelvis/waist_yaw_link/waist_pitch_link/torso_link/left_shoulder_pitch_link/left_shoulder_roll_link/left_shoulder_yaw_link/left_elbow_link/left_wrist_yaw_link/left_wrist_pitch_link")
    contact_left_wrist_roll_link = _contact_sensor("/Geometry/pelvis/waist_yaw_link/waist_pitch_link/torso_link/left_shoulder_pitch_link/left_shoulder_roll_link/left_shoulder_yaw_link/left_elbow_link/left_wrist_yaw_link/left_wrist_pitch_link/left_wrist_roll_link")
    contact_right_shoulder_pitch_link = _contact_sensor("/Geometry/pelvis/waist_yaw_link/waist_pitch_link/torso_link/right_shoulder_pitch_link")
    contact_right_shoulder_roll_link = _contact_sensor("/Geometry/pelvis/waist_yaw_link/waist_pitch_link/torso_link/right_shoulder_pitch_link/right_shoulder_roll_link")
    contact_right_shoulder_yaw_link = _contact_sensor("/Geometry/pelvis/waist_yaw_link/waist_pitch_link/torso_link/right_shoulder_pitch_link/right_shoulder_roll_link/right_shoulder_yaw_link")
    contact_right_elbow_link = _contact_sensor("/Geometry/pelvis/waist_yaw_link/waist_pitch_link/torso_link/right_shoulder_pitch_link/right_shoulder_roll_link/right_shoulder_yaw_link/right_elbow_link")
    contact_right_wrist_yaw_link = _contact_sensor("/Geometry/pelvis/waist_yaw_link/waist_pitch_link/torso_link/right_shoulder_pitch_link/right_shoulder_roll_link/right_shoulder_yaw_link/right_elbow_link/right_wrist_yaw_link")
    contact_right_wrist_pitch_link = _contact_sensor("/Geometry/pelvis/waist_yaw_link/waist_pitch_link/torso_link/right_shoulder_pitch_link/right_shoulder_roll_link/right_shoulder_yaw_link/right_elbow_link/right_wrist_yaw_link/right_wrist_pitch_link")
    contact_right_wrist_roll_link = _contact_sensor("/Geometry/pelvis/waist_yaw_link/waist_pitch_link/torso_link/right_shoulder_pitch_link/right_shoulder_roll_link/right_shoulder_yaw_link/right_elbow_link/right_wrist_yaw_link/right_wrist_pitch_link/right_wrist_roll_link")
    contact_head_yaw_link = _contact_sensor("/Geometry/pelvis/waist_yaw_link/waist_pitch_link/torso_link/head_yaw_link")
    contact_head_pitch_link = _contact_sensor("/Geometry/pelvis/waist_yaw_link/waist_pitch_link/torso_link/head_yaw_link/head_pitch_link")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=750.0),
    )


@configclass
class ActionsCfg:
    # HoST Eq. (1): q_target = q_current + beta*a.  beta=0.25 is its final
    # hardware-conscious action bound; the task-local class clips final targets.
    joint_position = BoundedRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        clip={".*": (-0.25, 0.25)},
        use_zero_offset=True,
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
            params={"sensor_names": FOOT_CONTACT_SENSORS},
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
        func=mdp.reset_root_state_recovery_curriculum,
        mode="reset",
        params={
            "supine_pose_range": {
                "x": (-0.04, 0.04),
                "y": (-0.04, 0.04),
                "z": (-0.001, 0.001),
                "roll": (-0.005, 0.005),
                "pitch": (-0.005, 0.005),
                "yaw": (-0.02, 0.02),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            # HumanUP Stage-I mixes standing starts into discovery.  The mix
            # disappears by iteration ~500 (32 control steps/iteration).
            "standing_probability_start": 0.50,
            "standing_probability_end": 0.0,
            "standing_probability_anneal_steps": 16_000,
            # X2's audited standing root is 0.68 m versus 0.19 m supine.
            "standing_height_offset": 0.49,
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        # The geometry audit bounds the official zero-joint collision hull.
        # Keep this exact so every episode is proven non-intersecting; root
        # X/Y/yaw jitter still gives the fixed evaluator seeds distinct starts.
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )
    lift_assist = EventTerm(
        func=mdp.apply_vertical_force_curriculum,
        mode="interval",
        interval_range_s=(0.05, 0.05),
        params={
            "start_force_n": HOST_ASSIST_FORCE_N,
            "end_force_n": 0.0,
            "anneal_steps": 24_000,
            "orientation_threshold": 0.80,
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
        },
    )


@configclass
class RewardsCfg:
    # HumanUP Stage-I discovery terms (RSS 2025, Appendix Table II).
    base_height = RewTerm(func=mdp.humanup_base_height, weight=5.0)
    head_height = RewTerm(
        func=mdp.humanup_head_height,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="head_pitch_link")},
    )
    height_increase = RewTerm(func=mdp.humanup_height_increase, weight=1.0)
    upright = RewTerm(func=mdp.humanup_body_upright, weight=0.25)
    # HoST's task-orientation term has group weight 2.5.  Unlike its later
    # tilt-only post term, this signed target rejects a 180-degree inversion.
    base_orientation = RewTerm(func=mdp.upright_exp, weight=2.5, params={"std": 0.10})
    standing_on_feet = RewTerm(
        func=mdp.both_feet_when_high,
        weight=2.5,
        params={
            "sensor_names": FOOT_CONTACT_SENSORS,
            "threshold": 15.0,
            "gate_start_height": 0.58,
            "target_height": 0.68,
        },
    )
    unsupported_support = RewTerm(
        func=mdp.unsupported_contacts_when_high,
        weight=-2.0,
        params={
            "sensor_names": ALL_CONTACT_SENSORS,
            "feet_sensor_names": FOOT_CONTACT_SENSORS,
            "threshold": 15.0,
            "gate_start_height": 0.58,
            "target_height": 0.68,
        },
    )
    # HoST post-task group (RSS 2025, Table VI), active above stage two.
    post_angular_velocity = RewTerm(
        func=mdp.host_post_base_angular_velocity, weight=10.0, params={"stage_height": 0.62}
    )
    post_linear_velocity = RewTerm(
        func=mdp.host_post_base_linear_velocity, weight=10.0, params={"stage_height": 0.62}
    )
    post_orientation = RewTerm(
        func=mdp.host_post_base_orientation, weight=10.0, params={"stage_height": 0.62}
    )
    post_height = RewTerm(
        func=mdp.host_post_base_height,
        weight=10.0,
        params={"stage_height": 0.62, "target_height": 0.68},
    )
    # HumanUP's weak Stage-I regularization.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.10)
    joint_acceleration = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    joint_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-6.0e-7)
    base_angular_velocity = RewTerm(func=mdp.base_angular_velocity_l2, weight=-0.10)
    base_linear_velocity = RewTerm(func=mdp.base_linear_velocity_l2, weight=-0.10)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)


@configclass
class TerminationsCfg:
    # Do not end when first upright: the policy must learn to remain stable.
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class X2RecoveryEnvCfg(ManagerBasedRLEnvCfg):
    scene: X2RecoverySceneCfg = X2RecoverySceneCfg(
        num_envs=MAX_CONFIGURED_ENVS,
        env_spacing=ENV_SPACING_M,
        clone_in_fabric=True,
    )
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
        self.events.lift_assist = None
        self.events.reset_back_pose.params["standing_probability_start"] = 0.0
        self.events.reset_back_pose.params["standing_probability_end"] = 0.0
        # Keep the narrow reset distribution: fixed evaluator seeds then exercise
        # five reproducible back-lying states instead of repeating one state.
