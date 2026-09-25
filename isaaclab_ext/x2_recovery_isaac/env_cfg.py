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
from .agents.absolute_action import BoundedAbsoluteJointPositionActionCfg
from .exact_contact_sensor import ExactPathContactSensor
from .relative_action import BoundedRelativeJointPositionActionCfg
from .x2_robot_cfg import X2_CFG


MAX_CONFIGURED_ENVS = 4096
ENV_SPACING_M = 2.5
GROUND_SIDE_M = 200.0
_required_ground_side = math.ceil(math.sqrt(MAX_CONFIGURED_ENVS)) * ENV_SPACING_M + 2.0
if GROUND_SIDE_M < _required_ground_side:
    raise ValueError(
        f"ground side {GROUND_SIDE_M} m does not cover the configured environment grid "
        f"({_required_ground_side:.1f} m required)"
    )


CONTACT_SENSOR_NAME = "contact_all"
FOOT_CONTACT_BODIES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)
ALL_CONTACT_BODIES = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_pitch_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "left_wrist_pitch_link",
    "left_wrist_roll_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
    "right_wrist_pitch_link",
    "right_wrist_roll_link",
    "head_yaw_link",
    "head_pitch_link",
)


def foot_contact_cfg() -> SceneEntityCfg:
    """Return a fresh, order-preserving selector for both feet."""

    return SceneEntityCfg(
        CONTACT_SENSOR_NAME,
        body_names=list(FOOT_CONTACT_BODIES),
        preserve_order=True,
    )


def all_contact_cfg() -> SceneEntityCfg:
    """Return a fresh, order-preserving selector for all X2 bodies."""

    return SceneEntityCfg(
        CONTACT_SENSOR_NAME,
        body_names=list(ALL_CONTACT_BODIES),
        preserve_order=True,
    )

# Sum of the masses in the pinned official X2 Ultra v1.3.0 URDF. HoST's
# official cross-robot guidance recommends a pull near 60% of robot weight.
X2_TOTAL_MASS_KG = 41.966521
# HoST applies its nominal pull on two G1 torso bodies. X2 likewise uses the
HOST_ASSIST_FORCE_N = mdp.scaled_assist_force_n(X2_TOTAL_MASS_KG)


def _all_contacts_sensor() -> ContactSensorCfg:
    """Create one recursive view for every body in X2's hierarchical USD."""
    return ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/pelvis/**",
        class_type=ExactPathContactSensor,
        # HumanUP's Stage-I foot-force-increase reward compares the current
        # and preceding contact sample.  Other contact terms remain unchanged
        # because they already reduce over the available history.
        history_length=2,
        track_air_time=False,
        # Success means support by the floor.  The unfiltered net-force
        # buffer also contains X2 self-collisions, so all reward, observation,
        # and validation contact masks use the per-partner ground matrix.
        filter_prim_paths_expr=["/World/ground"],
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
            # entire 4096-env grid, not just the central 20 m patch.
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
    contact_all = _all_contacts_sensor()
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=750.0),
    )


@configclass
class ActionsCfg:
    # HoST Eq. (1): q_target = q_current + beta*a.  A smooth tanh enforces the
    # paper's normalized action assumption, beta=0.25 bounds one policy step,
    # and the task-local class clips the final target to imported joint limits.
    joint_position = BoundedRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        clip=None,
        use_zero_offset=True,
    )


@configclass
class HumanUpActionsCfg:
    """HumanUP's absolute default-centred position target for every X2 joint."""

    joint_position = BoundedAbsoluteJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        clip=None,
        use_default_offset=True,
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
            params={
                "sensor_cfg": foot_contact_cfg()
            },
        )
        # Whole-body recovery depends on knowing whether the back, pelvis,
        # knees, elbows, or hands are supporting the robot. Foot-only contact
        # made those physically distinct states observationally ambiguous.
        body_contact = ObsTerm(
            func=mdp.body_contacts,
            params={"sensor_cfg": all_contact_cfg(), "threshold": 15.0},
        )
        previous_actions = ObsTerm(func=mdp.last_action, history_length=2)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class HumanUpObservationsCfg:
    """HumanUP's current proprioception, zero extrinsics, and 10-step history."""

    @configclass
    class PolicyCfg(ObsGroup):
        proprio = ObsTerm(func=mdp.humanup_proprioception)
        # HumanUP reserves 4 mass + 1 friction + 2*n_actions motor + 3
        # base-velocity dimensions. Stage I disables randomization and fills
        # this section with zeros; the history encoder is used by the actor.
        privileged_extrinsics = ObsTerm(func=mdp.humanup_privileged_zeros, params={"dimension": 70})
        proprio_history = ObsTerm(func=mdp.humanup_proprioception, history_length=10)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ImitationCollectionObservationsCfg:
    """Expose HumanUP student and legacy expert observations together."""

    policy: HumanUpObservationsCfg.PolicyCfg = HumanUpObservationsCfg.PolicyCfg()
    teacher_policy: ObservationsCfg.PolicyCfg = ObservationsCfg.PolicyCfg()


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
            # HumanUP mixes reference starts into discovery.  X2 adds three
            # collision-audited squat depths between supine and standing so
            # the policy observes the full rising manifold.  The schedule is
            # measured in policy steps (300 PPO updates at 12 steps/update).
            "reference_probability_start": 0.95,
            "reference_probability_end": 0.35,
            "reference_probability_anneal_policy_steps": 3_600,
            # Offsets are relative to the audited 0.190 m supine pelvis.
            "reference_root_height_offsets": (
                0.49000,
                0.41214,
                0.31129,
                0.20152,
                0.12452,
                0.06245,
                0.02527,
                -0.04046,
                -0.09806,
            ),
            # Symmetric hip/knee/ankle and arm references connect standing,
            # squatting, folded sitting, and low long-sitting states.  Every
            # height comes from the official collision geometry with 2 mm
            # clearance; shoulder/elbow flexion keeps wrists off the floor.
            "reference_body_angles": (
                (0.0, 0.0, 0.0, -1.0, -1.5),
                (-0.5, 1.0, -0.5, -1.0, -1.5),
                (-0.78, 1.56, -0.78, -1.0, -1.5),
                (-1.2, 1.98, -0.78, -1.0, -1.5),
                (-1.5, 2.2, -0.7, -1.0, -1.5),
                (-1.8, 2.3, -0.5, -1.0, -1.5),
                (-2.0, 2.3, -0.3, -1.0, -1.5),
                (-2.3, 2.3, 0.0, -1.0, -1.5),
                (-1.57, 0.0, 0.0, -1.0, -1.5),
            ),
        },
    )
    lift_assist = EventTerm(
        func=mdp.apply_vertical_force_curriculum,
        mode="interval",
        interval_range_s=(0.05, 0.05),
        params={
            "start_force_n": HOST_ASSIST_FORCE_N,
            "end_force_n": 0.0,
            "anneal_policy_steps": 4_000,
            "orientation_threshold": 0.80,
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
        },
    )


@configclass
class RewardsCfg:
    # Dense, orientation-gated height progress closes the two exploits seen in
    # the first run: lifting only the head and collecting upward-velocity bits
    # while the pelvis remains on the floor.
    base_height_progress = RewTerm(
        func=mdp.base_height_progress,
        weight=40.0,
        params={"target_height": 0.68},
    )
    upright = RewTerm(func=mdp.humanup_body_upright, weight=0.25)
    # HoST's task-orientation term has group weight 2.5.  Unlike its later
    # tilt-only post term, this signed target rejects a 180-degree inversion.
    base_orientation = RewTerm(func=mdp.upright_exp, weight=2.5, params={"std": 0.25})
    standing_on_feet = RewTerm(
        func=mdp.both_feet_when_high,
        weight=10.0,
        params={
            "sensor_cfg": foot_contact_cfg(),
            "threshold": 15.0,
            "gate_start_height": 0.08,
            "target_height": 0.68,
        },
    )
    unsupported_support = RewTerm(
        func=mdp.unsupported_contacts_when_high,
        weight=-2.0,
        params={
            "all_bodies_cfg": all_contact_cfg(),
            "feet_cfg": foot_contact_cfg(),
            "threshold": 15.0,
            "gate_start_height": 0.45,
            "target_height": 0.68,
        },
    )
    # HoST post-task group (RSS 2025, Table VI), active above stage two.
    post_angular_velocity = RewTerm(
        func=mdp.host_post_base_angular_velocity, weight=10.0, params={"stage_height": 0.58}
    )
    post_linear_velocity = RewTerm(
        func=mdp.host_post_base_linear_velocity, weight=10.0, params={"stage_height": 0.58}
    )
    post_orientation = RewTerm(
        func=mdp.host_post_base_orientation, weight=10.0, params={"stage_height": 0.58}
    )
    post_height = RewTerm(
        func=mdp.host_post_base_height,
        weight=10.0,
        params={"stage_height": 0.58, "target_height": 0.68},
    )
    standing_still = RewTerm(func=mdp.standing_still, weight=5.0, params={"target_height": 0.68})
    # Optimize the same instantaneous predicate used by evaluation.  The
    # sustained 0.5 s requirement remains evaluator state, but paying this
    # term on every valid step makes balance duration, foot loading, velocity,
    # and removal of hand/knee support visible to PPO instead of only to the
    # post-training report.
    exact_stance = RewTerm(
        func=mdp.strict_success,
        weight=25.0,
        params={
            "feet_cfg": foot_contact_cfg(),
            "all_bodies_cfg": all_contact_cfg(),
        },
    )
    # HumanUP's weak Stage-I regularization.
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-0.01)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    joint_acceleration = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    joint_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-6.0e-7)
    base_angular_velocity = RewTerm(func=mdp.base_angular_velocity_l2, weight=-0.10)
    base_linear_velocity = RewTerm(func=mdp.base_linear_velocity_l2, weight=-0.10)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    # Keep recovery in a coherent sagittal plane while allowing the large hip,
    # knee, shoulder-pitch, and elbow motions needed to get up.
    bilateral_symmetry = RewTerm(
        func=mdp.bilateral_joint_symmetry_l2,
        weight=-0.05,
        params={
            "left_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_hip_pitch_joint",
                    "left_knee_joint",
                    "left_ankle_pitch_joint",
                    "left_shoulder_pitch_joint",
                    "left_elbow_joint",
                ],
                preserve_order=True,
            ),
            "right_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "right_hip_pitch_joint",
                    "right_knee_joint",
                    "right_ankle_pitch_joint",
                    "right_shoulder_pitch_joint",
                    "right_elbow_joint",
                ],
                preserve_order=True,
            ),
        },
    )
    non_sagittal_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_roll_joint",
                    ".*_hip_yaw_joint",
                    ".*_ankle_roll_joint",
                    "waist_yaw_joint",
                    "waist_roll_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_wrist_.*",
                    "head_.*",
                ],
            )
        },
    )


def _humanup_feet_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=list(FOOT_CONTACT_BODIES), preserve_order=True)


def _humanup_head_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=["head_pitch_link"], preserve_order=True)


def _humanup_left_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
        "robot",
        joint_names=[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
        ],
        preserve_order=True,
    )


def _humanup_right_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
        "robot",
        joint_names=[
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
        ],
        preserve_order=True,
    )


@configclass
class HumanUpStageIRewardsCfg:
    """HumanUP official Stage-I code-release reward terms and weights.

    Body/joint names and reachable height caps are the only X2-specific
    substitutions.  The formulas and scalar weights match the official
    ``G1WaistHumanUPCfg.rewards.scales`` configuration.
    """

    base_height_exp = RewTerm(
        func=mdp.humanup_base_height_exp_clipped,
        weight=5.0,
        params={"target_height": 0.68},
    )
    head_height_exp = RewTerm(
        func=mdp.humanup_head_height_exp_clipped,
        weight=5.0,
        params={"target_height": 1.18, "asset_cfg": _humanup_head_cfg()},
    )
    delta_base_height = RewTerm(func=mdp.humanup_height_increase, weight=1.0)
    feet_contact_forces_increase = RewTerm(
        func=mdp.humanup_feet_contact_force_increase,
        weight=1.0,
        params={"sensor_cfg": foot_contact_cfg()},
    )
    stand_on_feet = RewTerm(
        func=mdp.humanup_stand_on_feet,
        weight=2.5,
        params={"sensor_cfg": foot_contact_cfg(), "feet_cfg": _humanup_feet_cfg()},
    )
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    body_up_exp = RewTerm(func=mdp.humanup_body_upright, weight=0.25)
    feet_height = RewTerm(
        func=mdp.humanup_feet_height,
        weight=2.5,
        params={"feet_cfg": _humanup_feet_cfg()},
    )
    feet_distance = RewTerm(
        func=mdp.humanup_feet_distance,
        weight=2.0,
        params={"feet_cfg": _humanup_feet_cfg(), "min_distance": 0.25, "max_distance": 1.0},
    )
    soft_symmetry_body = RewTerm(
        func=mdp.humanup_action_symmetry,
        weight=-1.0,
        params={
            "left_cfg": _humanup_left_cfg(),
            "right_cfg": _humanup_right_cfg(),
            "sign_flip_indices": (1, 2, 5, 7, 8),
            "head_cfg": _humanup_head_cfg(),
        },
    )
    soft_symmetry_waist = RewTerm(
        func=mdp.humanup_waist_action_symmetry,
        weight=-1.0,
        params={
            "waist_cfg": SceneEntityCfg(
                "robot", joint_names=["waist_yaw_joint", "waist_roll_joint"], preserve_order=True
            ),
            "head_cfg": _humanup_head_cfg(),
        },
    )
    feet_orientation = RewTerm(
        func=mdp.humanup_feet_orientation,
        weight=-0.5,
        params={"feet_cfg": _humanup_feet_cfg()},
    )
    foot_slip = RewTerm(
        func=mdp.humanup_foot_slip,
        weight=-1.0,
        params={"sensor_cfg": foot_contact_cfg(), "feet_cfg": _humanup_feet_cfg()},
    )
    termination = RewTerm(func=mdp.is_terminated, weight=-500.0)
    dof_error = RewTerm(
        func=mdp.humanup_joint_position_error,
        weight=-0.03,
        params={"head_cfg": _humanup_head_cfg()},
    )
    base_lin_vel = RewTerm(func=mdp.humanup_base_linear_velocity_norm, weight=-0.1)
    ang_vel = RewTerm(func=mdp.base_angular_velocity_l2, weight=-0.1)
    dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4)
    action_rate = RewTerm(func=mdp.humanup_action_rate_norm, weight=-0.1)
    torques = RewTerm(func=mdp.humanup_joint_torque_norm, weight=-1.0e-6)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    dof_torque_limits = RewTerm(
        func=mdp.humanup_joint_torque_limits,
        weight=-0.1,
        params={"soft_ratio": 1.0},
    )
    energy = RewTerm(func=mdp.humanup_energy, weight=-1.0e-4)
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)


@configclass
class HumanUpCurriculumRewardsCfg(HumanUpStageIRewardsCfg):
    """Published Stage-I terms plus the take-home's explicit success target."""

    x2_height_progress = RewTerm(
        func=mdp.base_height_progress,
        weight=40.0,
        params={"target_height": 0.68},
    )
    x2_final_leg_pose = RewTerm(
        func=mdp.final_leg_pose_exp,
        weight=50.0,
        params={
            "gate_start_height": 0.45,
            "target_height": 0.68,
            "std": 1.0,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_pitch_joint",
                    ".*_knee_joint",
                    ".*_ankle_pitch_joint",
                ],
            ),
        },
    )
    x2_final_alignment_pose = RewTerm(
        func=mdp.final_leg_pose_exp,
        weight=50.0,
        params={
            "gate_start_height": 0.55,
            "target_height": 0.68,
            "std": 1.0,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_roll_joint",
                    ".*_hip_yaw_joint",
                    ".*_ankle_roll_joint",
                    "waist_.*",
                    ".*_shoulder_.*",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                    "head_.*",
                ],
            ),
        },
    )
    x2_orientation = RewTerm(func=mdp.upright_exp, weight=2.5, params={"std": 0.25})
    x2_both_feet = RewTerm(
        func=mdp.both_feet_when_high,
        weight=10.0,
        params={
            "sensor_cfg": foot_contact_cfg(),
            "threshold": 15.0,
            "gate_start_height": 0.08,
            "target_height": 0.68,
        },
    )
    x2_other_support = RewTerm(
        func=mdp.unsupported_contacts_when_high,
        weight=-10.0,
        params={
            "all_bodies_cfg": all_contact_cfg(),
            "feet_cfg": foot_contact_cfg(),
            "threshold": 15.0,
            "gate_start_height": 0.45,
            "target_height": 0.68,
        },
    )
    host_post_angular_velocity = RewTerm(
        func=mdp.host_post_base_angular_velocity, weight=50.0, params={"stage_height": 0.58}
    )
    host_post_linear_velocity = RewTerm(
        func=mdp.host_post_base_linear_velocity, weight=50.0, params={"stage_height": 0.58}
    )
    host_post_orientation = RewTerm(
        func=mdp.host_post_base_orientation, weight=10.0, params={"stage_height": 0.58}
    )
    host_post_height = RewTerm(
        func=mdp.host_post_base_height,
        weight=10.0,
        params={"stage_height": 0.58, "target_height": 0.68},
    )
    x2_standing_still = RewTerm(func=mdp.standing_still, weight=5.0, params={"target_height": 0.68})
    x2_stance_proximity = RewTerm(
        func=mdp.strict_stance_proximity,
        weight=100.0,
        params={"feet_cfg": foot_contact_cfg(), "all_bodies_cfg": all_contact_cfg()},
    )
    exact_stance = RewTerm(
        func=mdp.strict_success,
        weight=50.0,
        params={"feet_cfg": foot_contact_cfg(), "all_bodies_cfg": all_contact_cfg()},
    )


@configclass
class TerminationsCfg:
    # Do not end when first upright: the policy must learn to remain stable.
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class HumanUpTerminationsCfg(TerminationsCfg):
    # HumanUP's released G1 task adds only broad root-state safety bounds.
    # This is a true failure termination; the target stance is never terminal.
    root_speed = DoneTerm(func=mdp.humanup_root_speed_too_high)
    root_height = DoneTerm(func=mdp.humanup_root_height_out_of_bounds)
    nonfinite_state = DoneTerm(func=mdp.root_state_nonfinite)


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
        # FRASA uses 20 Hz decisions and 100 Hz physical control. Five contact
        # substeps preserve that published ratio without spending half the run
        # on an unsupported 200 Hz oversampling choice.
        self.decimation = 5
        self.episode_length_s = 8.0
        self.sim.dt = 1.0 / 100.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 0.9
        self.sim.physics_material.dynamic_friction = 0.8
        self.sim.physics_material.restitution = 0.0
        # The assignment evaluates nominal simulation recovery.  Keep the
        # official mass/inertia, fixed floor friction and configured gains
        # deterministic until a nominal policy succeeds; randomization is a
        # later sim-to-real extension, not part of the required experiment.
        self.events.material = None
        self.events.mass = None
        self.events.pelvis_com = None
        self.events.actuator_gains = None
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
        self.events.reset_back_pose.params["reference_probability_start"] = 0.0
        self.events.reset_back_pose.params["reference_probability_end"] = 0.0
        # Keep the narrow reset distribution: fixed evaluator seeds then exercise
        # five reproducible back-lying states instead of repeating one state.


@configclass
class X2HumanUpDiscoveryEnvCfg(X2RecoveryEnvCfg):
    """HumanUP Stage-I reproduction with X2 body names and height caps."""

    rewards: HumanUpStageIRewardsCfg = HumanUpStageIRewardsCfg()
    observations: HumanUpObservationsCfg = HumanUpObservationsCfg()
    actions: HumanUpActionsCfg = HumanUpActionsCfg()
    terminations: HumanUpTerminationsCfg = HumanUpTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        params = self.events.reset_back_pose.params
        # Official HumanUP mixes 10--20% standing starts during discovery.
        params["reference_probability_start"] = 0.20
        params["reference_probability_end"] = 0.10
        params["reference_probability_anneal_policy_steps"] = 50_000
        params["reference_root_height_offsets"] = (0.49000,)
        params["reference_body_angles"] = ((0.0, 0.0, 0.0, -1.0, -1.5),)
        params["reference_min_stage"] = 0
        params["reference_max_stage_start"] = 0
        params["reference_max_stage_end"] = 0
        params["reference_stage_anneal_policy_steps"] = 1
        # The release contains optional 1500 N drag-force utilities, but its
        # published discovery configuration sets drag_robot_up=False.
        self.events.lift_assist = None
        self.episode_length_s = 10.0


@configclass
class X2HumanUpStandingEnvCfg(X2HumanUpDiscoveryEnvCfg):
    """Compute-efficient standing pretraining with the HumanUP model/rewards."""

    rewards: HumanUpCurriculumRewardsCfg = HumanUpCurriculumRewardsCfg()
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        params = self.events.reset_back_pose.params
        params["reference_probability_start"] = 1.0
        params["reference_probability_end"] = 1.0
        params["reference_probability_anneal_policy_steps"] = 1
        self.rewards.exact_stance.weight = 200.0
        self.episode_length_s = 4.0


@configclass
class X2HumanUpRiseEnvCfg(X2HumanUpDiscoveryEnvCfg):
    """Bridge-pose curriculum retaining HumanUP observations and rewards."""

    rewards: HumanUpCurriculumRewardsCfg = HumanUpCurriculumRewardsCfg()
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_position.brake_start_height = 0.55
        self.actions.joint_position.brake_end_height = 0.62
        self.actions.joint_position.brake_min_upright = 0.90
        self.actions.joint_position.maximum_brake = 1.00
        params = self.events.reset_back_pose.params
        params["reference_probability_start"] = 1.0
        params["reference_probability_end"] = 1.0
        params["reference_probability_anneal_policy_steps"] = 1
        params["reference_root_height_offsets"] = (
            0.49000, 0.41214, 0.31129, 0.20152, 0.12452, 0.06245, 0.02527, -0.04046, -0.09806
        )
        params["reference_body_angles"] = (
            (0.0, 0.0, 0.0, -1.0, -1.5),
            (-0.5, 1.0, -0.5, -1.0, -1.5),
            (-0.78, 1.56, -0.78, -1.0, -1.5),
            (-1.2, 1.98, -0.78, -1.0, -1.5),
            (-1.5, 2.2, -0.7, -1.0, -1.5),
            (-1.8, 2.3, -0.5, -1.0, -1.5),
            (-2.0, 2.3, -0.3, -1.0, -1.5),
            (-2.3, 2.3, 0.0, -1.0, -1.5),
            (-1.57, 0.0, 0.0, -1.0, -1.5),
        )
        params["reference_min_stage"] = 0
        params["reference_max_stage_start"] = 8
        params["reference_max_stage_end"] = 8
        params["reference_stage_anneal_policy_steps"] = 1
        self.rewards.exact_stance.weight = 200.0


@configclass
class X2StandingEnvCfg(X2RecoveryEnvCfg):
    """Phase 1: learn active balance from the audited straight stance."""

    def __post_init__(self) -> None:
        super().__post_init__()
        params = self.events.reset_back_pose.params
        params["reference_probability_start"] = 1.0
        params["reference_probability_end"] = 1.0
        params["reference_probability_anneal_policy_steps"] = 1
        params["reference_root_height_offsets"] = (0.49000,)
        params["reference_body_angles"] = ((0.0, 0.0, 0.0, -1.0, -1.5),)
        params["reference_max_stage_start"] = 0
        params["reference_max_stage_end"] = 0
        params["reference_stage_anneal_policy_steps"] = 1
        self.events.lift_assist = None
        # This teacher phase starts inside the valid set.  A stronger exact
        # reward teaches the actor to remain there before harder reset poses
        # are introduced.
        self.rewards.exact_stance.weight = 50.0
        self.episode_length_s = 4.0


@configclass
class X2RiseEnvCfg(X2RecoveryEnvCfg):
    """Phase 2: extend the standing policy into progressively deeper poses."""

    def __post_init__(self) -> None:
        super().__post_init__()
        params = self.events.reset_back_pose.params
        params["reference_probability_start"] = 1.0
        params["reference_probability_end"] = 1.0
        params["reference_probability_anneal_policy_steps"] = 1
        # At 12 decisions/update, unlock one deeper pose roughly every 31
        # updates and expose all nine poses after 250 updates.
        params["reference_max_stage_start"] = 0
        params["reference_max_stage_end"] = 8
        params["reference_stage_anneal_policy_steps"] = 3_000
        self.events.lift_assist = None
        # Once a rising trajectory enters the exact evaluation set, strongly
        # favor braking and remaining there over a high-speed pass-through.
        self.rewards.exact_stance.weight = 100.0


@configclass
class X2ImitationCollectionEnvCfg(X2RiseEnvCfg):
    """Read-only rollout environment for successful expert trajectories."""

    observations: ImitationCollectionObservationsCfg = ImitationCollectionObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.observations.policy.enable_corruption = False
        self.observations.teacher_policy.enable_corruption = False
