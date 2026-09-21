"""AgiBot X2 Ultra articulation configuration.

The USD is generated locally from AgiBot's official v1.3.0 simplified-collision
URDF. Nothing is copied into or modified inside the Isaac Lab checkout.
"""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_USD = (
    PROJECT_ROOT
    / "assets"
    / "isaac"
    / "x2_ultra_simple_collision"
    / "x2_ultra_simple_collision.usda"
)


def x2_usd_path() -> str:
    """Return a validated repo-local USD path, with an explicit override hook."""

    path = Path(os.environ.get("HRS_X2_USD", DEFAULT_USD)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"X2 USD not found at {path}. Run scripts/fetch_agibot_model.sh and "
            "scripts/import_x2_isaac.sh first, or set HRS_X2_USD."
        )
    return str(path)


def spawn_x2_with_nested_contacts(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn X2 and enable contact reporting on every hierarchical link.

    The current URDF converter preserves the kinematic link tree.  Isaac Lab's
    generic contact activator stops at the first rigid body, which only enables
    the pelvis on this asset.  Enumerating the composed rigid-body prims fixes
    that importer/sensor mismatch without editing the vendor URDF or USD.
    """

    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)

    from pxr import Usd, UsdPhysics

    from isaaclab.sim.schemas import activate_contact_sensors
    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    rigid_bodies = [
        child for child in Usd.PrimRange(prim) if child.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if not rigid_bodies:
        raise RuntimeError(f"No rigid bodies found below imported X2 prim: {prim_path}")
    for body in rigid_bodies:
        activate_contact_sensors(str(body.GetPath()), stage=stage)
    return prim


X2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        func=spawn_x2_with_nested_contacts,
        usd_path=x2_usd_path(),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    # X2 uses ROS FLU axes (X forward, Y left, Z up). Rotating -90 degrees
    # about Y points the chest/forward axis upward and places the back down.
    # This installed Isaac Lab release stores InitialStateCfg quaternions in
    # scalar-last XYZW order, as does its root-state tensor API.
    init_state=ArticulationCfg.InitialStateCfg(
        # Collision-hull audit: the supine model extends 0.18030 m below the
        # pelvis.  A 0.190 m root height leaves millimetres of clearance under
        # the bounded reset jitter instead of dropping the robot from 0.10 m.
        pos=(0.0, 0.0, 0.190),
        rot=(0.0, -0.7071068, 0.0, 0.7071068),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    # Keep a 2% numerical margin while preserving near-extension.  The X2
    # knees and elbows use zero as a hard endpoint; larger generic margins
    # made a straight-leg standing pose unreachable by the policy.
    soft_joint_pos_limit_factor=0.98,
    actuators={
        # Official URDF torque and speed limits remain authoritative in the USD.
        # Gains are grouped by load and joint size: the weight-bearing leg and
        # waist drives need more stiffness than the arm/head drives.  The
        # startup gain randomization still covers a +/-10% model mismatch.
        "legs_waist": ImplicitActuatorCfg(
            joint_names_expr=[".*hip.*", ".*knee.*", "waist_.*"],
            effort_limit_sim=None,
            velocity_limit_sim=None,
            stiffness=120.0,
            damping=6.0,
            armature=0.01,
        ),
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=[".*ankle.*"],
            effort_limit_sim=None,
            velocity_limit_sim=None,
            stiffness=80.0,
            damping=5.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*shoulder.*", ".*elbow.*"],
            effort_limit_sim=None,
            velocity_limit_sim=None,
            stiffness=40.0,
            damping=3.0,
            armature=0.01,
        ),
        "small_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*wrist.*", "head_.*"],
            effort_limit_sim=None,
            velocity_limit_sim=None,
            stiffness=15.0,
            damping=1.5,
            armature=0.01,
        ),
    },
)
