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


X2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=x2_usd_path(),
        activate_contact_sensors=True,
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
    # The pelvis is the root. A +90 degree rotation about Y places X2 on its back.
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.28),
        rot=(0.7071068, 0.0, 0.7071068, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        # Official URDF torque and speed limits remain authoritative in the USD.
        # One PD group keeps the task small while gain randomization covers mismatch.
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=None,
            velocity_limit_sim=None,
            stiffness=60.0,
            damping=4.0,
            armature=0.01,
        )
    },
)
