#!/usr/bin/env python3
"""Convert the official X2 Ultra v1.3.0 simplified URDF to repo-local USD."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402


def main() -> None:
    urdf_path = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Official X2 URDF not found: {urdf_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = UrdfConverterCfg(
        asset_path=str(urdf_path),
        usd_dir=str(output_dir),
        fix_base=False,
        merge_fixed_joints=True,
        force_usd_conversion=True,
        make_instanceable=True,
        self_collision=True,
        collision_from_visuals=False,
        robot_type="Humanoid",
        run_asset_transformer=True,
        run_multi_physics_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=60.0, damping=4.0),
        ),
    )
    converter = UrdfConverter(config)
    print(f"Generated repo-local X2 USD: {converter.usd_path}")


try:
    main()
finally:
    simulation_app.close()
