#!/usr/bin/env python3
"""Audit X2 collision bounds, supine frame and reset-floor clearance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def origin_transform(node: ET.Element | None) -> np.ndarray:
    transform = np.eye(4)
    if node is not None:
        transform[:3, 3] = np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
        transform[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(node.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
    return transform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/x2_geometry_audit.json"))
    args = parser.parse_args()
    urdf = args.urdf.expanduser().resolve(strict=True)
    document = ET.parse(urdf).getroot()

    children: dict[str, list[tuple[str, np.ndarray]]] = {}
    for joint in document.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        children.setdefault(parent, []).append((child, origin_transform(joint.find("origin"))))
    link_transforms = {"pelvis": np.eye(4)}
    pending = ["pelvis"]
    while pending:
        parent = pending.pop()
        for child, transform in children.get(parent, []):
            link_transforms[child] = link_transforms[parent] @ transform
            pending.append(child)

    collision_points = []
    for link in document.findall("link"):
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            geometry = collision.find("geometry")
            if mesh is not None:
                mesh_path = (urdf.parent / mesh.get("filename")).resolve(strict=True)
                vertices = np.asarray(trimesh.load(mesh_path, force="mesh", process=False).vertices)
                vertices *= np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            elif geometry.find("box") is not None:
                vertices = trimesh.creation.box(
                    np.fromstring(geometry.find("box").get("size"), sep=" ")
                ).vertices
            elif geometry.find("cylinder") is not None:
                cylinder = geometry.find("cylinder")
                vertices = trimesh.creation.cylinder(
                    radius=float(cylinder.get("radius")),
                    height=float(cylinder.get("length")),
                ).vertices
            elif geometry.find("sphere") is not None:
                vertices = trimesh.creation.icosphere(
                    radius=float(geometry.find("sphere").get("radius"))
                ).vertices
            else:
                raise ValueError("Unsupported collision geometry")
            transform = link_transforms[link.get("name")] @ origin_transform(collision.find("origin"))
            collision_points.append((transform[:3, :3] @ vertices.T).T + transform[:3, 3])
    points = np.vstack(collision_points)

    supine_rotation = Rotation.from_euler("y", -90.0, degrees=True).as_matrix()
    standing_min_z = float(points[:, 2].min())
    supine_min_z = float((supine_rotation @ points.T)[2].min())

    hull = points[ConvexHull(points).vertices]
    generator = np.random.default_rng(42)
    angles = generator.uniform(
        [-0.005, -0.005, -0.02],
        [0.005, 0.005, 0.02],
        size=(20_000, 3),
    )
    rotations = np.einsum(
        "ij,njk->nik", supine_rotation, Rotation.from_euler("xyz", angles).as_matrix()
    )
    sampled_min_z = np.einsum("nij,pj->nip", rotations, hull)[:, 2, :].min(axis=1)
    worst_sampled_clearance = float(0.190 - 0.001 + sampled_min_z.min())

    try:
        reported_urdf = str(urdf.relative_to(Path.cwd().resolve()))
    except ValueError:
        reported_urdf = str(urdf)

    result = {
        "urdf": reported_urdf,
        "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "links": len(document.findall("link")),
        "joints": len(document.findall("joint")),
        "collisions": len(document.findall(".//collision")),
        "standing_collision_extent_below_pelvis_m": round(-standing_min_z, 10),
        "standing_target_pelvis_height_m": 0.68,
        "standing_target_foot_clearance_m": round(0.68 + standing_min_z, 10),
        "supine_collision_extent_below_pelvis_m": round(-supine_min_z, 10),
        "supine_reset_pelvis_height_m": 0.190,
        "supine_nominal_floor_clearance_m": round(0.190 + supine_min_z, 10),
        "sampled_reset_count": len(angles),
        "sampled_worst_floor_clearance_m": round(worst_sampled_clearance, 10),
        "supine_world_forward_axis": (supine_rotation @ np.array([1.0, 0.0, 0.0])).tolist(),
        "passes": bool(worst_sampled_clearance > 0.0 and abs(0.68 + standing_min_z) < 0.01),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
