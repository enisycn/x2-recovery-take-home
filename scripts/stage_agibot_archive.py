#!/usr/bin/env python3
"""Safely stage a user-downloaded official AgiBot X2 ZIP inside this repository."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import xml.etree.ElementTree as ET
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "agibot_x2_urdf"
EXPECTED_URDF = Path("X2_URDF-v1.3.0/x2_ultra_simple_collision.urdf")
MAX_FILES = 10_000
MAX_UNCOMPRESSED_BYTES = 2 * 1024**3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    if len(members) > MAX_FILES:
        raise ValueError(f"archive contains too many entries: {len(members)}")
    if sum(member.file_size for member in members) > MAX_UNCOMPRESSED_BYTES:
        raise ValueError("archive exceeds the 2 GiB uncompressed limit")
    for member in members:
        path = PurePosixPath(member.filename)
        mode = member.external_attr >> 16
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe archive path: {member.filename}")
        if stat.S_ISLNK(mode):
            raise ValueError(f"symbolic links are not accepted: {member.filename}")
    return members


def extract_archive(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for member in safe_members(archive):
            relative = PurePosixPath(member.filename)
            target = destination.joinpath(*relative.parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)


def validate_model(source_root: Path) -> dict[str, int | str]:
    urdf_path = source_root / EXPECTED_URDF
    if not urdf_path.is_file():
        raise ValueError(f"missing expected official model: {EXPECTED_URDF}")
    root = ET.parse(urdf_path).getroot()
    if root.tag != "robot" or root.attrib.get("name") != "x2_ultra":
        raise ValueError("unexpected URDF robot identity")
    links = root.findall("link")
    joints = root.findall("joint")
    collisions = root.findall(".//collision")
    if len(links) < 25 or len(joints) < 24 or len(collisions) < 20:
        raise ValueError("URDF structure is incomplete for X2 Ultra")

    meshes = root.findall(".//mesh")
    missing_meshes: list[str] = []
    urdf_dir = urdf_path.parent.resolve()
    for mesh in meshes:
        filename = mesh.attrib.get("filename", "")
        mesh_ref = PurePosixPath(filename)
        if not filename or mesh_ref.is_absolute() or ".." in mesh_ref.parts:
            raise ValueError(f"unsafe mesh reference: {filename!r}")
        mesh_path = urdf_dir.joinpath(*mesh_ref.parts).resolve()
        if urdf_dir not in mesh_path.parents or not mesh_path.is_file():
            missing_meshes.append(filename)
    if missing_meshes:
        raise ValueError(f"missing referenced meshes, first entry: {missing_meshes[0]}")

    return {
        "robot": "x2_ultra",
        "links": len(links),
        "joints": len(joints),
        "collisions": len(collisions),
        "mesh_references": len(meshes),
        "urdf_sha256": sha256(urdf_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="ZIP downloaded from AgiBot's official docs or GitHub repository")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source-url",
        default="https://x2-aimdk.agibot.com/zh-cn/latest/get_sdk/index.html",
        help="Official page from which the archive was downloaded",
    )
    args = parser.parse_args()

    archive_path = args.archive.expanduser().resolve()
    output = args.output.expanduser().resolve()
    models_root = (PROJECT_ROOT / "models").resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if output.exists():
        raise FileExistsError(f"refusing to replace existing model directory: {output}")
    if output.parent.resolve() != models_root:
        raise ValueError(f"output must be a direct child of {models_root}")

    models_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="agibot-stage-", dir=models_root) as temporary:
        extraction_root = Path(temporary)
        extract_archive(archive_path, extraction_root)
        matches = list(extraction_root.rglob(str(EXPECTED_URDF)))
        if len(matches) != 1:
            raise ValueError(f"expected exactly one {EXPECTED_URDF}, found {len(matches)}")
        source_root = matches[0].parents[1]
        model_stats = validate_model(source_root)
        shutil.copytree(source_root, output)

    manifest = {
        "validation": "safe-official-archive-structure-v1",
        "source_url": args.source_url,
        "staged_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": sha256(archive_path),
        "urdf_relative_path": str(EXPECTED_URDF),
        **model_stats,
    }
    (output / ".hrs-source.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Staged verified X2 model: {output}")


if __name__ == "__main__":
    main()
