"""Collision support bounds for the Go2W default joint reset (URDF primitives only)."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)


def _origin(element: ET.Element | None) -> np.ndarray:
    result = np.eye(4)
    if element is not None:
        result[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        roll, pitch, yaw = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        result[:3, :3] = (
            _rotation(np.array([0, 0, 1]), yaw)
            @ _rotation(np.array([0, 1, 0]), pitch)
            @ _rotation(np.array([1, 0, 0]), roll)
        )
    return result


@dataclass
class CollisionGeometry:
    centers: np.ndarray
    rotations: np.ndarray
    half_sizes: np.ndarray
    radii: np.ndarray
    half_lengths: np.ndarray
    kinds: np.ndarray
    bounding_radius: float

    def tensors(self, device: str | torch.device) -> dict[str, torch.Tensor]:
        return {
            name: torch.as_tensor(getattr(self, name), dtype=torch.float32, device=device)
            for name in ("centers", "rotations", "half_sizes", "radii", "half_lengths", "kinds")
        }


def load_collision_geometry(urdf_path: Path, joint_positions: dict[str, float]) -> CollisionGeometry:
    """Compute every collision primitive's pose in the root frame at the configured reset joints."""
    robot = ET.parse(urdf_path).getroot()
    joints = robot.findall("joint")
    children = {joint.find("child").get("link") for joint in joints}
    roots = [link.get("name") for link in robot.findall("link") if link.get("name") not in children]
    if len(roots) != 1:
        raise ValueError("Reset geometry requires one URDF root link.")
    poses = {roots[0]: np.eye(4)}
    pending = list(joints)
    while pending:
        remaining = []
        for joint in pending:
            parent = joint.find("parent").get("link")
            if parent not in poses:
                remaining.append(joint)
                continue
            transform = _origin(joint.find("origin"))
            kind = joint.get("type")
            if kind in ("revolute", "continuous"):
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                transform[:3, :3] = transform[:3, :3] @ _rotation(axis, joint_positions[joint.get("name")])
            elif kind != "fixed":
                raise ValueError(f"Unsupported reset-geometry joint type: {kind}")
            poses[joint.find("child").get("link")] = poses[parent] @ transform
        if len(remaining) == len(pending):
            raise ValueError("URDF has unresolved joint parents.")
        pending = remaining

    centers, rotations, half_sizes, radii, half_lengths, kinds = [], [], [], [], [], []
    bounding_radius = 0.0
    for link in robot.findall("link"):
        for collision in link.findall("collision"):
            transform = poses[link.get("name")] @ _origin(collision.find("origin"))
            shape = list(collision.find("geometry"))[0]
            half_size = np.zeros(3)
            radius = half_length = 0.0
            if shape.tag == "box":
                kind = 0
                half_size = np.fromstring(shape.get("size"), sep=" ") / 2
                shape_radius = np.linalg.norm(half_size)
            elif shape.tag == "sphere":
                kind = 1
                radius = shape_radius = float(shape.get("radius"))
            elif shape.tag == "cylinder":
                kind = 2
                radius = float(shape.get("radius"))
                half_length = float(shape.get("length")) / 2
                shape_radius = math.hypot(radius, half_length)
            else:
                raise ValueError(f"Cannot guarantee reset clearance for collision shape: {shape.tag}")
            centers.append(transform[:3, 3])
            rotations.append(transform[:3, :3])
            half_sizes.append(half_size)
            radii.append(radius)
            half_lengths.append(half_length)
            kinds.append(kind)
            bounding_radius = max(bounding_radius, float(np.linalg.norm(transform[:3, 3]) + shape_radius))
    if not centers:
        raise ValueError("URDF contains no supported collision geometry.")
    return CollisionGeometry(
        np.asarray(centers),
        np.asarray(rotations),
        np.asarray(half_sizes),
        np.asarray(radii),
        np.asarray(half_lengths),
        np.asarray(kinds),
        bounding_radius,
    )


def collision_lower_bound(vertical_b: torch.Tensor, geometry: dict[str, torch.Tensor]) -> torch.Tensor:
    """Exact lowest z relative to the root for the cached boxes, spheres and cylinders."""
    directions = torch.einsum("ni,kij->nkj", vertical_b, geometry["rotations"])
    support = (directions.abs() * geometry["half_sizes"]).sum(dim=-1)
    support = torch.where(geometry["kinds"] == 1, geometry["radii"], support)
    cylinder = (
        geometry["radii"] * torch.linalg.vector_norm(directions[..., :2], dim=-1)
        + geometry["half_lengths"] * directions[..., 2].abs()
    )
    support = torch.where(geometry["kinds"] == 2, cylinder, support)
    return (vertical_b @ geometry["centers"].T - support).amin(dim=1)
