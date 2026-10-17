"""Single-scan inverse sensor model. Unknown space is never filled as free.

Local coordinates follow base yaw, with gravity-aligned Z and origin at base.
This uses first-return rays, not a height-map extrusion; overhang occlusion is
preserved. No temporal fusion or simulated robot self-occlusion is applied yet.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OccupancySpec:
    minimum: tuple = (-1.6, -2.4, -1.6)
    shape_xyz: tuple = (32, 24, 16)
    resolution: float = 0.2
    max_range: float = 8.0

    @property
    def tensor_shape(self):
        return (2, *reversed(self.shape_xyz))


def yaw_rotate(points, quaternion, inverse=False):
    w, x, y, z = quaternion.unbind(-1)
    angle = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    if inverse:
        angle = -angle
    while angle.ndim < points.ndim - 1:
        angle = angle.unsqueeze(-1)
    c, s = angle.cos(), angle.sin()
    return torch.stack(
        (c * points[..., 0] - s * points[..., 1], s * points[..., 0] + c * points[..., 1], points[..., 2]), -1
    )


def lidar_directions(device):
    elevation = torch.linspace(-85, 35, 24, device=device) * torch.pi / 180
    azimuth = torch.arange(120, device=device) * (2 * torch.pi / 120)
    el, az = torch.meshgrid(elevation, azimuth, indexing="ij")
    return torch.stack((el.cos() * az.cos(), el.cos() * az.sin(), el.sin()), -1).reshape(-1, 3)


def voxelize_rays(directions, distances, spec=OccupancySpec()):
    """Distances (N,R): finite first hit, +inf no return, NaN invalid sensor ray.

    All rays originate at local (0,0,0); hit voxels override free ray samples.
    Ray marching at half-voxel spacing approximates free-space carving; untouched
    voxels, including surfaces behind the first return, retain known=0.
    """
    n, rays = distances.shape
    device = distances.device
    nx, ny, nz = spec.shape_xyz
    count = nx * ny * nz
    known = torch.zeros((n, count), dtype=torch.bool, device=device)
    occupied = torch.zeros_like(known)
    lower = distances.new_tensor(spec.minimum)
    shape = torch.tensor(spec.shape_xyz, device=device)
    batch_offset = torch.arange(n, device=device)[:, None] * count

    def scatter(buffer, points, valid):
        indices = torch.floor((points - lower) / spec.resolution).long()
        inside = ((indices >= 0) & (indices < shape)).all(-1) & valid
        flat = indices[..., 0] + nx * (indices[..., 1] + ny * indices[..., 2])
        addresses = flat.reshape(n, -1) + batch_offset
        buffer.view(-1)[addresses[inside.reshape(n, -1)]] = True

    samples = torch.arange(0, spec.max_range, spec.resolution / 2, device=device)
    for first in range(0, rays, 128):
        distance = distances[:, first : first + 128]
        direction = directions[first : first + 128]
        valid = ~torch.isnan(distance) & (distance > 0)
        length = distance.clamp(max=spec.max_range)
        points = direction[None, :, None, :] * samples[None, None, :, None]
        points = points.expand(n, -1, -1, -1)
        free = valid[..., None] & (samples[None, None] < length[..., None])
        scatter(known, points, free)
        hits = direction[None] * torch.nan_to_num(length)[..., None]
        hit_mask = valid & torch.isfinite(distance) & (distance < spec.max_range)
        scatter(occupied, hits, hit_mask)
    known |= occupied
    return torch.stack((occupied, known), 1).reshape(n, *spec.tensor_shape).float()
