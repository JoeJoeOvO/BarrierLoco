"""Closed trench meshes for continuous navigation, independent of Isaac imports."""

import numpy as np
import trimesh

GAP_FAMILIES = ("single_gap", "double_gap", "diagonal_gap", "zigzag_gap", "variable_gap")


def gap_bands(difficulty, family, size=(16.0, 16.0)):
    """Return y bands and their ordered trench intervals at both endpoints."""
    rng = np.random.default_rng(int(difficulty * 1e9) + 719 + GAP_FAMILIES.index(family))
    width = float(rng.uniform(0.15, 0.9))
    depth = float(rng.uniform(0.6, 1.4))
    centers = [float(rng.uniform(6.0, 8.0))]
    if family == "double_gap":
        centers = [float(rng.uniform(4.5, 5.5)), float(rng.uniform(10, 12))]
    slope = float(rng.choice([-1, 1]) * rng.uniform(0.12, 0.25)) if family == "diagonal_gap" else 0.0
    ys = np.linspace(0, size[1], 17 if family in ("zigzag_gap", "variable_gap") else 2)
    shifts = slope * (ys - size[1] / 2)
    if family == "zigzag_gap":
        shifts = rng.uniform(0.4, 1.0) * np.interp(ys, [0, 4, 8, 12, 16], [-1, 1, -1, 1, -1])
    widths = np.full(len(ys), width)
    if family == "variable_gap":
        widths = rng.uniform(0.15, 0.9, len(ys))
    bands = []
    for i in range(len(ys) - 1):
        intervals = [(np.array([c + shifts[i], c + shifts[i + 1]]), widths[i : i + 2].copy()) for c in centers]
        bands.append((ys[i], ys[i + 1], intervals))
    return bands, depth


def trench_prism(left, right, y0, y1, bottom, top):
    """Watertight trapezoidal prism; no support slab bridges the trench."""
    xy = [(left[0], y0), (right[0], y0), (right[1], y1), (left[1], y1)]
    vertices = [(x, y, z) for z in (bottom, top) for x, y in xy]
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7)]
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        faces += [(a, b, b + 4), (a, b + 4, a + 4)]
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def diverse_gap_terrain(difficulty, cfg):
    bands, depth = gap_bands(difficulty, cfg.family, cfg.size)
    meshes = []
    for y0, y1, intervals in bands:
        start = np.zeros(2)
        for left, width in intervals:
            right = left + width
            meshes.append(trench_prism(start, left, y0, y1, -depth - 0.2, 0))
            meshes.append(trench_prism(left, right, y0, y1, -depth - 0.2, -depth))
            start = right
        meshes.append(trench_prism(start, np.full(2, cfg.size[0]), y0, y1, -depth - 0.2, 0))
    return [trimesh.util.concatenate(meshes)], np.array([1.5, cfg.size[1] / 2, 0])
