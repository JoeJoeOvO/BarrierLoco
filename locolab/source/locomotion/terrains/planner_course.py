"""Deterministic lane geometry for Planner development, independent of Isaac imports."""

from dataclasses import dataclass

import numpy as np
import trimesh

FAMILIES = ("flat", "platform", "gap", "mixed")


@dataclass(frozen=True)
class CourseLayout:
    family: str
    difficulty: float
    size: tuple[float, float] = (16.0, 6.0)
    spawn_x: float = 1.5
    goal_x: float = 14.5
    height_override: float | None = None
    gap_width_override: float | None = None
    gap_start_override: float | None = None
    platform_start: float = 4.0
    platform_length: float = 2.0

    def __post_init__(self):
        if self.family not in FAMILIES or not 0 <= self.difficulty <= 1:
            raise ValueError("Unknown family or difficulty outside [0, 1]")
        if self.size[0] != 16.0 or self.size[1] < 6.0:
            raise ValueError("This course uses a 16 m length and at least 6 m width")

    @property
    def height(self):
        return self.height_override if self.height_override is not None else 0.20 + 0.40 * self.difficulty

    @property
    def gap_width(self):
        return self.gap_width_override if self.gap_width_override is not None else 0.20 + 0.40 * self.difficulty

    @property
    def gaps(self):
        start = 10.0 if self.family == "mixed" else 6.0
        start = self.gap_start_override if self.gap_start_override is not None else start
        return [(start, start + self.gap_width)] if self.family in ("gap", "mixed") else []

    @property
    def platforms(self):
        return (
            [(self.platform_start, self.platform_start + self.platform_length, self.height)]
            if self.family in ("platform", "mixed")
            else []
        )

    def surface_height(self, x):
        """Analytic collision-surface oracle for tests and dataset metadata."""
        if any(left < x < right for left, right in self.gaps):
            return -1.0
        for left, right, height in self.platforms:
            if left < x < right:
                return height
        return 0.0


def build_course(layout):
    length, width = layout.size
    meshes = []

    def box(left, right, bottom, top):
        mesh = trimesh.creation.box(extents=(right - left, width, top - bottom))
        mesh.apply_translation(((left + right) / 2, width / 2, (bottom + top) / 2))
        meshes.append(mesh)

    # Real gaps: split the support floor, never place a continuous z=0 slab below.
    start = 0.0
    for left, right in layout.gaps:
        box(start, left, -0.2, 0.0)
        box(left, right, -1.2, -1.0)
        start = right
    box(start, length, -0.2, 0.0)
    for left, right, height in layout.platforms:
        box(left, right, 0.0, height)
    return meshes, np.array([layout.spawn_x, width / 2, 0.0])


def planner_course_terrain(difficulty, cfg):
    layout = CourseLayout(cfg.family, float(np.clip(difficulty, 0, 1)), tuple(cfg.size))
    if getattr(cfg, "vary_geometry", False):
        from dataclasses import replace

        # Local RNG makes geometry reproducible from tile difficulty and seed,
        # without depending on unrelated global NumPy calls.
        rng = np.random.default_rng(int(difficulty * 1e9) + 137)
        layout = replace(
            layout,
            height_override=float(rng.uniform(0.15, 0.65)),
            gap_width_override=float(rng.uniform(0.2, 0.6)),
            gap_start_override=float(rng.uniform(9.0, 12.0) if cfg.family == "mixed" else rng.uniform(4.0, 11.0)),
            platform_start=float(rng.uniform(3.0, 4.5)),
            platform_length=float(rng.uniform(1.0, 3.0)),
        )
    return build_course(layout)
