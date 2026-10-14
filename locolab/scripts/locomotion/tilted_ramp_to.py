"""Independent jumps onto left/right cross-sloped Tilted Ramp landing surfaces.

Inspired by the bounded goal-centred stones in Extreme Parkour's terrain, not
an implementation of that paper's RL algorithm. Both ramps exist in each scene;
touching the other pad, straddling both, or running off the end is not success.
"""

import math
from dataclasses import asdict, dataclass, fields

import torch
from gap_leap_to import GapLeapTarget, gap_leap_metrics, gap_proposals, geometry_fingerprint


@dataclass(frozen=True)
class TiltedRampTarget(GapLeapTarget):
    side: int = 1  # +Y left / -Y right, relative to the fixed forward +X axis
    platform_length: float = 1.6
    platform_width: float = 0.7
    offset_y: float = 0.4
    center_tolerance_x: float = 0.35
    center_tolerance_y: float = 0.075
    terminal_speed_tolerance: float = 0.15
    terminal_duration: float = 0.25
    air_lateral_fraction: float = 0.5
    tilt_degrees: float = 20.0
    maximum_roll: float = 0.70
    maximum_yaw_error: float = 0.60
    lateral_corridor: float = 1.2

    def __post_init__(self):
        # The base validates its own physical parameters without treating the
        # signed side selector as a positive physical quantity.
        base_names = {field.name for field in fields(GapLeapTarget)}
        GapLeapTarget(**{name: getattr(self, name) for name in base_names})
        if type(self.side) is not int or self.side not in (-1, 1):
            raise ValueError("Platform side must be +1 (left) or -1 (right)")
        for name, value in asdict(self).items():
            if name not in base_names and name != "side" and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"Invalid platform parameter: {name}")
        if 2 * self.offset_y <= self.platform_width + 0.02:
            raise ValueError("Left/right platforms must be physically separate, not overlapping or touching")
        if self.offset_y + self.platform_width / 2 >= self.bank_half_width:
            raise ValueError("Landing pads must fit inside the pit floor bounds")
        if self.platform_width <= 2 * self.landing_margin or self.platform_length <= 2 * self.landing_margin:
            raise ValueError("Landing margin consumes the platform")
        if not 0 < self.air_lateral_fraction <= 1:
            raise ValueError("Airborne lateral fraction must be in (0, 1]")
        if not 0 < self.tilt_degrees <= 30:
            raise ValueError("Cross-slope angle must be in (0, 30] degrees")
        if self.platform_width * self.slope_magnitude / 2 >= self.depth - 0.05:
            raise ValueError("The ramp's low edge must stay above the pit floor")

    @property
    def landing_y(self):
        return self.side * self.offset_y

    @property
    def end_x(self):
        return self.far_x + self.platform_length

    @property
    def center_x(self):
        return self.far_x + self.platform_length / 2

    def rectangle(self, side=None):
        y = (self.side if side is None else side) * self.offset_y
        return (self.far_x, self.end_x, y - self.platform_width / 2, y + self.platform_width / 2)

    @property
    def slope_magnitude(self):
        return math.tan(math.radians(self.tilt_degrees))

    def plane(self, side=None):
        side = self.side if side is None else side
        slope = side * self.slope_magnitude
        scale = math.sqrt(1 + slope * slope)
        return (self.center_x, side * self.offset_y, 0.0), (0.0, -slope / scale, 1.0 / scale)


def rectangle_margins(x, y, rectangle, radius_x=0.0, radius_y=0.0):
    """Signed distances to all FOUR boundaries; optional wheel AABB extents."""
    lo_x, hi_x, lo_y, hi_y = rectangle
    return torch.stack((x - lo_x - radius_x, hi_x - x - radius_x, y - lo_y - radius_y, hi_y - y - radius_y), -1)


def deck_membership(wheels, target, tolerance=0.005):
    x, y = wheels[..., 0], wheels[..., 1]
    source = (x >= target.near_x - target.bank_length - tolerance) & (x <= target.near_x + tolerance)
    source &= y.abs() <= target.bank_half_width + tolerance
    left = (rectangle_margins(x, y, target.rectangle(1)) >= -tolerance).all(-1)
    right = (rectangle_margins(x, y, target.rectangle(-1)) >= -tolerance).all(-1)
    return source, left, right


def tilted_ramp_mesh(target):
    import trimesh

    boxes = [
        (
            target.near_x - target.bank_length,
            target.near_x,
            -target.bank_half_width,
            target.bank_half_width,
            0.0,
            target.depth,
            [155, 165, 170, 255],
        )
    ]
    for side, color in ((1, [100, 140, 185, 255]), (-1, [105, 170, 125, 255])):
        boxes.append((*target.rectangle(side), 0.0, target.depth, color))
    boxes.append(
        (
            target.near_x,
            target.end_x + 2.0,
            -target.bank_half_width,
            target.bank_half_width,
            -target.depth,
            0.1,
            [45, 48, 52, 255],
        )
    )
    parts = []
    for index, (x0, x1, y0, y1, top, thickness, color) in enumerate(boxes):
        part = trimesh.creation.box(extents=(x1 - x0, y1 - y0, thickness))
        part.apply_translation(((x0 + x1) / 2, (y0 + y1) / 2, top - thickness / 2))
        if index in (1, 2):
            side = 1 if index == 1 else -1
            vertices = part.vertices.copy()
            upper = vertices[:, 2] > -target.depth / 2
            vertices[upper, 2] = side * target.slope_magnitude * (vertices[upper, 1] - side * target.offset_y)
            part.vertices = vertices
        part.visual.vertex_colors = color
        parts.append(part)
    return trimesh.util.concatenate(parts)


def validate_ramp_geometry(certificate, target):
    mesh = tilted_ramp_mesh(target)
    if (
        certificate.get("target") != asdict(target)
        or certificate.get("mesh_sha256") != geometry_fingerprint(mesh.vertices, mesh.faces)
        or certificate.get("collision_approximation") != "none"
        or not certificate.get("collision_enabled")
        or not certificate.get("no_extra_static_colliders")
    ):
        raise ValueError("Tilted-ramp collider geometry does not match the requested scene")


def ramp_proposals(samples, low, high, side):
    result = gap_proposals(samples, low, high)
    p = result[..., :70].reshape(result.shape[0], result.shape[1], 7, 10)
    original = samples[..., :70].reshape_as(p)
    # Preserve braking instead of adding the gap solver's rolling-exit boost.
    p[..., 4:, 9] = original[..., 4:, 9]
    strength = torch.linspace(0, 1, result.shape[1], device=result.device)[None, :, None]
    # Couple lateral push with an outward stance proposal. A straight-gap
    # seed can otherwise adduct the uphill legs when common hip is shifted.
    # These are bounded control guesses, not imposed body velocities/forces.
    p[..., :2, 0] -= side * 0.25 * strength
    p[..., :2, 1] += 0.10 * strength
    p[..., 4:, 0] *= 1 - 0.5 * strength
    return result.clamp(low, high)


def cylinder_plane_contact(center, axis, origin, normal):
    """Analytic cylinder support distance and point along an arbitrary normal.

    World-Z wheel bottom minus z(y) is incorrect on a cross slope. Use actual
    wheel axes and the same 86 mm radius / 22.5 mm half-width as the robot.
    """
    origin = center.new_tensor(origin)
    normal = center.new_tensor(normal)
    cosine = (axis * normal).sum(-1).clamp(-1, 1)
    perpendicular = normal - cosine[..., None] * axis
    norm = perpendicular.norm(dim=-1, keepdim=True)
    radial = torch.where(norm > 1e-6, perpendicular / norm.clamp_min(1e-8), 0)
    support = 0.0225 * cosine.sign()[..., None] * axis + 0.086 * radial
    point = center - support
    radius = 0.0225 * cosine.abs() + 0.086 * (1 - cosine.square()).clamp_min(0).sqrt()
    separation = ((center - origin) * normal).sum(-1) - radius
    point += (separation - ((point - origin) * normal).sum(-1))[..., None] * normal
    return separation, point


def edge_separation(center, axis, rectangle, origin, normal):
    """Conservative cylinder-to-finite-plane clearance outside the deck.

    Closest point on the inclined rectangle is found in its orthonormal surface
    coordinates. This checks physical edges without inventing a horizontal
    clearance plane over the void at z=0.
    """
    origin, normal = center.new_tensor(origin), center.new_tensor(normal)
    along = torch.stack((torch.zeros_like(normal[0]), normal[2], -normal[1]))
    relative = center - origin
    u = relative[..., 0].clamp(rectangle[0] - origin[0], rectangle[1] - origin[0])
    v = (relative * along).sum(-1)
    v = v.clamp((rectangle[2] - origin[1]) / along[1], (rectangle[3] - origin[1]) / along[1])
    closest = origin + v[..., None] * along
    closest[..., 0] += u
    delta = center - closest
    distance = delta.norm(dim=-1)
    direction = delta / distance[..., None].clamp_min(1e-8)
    cosine = (axis * direction).sum(-1).clamp(-1, 1)
    radius = 0.0225 * cosine.abs() + 0.086 * (1 - cosine.square()).clamp_min(0).sqrt()
    return distance - radius


def tilted_ramp_metrics(states, target, dt=0.005):
    # Reuse flight-event detection and the unchanged actuator/contact checks,
    # not the gap task's infinite far bank or moving-exit acceptance predicate.
    common = gap_leap_metrics(states, target, dt)
    pos, vel, contacts = states["position"], states["velocity"], states["contacts"]
    wheels = states["wheel_center_position"]
    n, steps, _ = pos.shape
    ids, feet = torch.arange(n, device=pos.device), torch.arange(4, device=pos.device)[None]
    time = torch.arange(steps, device=pos.device)[None]
    takeoff = (common["takeoff_time"] / dt).round().long()
    landing = (common["landing_time"] / dt).round().long()
    touch_indices = (common["first_touchdown_time"] / dt).round().long()
    axes = states["wheel_axis"]
    separation, surface_points = cylinder_plane_contact(wheels, axes, *target.plane())
    touch = surface_points[ids[:, None], touch_indices, feet]
    touch_separation = separation[ids[:, None], touch_indices, feet]
    touch_margins = rectangle_margins(touch[..., 0], touch[..., 1], target.rectangle())
    touch_found = contacts[ids[:, None], touch_indices, feet] & common["landing_observed"][:, None]
    chosen_touch = touch_found & (touch_separation.abs() <= 0.02) & (touch_margins >= target.landing_margin).all(-1)
    center_margins = rectangle_margins(wheels[..., 0], wheels[..., 1], target.rectangle())
    full_margins = rectangle_margins(
        wheels[..., 0],
        wheels[..., 1],
        target.rectangle(),
        states["wheel_forward_radius"],
        states["wheel_lateral_radius"],
    )
    source, left, right = deck_membership(wheels, target)
    on_deck = source | left | right
    source_rectangle = (
        target.near_x - target.bank_length,
        target.near_x,
        -target.bank_half_width,
        target.bank_half_width,
    )
    source_sep, source_point = cylinder_plane_contact(wheels, axes, (0, 0, 0), (0, 0, 1))
    valid_top = (source_sep.abs() <= 0.02) & (
        rectangle_margins(source_point[..., 0], source_point[..., 1], source_rectangle) >= -0.005
    ).all(-1)
    edge_distances = [edge_separation(wheels, axes, source_rectangle, (0, 0, 0), (0, 0, 1))]
    for side in (-1, 1):
        surface_sep, surface_point = cylinder_plane_contact(wheels, axes, *target.plane(side))
        surface_inside = (
            rectangle_margins(surface_point[..., 0], surface_point[..., 1], target.rectangle(side)) >= -0.005
        ).all(-1)
        valid_top |= (surface_sep.abs() <= 0.02) & surface_inside
        edge_distances.append(edge_separation(wheels, axes, target.rectangle(side), *target.plane(side)))
    off_surface_contact = (contacts & ~valid_top).any((1, 2))
    chosen = left if target.side == 1 else right
    after_takeoff = (time >= takeoff[:, None]) & common["landing_observed"][:, None]
    wrong_landing = (contacts & after_takeoff[..., None] & ~chosen).any((1, 2))
    nearest_edge = torch.stack(edge_distances, -1).amin(-1)
    clearance_deficit = (target.edge_clearance - nearest_edge).clamp_min(0).masked_fill(on_deck, 0).amax((1, 2))
    recent = contacts.clone()
    for lag in range(1, round(target.recovery_contact_memory / dt) + 1):
        recent[:, lag:] |= contacts[:, :-lag]
    supported = recent & (separation.abs() <= target.recovery_ground_tolerance)
    secure = (full_margins >= target.landing_margin).all((-1, -2))
    # The landing goal is the support footprint, not the torso origin. On a
    # cross slope a valid tilted stance puts the torso downhill of its feet.
    contact_centroid = surface_points.mean(-2)
    center_x_error = (contact_centroid[..., 0] - target.center_x).abs()
    center_y_error = (contact_centroid[..., 1] - target.landing_y).abs()
    w, qx, qy, qz = states["quaternion"].unbind(-1)
    up = torch.stack((2 * (qx * qz + w * qy), 2 * (qy * qz - w * qx), 1 - 2 * (qx.square() + qy.square())), -1)
    origin, normal = target.plane()
    normal = pos.new_tensor(normal)
    surface_tilt = torch.acos((up * normal).sum(-1).clamp(-1, 1))
    stable = (
        supported.all(-1)
        & secure
        & (vel.norm(dim=-1) <= target.terminal_speed_tolerance)
        & (states["tilt"] < math.radians(target.tilt_degrees) + 0.20)
        & (surface_tilt < math.radians(target.tilt_degrees) + 0.12)
        & (states["omega"].norm(dim=-1) < 1.0)
        & (center_x_error <= target.center_tolerance_x)
        & (center_y_error <= target.center_tolerance_y)
        & (time > landing[:, None])
        & common["landing_observed"][:, None]
    )
    run = torch.zeros(n, device=pos.device, dtype=torch.long)
    for i in range(steps):
        run = torch.where(stable[:, i], run + 1, 0)
    stable_time = run * dt
    lateral_air = (pos[ids, landing, 1] - pos[ids, takeoff, 1]) * target.side
    required_lateral = (target.landing_y - pos[:, 0, 1]).abs() * target.air_lateral_fraction
    initial = common["initial_stable"] & (pos[:, 0, 1].abs() <= 0.05)
    end_speed = vel[:, -1, :2].norm(dim=-1)
    touch_deficit = (target.landing_margin - touch_margins).clamp_min(0)
    final_deficit = (target.landing_margin - full_margins[:, -1]).clamp_min(0)
    body_clearance = ((pos - pos.new_tensor(origin)) * normal).sum(-1)
    after = (time >= landing[:, None]) & common["landing_observed"][:, None]
    sink = (pos[:, :1, 2] - body_clearance).clamp_min(0).masked_fill(~after, 0).amax(-1)
    violations = torch.stack(
        (
            (~initial).float(),
            (~common["landing_observed"]).float(),
            (~common["near_bank_takeoff"]).float().mean(-1),
            touch_deficit.amax((1, 2)) / 0.15,
            final_deficit.amax((1, 2)) / 0.15,
            (touch_separation.abs() - 0.02).clamp_min(0).amax(-1) / 0.1,
            off_surface_contact.float(),
            wrong_landing.float(),
            common["near_bank_return"].float(),
            clearance_deficit / 0.1,
            (required_lateral - lateral_air).clamp_min(0) / 0.1,
            (target.terminal_duration - stable_time).clamp_min(0) / target.terminal_duration,
            (end_speed - target.terminal_speed_tolerance).clamp_min(0) / 0.3,
            (center_x_error[:, -1] - target.center_tolerance_x).clamp_min(0) / 0.2,
            (center_y_error[:, -1] - target.center_tolerance_y).clamp_min(0) / 0.1,
            (common["max_roll"] / target.maximum_roll - 1).clamp_min(0),
            (common["max_pitch"] / target.maximum_pitch - 1).clamp_min(0),
            (common["max_yaw_error"] / target.maximum_yaw_error - 1).clamp_min(0),
            (common["maximum_lateral_excursion"] - target.lateral_corridor).clamp_min(0) / 0.2,
            (sink - target.maximum_landing_sink).clamp_min(0) / 0.1,
            (common["max_nonfoot_contact_force"] - 5).clamp_min(0) / 200,
            (common["max_joint_speed_excess"] - 0.05).clamp_min(0),
            (common["max_joint_limit_violation"] - 0.002).clamp_min(0) / 0.02,
        ),
        -1,
    )
    accepted = (
        initial
        & common["landing_observed"]
        & common["near_bank_takeoff"].all(-1)
        & chosen_touch.all(-1)
        & secure[:, -1]
        & ~wrong_landing
        & ~off_surface_contact
        & ~common["near_bank_return"]
        & (clearance_deficit <= 1e-6)
        & (lateral_air >= required_lateral - 1e-6)
        & (stable_time >= target.terminal_duration - 1e-6)
        & (common["max_roll"] <= target.maximum_roll)
        & (common["max_pitch"] <= target.maximum_pitch)
        & (common["max_yaw_error"] <= target.maximum_yaw_error)
        & (common["maximum_lateral_excursion"] <= target.lateral_corridor)
        & (sink <= target.maximum_landing_sink)
        & (common["max_nonfoot_contact_force"] <= 5)
        & (common["max_joint_speed_excess"] <= 0.05)
        & (common["max_joint_limit_violation"] <= 0.002)
    )
    total = violations.square().sum(-1)
    score = 100 * total + 10 * common["max_pitch"].square() + 5 * center_y_error[:, -1].square()
    result = dict(common)
    result.update(
        score=score,
        accepted=accepted,
        initial_stable=initial,
        far_bank_touchdown=chosen_touch,
        wheel_fully_crossed=(full_margins[:, -1] >= target.landing_margin).all(-1),
        gap_clearance_deficit=clearance_deficit,
        gap_contact=off_surface_contact,
        chosen_platform_touchdown=chosen_touch,
        first_touchdown_y=touch[..., 1],
        first_touchdown_surface_clearance=touch_separation,
        final_surface_tilt=surface_tilt[:, -1],
        maximum_landing_sink=sink,
        first_touchdown_margins=touch_margins,
        final_wheel_margins=full_margins[:, -1],
        minimum_selected_pad_center_margin=center_margins[:, -1].amin((1, 2)),
        wrong_platform_contact=wrong_landing,
        off_surface_contact=off_surface_contact,
        platform_clearance_deficit=clearance_deficit,
        airborne_lateral_distance=lateral_air,
        required_airborne_lateral_distance=required_lateral,
        terminal_stable_duration=stable_time,
        final_speed_xy=end_speed,
        final_y=pos[:, -1, 1],
        final_contact_centroid=contact_centroid[:, -1],
        final_center_error_x=center_x_error[:, -1],
        final_center_error_y=center_y_error[:, -1],
        search_safety_violation=torch.zeros_like(total),
        search_max_violation=violations.amax(-1),
        search_total_violation=total,
    )
    return result
