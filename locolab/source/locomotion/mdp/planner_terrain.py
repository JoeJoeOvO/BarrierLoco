"""Ground-truth terrain and destination interface; no learned skill routing yet."""

import torch


def custom_course(env):
    # First four columns are the explicit forward courses; Rough starts centrally.
    return env.scene.terrain.terrain_types < 4


def goal_distance(env):
    return torch.where(custom_course(env), 13.0, 6.0)


def goal_vector(env):
    robot = env.scene["robot"]
    remaining = env.scene.env_origins[:, :2] - robot.data.root_pos_w[:, :2]
    remaining[:, 0] += goal_distance(env)
    w, x, y, z = robot.data.root_quat_w.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = yaw.cos(), yaw.sin()
    return torch.stack((c * remaining[:, 0] + s * remaining[:, 1], -s * remaining[:, 0] + c * remaining[:, 1]), -1)


def height_map(env):
    sensor = env.scene["height_scanner"]
    height = env.scene["robot"].data.root_pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
    return torch.nan_to_num(height, nan=2.0, posinf=2.0, neginf=-2.0).clamp(-2.0, 2.0)


def height_validity(env):
    return torch.isfinite(env.scene["height_scanner"].data.ray_hits_w[..., 2]).float()


def reached_goal(env):
    return goal_vector(env).norm(dim=-1) < 0.5


def left_course(env):
    local = env.scene["robot"].data.root_pos_w - env.scene.env_origins
    # Leave a 1 m margin to neighboring lanes; bypassing the obstacle is not success.
    custom = custom_course(env)
    minimum = torch.where(custom, -1.0, -7.0)
    maximum = torch.where(custom, 14.0, 7.0)
    lateral = torch.where(custom, 2.0, 7.0)
    return (local[:, 0] < minimum) | (local[:, 0] > maximum) | (local[:, 1].abs() > lateral)


def fell_into_gap(env):
    # Descending Rough stairs/slopes can legitimately be below their spawn elevation.
    gap = (env.scene.terrain.terrain_types == 2) | (env.scene.terrain.terrain_types == 3)
    return gap & (env.scene["robot"].data.root_pos_w[:, 2] < env.scene.env_origins[:, 2] - 0.35)
