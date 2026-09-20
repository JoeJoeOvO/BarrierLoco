"""WTW parameter-conditioned rewards, with finite-cylinder wheel adaptation.

Released CoRLRewards costs are expressed as non-negative costs here; environment
weights carry the negative sign. This preserves the original optimization direction.
"""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

from .wtw_math import contact_force_cost, contact_velocity_cost, swing_height_target, wheel_contact_geometry


def wtw_commands(env, command_name: str = "base_velocity"):
    """Ten scaled command entries, no categorical gait label."""
    command = env.command_manager.get_command(command_name)
    scales = command.new_tensor([2.0, 2.0, 0.25, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    return command * scales


def wtw_clocks(env, command_name: str = "base_velocity"):
    return env.command_manager.get_term(command_name).clock_inputs


def wtw_joint_deviation_l1(env, asset_cfg: SceneEntityCfg, command_name: str = "base_velocity"):
    """Pose cost relative to height IK, independent of action/observation offsets."""
    term = env.command_manager.get_term(command_name)
    robot = env.scene[asset_cfg.name]
    error = robot.data.joint_pos[:, asset_cfg.joint_ids] - term.reward_joint_pos[:, asset_cfg.joint_ids]
    return error.abs().sum(-1)


def wtw_stand_still(env, asset_cfg: SceneEntityCfg, command_name: str = "base_velocity"):
    """Height-conditioned nominal-leg-pose cost for near-zero velocity and no stepping.

    Height/phase parameters are not velocities. Including them in the zero-command
    norm would silently disable this cost for every ten-dimensional command.
    """
    command = env.command_manager.get_command(command_name)
    error = wtw_joint_deviation_l1(env, asset_cfg, command_name)
    stationary = (command[:, :3].norm(dim=-1) < 0.1) & (command[:, 8] == 1)
    return error * stationary


def _wheel_state(term):
    robot = term.robot.data
    ids = term.foot_body_ids
    orientation = robot.body_link_quat_w[:, ids]
    local_axis = torch.zeros_like(robot.body_link_pos_w[:, ids])
    local_axis[..., 1] = 1
    axis = quat_apply(orientation, local_axis)
    displacement, velocity = wheel_contact_geometry(
        axis,
        robot.body_link_lin_vel_w[:, ids],
        robot.body_ang_vel_w[:, ids],
        term.cfg.wheel_radius,
        term.cfg.wheel_half_width,
        term.center_offsets,
    )
    return robot.body_link_pos_w[:, ids] + displacement, velocity


def wtw_swing_force(env, command_name: str = "base_velocity"):
    term = env.command_manager.get_term(command_name)
    force = term.sensor.data.net_forces_w[:, term.sensor_body_ids]
    return contact_force_cost(force, term.desired_contact)


def wtw_stance_velocity(env, command_name: str = "base_velocity"):
    term = env.command_manager.get_term(command_name)
    _, velocity = _wheel_state(term)
    # A rolling wheel center moves while the material point at contact does not.
    return contact_velocity_cost(velocity, term.desired_contact)


def wtw_clearance(env, command_name: str = "base_velocity"):
    term = env.command_manager.get_term(command_name)
    bottom, _ = _wheel_state(term)
    height = bottom[..., 2] - env.scene.env_origins[:, 2:3]
    target = swing_height_target(term.leg_phase, term.command)
    return ((height - target).square() * (1 - term.desired_contact)).sum(-1)


def wtw_height(env, command_name: str = "base_velocity"):
    term = env.command_manager.get_term(command_name)
    height = term.robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (height - term.command[:, 3]).square()


def wtw_swing_wheel_velocity(env, command_name: str = "base_velocity"):
    term = env.command_manager.get_term(command_name)
    velocity = term.robot.data.joint_vel[:, term.wheel_joint_ids]
    return ((1 - term.desired_contact) * velocity.square()).mean(-1)


def wtw_raibert(env, command_name: str = "base_velocity", stance_width: float = 0.38, stance_length: float = 0.3868):
    """WTW phase-based foot placement, with Go2W dimensions; disabled for rolling."""
    term = env.command_manager.get_term(command_name)
    robot = term.robot.data
    # Use collision centers, not link frames offset from the wheel centers.
    local = torch.zeros_like(robot.body_link_pos_w[:, term.foot_body_ids])
    local[..., 1] = term.center_offsets
    centers = robot.body_link_pos_w[:, term.foot_body_ids] + quat_apply(
        robot.body_link_quat_w[:, term.foot_body_ids], local
    )
    quat = yaw_quat(robot.root_quat_w)[:, None, :].expand(-1, 4, -1)
    positions = quat_apply_inverse(quat, centers - robot.root_pos_w[:, None, :])
    nominal = positions.new_tensor(
        [
            [stance_length / 2, -stance_width / 2],
            [stance_length / 2, stance_width / 2],
            [-stance_length / 2, -stance_width / 2],
            [-stance_length / 2, stance_width / 2],
        ]
    )
    phase = (1 - 2 * term.leg_phase).abs() - 0.5
    stepping = term.command[:, 8:9] < 1
    frequency = torch.where(stepping, term.command[:, 4:5], 1.0)
    half_period = 0.5 / frequency.clamp_min(1e-6)
    x_offset = phase * term.command[:, 0:1] * half_period
    y_offset = phase * term.command[:, 2:3] * stance_length / 2 * half_period
    y_offset[:, 2:] *= -1
    target = nominal[None] + torch.stack((x_offset, y_offset), dim=-1)
    error = (positions[..., :2] - target).square().sum((1, 2))
    return torch.where(stepping.squeeze(-1), error, 0.0)
