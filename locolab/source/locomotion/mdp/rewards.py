from __future__ import annotations

from typing import TYPE_CHECKING

import torch

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ==============================================================================
# TASK REWARDS
# Rewards directly tied to achieving locomotion objectives
# ==============================================================================

"""
Survival rewards.
"""


def stay_alive_startup(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for staying alive during the startup phase."""
    mean_episode_length = env.episode_length_buf.float().mean()
    max_episode_length = env.max_episode_length
    ratio = ((max_episode_length - mean_episode_length * 2) / max_episode_length).clip(0, 1)
    return torch.ones(env.num_envs, device=env.device) * ratio


"""
Velocity tracking rewards.
"""


def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy) in the gravity aligned robot frame using exponential kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    # Transform velocity to yaw-aligned frame (gravity aligned, only yaw rotation)
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


"""
Base stability rewards.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base height deviation from target using L2 squared kernel.

    Uses height scanner to measure ground-relative height.

    Args:
        env: The environment object.
        target_height: The desired base height above ground.
        sensor_cfg: Configuration for height scanner sensor.
        asset_cfg: Configuration for the robot asset.
    """
    from isaaclab.sensors import RayCaster

    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    # Compute height above ground: root_z - ray_hit_z
    height = asset.data.root_pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[:, :, 2]
    # ray_hits_w can be inf when no hit, replace with target_height to avoid NaN
    height = torch.nan_to_num(height, nan=target_height, posinf=target_height, neginf=target_height)

    return torch.square(height.mean(dim=1) - target_height)


def base_height_tolerance(
    env: ManagerBasedRLEnv,
    target_height: float,
    tolerance: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base height deviation from target with a deadzone around the target height."""
    from isaaclab.sensors import RayCaster

    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    height = asset.data.root_pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[:, :, 2]
    height = torch.nan_to_num(height, nan=target_height, posinf=target_height, neginf=target_height)
    mean_height = height.mean(dim=1)

    return torch.clamp(torch.abs(mean_height - target_height) - tolerance, min=0.0)


# ==============================================================================
# REGULARIZATION REWARDS
# Constraints for smooth and efficient behavior
# ==============================================================================

"""
Joint regularization.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint deviation from default when command is near zero."""
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


def joint_position_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    ang_vel_threshold: float = 0.1,
    moving_scale: float = 0.25,
) -> torch.Tensor:
    """Penalize joint position error from default with command-based smooth gating.

    The penalty is strongest near zero command and decays smoothly as linear and angular
    velocity commands increase.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command("base_velocity")
    cmd_lin = torch.linalg.norm(cmd[:, :2], dim=1)
    cmd_ang = torch.abs(cmd[:, 2])

    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_deviation = torch.linalg.norm(joint_pos - default_pos, dim=1)

    lin_gate = torch.clamp(1.0 - cmd_lin / velocity_threshold, min=0.0, max=1.0)
    ang_gate = torch.clamp(1.0 - cmd_ang / ang_vel_threshold, min=0.0, max=1.0)
    idle_gate = lin_gate * ang_gate

    scale = moving_scale + (stand_still_scale - moving_scale) * idle_gate
    return scale * joint_deviation


"""
Feet regularization.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_lift_on_stumble(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    stumble_ratio: float = 2.5,
    target_height: float = 0.08,
) -> torch.Tensor:
    """当脚撞侧面（stumble）时，奖励该脚抬高到 target_height 以上。

    逻辑：
      1. 检测哪只脚正在 stumble（水平力 > stumble_ratio * 竖直力）
      2. 只对这些脚，奖励其绝对高度超过 target_height 的部分
      3. 高度信号比速度信号更稳定，策略可以学到"撞到后快速抬腿"

    改动说明（vs 原版）：
      - 奖励由 vel_z 改为 clamp(foot_height - target_height, min=0)
      - stumble_ratio 从 4.0 降到 2.5，让 stumble 检测更容易触发
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    # stumble 检测：哪些脚撞了侧面 (N, num_feet)
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    is_stumble = forces_xy > stumble_ratio * forces_z  # (N, num_feet)

    # 脚的绝对高度（世界坐标 z）(N, num_feet)
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    # 只在 stumble 的脚上，奖励高于 target_height 的部分
    lift_reward = torch.clamp(foot_height - target_height, min=0.0) * is_stumble.float()

    return torch.sum(lift_reward, dim=1)


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Gait regularization.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Symmetry regularization.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward


"""
DOF limits and power regularization.
"""


def dof_power_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint power consumption using L1 kernel (|torque * velocity|)."""
    asset: Articulation = env.scene[asset_cfg.name]
    power = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]) * torch.abs(
        asset.data.joint_vel[:, asset_cfg.joint_ids]
    )
    return torch.sum(power, dim=-1)


def joint_pos_limits_torque_regularization(
    env: ManagerBasedRLEnv,
    pos_limit_threshold: float,
    torque_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joints that apply torque while near position limits.

    This prevents the robot from pushing against joint limits.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]

    # Get joint limits
    soft_lower = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    soft_upper = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]

    # Check if near limits
    near_lower = joint_pos < (soft_lower + pos_limit_threshold)
    near_upper = joint_pos > (soft_upper - pos_limit_threshold)

    # Penalize torque in wrong direction near limits
    # Near lower limit: penalize negative torque (pushing into limit)
    # Near upper limit: penalize positive torque (pushing into limit)
    torque_violation = torch.zeros_like(joint_pos)
    torque_violation = torch.where(
        near_lower & (joint_torque < -torque_threshold), torch.abs(joint_torque), torque_violation
    )
    torque_violation = torch.where(
        near_upper & (joint_torque > torque_threshold), torch.abs(joint_torque), torque_violation
    )

    return torch.sum(torque_violation, dim=-1)


def joint_pos_limits_vel_regularization(
    env: ManagerBasedRLEnv,
    pos_limit_threshold: float,
    vel_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joints moving towards limits at high velocity.

    This prevents the robot from slamming into joint limits.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    # Get joint limits
    soft_lower = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    soft_upper = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]

    # Check if near limits
    near_lower = joint_pos < (soft_lower + pos_limit_threshold)
    near_upper = joint_pos > (soft_upper - pos_limit_threshold)

    # Penalize velocity towards limits
    # Near lower limit: penalize negative velocity (moving towards limit)
    # Near upper limit: penalize positive velocity (moving towards limit)
    vel_violation = torch.zeros_like(joint_pos)
    vel_violation = torch.where(near_lower & (joint_vel < -vel_threshold), torch.abs(joint_vel), vel_violation)
    vel_violation = torch.where(near_upper & (joint_vel > vel_threshold), torch.abs(joint_vel), vel_violation)

    return torch.sum(vel_violation, dim=-1)


def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize deviation from default joint positions using L1 kernel.

    Uses body_names to identify which joints to penalize based on connected links.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Use joint_ids if available, otherwise use all joints
    if asset_cfg.joint_ids is not None:
        joint_ids = asset_cfg.joint_ids
    else:
        joint_ids = slice(None)

    deviation = torch.abs(asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids])
    return torch.sum(deviation, dim=-1)


"""
Action regularization.
"""


def action_rate_l2_mask(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the rate of change of specific actions using L2 squared kernel.

    Args:
        env: The environment object.
        asset_cfg: Configuration for the asset with joint_names to specify which actions to penalize.
    """
    action_diff = (
        env.action_manager.action[:, asset_cfg.joint_ids] - env.action_manager.prev_action[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.square(action_diff), dim=1)


def action_smoothness_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize action jerk using second-order finite difference.

    Computes the second derivative of actions: current - 2*prev + prev_prev.
    Returns zero penalty for the first 3 steps (startup phase masking).

    Args:
        env: The environment object.
        asset_cfg: Configuration for the asset with joint_names to specify which actions to penalize.
    """
    # Get full action tensors
    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action

    # Initialize prev_prev_action on first call (store full action, not sliced)
    if not hasattr(env, "_action_smoothness_prev_prev_action"):
        env._action_smoothness_prev_prev_action = prev_action.clone()
        return torch.zeros(env.num_envs, device=env.device)

    # Check for shape mismatch (can happen after reset) and reinitialize if needed
    if env._action_smoothness_prev_prev_action.shape != prev_action.shape:
        env._action_smoothness_prev_prev_action = prev_action.clone()
        return torch.zeros(env.num_envs, device=env.device)

    # Compute second-order difference (jerk) on selected joints: current - 2*prev + prev_prev
    second_diff = (
        current_action[:, asset_cfg.joint_ids]
        - 2 * prev_action[:, asset_cfg.joint_ids]
        + env._action_smoothness_prev_prev_action[:, asset_cfg.joint_ids]
    )

    # Compute L2 squared penalty
    penalty = torch.sum(torch.square(second_diff), dim=1)

    # Mask penalty for startup phase (first 3 steps)
    penalty = torch.where(env.episode_length_buf < 3, torch.zeros_like(penalty), penalty)

    # Update prev_prev_action for next step (store full action)
    env._action_smoothness_prev_prev_action = prev_action.clone()

    return penalty


# ==============================================================================
# WHEEL-LEGGED ROBOT REWARDS
# Rewards specifically designed for wheel-legged robots
# ==============================================================================


def wheels_velocity_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    wheel_radius: float = 0.05,
    wheel_base: float = 0.3,
    std: float = 0.5,
) -> torch.Tensor:
    """Reward wheels for matching the expected velocity based on body velocity command.

    For a differential drive robot:
    - v_left = v_body - omega * wheel_base / 2
    - v_right = v_body + omega * wheel_base / 2
    - wheel_angular_vel = v_wheel / wheel_radius

    Args:
        env: The environment object.
        command_name: Name of the velocity command.
        asset_cfg: Configuration with wheel joint names (expects 4 wheels: FL, FR, RL, RR foot joints).
        wheel_radius: Radius of the wheels in meters.
        wheel_base: Distance between left and right wheels in meters.
        std: Standard deviation for exponential kernel.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)

    # Get commanded linear and angular velocity
    cmd_lin_vel_x = cmd[:, 0]  # Forward velocity
    cmd_ang_vel_z = cmd[:, 2]  # Yaw rate

    # Compute expected wheel linear velocities (differential drive)
    # Left wheels: v - omega * d/2, Right wheels: v + omega * d/2
    expected_vel_left = cmd_lin_vel_x - cmd_ang_vel_z * wheel_base / 2
    expected_vel_right = cmd_lin_vel_x + cmd_ang_vel_z * wheel_base / 2

    # Convert to wheel angular velocity: omega = v / r
    # Positive angular velocity = forward motion (verified by testing)
    expected_omega_left = expected_vel_left / wheel_radius
    expected_omega_right = expected_vel_right / wheel_radius

    # Get actual wheel velocities (assuming order: FL, FR, RL, RR)
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    # Compute error for each wheel (FL=0, FR=1, RL=2, RR=3)
    # Left wheels: FL(0), RL(2); Right wheels: FR(1), RR(3)
    error_fl = torch.square(wheel_vel[:, 0] - expected_omega_left)
    error_fr = torch.square(wheel_vel[:, 1] - expected_omega_right)
    error_rl = torch.square(wheel_vel[:, 2] - expected_omega_left)
    error_rr = torch.square(wheel_vel[:, 3] - expected_omega_right)

    total_error = (error_fl + error_fr + error_rl + error_rr) / 4.0
    # Use softer exponential to avoid vanishing reward
    # Scale error by std^2 before exp to handle large angular velocities
    return torch.exp(-total_error / (std**2 * 100))


def legs_static_when_moving(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    velocity_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize leg joint velocity when robot should be moving with wheels.

    When the robot has a non-zero velocity command, the legs should remain relatively
    static while the wheels do the work.

    Args:
        env: The environment object.
        command_name: Name of the velocity command.
        asset_cfg: Configuration with leg joint names.
        velocity_threshold: Command magnitude threshold to activate penalty.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_norm = torch.linalg.norm(cmd[:, :2], dim=1)

    # Get leg joint velocities
    leg_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    leg_vel_penalty = torch.sum(torch.square(leg_vel), dim=1)

    # Only penalize when there's a velocity command (robot should be using wheels)
    return torch.where(cmd_norm > velocity_threshold, leg_vel_penalty, torch.zeros_like(leg_vel_penalty))


def wheels_air_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    min_force: float = 1.0,
    ang_vel_threshold: float = 0.2,
) -> torch.Tensor:
    """Penalize if any wheel loses ground contact when not turning sharply.

    When angular velocity command is low (< threshold), all wheels should maintain
    ground contact. Returns 1.0 if ANY wheel is in the air, 0.0 otherwise.

    Args:
        env: The environment object.
        command_name: Name of the velocity command.
        sensor_cfg: Configuration for contact sensor with wheel body names.
        min_force: Minimum contact force to consider as "in contact".
        ang_vel_threshold: Angular velocity threshold. Penalty only applies when |ang_vel| < threshold.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2]

    # Check if each wheel is in contact (positive z force)
    max_forces = torch.max(contact_forces, dim=1)[0]
    not_in_contact = (max_forces < min_force).float()

    # Binary: 1.0 if ANY wheel is in air, 0.0 if all wheels on ground
    any_wheel_in_air = (torch.sum(not_in_contact, dim=1) > 0).float()

    # Only penalize when angular velocity command is low (not turning)
    cmd = env.command_manager.get_command(command_name)
    cmd_ang = torch.abs(cmd[:, 2])
    is_not_turning = (cmd_ang < ang_vel_threshold).float()

    return any_wheel_in_air * is_not_turning


def no_leg_ground_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize leg bodies (hip, thigh, calf) for contacting the ground.

    For a wheeled robot, only wheels should contact the ground.

    Args:
        env: The environment object.
        sensor_cfg: Configuration for contact sensor with leg body names (hip, thigh, calf).
        threshold: Force threshold to consider as contact.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)

    # Penalize any leg contact
    leg_contact = (contact_forces > threshold).float()
    return torch.sum(leg_contact, dim=1)


def differential_drive_symmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward symmetric wheel velocities between front and rear wheels on same side.

    For stable wheeled locomotion, FL should match RL and FR should match RR.

    Args:
        env: The environment object.
        asset_cfg: Configuration with wheel joint names (FL, FR, RL, RR order).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    # FL(0) should match RL(2), FR(1) should match RR(3)
    left_diff = torch.square(wheel_vel[:, 0] - wheel_vel[:, 2])
    right_diff = torch.square(wheel_vel[:, 1] - wheel_vel[:, 3])

    # Return negative of error (higher is better)
    return torch.exp(-(left_diff + right_diff) / 2.0)


def wheels_spinning_when_commanded(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    min_wheel_vel: float = 0.5,
) -> torch.Tensor:
    """Reward wheels for spinning when there's a velocity command.

    This encourages the robot to use wheels instead of legs for locomotion.

    Args:
        env: The environment object.
        command_name: Name of the velocity command.
        asset_cfg: Configuration with wheel joint names.
        min_wheel_vel: Minimum wheel velocity to consider as "spinning".
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_norm = torch.linalg.norm(cmd[:, :2], dim=1)

    # Get wheel velocities
    wheel_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    avg_wheel_vel = torch.mean(wheel_vel, dim=1)

    # Reward when: command exists AND wheels are spinning
    has_command = (cmd_norm > 0.1).float()
    wheels_active = (avg_wheel_vel > min_wheel_vel).float()

    return has_command * wheels_active
