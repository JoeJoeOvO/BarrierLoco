"""自定义 event 函数，补充 isaaclab 未提供的随机化功能。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """随机化刚体惯量张量的对角分量（Ixx, Iyy, Izz）。

    参考 LocoLeggedWheel/locoleggedwheel/mdp/events.py 实现移植，
    isaaclab 未提供此函数，需要在 OmniBot 中自行实现。
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # 从物理引擎获取当前惯量张量 (num_envs, num_bodies, 9)
    inertias = asset.root_physx_view.get_inertias()

    # 先恢复到默认值，再随机化（避免累积误差）
    inertias[env_ids[:, None], body_ids, :] = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()

    # 对对角分量（indices 0, 4, 8 对应 Ixx, Iyy, Izz）分别随机化
    for idx in [0, 4, 8]:
        inertias[env_ids[:, None], body_ids, idx] = _randomize_prop_by_op(
            inertias[:, :, idx],
            inertia_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )

    asset.root_physx_view.set_inertias(inertias, env_ids)


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """按指定分布和操作对数据进行随机化，返回随机化后的子张量。"""
    # 解析维度大小
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]

    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # 选择采样函数
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(f"Unknown distribution: '{distribution}'")

    # 执行操作
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(f"Unknown operation: '{operation}'")

    return data


def apply_external_force_torque_stochastic(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    force_range: dict[str, tuple[float, float]],
    torque_range: dict[str, tuple[float, float]],
    probability: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Apply a one-step stochastic external wrench to the configured bodies."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    asset._external_force_b *= 0
    asset._external_torque_b *= 0

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    random_values = torch.rand(env_ids.shape, device=env_ids.device)
    masked_env_ids = env_ids[random_values < probability]
    if len(masked_env_ids) == 0:
        return

    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies
    size = (len(masked_env_ids), num_bodies, 3)

    force_bounds = torch.tensor([force_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]], device=asset.device)
    torque_bounds = torch.tensor(
        [torque_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]],
        device=asset.device,
    )
    forces = math_utils.sample_uniform(force_bounds[:, 0], force_bounds[:, 1], size, asset.device)
    torques = math_utils.sample_uniform(torque_bounds[:, 0], torque_bounds[:, 1], size, asset.device)
    asset.set_external_force_and_torque(forces, torques, env_ids=masked_env_ids, body_ids=asset_cfg.body_ids)


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the default joint positions to model calibration offsets."""
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is None:
        return

    joint_pos = asset.data.default_joint_pos.to(asset.device).clone()
    joint_pos = _randomize_prop_by_op(
        joint_pos,
        pos_distribution_params,
        env_ids,
        joint_ids,
        operation=operation,
        distribution=distribution,
    )[env_ids][:, joint_ids]
    env_index = env_ids
    if not isinstance(env_ids, slice) and not isinstance(joint_ids, slice):
        env_index = env_ids[:, None]
    asset.data.default_joint_pos[env_index, joint_ids] = joint_pos


def randomize_joint_armature(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    armature_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the joint armature values."""
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if armature_distribution_params is None:
        return

    armature = asset.data.joint_armature.to(asset.device).clone()
    armature = _randomize_prop_by_op(
        armature,
        armature_distribution_params,
        env_ids,
        joint_ids,
        operation=operation,
        distribution=distribution,
    )[env_ids][:, joint_ids]
    asset.write_joint_armature_to_sim(armature, joint_ids=joint_ids, env_ids=env_ids)


def align_actuators_delay(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Align delays across all actuators after reset."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    time_lags = None
    sensor_time_lags = None
    for index, (_, actuator) in enumerate(asset.actuators.items()):
        if index == 0:
            time_lags = actuator.positions_delay_buffer.time_lags[env_ids].clone()
            if hasattr(actuator, "sensor_pos_delay_buffer"):
                sensor_time_lags = actuator.sensor_pos_delay_buffer.time_lags[env_ids].clone()
            continue
        actuator.positions_delay_buffer.time_lags[env_ids] = time_lags
        actuator.velocities_delay_buffer.time_lags[env_ids] = time_lags
        actuator.efforts_delay_buffer.time_lags[env_ids] = time_lags
        if sensor_time_lags is not None and hasattr(actuator, "sensor_pos_delay_buffer"):
            actuator.sensor_pos_delay_buffer.time_lags[env_ids] = sensor_time_lags
            actuator.sensor_vel_delay_buffer.time_lags[env_ids] = sensor_time_lags


class wheel_resistance_xy(ManagerTermBase):
    """Apply random wheel resistance in the contact plane."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.linear_resistance = torch.zeros(env.num_envs, device=self.device)
        self.quadratic_resistance = torch.zeros(env.num_envs, device=self.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        resample_probability: float,
        linear_res_coef_range: tuple[float, float] = (0.0, 0.0),
        quadratic_res_coef_range: tuple[float, float] = (0.0, 0.0),
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]

        random_values = torch.rand(env.num_envs, device=env.device)
        env_ids = (random_values < resample_probability).nonzero(as_tuple=True)[0]

        self.linear_resistance[env_ids] = (
            torch.rand(len(env_ids), device=self.device) * (linear_res_coef_range[1] - linear_res_coef_range[0])
            + linear_res_coef_range[0]
        ).clip(min=0.0)
        self.quadratic_resistance[env_ids] = (
            torch.rand(len(env_ids), device=self.device) * (quadratic_res_coef_range[1] - quadratic_res_coef_range[0])
            + quadratic_res_coef_range[0]
        ).clip(min=0.0)

        wheel_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids]
        linear_force = wheel_vel_w * self.linear_resistance[:, None, None]
        quadratic_force = wheel_vel_w.square() * wheel_vel_w.sign() * self.quadratic_resistance[:, None, None]
        wheel_force_w = -(linear_force + quadratic_force)
        wheel_force_w[..., 2] = 0.0
        wheel_force_b = math_utils.quat_apply_inverse(asset.data.body_quat_w[:, asset_cfg.body_ids], wheel_force_w)

        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        contact_history = contact_sensor.data.net_forces_w_history
        contact_mask = contact_history[:, :, sensor_cfg.body_ids, 2].clip(min=0.0, max=1.0).mean(dim=1)
        wheel_force_b *= contact_mask.unsqueeze(-1)

        asset.has_external_wrench = True
        asset._external_force_b[:, asset_cfg.body_ids] = wheel_force_b
