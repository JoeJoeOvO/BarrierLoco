"""Delayed actuator implementations used by WF_TRON1."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.actuators import DCMotor, ImplicitActuator
from isaaclab.utils import DelayBuffer
from isaacsim.core.utils.types import ArticulationActions

if TYPE_CHECKING:
    from .actuator_cfg import DelayedDCMotorCfg, DelayedImplicitActuatorCfg


class DelayedImplicitActuator(ImplicitActuator):
    """Implicit actuator that applies delayed commands."""

    cfg: DelayedImplicitActuatorCfg

    def __init__(self, cfg: DelayedImplicitActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)

        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        return super().compute(control_action, joint_pos, joint_vel)


class DelayedDCMotor(DCMotor):
    """DC motor actuator that delays both commands and sensor feedback."""

    cfg: DelayedDCMotorCfg

    def __init__(self, cfg: DelayedDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.sensor_pos_delay_buffer = DelayBuffer(cfg.max_sensor_delay, self._num_envs, device=self._device)
        self.sensor_vel_delay_buffer = DelayBuffer(cfg.max_sensor_delay, self._num_envs, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)

        command_time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        sensor_time_lags = torch.randint(
            low=self.cfg.min_sensor_delay,
            high=self.cfg.max_sensor_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )

        self.positions_delay_buffer.set_time_lag(command_time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(command_time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(command_time_lags, env_ids)
        self.sensor_pos_delay_buffer.set_time_lag(sensor_time_lags, env_ids)
        self.sensor_vel_delay_buffer.set_time_lag(sensor_time_lags, env_ids)

        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)
        self.sensor_pos_delay_buffer.reset(env_ids)
        self.sensor_vel_delay_buffer.reset(env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        delayed_joint_pos = self.sensor_pos_delay_buffer.compute(joint_pos)
        delayed_joint_vel = self.sensor_vel_delay_buffer.compute(joint_vel)
        return super().compute(control_action, delayed_joint_pos, delayed_joint_vel)
