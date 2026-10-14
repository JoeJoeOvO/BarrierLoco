"""Continuous ten-parameter command for Go2W gait-conditioned locomotion."""

from __future__ import annotations

import math

import torch
from isaaclab.managers import CommandTerm

from ..wtw_curriculum import curriculum_limits, sample_curriculum_commands, validate_schedule
from ..wtw_kinematics import LEG_JOINT_NAMES, height_reference_pose
from ..wtw_math import (
    FOOT_NAMES,
    PHASE_PRESETS,
    contact_schedule,
    sample_phase_neighborhood,
    validate_commands,
    validate_phase_neighborhood,
)


class WTWCommand(CommandTerm):
    """One shared command/time source for actor, critic and rewards.

    Resampling changes behavior without resetting time. Only episode reset
    resets time; reads never advance it. Categories are sampling metadata only.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._curriculum_step_offset = 0
        if getattr(cfg, "curriculum_enabled", False):
            validate_schedule(cfg.curriculum_schedule)
            if cfg.curriculum_steps_per_iteration <= 0:
                raise ValueError("Curriculum steps per iteration must be positive.")
            final_speed, final_swing = cfg.curriculum_schedule[-1][1:]
            if cfg.lin_vel_x != (-final_speed, final_speed) or cfg.footswing_height != (0.0, final_swing):
                raise ValueError("Command bounds must match the final curriculum limits.")
        if cfg.debug_vis:
            raise ValueError("WTW command does not implement debug markers.")
        if len(cfg.sampling_weights) != 3 or any(x < 0 for x in cfg.sampling_weights) or sum(cfg.sampling_weights) <= 0:
            raise ValueError("Provide three non-negative rolling/trot/bound weights with a positive sum.")
        if not 0 <= cfg.standing_probability <= 1 or cfg.kappa <= 0:
            raise ValueError("Invalid standing probability or contact smoothing.")
        if (
            not 0 <= cfg.rel_heading_envs <= 1
            or not math.isfinite(cfg.heading_control_stiffness)
            or cfg.heading_control_stiffness < 0
        ):
            raise ValueError("Heading probability must be in [0, 1] and heading gain finite/non-negative.")
        for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z", "heading", "body_height", "frequency", "footswing_height"):
            lower, upper = getattr(cfg, name)
            if not math.isfinite(lower) or not math.isfinite(upper) or not lower <= upper:
                raise ValueError(f"Invalid {name} range.")
        if cfg.frequency[0] <= 0 or cfg.body_height[0] <= 0 or cfg.footswing_height[0] < 0:
            raise ValueError("Periodic frequency/height must be positive and clearance non-negative.")
        if not 0 < cfg.gait_duration < 1:
            raise ValueError("Periodic gait_duration must be in (0, 1).")
        validate_phase_neighborhood(cfg.phase_jitter_deg, cfg.max_pair_phase_error_deg)
        self.robot = env.scene[cfg.asset_name]
        self.sensor = env.scene.sensors[cfg.contact_sensor_name]
        self.foot_body_ids, names = self.robot.find_bodies(list(FOOT_NAMES), preserve_order=True)
        self.sensor_body_ids, sensor_names = self.sensor.find_bodies(list(FOOT_NAMES), preserve_order=True)
        if tuple(names) != FOOT_NAMES or tuple(sensor_names) != FOOT_NAMES:
            raise ValueError("WTW requires exactly FR_foot, FL_foot, RR_foot, RL_foot in that order.")
        joint_names = [name + "_joint" for name in FOOT_NAMES]
        self.wheel_joint_ids, names = self.robot.find_joints(joint_names, preserve_order=True)
        if names != joint_names:
            raise ValueError("Could not resolve the four ordered wheel joints.")
        self.leg_joint_ids, names = self.robot.find_joints(list(LEG_JOINT_NAMES), preserve_order=True)
        if tuple(names) != LEG_JOINT_NAMES:
            raise ValueError("Could not resolve Go2W hip/thigh/calf joints in canonical leg order.")
        # Keep Flat's action/observation/reset pose immutable. The separate
        # per-env height reference is consumed only by default-pose rewards.
        self._asset_default_joint_pos = self.robot.data.default_joint_pos.clone()
        self._command = torch.zeros(self.num_envs, 10, device=self.device)
        self._command[:, 3] = sum(cfg.body_height) / 2
        self._command[:, 4] = sum(cfg.frequency) / 2
        self._command[:, 8] = 1
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._reward_joint_pos = self._reward_reference_for_commands(self._command, all_ids)
        # Reject infeasible sampling bounds during construction, not mid-training.
        bounds = self._command[[0, 0]].clone()
        bounds[:, 3] = bounds.new_tensor(cfg.body_height)
        self._reward_reference_for_commands(bounds, torch.zeros(2, dtype=torch.long, device=self.device))
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self._just_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.category = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        self.center_offsets = torch.tensor(cfg.wheel_center_offsets, device=self.device)
        for name in ("contact_mismatch", "velocity_error", "height_error"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        self._refresh_targets()

    @property
    def command(self):
        return self._command

    @property
    def reward_joint_pos(self):
        """Height-conditioned pose for rewards, never an action/observation offset."""
        return self._reward_joint_pos

    def _reward_reference_for_commands(self, command, env_ids):
        reference = self._asset_default_joint_pos[env_ids].clone()
        if not self.cfg.height_conditioned_reward_reference:
            return reference
        legs = height_reference_pose(
            command[:, 3], reference[:, self.leg_joint_ids].reshape(-1, 4, 3), self.cfg.wheel_radius
        ).reshape(-1, 12)
        limits = self.robot.data.soft_joint_pos_limits[env_ids][:, self.leg_joint_ids]
        if ((legs < limits[..., 0] - 1e-6) | (legs > limits[..., 1] + 1e-6)).any():
            raise ValueError("Height-conditioned reference exceeds Go2W soft joint limits; request another height.")
        reference[:, self.leg_joint_ids] = legs
        return reference

    def _ids(self, env_ids):
        if env_ids is None or isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[slice(None) if env_ids is None else env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def reset(self, env_ids=None):
        ids = self._ids(env_ids)
        self.phase[ids] = 0
        self._just_reset[ids] = True
        metrics = super().reset(ids)
        if getattr(self.cfg, "curriculum_enabled", False):
            iteration = self.curriculum_step / self.cfg.curriculum_steps_per_iteration
            speed, swing = curriculum_limits(iteration, self.cfg.curriculum_schedule)
            metrics.update(curriculum_iteration=iteration, curriculum_vx_limit=speed, curriculum_swing_limit=swing)
        return metrics

    @property
    def curriculum_step(self):
        return self._curriculum_step_offset + self._env.common_step_counter

    def curriculum_state_dict(self):
        return {
            "step": self.curriculum_step,
            "schedule": self.cfg.curriculum_schedule,
            "steps_per_iteration": self.cfg.curriculum_steps_per_iteration,
        }

    def load_curriculum_state_dict(self, state):
        if tuple(map(tuple, state["schedule"])) != tuple(map(tuple, self.cfg.curriculum_schedule)):
            raise ValueError("Resume curriculum schedule does not match this checkpoint.")
        if state["steps_per_iteration"] != self.cfg.curriculum_steps_per_iteration or state["step"] < 0:
            raise ValueError("Invalid or incompatible curriculum clock.")
        self._curriculum_step_offset = int(state["step"]) - self._env.common_step_counter

    def _resample_command(self, env_ids):
        ids = self._ids(env_ids)
        count = len(ids)
        if count == 0:
            return
        weights = torch.tensor(self.cfg.sampling_weights, device=self.device)
        category = torch.multinomial(weights, count, replacement=True)
        standing = torch.rand(count, device=self.device) < self.cfg.standing_probability
        category[standing] = 0
        sample = torch.zeros(count, 10, device=self.device)
        for index, name in (
            (0, "lin_vel_x"),
            (1, "lin_vel_y"),
            (2, "ang_vel_z"),
            (3, "body_height"),
            (4, "frequency"),
            (9, "footswing_height"),
        ):
            sample[:, index].uniform_(*getattr(self.cfg, name))
        if getattr(self.cfg, "curriculum_enabled", False):
            sample[:, 0], sample[:, 9] = sample_curriculum_commands(
                count,
                self.device,
                self.curriculum_step / self.cfg.curriculum_steps_per_iteration,
                self.cfg.curriculum_schedule,
            )
        presets = torch.tensor(list(PHASE_PRESETS.values()), device=self.device)
        sample[:, 5:8] = sample_phase_neighborhood(
            presets[category], self.cfg.phase_jitter_deg, self.cfg.max_pair_phase_error_deg
        )
        sample[:, 8] = self.cfg.gait_duration
        rolling = category == 0
        # Frequency has the same continuous distribution for all categories.
        # Full support (duration=1), rather than frequency=0, encodes rolling.
        sample[rolling, 8] = 1
        sample[rolling, 9] = 0
        sample[standing, :3] = 0
        reference = self._reward_reference_for_commands(sample, ids)
        self.category[ids] = category
        self._command[ids] = sample
        self._reward_joint_pos[ids] = reference
        self.is_standing_env[ids] = standing
        self.is_heading_env[ids] = False
        if self.cfg.heading_command:
            self.heading_target[ids] = torch.empty(count, device=self.device).uniform_(*self.cfg.heading)
            self.is_heading_env[ids] = torch.rand(count, device=self.device) < self.cfg.rel_heading_envs
        self._apply_heading()
        self._refresh_targets()

    def set_command(self, command, env_ids=None):
        """Set continuous commands and suspend random resampling for these environments.

        Callers recompute observations before the next policy inference. An
        episode reset resumes the configured training distribution. Column 2
        remains a direct yaw-rate request: manual commands disable automatic
        heading and standing masks on the selected environments.
        """
        ids = self._ids(env_ids)
        values = torch.as_tensor(command, dtype=self._command.dtype, device=self.device)
        if values.ndim == 1:
            values = values.unsqueeze(0).expand(len(ids), -1)
        if values.shape != (len(ids), 10):
            raise ValueError("Expected one ten-dimensional command per selected environment.")
        validate_commands(values)
        # Validate IK/limits before committing either buffer, so rejected manual
        # requests cannot leave command and reward references inconsistent.
        reference = self._reward_reference_for_commands(values, ids)
        self._command[ids] = values
        self._reward_joint_pos[ids] = reference
        self.is_heading_env[ids] = False
        self.is_standing_env[ids] = False
        self.time_left[ids] = torch.inf
        self._refresh_targets()

    def _apply_heading(self):
        """Match Flat's heading P controller while retaining a 10-D command."""
        if self.cfg.heading_command:
            error = self.heading_target - self.robot.data.heading_w
            wrapped = torch.remainder(error + math.pi, 2 * math.pi) - math.pi
            wrapped = torch.where((wrapped == -math.pi) & (error > 0), math.pi, wrapped)
            yaw_rate = torch.clamp(self.cfg.heading_control_stiffness * wrapped, *self.cfg.ang_vel_z)
            self._command[:, 2] = torch.where(self.is_heading_env, yaw_rate, self._command[:, 2])
        # A sampled standing environment must not turn to a random heading.
        self._command[self.is_standing_env, :3] = 0

    def _refresh_targets(self):
        self.raw_phase, self.leg_phase, self.clock_inputs, self.desired_contact = contact_schedule(
            self.phase, self._command, self.cfg.kappa
        )

    def _update_command(self):
        self._apply_heading()
        advance = self._command[:, 4] * self._env.step_dt
        self.phase = torch.remainder(self.phase + torch.where(self._just_reset, 0.0, advance), 1.0)
        self._just_reset[:] = False
        self._refresh_targets()

    def _update_metrics(self):
        forces = self.sensor.data.net_forces_w[:, self.sensor_body_ids]
        contact = forces.norm(dim=-1) > 1.0
        normalizer = self._env.max_episode_length
        self.metrics["contact_mismatch"] += (contact.float() - self.desired_contact).abs().mean(-1) / normalizer
        self.metrics["velocity_error"] += (self.robot.data.root_lin_vel_b[:, :2] - self._command[:, :2]).norm(
            dim=-1
        ) / normalizer
        self.metrics["height_error"] += (
            self.robot.data.root_pos_w[:, 2] - self._env.scene.env_origins[:, 2] - self._command[:, 3]
        ).abs() / normalizer
