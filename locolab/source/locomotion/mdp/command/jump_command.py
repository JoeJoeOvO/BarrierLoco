"""Binary one-shot jump command with a fixed time schedule."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .jump_command_cfg import BinaryJumpCommandCfg


class JumpPhase(IntEnum):
    """Phases of one commanded jump."""

    STANCE = 0
    IN_FLIGHT = 1
    LANDED = 2


class BinaryJumpCommand(CommandTerm):
    """Generate one binary trigger and advance it on a fixed schedule.

    Contact is deliberately not used as the clock.  A sampled or manually
    supplied value of one starts at a configured phase time, remains one until
    ``sequence_duration``, and then clears.  Contacts are measured only as task
    outcomes (four-wheel flight and stable four-wheel landing).
    """

    cfg: BinaryJumpCommandCfg

    def __init__(self, cfg: BinaryJumpCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        if not 0.0 <= self.cfg.jump_probability <= 1.0:
            raise ValueError(f"jump_probability must be in [0, 1], got {self.cfg.jump_probability}.")
        if self.cfg.contact_force_threshold < 0.0:
            raise ValueError("contact_force_threshold must be non-negative.")
        if self.cfg.landing_recovery_time < 0.0:
            raise ValueError("landing_recovery_time must be non-negative.")
        schedule = (
            self.cfg.takeoff_time,
            self.cfg.apex_time,
            self.cfg.landing_time,
            self.cfg.sequence_duration,
        )
        if not 0.0 < schedule[0] < schedule[1] < schedule[2] < schedule[3]:
            raise ValueError(
                "Jump timing must satisfy 0 < takeoff_time < apex_time < landing_time < sequence_duration, "
                f"got {schedule}."
            )

        self.robot = env.scene[cfg.asset_name]
        self.contact_sensor = env.scene.sensors[cfg.contact_sensor_name]
        self.foot_body_ids = self.contact_sensor.find_bodies(cfg.foot_body_names)[0]
        if len(self.foot_body_ids) == 0:
            raise ValueError(f"No contact-sensor bodies matched {cfg.foot_body_names!r}.")

        self._command = torch.zeros((self.num_envs, 1), device=self.device)
        self._active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.phase = torch.full((self.num_envs,), int(JumpPhase.STANCE), dtype=torch.long, device=self.device)
        self.contact_state = torch.zeros((self.num_envs, len(self.foot_body_ids)), dtype=torch.bool, device=self.device)
        self.takeoff_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.apex_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.touchdown_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.timeout_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.completion_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.apex_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.all_wheels_airborne_achieved = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stable_landing_achieved = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.history_prior_airborne = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.history_prior_apex = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.history_prior_landed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.trigger_position = self.robot.data.root_pos_w.clone()
        self.trigger_height = self.robot.data.root_pos_w[:, 2].clone()
        self.target_jump_height = torch.full((self.num_envs,), float(self.cfg.minimum_jump_height), device=self.device)
        self.max_height = self.trigger_height.clone()
        self.touchdown_velocity = torch.zeros(self.num_envs, device=self.device)
        self.recovery_time = torch.zeros(self.num_envs, device=self.device)
        self.reference_time = torch.zeros(self.num_envs, device=self.device)
        self._last_update_step = -1

        self.metrics["jump_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["touchdown_velocity"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["jump_success"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["jump_timeout"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            "BinaryJumpCommand:\n"
            "\tCommand dimension: (1,)\n"
            f"\tJump probability: {self.cfg.jump_probability}\n"
            f"\tResampling time range: {self.cfg.resampling_time_range}"
        )

    @property
    def command(self) -> torch.Tensor:
        """Binary jump trigger with shape ``(num_envs, 1)``."""
        return self._command

    @property
    def active(self) -> torch.Tensor:
        """Whether a jump request is currently latched."""
        return self._active

    @property
    def is_in_flight(self) -> torch.Tensor:
        return (
            self._active
            & (self.reference_time >= self.cfg.takeoff_time)
            & (self.reference_time < self.cfg.landing_time)
        )

    @property
    def has_landed(self) -> torch.Tensor:
        return self._active & (self.reference_time >= self.cfg.landing_time)

    @property
    def normalized_phase(self) -> torch.Tensor:
        """One-shot phase in ``[0, 1]``; zero for non-jump episodes."""
        phase = torch.clamp(self.reference_time / self.cfg.sequence_duration, min=0.0, max=1.0)
        has_schedule = self._active | (self.reference_time > 0.0)
        return phase * has_schedule.to(phase.dtype)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        ids = self._resolve_env_ids(env_ids)
        metrics = super().reset(ids)
        # Reset-time observation computation may query this term before the
        # environment has advanced.  Anchor the clock to the current step so
        # such reads do not consume a schedule tick.
        self._last_update_step = int(getattr(self._env, "common_step_counter", -1))
        return metrics

    def set_trigger(self, trigger: int | float | torch.Tensor, env_ids: Sequence[int] | None = None) -> None:
        """Set external 0/1 input for selected environments.

        This is useful for deterministic play and later deployment integration.
        Values other than exact zero or one are rejected to keep policy semantics
        identical between training and deployment.
        """
        ids = self._resolve_env_ids(env_ids)
        values = torch.as_tensor(trigger, dtype=self._command.dtype, device=self.device).flatten()
        if values.numel() == 1:
            values = values.expand(ids.numel())
        if values.numel() != ids.numel():
            raise ValueError(f"Expected one trigger or {ids.numel()} triggers, got {values.numel()}.")
        if not torch.all((values == 0.0) | (values == 1.0)):
            raise ValueError("Jump trigger must contain only 0 or 1.")

        self._reset_jump_state(ids)
        active_ids = ids[values.bool()]
        if active_ids.numel() > 0:
            self._activate_schedule(active_ids, torch.zeros(active_ids.numel(), device=self.device))
        self._last_update_step = int(getattr(self._env, "common_step_counter", -1))

    def update_state(self) -> None:
        """Advance the fixed jump clock exactly once per RL step."""
        step = int(getattr(self._env, "common_step_counter", self._last_update_step + 1))
        if step == self._last_update_step:
            return
        self._last_update_step = step

        self.takeoff_event.zero_()
        self.apex_event.zero_()
        self.touchdown_event.zero_()
        self.timeout_event.zero_()
        self.completion_event.zero_()

        forces = self.contact_sensor.data.net_forces_w[:, self.foot_body_ids]
        self.contact_state[:] = torch.linalg.norm(forces, dim=-1) > self.cfg.contact_force_threshold
        root_height = self.robot.data.root_pos_w[:, 2]
        root_velocity_z = self.robot.data.root_lin_vel_w[:, 2]
        previous_time = self.reference_time.clone()
        was_active = self._active.clone()
        self.reference_time[was_active] = torch.clamp(
            self.reference_time[was_active] + self._env.step_dt,
            max=self.cfg.sequence_duration,
        )

        takeoff = was_active & (previous_time < self.cfg.takeoff_time) & (self.reference_time >= self.cfg.takeoff_time)
        self.takeoff_event[takeoff] = True
        apex = was_active & (previous_time < self.cfg.apex_time) & (self.reference_time >= self.cfg.apex_time)
        self.apex_event[apex] = True
        self.apex_reached[apex] = True
        touchdown = (
            was_active & (previous_time < self.cfg.landing_time) & (self.reference_time >= self.cfg.landing_time)
        )
        self.touchdown_event[touchdown] = True
        self.touchdown_velocity[touchdown] = torch.abs(root_velocity_z[touchdown])

        self._set_phase_from_time(torch.arange(self.num_envs, device=self.device))
        self.max_height[was_active] = torch.maximum(self.max_height[was_active], root_height[was_active])
        jump_height = self.max_height - self.trigger_height

        actual_airborne = ~self.contact_state.any(dim=-1)
        self.all_wheels_airborne_achieved |= self.is_in_flight & actual_airborne
        landed = self.has_landed
        all_wheels_contact = self.contact_state.all(dim=-1)
        stable_landing = (
            landed & all_wheels_contact & (torch.abs(root_velocity_z) <= self.cfg.landing_velocity_threshold)
        )
        self.recovery_time[landed & ~stable_landing] = 0.0
        self.recovery_time[stable_landing] += self._env.step_dt
        self.stable_landing_achieved |= self.recovery_time >= self.cfg.landing_recovery_time

        schedule_epsilon = 1.0e-6
        completed = (
            was_active
            & (previous_time < self.cfg.sequence_duration)
            & (self.reference_time >= self.cfg.sequence_duration - schedule_epsilon)
        )
        self.reference_time[completed] = self.cfg.sequence_duration
        successful = (
            completed
            & self.all_wheels_airborne_achieved
            & self.stable_landing_achieved
            & (jump_height >= self.target_jump_height)
        )
        failed = completed & ~successful
        self.completion_event[completed] = True
        self.timeout_event[failed] = True
        self.metrics["jump_success"][successful] = 1.0
        self.metrics["jump_timeout"][failed] = 1.0
        self._command[completed, 0] = 0.0
        self._active[completed] = False

        self.metrics["jump_height"][:] = torch.clamp_min(jump_height, 0.0)
        self.metrics["touchdown_velocity"][:] = self.touchdown_velocity

    def _update_metrics(self) -> None:
        self.update_state()

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = self._resolve_env_ids(env_ids)
        if ids.numel() == 0:
            return

        self._reset_jump_state(ids)
        pending = getattr(self._env, "_jump_reset_pending", None)
        if pending is None:
            sampled = torch.rand(ids.numel(), device=self.device) < self.cfg.jump_probability
            jump_ids = ids[sampled]
            if jump_ids.numel() > 0:
                self._activate_schedule(jump_ids, torch.zeros(jump_ids.numel(), device=self.device))
            return

        pending_ids = ids[pending[ids]]
        fallback_ids = ids[~pending[ids]]
        if pending_ids.numel() > 0:
            jump_mask = self._env._jump_episode_mask[pending_ids]
            jump_ids = pending_ids[jump_mask]
            if jump_ids.numel() > 0:
                self._activate_schedule(
                    jump_ids,
                    self._env._jump_initial_phase_time[jump_ids],
                    trigger_position=self._env._jump_trigger_position[jump_ids],
                    target_height=self._env._jump_target_height[jump_ids],
                )
            pending[pending_ids] = False

        if fallback_ids.numel() > 0:
            sampled = torch.rand(fallback_ids.numel(), device=self.device) < self.cfg.jump_probability
            jump_ids = fallback_ids[sampled]
            if jump_ids.numel() > 0:
                self._activate_schedule(jump_ids, torch.zeros(jump_ids.numel(), device=self.device))

    def _update_command(self) -> None:
        self.update_state()

    def _resolve_env_ids(self, env_ids: Sequence[int] | None) -> torch.Tensor:
        all_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if env_ids is None:
            return all_ids
        if isinstance(env_ids, slice):
            return all_ids[env_ids]
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()

    def _reset_jump_state(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        position = self.robot.data.root_pos_w[env_ids]
        height = position[:, 2]
        self._command[env_ids, 0] = 0.0
        self._active[env_ids] = False
        self.phase[env_ids] = int(JumpPhase.STANCE)
        self.contact_state[env_ids] = False
        self.takeoff_event[env_ids] = False
        self.apex_event[env_ids] = False
        self.touchdown_event[env_ids] = False
        self.timeout_event[env_ids] = False
        self.completion_event[env_ids] = False
        self.apex_reached[env_ids] = False
        self.all_wheels_airborne_achieved[env_ids] = False
        self.stable_landing_achieved[env_ids] = False
        self.history_prior_airborne[env_ids] = False
        self.history_prior_apex[env_ids] = False
        self.history_prior_landed[env_ids] = False
        self.trigger_position[env_ids] = position
        self.trigger_height[env_ids] = height
        self.target_jump_height[env_ids] = float(self.cfg.minimum_jump_height)
        self.max_height[env_ids] = height
        self.touchdown_velocity[env_ids] = 0.0
        self.recovery_time[env_ids] = 0.0
        self.reference_time[env_ids] = 0.0
        self.metrics["jump_height"][env_ids] = 0.0
        self.metrics["touchdown_velocity"][env_ids] = 0.0
        self.metrics["jump_success"][env_ids] = 0.0
        self.metrics["jump_timeout"][env_ids] = 0.0

    def _activate_schedule(
        self,
        env_ids: torch.Tensor,
        initial_time: torch.Tensor,
        trigger_position: torch.Tensor | None = None,
        target_height: torch.Tensor | None = None,
    ) -> None:
        if env_ids.numel() == 0:
            return
        time = initial_time.to(device=self.device, dtype=self.reference_time.dtype).clamp(
            0.0, self.cfg.sequence_duration
        )
        self.reference_time[env_ids] = time
        self._command[env_ids, 0] = (time < self.cfg.sequence_duration).to(self._command.dtype)
        self._active[env_ids] = time < self.cfg.sequence_duration
        if trigger_position is not None:
            self.trigger_position[env_ids] = trigger_position
            self.trigger_height[env_ids] = trigger_position[:, 2]
        if target_height is not None:
            self.target_jump_height[env_ids] = target_height
        self.max_height[env_ids] = torch.maximum(self.max_height[env_ids], self.robot.data.root_pos_w[env_ids, 2])

        # A phase-randomized reset records schedule history as a training prior.
        # It must not pre-populate observed outcomes used by success metrics.
        after_takeoff = time >= self.cfg.takeoff_time
        after_apex = time >= self.cfg.apex_time
        landed = time >= self.cfg.landing_time
        self.history_prior_airborne[env_ids] = after_takeoff
        self.history_prior_apex[env_ids] = after_apex
        self.history_prior_landed[env_ids] = landed
        self._set_phase_from_time(env_ids)

    def _set_phase_from_time(self, env_ids: torch.Tensor) -> None:
        time = self.reference_time[env_ids]
        self.phase[env_ids] = int(JumpPhase.STANCE)
        in_flight = (time >= self.cfg.takeoff_time) & (time < self.cfg.landing_time)
        landed = time >= self.cfg.landing_time
        self.phase[env_ids[in_flight]] = int(JumpPhase.IN_FLIGHT)
        self.phase[env_ids[landed]] = int(JumpPhase.LANDED)
