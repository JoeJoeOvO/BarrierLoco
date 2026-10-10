"""Binary one-shot jump command with a fixed time schedule."""

from __future__ import annotations

import math
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
    outcomes and eligibility for a new request, never to stretch the clock.
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
        delay = getattr(cfg, "initial_trigger_delay_range", (0.0, 0.0))
        if not 0.0 <= delay[0] <= delay[1]:
            raise ValueError("initial_trigger_delay_range must be ordered and non-negative.")
        if getattr(cfg, "minimum_forward_speed", 0.0) < 0.0:
            raise ValueError("minimum_forward_speed must be non-negative.")
        heading_duration = getattr(cfg, "heading_hold_duration", 0.0)
        if not math.isfinite(heading_duration) or heading_duration < 0.0:
            raise ValueError("heading_hold_duration must be finite and non-negative.")
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
        self.trigger_yaw = torch.zeros(self.num_envs, device=self.device)
        self.heading_time_remaining = torch.zeros(self.num_envs, device=self.device)
        self._last_update_step = -1
        self._resetting = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._external_level = torch.zeros_like(self._resetting)
        self._ground_attempt = torch.zeros_like(self._resetting)
        self._touchdown_recorded = torch.zeros_like(self._resetting)
        self._previous_velocity_z = self.robot.data.root_lin_vel_w[:, 2].clone()

        self.metrics["jump_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["touchdown_velocity"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["jump_success"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["jump_timeout"] = torch.zeros(self.num_envs, device=self.device)
        for name in (
            "jump_attempts",
            "jump_completions",
            "jump_successes",
            "ground_attempts",
            "ground_completions",
            "ground_takeoffs",
            "ground_successes",
            "ground_jump_success",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

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
    def heading_tracking_mask(self) -> torch.Tensor:
        return self.heading_time_remaining > 0.0

    def _current_yaw(self) -> torch.Tensor:
        w, x, y, z = self.robot.data.root_quat_w.unbind(dim=-1)
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))

    def heading_error(self) -> torch.Tensor:
        difference = self._current_yaw() - self.trigger_yaw
        return torch.atan2(difference.sin(), difference.cos())

    def velocity_in_trigger_frame(self, velocity_world: torch.Tensor) -> torch.Tensor:
        """Horizontal motion in the fixed yaw frame captured at jump trigger."""
        vx, vy = velocity_world[:, :2].unbind(dim=-1)
        cosine, sine = self.trigger_yaw.cos(), self.trigger_yaw.sin()
        return torch.stack((cosine * vx + sine * vy, -sine * vx + cosine * vy), dim=-1)

    def heading_velocity_command(self, velocity_command: torch.Tensor) -> torch.Tensor:
        """Expose the held direction through existing body-frame vx/vy inputs.

        The external command remains unchanged. This conversion lets the MLP
        observe a heading error without adding observation or command columns.
        """
        difference = self.heading_error()
        cosine, sine = difference.cos(), difference.sin()
        vx, vy = velocity_command[:, :2].unbind(dim=-1)
        held = torch.stack((cosine * vx + sine * vy, -sine * vx + cosine * vy, velocity_command[:, 2]), dim=-1)
        return torch.where(self.heading_tracking_mask[:, None], held, velocity_command[:, :3])

    @property
    def normalized_phase(self) -> torch.Tensor:
        """One-shot phase in ``[0, 1]``; zero for non-jump episodes."""
        phase = torch.clamp(self.reference_time / self.cfg.sequence_duration, min=0.0, max=1.0)
        has_schedule = self._active | (self.reference_time > 0.0)
        return phase * has_schedule.to(phase.dtype)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        ids = self._resolve_env_ids(env_ids)
        completed = self.metrics["ground_completions"][ids].sum().item()
        successes = self.metrics["ground_successes"][ids].sum().item()
        self._resetting[ids] = True
        try:
            metrics = super().reset(ids)
        finally:
            self._resetting[ids] = False
        metrics["ground_jump_success"] = successes / max(completed, 1.0)
        # Reset-time observation computation may query this term before the
        # environment has advanced.  Anchor the clock to the current step so
        # such reads do not consume a schedule tick.
        self._last_update_step = int(getattr(self._env, "common_step_counter", -1))
        return metrics

    def set_trigger(self, trigger: int | float | torch.Tensor, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Accept a rising-edge request without resetting the physical robot.

        This is useful for deterministic play and later deployment integration.
        Values other than exact zero or one are rejected to keep policy semantics
        identical between training and deployment. Holding one cannot restart a
        running clock. Zero rearms the interface but does not cancel a jump.
        Returns an acceptance mask in the supplied environment order.
        """
        ids = self._resolve_env_ids(env_ids)
        values = torch.as_tensor(trigger, dtype=self._command.dtype, device=self.device).flatten()
        if values.numel() == 1:
            values = values.expand(ids.numel())
        if values.numel() != ids.numel():
            raise ValueError(f"Expected one trigger or {ids.numel()} triggers, got {values.numel()}.")
        if not torch.all((values == 0.0) | (values == 1.0)):
            raise ValueError("Jump trigger must contain only 0 or 1.")

        rising = values.bool() & ~self._external_level[ids]
        self._external_level[ids] = values.bool()
        accepted = rising & ~self.active[ids] & self._can_trigger(ids)
        self._start_ground_jump(ids[accepted])
        return accepted

    def update_state(self) -> None:
        """Advance the fixed jump clock exactly once per RL step."""
        step = int(getattr(self._env, "common_step_counter", self._last_update_step + 1))
        if step == self._last_update_step:
            return
        self._last_update_step = step
        self.heading_time_remaining.sub_(self._env.step_dt).clamp_(min=0.0)
        self.heading_time_remaining.masked_fill_(self.heading_time_remaining < self._env.step_dt * 1e-3, 0.0)

        self.takeoff_event.zero_()
        self.apex_event.zero_()
        self.touchdown_event.zero_()
        self.timeout_event.zero_()
        self.completion_event.zero_()

        previously_airborne = ~self.contact_state.any(dim=-1)
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

        apex = was_active & (previous_time < self.cfg.apex_time) & (self.reference_time >= self.cfg.apex_time)
        self.apex_event[apex] = True
        self.apex_reached[apex] = True

        self._set_phase_from_time(torch.arange(self.num_envs, device=self.device))
        self.max_height[was_active] = torch.maximum(self.max_height[was_active], root_height[was_active])
        jump_height = self.max_height - self.trigger_height

        actual_airborne = ~self.contact_state.any(dim=-1)
        first_airborne = was_active & actual_airborne & ~self.all_wheels_airborne_achieved
        self.takeoff_event[:] = first_airborne & ~previously_airborne
        self.metrics["ground_takeoffs"] += (first_airborne & self._ground_attempt).float()
        self.all_wheels_airborne_achieved |= was_active & actual_airborne
        touchdown = (
            was_active
            & previously_airborne
            & ~actual_airborne
            & self.all_wheels_airborne_achieved
            & ~self._touchdown_recorded
        )
        self.touchdown_event[:] = touchdown
        self.touchdown_velocity[touchdown] = self._previous_velocity_z[touchdown].abs()
        self._touchdown_recorded |= touchdown
        self._previous_velocity_z[:] = root_velocity_z
        landed = was_active & self._touchdown_recorded & (self.reference_time >= self.cfg.apex_time)
        all_wheels_contact = self.contact_state.all(dim=-1)
        stable_landing = (
            landed & all_wheels_contact & (torch.abs(root_velocity_z) <= self.cfg.landing_velocity_threshold)
        )
        if getattr(self.cfg, "minimum_forward_speed", 0.0) > 0.0:
            stable_landing &= self.robot.data.root_lin_vel_b[:, 0] >= self.cfg.minimum_forward_speed
        gravity = getattr(self.robot.data, "projected_gravity_b", None)
        if gravity is not None:
            stable_landing &= gravity[:, 2] < -math.cos(getattr(self.cfg, "landing_attitude_limit", 1.0))
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
            & stable_landing
            & (jump_height >= self.target_jump_height)
        )
        failed = completed & ~successful
        self.completion_event[completed] = True
        self.timeout_event[failed] = True
        self.metrics["jump_completions"] += completed.float()
        self.metrics["jump_successes"] += successful.float()
        self.metrics["jump_success"][:] = self.metrics["jump_successes"] / self.metrics["jump_completions"].clamp_min(1)
        self.metrics["jump_timeout"][:] = (
            self.metrics["jump_completions"] - self.metrics["jump_successes"]
        ) / self.metrics["jump_completions"].clamp_min(1)
        self.metrics["ground_completions"] += (completed & self._ground_attempt).float()
        self.metrics["ground_successes"] += (successful & self._ground_attempt).float()
        self.metrics["ground_jump_success"][:] = self.metrics["ground_successes"] / self.metrics[
            "ground_completions"
        ].clamp_min(1)
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

        reset_ids = ids[self._resetting[ids]]
        self._reset_jump_state(reset_ids)
        self._external_level[reset_ids] = False
        pending = getattr(self._env, "_jump_reset_pending", None)
        pending_ids = reset_ids[pending[reset_ids]] if pending is not None else reset_ids[:0]
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
        delay = getattr(self.cfg, "initial_trigger_delay_range", (0.0, 0.0))
        if delay[1] > 0 and reset_ids.numel() > 0:
            self.time_left[reset_ids] = torch.empty(reset_ids.numel(), device=self.device).uniform_(*delay)
        fallback_ids = ids[~self._resetting[ids]]
        if delay[1] == 0 and pending is None:
            fallback_ids = ids
        ready_ids = fallback_ids[~self.active[fallback_ids] & self._can_trigger(fallback_ids)]
        sampled = torch.rand(ready_ids.numel(), device=self.device) < self.cfg.jump_probability
        self._start_ground_jump(ready_ids[sampled])

    def _can_trigger(self, env_ids: torch.Tensor) -> torch.Tensor:
        if not getattr(self.cfg, "require_grounded_trigger", False):
            return torch.ones(env_ids.numel(), device=self.device, dtype=torch.bool)
        forces = self.contact_sensor.data.net_forces_w[env_ids][:, self.foot_body_ids]
        grounded = (torch.linalg.vector_norm(forces, dim=-1) > self.cfg.contact_force_threshold).all(dim=-1)
        velocity = self.robot.data.root_lin_vel_b[env_ids]
        upright = self.robot.data.projected_gravity_b[env_ids, 2] < -math.cos(self.cfg.landing_attitude_limit)
        return (
            grounded
            & upright
            & (velocity[:, 0] >= self.cfg.minimum_forward_speed)
            & (self.robot.data.root_lin_vel_w[env_ids, 2].abs() <= self.cfg.landing_velocity_threshold)
        )

    def _start_ground_jump(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self._reset_jump_state(env_ids, clear_metrics=False)
        self._ground_attempt[env_ids] = True
        self.metrics["ground_attempts"][env_ids] += 1
        # Preserve root/joint state and current momentum; only the clock starts.
        self._activate_schedule(env_ids, torch.zeros(env_ids.numel(), device=self.device))

    def _update_command(self) -> None:
        self.update_state()

    def _resolve_env_ids(self, env_ids: Sequence[int] | None) -> torch.Tensor:
        all_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if env_ids is None:
            return all_ids
        if isinstance(env_ids, slice):
            return all_ids[env_ids]
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()

    def _reset_jump_state(self, env_ids: torch.Tensor, clear_metrics: bool = True) -> None:
        if env_ids.numel() == 0:
            return
        position = self.robot.data.root_pos_w[env_ids]
        height = position[:, 2]
        self._command[env_ids, 0] = 0.0
        self._active[env_ids] = False
        self.phase[env_ids] = int(JumpPhase.STANCE)
        forces = self.contact_sensor.data.net_forces_w[env_ids][:, self.foot_body_ids]
        self.contact_state[env_ids] = torch.linalg.vector_norm(forces, dim=-1) > self.cfg.contact_force_threshold
        self._ground_attempt[env_ids] = False
        self._touchdown_recorded[env_ids] = False
        self._previous_velocity_z[env_ids] = self.robot.data.root_lin_vel_w[env_ids, 2]
        self._reference_reward_cache = None
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
        self.trigger_yaw[env_ids] = 0.0
        self.heading_time_remaining[env_ids] = 0.0
        if clear_metrics:
            for value in self.metrics.values():
                value[env_ids] = 0.0

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
        heading_duration = getattr(self.cfg, "heading_hold_duration", 0.0)
        if heading_duration > 0.0:
            self.trigger_yaw[env_ids] = self._current_yaw()[env_ids]
            self.heading_time_remaining[env_ids] = torch.where(
                self._active[env_ids], (heading_duration - time).clamp_min(0.0), 0.0
            )
        self.metrics["jump_attempts"][env_ids] += 1
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
