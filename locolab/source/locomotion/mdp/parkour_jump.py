"""Seven-condition jumping with per-motion clocks, matched terrain and actual-state imitation."""

import math

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .parkour_reference import ParkourReferenceBank, sample_reset_phases


def bank_for(env):
    if not hasattr(env, "_parkour_bank"):
        env._parkour_bank = ParkourReferenceBank(env.cfg.reference_bank_path, env.device)
    required = getattr(getattr(env, "cfg", None), "required_reference_bank_version", None)
    if required is not None and env._parkour_bank.schema_version != required:
        raise ValueError("Task requires a validated goal-grid reference bank")
    return env._parkour_bank


def reset_parkour_state(
    env,
    env_ids,
    reference_probability=None,
    phase_range=(0.0, 1.8),
    reset_weights=(0.30, 0.40, 0.15, 0.10, 0.05),
    locomotion_slots=0,
    ground_reference_pose=False,
    resample_flat_motion=False,
):
    """Reference phases stay on their own collision scene; normal starts stay grounded."""
    bank, robot = bank_for(env), env.scene["robot"]
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    if (reference_probability is not None and not 0 <= reference_probability <= 1) or not 0 <= phase_range[
        0
    ] <= phase_range[1] <= bank.duration:
        raise ValueError("Invalid reference reset mixture or phase interval")
    motion, locomotion = bank.layout(ids, locomotion_slots)
    if resample_flat_motion:
        motion = bank.resample_flat_motions(motion, locomotion)
    if not hasattr(env, "_parkour_motion_ids"):
        env._parkour_motion_ids = bank.layout(torch.arange(env.num_envs, device=env.device), locomotion_slots)[0]
    env._parkour_motion_ids[ids] = motion
    if reference_probability is None:
        modes, times = sample_reset_phases(bank, motion, reset_weights)
        is_reference = modes != 0
    else:
        # Explicit overrides are reserved for play and exact-phase smoke tests.
        is_reference = torch.rand(len(ids), device=env.device) < reference_probability
        times = torch.empty(len(ids), device=env.device).uniform_(*phase_range) * is_reference
        modes = torch.where(times >= bank.times[motion, 2], 4, torch.where(times >= bank.times[motion, 0], 3, 2))
        modes = torch.where(times == 0, 1, modes)
        modes = torch.where(is_reference, modes, 0)
    # Ordinary locomotion lanes never receive an airborne reset or a jump request.
    is_reference &= ~locomotion
    times = torch.where(locomotion, 0, times)
    modes = torch.where(locomotion, 0, modes)
    sample = bank.sample(motion, times)
    pose = robot.data.default_root_state[ids, :7].clone()
    pose[:, :3] += env.scene.env_origins[ids]
    pose[:, 0] -= torch.empty(len(ids), device=env.device).uniform_(0.7, 0.9)
    pose[:, 1] += torch.empty(len(ids), device=env.device).uniform_(-0.02, 0.02)
    velocity = torch.zeros(len(ids), 6, device=env.device)
    q = robot.data.default_joint_pos[ids].clone()
    q *= torch.empty_like(q).uniform_(0.95, 1.05)
    qd = torch.zeros_like(q)
    leg_ids = robot.find_joints(bank.leg_names, preserve_order=True)[0]
    wheel_ids = robot.find_joints(bank.wheel_names, preserve_order=True)[0]
    if ground_reference_pose:
        # Use the independently settled teacher's support geometry, at rest.
        # No global simulator step, state injection at trigger, or pose retiming.
        initial = bank.sample(motion, torch.zeros_like(times))
        pose[:, 2] = env.scene.env_origins[ids, 2] + initial["position"][:, 2]
        pose[:, 3:7] = initial["quaternion"]
        q[:, leg_ids] = initial["joint_position"]
    # A 10 s, 1.5 m/s rolling episode fits inside the existing 24 m flat tile.
    pose[locomotion, 0] = env.scene.env_origins[ids[locomotion], 0] - 6.0
    chosen = is_reference.nonzero(as_tuple=False).flatten()
    pose[chosen, :3] = env.scene.env_origins[ids[chosen]] + sample["position"][chosen]
    pose[chosen, 3:7] = sample["quaternion"][chosen]
    velocity[chosen, :3] = sample["velocity"][chosen]
    velocity[chosen, 3:] = sample["omega"][chosen]
    q[chosen[:, None], leg_ids] = sample["joint_position"][chosen]
    qd[chosen[:, None], leg_ids] = sample["joint_velocity"][chosen]
    q[chosen[:, None], wheel_ids] = sample["wheel_position"][chosen]
    qd[chosen[:, None], wheel_ids] = sample["wheel_velocity"][chosen]
    # Preserve reference poses inside true limits; do not apply the flat
    # task's tighter soft bounds to a measured tucked-leg configuration.
    limits = robot.data.joint_pos_limits[ids]
    q.clamp_(limits[..., 0], limits[..., 1])
    speed = robot.data.joint_vel_limits[ids]
    qd.clamp_(-speed, speed)
    robot.write_root_pose_to_sim(pose, env_ids=ids)
    robot.write_root_velocity_to_sim(velocity, env_ids=ids)
    robot.write_joint_state_to_sim(q, qd, env_ids=ids)
    if not hasattr(env, "_parkour_reset_reference"):
        env._parkour_reset_reference = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._parkour_reset_time = torch.zeros(env.num_envs, device=env.device)
        env._parkour_reset_mode = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    env._parkour_reset_reference[ids] = is_reference
    env._parkour_reset_time[ids] = times
    env._parkour_reset_mode[ids] = modes


class ParkourJumpCommand(CommandTerm):
    """Public seven-column command; each environment has one matched terrain/motion."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if getattr(cfg, "landing_command_mode", "remaining") not in ("remaining", "fixed"):
            raise ValueError("landing_command_mode must be remaining or fixed")
        probability = getattr(cfg, "speed_endpoint_probability", 0.0)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("speed_endpoint_probability must be within [0, 1]")
        for name, default, positive in (
            ("approach_velocity_std", 0.5, True),
            ("approach_speed_error_floor", 0.0, False),
            ("reference_xy_feedback_gain", 0.0, False),
            ("reference_xy_feedback_limit", 0.75, True),
        ):
            value = getattr(cfg, name, default)
            if not math.isfinite(value) or (value <= 0 if positive else value < 0):
                raise ValueError(f"Invalid {name}")
        self.bank = bank_for(env)
        self.robot = env.scene["robot"]
        self.sensor = env.scene["contact_forces"]
        self.leg_ids = self.robot.find_joints(self.bank.leg_names, preserve_order=True)[0]
        self.wheel_ids = self.robot.find_joints(self.bank.wheel_names, preserve_order=True)[0]
        foot_names = [name.replace("_joint", "") for name in self.bank.wheel_names]
        self.foot_ids = self.sensor.find_bodies(foot_names, preserve_order=True)[0]
        self.foot_body_ids = self.robot.find_bodies(foot_names, preserve_order=True)[0]
        self.nonfoot_ids = [i for i, name in enumerate(self.sensor.body_names) if name not in foot_names]
        self.motion_ids, self.locomotion_only = self.bank.layout(
            torch.arange(self.num_envs, device=self.device), getattr(cfg, "locomotion_slots", 0)
        )
        self.forward_speed = torch.ones(self.num_envs, device=self.device)
        self.reference_time = torch.zeros(self.num_envs, device=self.device)
        self.exit_time = self.reference_time.clone()
        self.rolling_motion = torch.tensor(
            [entry["replay"]["motion_mode"] != "tilted_ramp" for entry in self.bank.entries], device=self.device
        )[self.motion_ids]
        self.active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.started = self.active.clone()
        self.ground_attempt = self.active.clone()
        self.air_seen = self.active.clone()
        self.finished = self.active.clone()
        self.contact_state = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.bool)
        self.air_time = self.reference_time.clone()
        self.stable_time = self.reference_time.clone()
        self.ground_success = self.active.clone()
        self.task_success = self.active.clone()
        self.upward_takeoff = self.active.clone()
        self.support_seen = self.active.clone()
        self.touchdown_seen = self.active.clone()
        self.touchdown_event = self.active.clone()
        self.touchdown_error = torch.full_like(self.reference_time, float("inf"))
        self.launch_height = self.bank.states["position"][self.motion_ids, 0, 2].clone()
        self.peak_height = self.reference_time.clone()
        self.flight_peak_height = self.reference_time.clone()
        self.reset_mode = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._last_step = -1
        self._sample = None
        for name in (
            "reference_position_error",
            "ground_attempt",
            "ground_takeoff",
            "ground_success",
            "jump_height",
            "task_success",
            "reference_recovery_success",
            "motion_id",
            "locomotion_only",
            "commanded_speed",
            "rolling_speed_error",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self):
        return self._command_with_landing(getattr(self.cfg, "landing_command_mode", "remaining"))

    @property
    def critic_command(self):
        """Legacy dynamic goal feedback, privileged when the actor uses fixed targets."""
        return self._command_with_landing("remaining")

    def _command_with_landing(self, mode):
        # Velocity keeps the fixed-world-heading contract in both versions.
        w, x, y, z = self.robot.data.root_quat_w.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
        if mode == "fixed":
            # Displacement from the fixed TO start frame, just like the height
            # descriptor. No live root position or current-yaw rotation enters
            # these two actor inputs; terrain/world targets remain unchanged.
            landing = self.bank.descriptors[self.motion_ids, 1:]
        else:
            local_xy = self.robot.data.root_pos_w[:, :2] - self._env.scene.env_origins[:, :2]
            remaining = self.bank.descriptors[self.motion_ids, 1:] - local_xy
            c, s = yaw.cos(), yaw.sin()
            landing = torch.stack(
                (c * remaining[:, 0] + s * remaining[:, 1], -s * remaining[:, 0] + c * remaining[:, 1]), -1
            )
        has_goal = ~(self.locomotion_only | self.rolling_exit)
        return torch.cat(
            (
                (self.forward_speed * yaw.cos())[:, None],
                (-self.forward_speed * yaw.sin())[:, None],
                torch.zeros_like(yaw[:, None]),
                self.active.float()[:, None],
                self.bank.descriptors[self.motion_ids, :1] * has_goal[:, None],
                landing * has_goal[:, None],
            ),
            -1,
        )

    @property
    def normalized_phase(self):
        return self.reference_time / self.bank.sequence_duration

    @property
    def rolling_exit(self):
        return (
            self.finished
            & self.task_success
            & self.rolling_motion
            & (getattr(self.cfg, "rolling_exit_duration", 0.0) > 0)
        )

    @property
    def completion_ready(self):
        return self.finished & (
            ~self.rolling_exit | (self.exit_time >= getattr(self.cfg, "rolling_exit_duration", 0.0))
        )

    @property
    def locomotion_active(self):
        return ~self.started | (self.rolling_exit & (self.exit_time > 0))

    def sample(self):
        if self._sample is None:
            self._sample = self.bank.sample(self.motion_ids, self.reference_time)
        return self._sample

    def reset(self, env_ids=None):
        ids = self._ids(env_ids)
        attempts = self.ground_attempt[ids].sum().item()
        successes = self.ground_success[ids].sum().item()
        ground_peak = self.flight_peak_height[ids][self.ground_attempt[ids]]
        ground_peak_mean = ground_peak.mean().item() if len(ground_peak) else 0.0
        result = super().reset(ids)
        if hasattr(self._env, "_parkour_motion_ids"):
            selected = self._env._parkour_motion_ids[ids]
            old_obstacle = ~self.bank.is_flat[self.motion_ids[ids]]
            if ((selected != self.motion_ids[ids]) & old_obstacle).any():
                raise ValueError("Cannot change an obstacle reference without changing its physical terrain")
            if (self.bank.is_flat[self.motion_ids[ids]] & ~self.bank.is_flat[selected]).any():
                raise ValueError("Cannot sample an obstacle reference on a flat tile")
            self.motion_ids[ids] = selected
        self.reference_time[ids] = self._env._parkour_reset_time[ids]
        self.exit_time[ids] = 0
        self.active[ids] = self._env._parkour_reset_reference[ids]
        self.started[ids] = self.active[ids]
        self.ground_attempt[ids] = False
        self.air_seen[ids] = False
        self.finished[ids] = False
        self.ground_success[ids] = False
        self.task_success[ids] = False
        self.upward_takeoff[ids] = False
        self.support_seen[ids] = False
        self.touchdown_seen[ids] = False
        self.touchdown_event[ids] = False
        self.touchdown_error[ids] = float("inf")
        self.reset_mode[ids] = self._env._parkour_reset_mode[ids]
        self.launch_height[ids] = self.bank.states["position"][self.motion_ids[ids], 0, 2]
        local_z = self.robot.data.root_pos_w[ids, 2] - self._env.scene.env_origins[ids, 2]
        self.peak_height[ids] = (local_z - self.launch_height[ids]).clamp_min(0)
        self.flight_peak_height[ids] = 0
        self.contact_state[ids] = False
        self.air_time[ids] = 0
        self.stable_time[ids] = 0
        self._sample = None
        self._last_step = int(self._env.common_step_counter)
        result["ground_success_rate"] = successes / max(attempts, 1)
        result["ground_peak_height"] = ground_peak_mean
        result["ground_attempt_count"] = attempts
        result["ground_success_count"] = successes
        return result

    def _ids(self, ids):
        all_ids = torch.arange(self.num_envs, device=self.device)
        return (
            all_ids
            if ids is None
            else all_ids[ids] if isinstance(ids, slice) else torch.as_tensor(ids, device=self.device)
        )

    def update_state(self):
        step = int(self._env.common_step_counter)
        if step == self._last_step:
            return
        self._last_step = step
        self._sample = None
        self.touchdown_event.zero_()
        self.exit_time[self.rolling_exit] += self._env.step_dt
        self.contact_state[:] = self.sensor.data.net_forces_w[:, self.foot_ids].norm(dim=-1) > 1.0
        local = self.robot.data.root_pos_w - self._env.scene.env_origins
        advance = self.active.clone()
        self.reference_time[advance] += self._env.step_dt
        self.reference_time.clamp_(max=self.bank.sequence_duration)
        # Fixed geometric trigger: the TO's actual settled start is x=0.
        # Never move/teleport the root to make a ground request valid.
        ready = (
            ~self.started
            & ~self.locomotion_only
            & (local[:, 0] >= 0)
            & (local[:, 0] <= 0.12)
            & (local[:, 1].abs() <= 0.12)
            & self.contact_state.all(-1)
            & ((self.robot.data.root_lin_vel_b[:, 0] - self.forward_speed).abs() <= 0.25)
            & (self.robot.data.root_lin_vel_b[:, 0] > 0.1)
            & ((local[:, 2] - self.bank.states["position"][self.motion_ids, 0, 2]).abs() <= 0.05)
            & (self.robot.data.root_lin_vel_w[:, 2].abs() < 0.2)
            & (self.robot.data.projected_gravity_b[:, 2] < -math.cos(0.5))
        )
        self.active[ready] = True
        self.started[ready] = True
        self.ground_attempt[ready] = True
        self.launch_height[ready] = local[ready, 2]
        self.peak_height[ready] = 0
        self.flight_peak_height[ready] = 0
        self.reference_time[ready] = 0
        airborne = advance & ~self.contact_state.any(-1)
        self.air_time = torch.where(airborne, self.air_time + self._env.step_dt, 0)
        self.peak_height = torch.where(
            self.active,
            torch.maximum(self.peak_height, (local[:, 2] - self.launch_height).clamp_min(0)),
            self.peak_height,
        )
        # Loss of contact while falling is NOT takeoff. A real attempt needs
        # observed ground support, positive launch velocity and body rise.
        lift_off = (
            airborne
            & self.support_seen
            & ~self.upward_takeoff
            & (self.robot.data.root_lin_vel_w[:, 2] > 0.35)
            & (local[:, 2] > self.launch_height + 0.025)
            & (self.reference_time < self.bank.times[self.motion_ids, 2] - 0.06)
        )
        self.upward_takeoff |= lift_off
        self.support_seen |= self.active & self.contact_state.any(-1)
        main_air = airborne & self.upward_takeoff & ~self.touchdown_seen
        self.flight_peak_height = torch.where(
            main_air,
            torch.maximum(self.flight_peak_height, (local[:, 2] - self.launch_height).clamp_min(0)),
            self.flight_peak_height,
        )
        self.air_seen |= self.upward_takeoff & (self.air_time >= 0.08) & (self.flight_peak_height >= 0.04)
        touchdown = (
            advance & self.contact_state.any(-1) & ~self.touchdown_seen & (self.air_seen | (self.reset_mode == 3))
        )
        self.touchdown_event[:] = touchdown
        self.touchdown_seen |= touchdown
        self.touchdown_error[touchdown] = (
            local[touchdown, :2] - self.bank.descriptors[self.motion_ids[touchdown], 1:]
        ).norm(dim=-1)
        reference = self.sample()
        error = (local - reference["position"]).norm(dim=-1)
        velocity_error = (self.robot.data.root_lin_vel_w - reference["velocity"]).norm(dim=-1)
        orientation_error = 1 - (self.robot.data.root_quat_w * reference["quaternion"]).sum(-1).square().clamp(0, 1)
        stable = (
            advance
            & self.contact_state.all(-1)
            & (error < 0.20)
            & (velocity_error < 0.30)
            & (orientation_error < 0.04)
            & (self.reference_time > self.bank.times[self.motion_ids, 2])
        )
        # Terminal wheel footprints must be on the selected ramp, not straddling both.
        foot_local = self.robot.data.body_pos_w[:, self.foot_body_ids] - self._env.scene.env_origins[:, None]
        for index, entry in enumerate(self.bank.entries):
            if entry["replay"]["motion_mode"] == "tilted_ramp":
                target = entry["replay"]["target"]
                origin = self.bank.origins[index]
                x0 = target["near_x"] + target["width"] - origin[0]
                y0 = target["side"] * target["offset_y"] - origin[1]
                inside = (
                    (foot_local[..., 0] > x0 + 0.09)
                    & (foot_local[..., 0] < x0 + target["platform_length"] - 0.09)
                    & ((foot_local[..., 1] - y0).abs() < target["platform_width"] / 2 - 0.07)
                ).all(-1)
                stable &= (self.motion_ids != index) | inside
        self.stable_time = torch.where(stable, self.stable_time + self._env.step_dt, 0)
        complete = advance & (self.reference_time >= self.bank.sequence_duration - self._env.step_dt * 1e-3)
        self.reference_time[complete] = self.bank.sequence_duration
        self.finished |= complete
        required_height = (self.bank.descriptors[self.motion_ids, 0] - 0.05).clamp_min(0.08)
        full_jump = (
            self.air_seen
            & (self.flight_peak_height >= required_height)
            & (self.flight_peak_height <= self.bank.descriptors[self.motion_ids, 0] + 0.08)
            & self.touchdown_seen
            & (self.touchdown_error <= 0.20)
        )
        # RSI recovery practice has its own result, never fabricated launch history.
        eligible = torch.where(
            self.reset_mode <= 2,
            full_jump,
            (self.reset_mode == 4) | (self.touchdown_seen & (self.touchdown_error <= 0.20)),
        )
        nonfoot_collision = self.sensor.data.net_forces_w[:, self.nonfoot_ids].norm(dim=-1).amax(-1) > 1.0
        self.task_success |= complete & eligible & (self.stable_time >= 0.10) & ~nonfoot_collision
        self.ground_success |= self.task_success & self.ground_attempt & self.completion_ready
        self.active[complete] = False
        self.metrics["reference_position_error"][:] = error * self.started
        self.metrics["ground_attempt"][:] = self.ground_attempt.float()
        self.metrics["ground_takeoff"][:] = (self.ground_attempt & self.air_seen).float()
        self.metrics["ground_success"][:] = self.ground_success.float()
        self.metrics["jump_height"][:] = self.flight_peak_height
        self.metrics["task_success"][:] = (self.task_success & (self.reset_mode <= 2)).float()
        self.metrics["reference_recovery_success"][:] = (self.task_success & (self.reset_mode >= 3)).float()
        self.metrics["motion_id"][:] = self.motion_ids.float()
        self.metrics["locomotion_only"][:] = self.locomotion_only.float()
        self.metrics["commanded_speed"][:] = self.forward_speed
        self.metrics["rolling_speed_error"][:] = (self.robot.data.root_lin_vel_w[:, 0] - self.forward_speed).abs() * (
            ~self.started
        )

    def _update_metrics(self):
        self.update_state()

    def _update_command(self):
        self.update_state()

    def _resample_command(self, env_ids):
        low, high = getattr(self.cfg, "forward_speed_range", (1.0, 1.0))
        if not 0.1 <= low <= high <= 1.5:
            raise ValueError("Expected forward speed range within [0.1, 1.5] m/s")
        ids = self._ids(env_ids)
        if low == high:
            self.forward_speed[ids] = low
            return
        speeds = torch.empty(len(ids), device=self.device).uniform_(low, high)
        probability = getattr(self.cfg, "speed_endpoint_probability", 0.0)
        if probability > 0:
            choose_endpoint = torch.rand(len(ids), device=self.device) < probability
            endpoints = torch.where(torch.rand(len(ids), device=self.device) < 0.5, low, high)
            speeds = torch.where(choose_endpoint, endpoints, speeds)
        self.forward_speed[ids] = speeds


@configclass
class ParkourJumpCommandCfg(CommandTermCfg):
    class_type: type = ParkourJumpCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    debug_vis: bool = False
    forward_speed_range: tuple[float, float] = (1.0, 1.0)
    locomotion_slots: int = 0
    approach_timeout: float = 3.0
    rolling_exit_duration: float = 0.0
    landing_command_mode: str = "remaining"
    speed_endpoint_probability: float = 0.0
    approach_velocity_std: float = 0.5
    approach_speed_error_floor: float = 0.0
    reference_xy_feedback_gain: float = 0.0
    reference_xy_feedback_limit: float = 0.75


def term_for(env):
    term = env.command_manager.get_term("parkour")
    term.update_state()
    return term


def parkour_phase(env):
    return term_for(env).normalized_phase[:, None]


def parkour_critic_commands(env):
    """Seven privileged values in the original critic layout; actor cmd may differ."""
    return env.command_manager.get_term("parkour").critic_command


def parkour_finished(env):
    return term_for(env).finished


def parkour_succeeded(env):
    term = term_for(env)
    return term.completion_ready & term.task_success


def parkour_deadline_failed(env):
    term = term_for(env)
    return term.completion_ready & ~term.task_success


def parkour_failure_cost(env):
    term = term_for(env)
    failure = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in ("base_contact", "bad_orientation", "missed_target_or_pit", "task_failed", "time_out"):
        event = env.termination_manager.get_term(name)
        # A normal rolling episode reaching its horizon is not a failed jump.
        failure |= event & (~term.locomotion_only if name == "time_out" else True)
    # get_term() retains the last episode's cause in this Isaac Lab version.
    # It must not charge an event again on subsequent living steps.
    failure &= env.termination_manager.dones
    return failure.float() * torch.where(term.reset_mode <= 2, 1.0, 1.0 / 6.0)


def parkour_failed(env):
    term = term_for(env)
    local = term.robot.data.root_pos_w - env.scene.env_origins
    approach = ~term.started & ~term.locomotion_only
    late_approach = approach & (env.episode_length_buf * env.step_dt > getattr(term.cfg, "approach_timeout", 3.0))
    return (
        (local[:, 2] < -0.2)
        | (local[:, 1].abs() > 2.0)
        | (approach & (local[:, 0] > 0.12))
        | late_approach
        | (term.locomotion_only & (local[:, 0].abs() > 11.5))
    )


def parkour_reward(env, kind):
    term = term_for(env)
    robot, ref = term.robot, term.sample()
    active = term.active | (term.finished & ~(term.rolling_exit & (term.exit_time > 0)))
    local = robot.data.root_pos_w - env.scene.env_origins
    ascent = active & (term.reference_time <= term.bank.times[term.motion_ids, 1])
    pose_weight = torch.where(term.reference_time > term.bank.times[term.motion_ids, 2] + 0.25, 0.25, 1.0)
    if kind == "alive":
        return torch.ones(env.num_envs, device=env.device)
    if kind == "success":
        return (term.completion_ready & term.task_success).float() * torch.where(term.reset_mode <= 2, 1.0, 1.0 / 6.0)
    if kind == "joint_position":
        err = (robot.data.joint_pos[:, term.leg_ids] - ref["joint_position"]).square().mean(-1)
        return active * pose_weight * torch.exp(-err / math.radians(15) ** 2)
    if kind == "joint_velocity":
        err = (robot.data.joint_vel[:, term.leg_ids] - ref["joint_velocity"]).square().mean(-1)
        return active * pose_weight * torch.exp(-err / 9.0)
    if kind == "base_position":
        err = ((local - ref["position"]) / local.new_tensor([0.15, 0.12, 0.08])).square().mean(-1)
        return active * torch.exp(-err)
    if kind == "base_velocity":
        target = ref["velocity"].clone()
        gain = getattr(term.cfg, "reference_xy_feedback_gain", 0.0)
        if gain:
            limit = getattr(term.cfg, "reference_xy_feedback_limit", 0.75)
            target[:, :2] += (gain * (ref["position"][:, :2] - local[:, :2])).clamp(-limit, limit)
        return active * torch.exp(-(robot.data.root_lin_vel_w - target).square().sum(-1) / 0.75**2)
    if kind == "landing_lead_error":
        # Estimate the consequence of excess momentum relative to the teacher,
        # not a constant-velocity extrapolation of its accelerated TO trajectory.
        # Exact teacher states have zero cost; TO timing and world goals stay fixed.
        time_left = (term.bank.times[term.motion_ids, 2] - term.reference_time).clamp_min(0)
        position_error = local[:, :2] - ref["position"][:, :2]
        velocity_error = robot.data.root_lin_vel_w[:, :2] - ref["velocity"][:, :2]
        lead_error = position_error + time_left[:, None] * velocity_error
        preparing_or_flying = active & ~term.touchdown_seen & (time_left > 0)
        return preparing_or_flying * lead_error.square().sum(-1)
    if kind == "recovery_position_error":
        recovering = active & (term.touchdown_seen | (term.reference_time >= term.bank.times[term.motion_ids, 2]))
        return recovering * (local[:, :2] - ref["position"][:, :2]).square().sum(-1)
    if kind == "landing_position_error":
        # One control-step event, with the same auxiliary-RSI scaling as the
        # success/failure bonuses. where() prevents 0 * inf before first contact.
        error = torch.where(term.touchdown_event, term.touchdown_error.square(), 0)
        return error * torch.where(term.reset_mode <= 2, 1.0, 1.0 / 6.0)
    if kind == "attitude":
        dot = (robot.data.root_quat_w * ref["quaternion"]).sum(-1)
        return active * (1 - dot.square().clamp(0, 1))
    if kind == "angular_velocity":
        return active * (robot.data.root_ang_vel_w - ref["omega"]).square().sum(-1)
    if kind == "upward_velocity":
        # Immediate grounded ascent shaping remains, but exceeding the teacher
        # speed earns no extra reward when learning a commanded low jump.
        return ascent * torch.minimum(robot.data.root_lin_vel_w[:, 2].clamp_min(0), ref["velocity"][:, 2].clamp_min(0))
    if kind == "height_error":
        # Keep feedback after the nominal apex too: delayed takeoff and late
        # over-jumping must not escape the height penalty.
        return active * (local[:, 2] - ref["position"][:, 2]).square()
    if kind == "height_tanh":
        return ascent * (1 - torch.tanh((local[:, 2] - ref["position"][:, 2]).abs() / 0.05))
    if kind == "contacts":
        return active * (term.contact_state == ref["contacts"].bool()).float().mean(-1)
    if kind == "airborne":
        return active * term.upward_takeoff * (~ref["contacts"].bool().any(-1)) * (~term.contact_state.any(-1))
    if kind == "approach_velocity":
        goal = torch.zeros_like(robot.data.root_lin_vel_w)
        goal[:, 0] = term.forward_speed
        std = getattr(term.cfg, "approach_velocity_std", 0.5)
        return term.locomotion_active * torch.exp(
            -(robot.data.root_lin_vel_w[:, :2] - goal[:, :2]).square().sum(-1) / std**2
        )
    if kind == "approach_yaw":
        return term.locomotion_active * torch.exp(-robot.data.root_ang_vel_w[:, 2].square() / 0.25)
    if kind == "approach_speed_error":
        error = robot.data.root_lin_vel_w[:, 0] - term.forward_speed
        floor = getattr(term.cfg, "approach_speed_error_floor", 0.0)
        if floor > 0:
            error = error / term.forward_speed.abs().clamp_min(floor)
        return term.locomotion_active * error.square()
    if kind == "approach_joint_position":
        initial = term.bank.states["joint_position"][term.motion_ids, 0]
        err = (robot.data.joint_pos[:, term.leg_ids] - initial).square().mean(-1)
        return term.locomotion_active * torch.exp(-err / math.radians(15) ** 2)
    if kind == "inactive_orientation":
        return term.locomotion_active * robot.data.projected_gravity_b[:, :2].square().sum(-1)
    if kind == "inactive_angular_velocity":
        return term.locomotion_active * robot.data.root_ang_vel_w[:, :2].square().sum(-1)
    if kind == "inactive_thigh_calf":
        initial = term.bank.states["joint_position"][term.motion_ids, 0]
        error = (robot.data.joint_pos[:, term.leg_ids] - initial).abs()
        return term.locomotion_active * (error[:, 1::3].sum(-1) + error[:, 2::3].sum(-1))
    if kind == "inactive_height":
        return term.locomotion_active * (local[:, 2] - 0.42).square()
    if kind == "inactive_vz":
        return term.locomotion_active * robot.data.root_lin_vel_w[:, 2].square()
    if kind == "inactive_hip":
        return term.locomotion_active * (
            robot.data.joint_pos[:, term.leg_ids[::3]] - robot.data.default_joint_pos[:, term.leg_ids[::3]]
        ).abs().sum(-1)
    if kind == "landing_vz":
        landed = active & (term.reference_time > term.bank.times[term.motion_ids, 1]) & term.contact_state.any(-1)
        return landed * (robot.data.root_lin_vel_w[:, 2].abs() - 0.2).clamp_min(0).square()
    raise ValueError(f"Unknown parkour reward: {kind}")
