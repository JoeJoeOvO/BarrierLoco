"""Playback-only off-grid commands and independent jump measurements.

Reference motions, training rewards and bank labels are never rewritten. Only
actor command columns are overridden. Teacher-specific success terminations must
be disabled by the caller; this inspector does not report teacher success rates.
"""

import math

import torch


def target_for_attempt(index):
    """Smooth target families, sampled once per attempt (not retargeted in flight)."""
    phase = 2 * math.pi * ((index % 32) / 32 + 0.037)
    group = (index // 32) % 4
    height, x, y, speed = 0.20, 0.60, 0.0, 1.0
    if group == 0:
        height = 0.20 + 0.05 * math.sin(phase)
    elif group == 1:
        x = 0.60 + 0.20 * math.sin(phase)
    elif group == 2:
        y = 0.15 * math.sin(phase)
    else:
        height = 0.20 + 0.05 * math.sin(phase)
        x = 0.60 + 0.20 * math.sin(phase * 2 + 0.7)
        y = 0.15 * math.cos(phase * 3)
        speed = 0.85 + 0.65 * math.sin(phase + 0.4)
    return ("height", "landing X", "landing Y", "combined")[group], (height, x, y), speed


class GoalGridInspector:
    def __init__(self, env):
        self.env = env
        self.term = env.command_manager.get_term("parkour")
        self.index = -1
        self.last_result = "No completed attempt"
        self.next_attempt()

    def next_attempt(self):
        self.index += 1
        self.group, self.target, self.speed = target_for_attempt(self.index)
        self.peak = 0.0
        self.airtime = 0.0
        self.seen_air = False
        self.landing = None
        self.start_z = None
        self.started = False
        print(
            f"[GENERALIZATION] attempt={self.index + 1} family={self.group} "
            f"HXY={self.target} speed={self.speed:.3f}",
            flush=True,
        )

    def actor_observation(self, obs):
        # Actor layout: omega(3), gravity(3), command(7), joints, previous action.
        # Keep velocity, yaw and the existing ground-trigger/2.2 s execution clock.
        obs = obs.clone()
        actor = obs if isinstance(obs, torch.Tensor) else obs["policy"]
        if not bool(self.term.finished[0]):
            actor[0, 10:13] = actor.new_tensor(self.target)
        else:
            actor[0, 9:13] = 0
        return obs

    def measure(self):
        robot = self.env.scene["robot"]
        local = robot.data.root_pos_w[0] - self.env.scene.env_origins[0]
        if bool(self.term.started[0]) and not self.started:
            self.started = True
            self.start_z = float(self.term.launch_height[0])
        if self.start_z is None:
            self.start_z = float(local[2])
        rise = float(local[2]) - self.start_z
        contact = bool(self.term.contact_state[0].any())
        if self.started and self.landing is None:
            self.peak = max(self.peak, rise)
            if not contact:
                self.airtime += self.env.step_dt
                if self.airtime >= 0.08 and rise > 0.025:
                    self.seen_air = True
            else:
                if self.seen_air:
                    self.landing = (float(local[0]), float(local[1]))
                self.airtime = 0.0
        stage = "approach" if not self.started else "landed" if self.landing else "jump"
        target = self.target
        landing_text = "pending"
        if self.landing is not None:
            error = math.dist(self.landing, target[1:])
            landing_text = f"X={self.landing[0]:.3f} Y={self.landing[1]:.3f}, error={error:.3f} m"
        details = (
            f"INTERPOLATION TEST | {stage} | speed={self.speed:.2f} m/s\n"
            f"Peak rise: {self.peak:.3f} / target {target[0]:.3f} m\n"
            f"First landing: {landing_text}\nLast: {self.last_result}"
        )
        return list(target), [rise, float(local[0]), float(local[1])], details

    def finish(self):
        if self.landing is None:
            result = f"#{self.index + 1}: no measured landing, peak={self.peak:.3f} m"
        else:
            result = (
                f"#{self.index + 1}: height error={self.peak - self.target[0]:+.3f} m, "
                f"landing error={math.dist(self.landing, self.target[1:]):.3f} m"
            )
        self.last_result = result
        print(f"[GENERALIZATION_RESULT] {result}", flush=True)
