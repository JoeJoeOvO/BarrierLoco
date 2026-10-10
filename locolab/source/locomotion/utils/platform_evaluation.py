"""Conservative, sequence-aware fixed-platform success checks (no Isaac imports)."""

import torch

START_X = 1.5  # Platform start 2.5 m minus terrain spawn origin 1.0 m.
END_X = 3.0
HALF_WIDTH = 1.0
WHEEL_RADIUS = 0.086


def support_masks(root, wheels, force_z, gravity_z, heights):
    """All coordinates are relative to the flat-ground terrain spawn origin."""
    support = (force_z > 5.0).sum(dim=1) >= 2
    upright = gravity_z < -0.7
    inside_x = ((wheels[:, :, 0] >= START_X + 0.02) & (wheels[:, :, 0] <= END_X - 0.02)).all(dim=1)
    inside_y = (wheels[:, :, 1].abs() <= HALF_WIDTH - 0.02).all(dim=1)
    top_height = (wheels[:, :, 2] - (heights[:, None] + WHEEL_RADIUS)).abs().amax(dim=1) <= 0.065
    on_top = inside_x & inside_y & top_height & support & upright
    beyond = (wheels[:, :, 0].amin(dim=1) >= END_X + 0.05) & (root[:, 0] >= END_X + 0.35)
    on_ground = (wheels[:, :, 2] - WHEEL_RADIUS).abs().amax(dim=1) <= 0.08
    landed_beyond = beyond & on_ground & support & upright
    corridor_exit = root[:, 1].abs() > 0.75
    return on_top, landed_beyond, corridor_exit


class PlatformProgress:
    """One attempt per environment: first failure is final, and bypasses cannot pass."""

    def __init__(self, heights, dt=0.02):
        self.heights = heights
        self.dt = dt
        self.mount_steps = round(0.1 / dt)
        self.land_steps = round(0.2 / dt)
        zeros = torch.zeros_like(heights, dtype=torch.bool)
        self.active = ~zeros
        self.mounted = zeros.clone()
        self.passed = zeros.clone()
        self.terminated = zeros.clone()
        self.timed_out = zeros.clone()
        self.left_corridor = zeros.clone()
        self.top_streak = torch.zeros_like(heights, dtype=torch.long)
        self.land_streak = self.top_streak.clone()
        self.mount_time = torch.full_like(heights, -1.0)
        self.pass_time = self.mount_time.clone()
        self.max_forward = torch.full_like(heights, -1e6)

    def update(self, root, wheels, force_z, gravity_z, terminated, truncated, time_s):
        on_top, beyond, exit_corridor = support_masks(root, wheels, force_z, gravity_z, self.heights)
        valid = self.active & ~terminated & ~truncated
        self.max_forward = torch.maximum(self.max_forward, torch.where(valid, root[:, 0], self.max_forward))
        left = valid & exit_corridor
        valid &= ~left
        self.top_streak = torch.where(valid & on_top, self.top_streak + 1, 0)
        just_mounted = valid & ~self.mounted & (self.top_streak >= self.mount_steps)
        self.mount_time[just_mounted] = time_s
        self.mounted |= just_mounted
        self.land_streak = torch.where(valid & self.mounted & beyond, self.land_streak + 1, 0)
        reached = valid & (self.land_streak >= self.land_steps)
        self.pass_time[reached] = time_s
        self.passed |= reached
        self.terminated |= self.active & terminated
        self.timed_out |= self.active & ~terminated & truncated
        self.left_corridor |= left
        self.active &= ~(terminated | truncated | left | reached)

    def finish(self):
        self.timed_out |= self.active
        self.active[:] = False
        outcomes = self.passed.int() + self.terminated.int() + self.timed_out.int() + self.left_corridor.int()
        assert (outcomes == 1).all().item()
