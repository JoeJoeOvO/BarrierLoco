"""10 Hz navigation over frozen 50 Hz skills.

Training uses manual resets at high-level boundaries: the terminal observation
is captured BEFORE any reset and supplied for timeout value bootstrapping.
The old debug mode retains the base environment's automatic resets.
"""

import time

import torch

from .occupancy import OccupancySpec, lidar_directions, voxelize_rays, yaw_rotate
from .skills import FrozenSkills


class NavigationRollout:
    def __init__(
        self, env, root, manual_reset=False, falls_only=False, cross_terrain=True, gap_only=False, reward_scales=None
    ):
        self.env = env.unwrapped
        self.manual_reset = manual_reset
        # Playback can run continuously; training retains bounded episodes.
        self.falls_only = falls_only
        self.cross_terrain = cross_terrain
        self.gap_only = gap_only
        self.all_gap_terrain = False
        self.forward_only = False  # Optional playback route; training is unchanged.
        self.reward_scales = reward_scales or (1.0, 5.0, 5.0, 0.01)
        if falls_only and not manual_reset:
            raise ValueError("falls_only requires manual_reset")
        self.realtime = False
        self.step_observer = None  # Optional read-only playback telemetry.
        self.log_resets = False
        self.fall_time = torch.zeros(self.env.num_envs, device=self.env.device)
        self.base_contact_ids, _ = self.env.scene["contact_forces"].find_bodies("base")
        if manual_reset and self.env.termination_manager.active_terms:
            raise ValueError("Disable native terminations when using high-level manual resets")
        self.skills = FrozenSkills(self.env, root)
        self.spec = OccupancySpec()
        self.directions = lidar_directions(self.env.device)
        self.goal_w = torch.zeros(self.env.num_envs, 3, device=self.env.device)
        self.goal_count = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.env.device)
        self.goal_valid = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        self.arrival_steps = torch.zeros_like(self.goal_count)
        self.last_occ = None
        self.directional_goals = False
        self.arrival_radius = 0.5
        self.arrival_speed = float("inf")
        self.arrival_hold_steps = 2
        self.route_heading = torch.rand(self.env.num_envs, device=self.env.device) * (2 * torch.pi)

    def raycast(self, starts, directions, distance=20.0):
        from isaaclab.utils.warp import raycast_mesh

        mesh = self.env.scene["height_scanner"].meshes["/World/ground"]
        return raycast_mesh(starts, directions, mesh, max_dist=distance)[0]

    def reset(self):
        self.env.reset()
        ids = torch.arange(self.env.num_envs, device=self.env.device)
        self.skills.reset(ids)
        self.fall_time[ids] = 0
        self.route_heading[ids] = torch.rand(len(ids), device=self.env.device) * (2 * torch.pi)
        self.goal_count[:] = 0
        self.sample_goals(ids)
        return self.observe()

    def sample_goals(self, ids):
        """Sample supported endpoints around the robot across terrain tiles.

        Footprint checks reject gap bottoms/edges, large steps and endpoints too
        close to the robot. Rejection exhaustion marks goal invalid, never gives
        a free arrival bonus. Terrain feasibility is not path reachability.
        """
        robot = self.env.scene["robot"].data
        origin = self.env.scene.env_origins
        self.goal_valid[ids] = False
        self.arrival_steps[ids] = 0
        pending = ids
        offsets = self.goal_w.new_tensor([[0, 0, 0], [0.3, 0, 0], [-0.3, 0, 0], [0, 0.3, 0], [0, -0.3, 0]])
        for _ in range(24):
            if not len(pending):
                break
            terrain = self.env.scene.terrain
            generator = terrain.cfg.terrain_generator
            uniform = torch.rand(len(pending), 2, device=self.env.device)
            if self.gap_only:
                # Every target is on the other bank. No skill is prescribed.
                position = robot.root_pos_w[pending].clone()
                half_x = generator.num_rows * generator.size[0] / 2
                row = ((position[:, 0] + half_x) / generator.size[0]).floor().clamp(0, generator.num_rows - 1)
                edge = row * generator.size[0] - half_x + 6.0
                before = position[:, 0] < edge
                offset = 1.0 + uniform[:, 0]
                if self.forward_only:
                    # Advance through successive tiles; intermediate ground goals
                    # bridge long approaches without prescribing any skill.
                    next_edge = torch.where(before, edge, edge + generator.size[0])
                    destination = (next_edge + 1.8).clamp(max=half_x - 1.0)
                    position[:, 0] = torch.minimum(position[:, 0] + 6.0, destination)
                    # Do not put an intermediate endpoint inside/next to a gap.
                    near_gap = (position[:, 0] > next_edge - 0.5) & (position[:, 0] < next_edge + 1.0)
                    position[:, 0] = torch.where(near_gap, next_edge + 1.8, position[:, 0])
                else:
                    position[:, 0] = torch.where(before, edge + 0.6 + offset, edge - offset)
                if not self.forward_only:
                    position[:, 1] += (uniform[:, 1] - 0.5) * 3.0
                half_y = generator.num_cols * generator.size[1] / 2
                in_map = position[:, 1].abs() < half_y - 0.5
                custom = torch.ones(len(pending), dtype=torch.bool, device=self.env.device)
            elif self.cross_terrain:
                # Local navigation goals can cross tile/family boundaries. Bounds
                # apply to the entire generated map, never the spawn tile.
                position = robot.root_pos_w[pending].clone()
                if self.directional_goals:
                    # Persist direction across goals, with gradual bends. Retry
                    # broader headings near boundaries or unsupported endpoints.
                    spread = torch.pi / 3 if _ < 12 else torch.pi
                    angle = self.route_heading[pending] + (uniform[:, 0] - 0.5) * 2 * spread
                    radius = 8.0 + 8.0 * uniform[:, 1]
                else:
                    angle = uniform[:, 0] * (2 * torch.pi)
                    radius = 4.0 + 16.0 * uniform[:, 1]
                position[:, 0] += radius * angle.cos()
                position[:, 1] += radius * angle.sin()
                half_x = generator.num_rows * generator.size[0] / 2
                half_y = generator.num_cols * generator.size[1] / 2
                in_map = (position[:, 0].abs() < half_x - 0.5) & (position[:, 1].abs() < half_y - 0.5)
                column = ((position[:, 1] + half_y) / generator.size[1]).long()
                custom = torch.ones_like(column, dtype=torch.bool) if self.all_gap_terrain else column < 4
            else:
                custom = terrain.terrain_types[pending] < 4
                position = origin[pending].clone()
                position[:, 0] += torch.where(custom, uniform[:, 0] * 12.0 + 0.5, uniform[:, 0] * 12.0 - 6.0)
                position[:, 1] += (uniform[:, 1] - 0.5) * torch.where(custom, 3.0, 12.0)
                in_map = torch.ones(len(pending), dtype=torch.bool, device=self.env.device)
            starts = position[:, None] + offsets[None]
            # Cast from above the whole map, even when the robot is in a valley.
            starts[:, :, 2] = terrain.terrain_origins[..., 2].max() + 10.0
            direction = torch.zeros_like(starts)
            direction[..., 2] = -1
            hits = self.raycast(starts, direction, distance=50.0)
            heights = hits[..., 2]
            distance = (position[:, :2] - robot.root_pos_w[pending, :2]).norm(dim=-1)
            accepted = torch.isfinite(hits).all(dim=(1, 2)) & (heights.amax(-1) - heights.amin(-1) < 0.35)
            accepted &= (distance > (0.5 if self.forward_only else 2.0)) & (
                distance < (21.0 if self.cross_terrain else 12.0)
            )
            accepted &= in_map
            accepted &= ~custom | (heights[:, 0] >= -0.2)
            chosen = pending[accepted]
            self.goal_w[chosen] = hits[accepted, 0]
            self.goal_valid[chosen] = True
            if self.directional_goals:
                delta = self.goal_w[chosen, :2] - robot.root_pos_w[chosen, :2]
                self.route_heading[chosen] = torch.atan2(delta[:, 1], delta[:, 0])
            pending = pending[~accepted]
        # Safe explicit missing-goal sentinel. No arrival can be credited to it.
        self.goal_w[pending] = robot.root_pos_w[pending]

    def observe(self):
        robot = self.env.scene["robot"].data
        starts = robot.root_pos_w[:, None].expand(-1, len(self.directions), -1).contiguous()
        directions = yaw_rotate(self.directions[None].expand(self.env.num_envs, -1, -1), robot.root_quat_w).contiguous()
        hits = self.raycast(starts, directions, self.spec.max_range)
        distance = (hits - starts).norm(dim=-1)
        occ = voxelize_rays(self.directions, distance, self.spec)
        relative = yaw_rotate(self.goal_w - robot.root_pos_w, robot.root_quat_w, inverse=True)
        # Goal XY in meters scaled by map range, plus height difference.
        goal = (relative / relative.new_tensor([8.0, 8.0, 2.0])).clamp(-2.0, 2.0)
        goal[~self.goal_valid] = 0
        self.last_occ = occ
        return {"occ": occ, "proprio": self.skills.proprio(), "goal": goal}

    @torch.no_grad()
    def step(self, skill, command):
        self.skills.select(skill, command)
        robot = self.env.scene["robot"].data
        old_distance = (self.goal_w[:, :2] - robot.root_pos_w[:, :2]).norm(dim=-1)
        failed = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        timed_out = torch.zeros_like(failed)
        for _ in range(5):
            tick = time.monotonic()
            action = self.skills.act()
            _, _, terminated, truncated, _ = self.env.step(action)
            # Use physical evidence for collapsed/rim-hung robots too. A short
            # landing impact or crouch alone must not trigger this condition.
            base_force = (
                self.env.scene["contact_forces"].data.net_forces_w[:, self.base_contact_ids].norm(dim=-1).amax(-1)
            )
            collapsed = (robot.root_pos_w[:, 2] < 0.12) if self.all_gap_terrain else torch.zeros_like(failed)
            bad_support = (base_force > 10.0) | collapsed
            self.fall_time = torch.where(bad_support, self.fall_time + self.env.step_dt, 0.0)
            if self.step_observer is not None:
                self.step_observer()
            if self.realtime:
                time.sleep(max(0.0, self.env.step_dt - (time.monotonic() - tick)))
            done = terminated | truncated
            failed |= terminated
            timed_out |= truncated
            if done.any():
                ids = done.nonzero().flatten()
                self.skills.reset(ids)
                self.sample_goals(ids)
        if self.manual_reset:
            from locolab.source.locomotion.mdp.planner_terrain import fell_into_gap, left_course

            if self.cross_terrain:
                generator = self.env.scene.terrain.cfg.terrain_generator
                half_y = generator.num_cols * generator.size[1] / 2
                column = ((robot.root_pos_w[:, 1] + half_y) / generator.size[1]).long()
                pit_column = (
                    torch.ones_like(column, dtype=torch.bool)
                    if self.gap_only or self.all_gap_terrain
                    else ((column == 2) | (column == 3))
                )
                in_pit = pit_column & (robot.root_pos_w[:, 2] < (0.0 if self.all_gap_terrain else -0.35))
            else:
                in_pit = fell_into_gap(self.env)
            tilted = robot.projected_gravity_b[:, 2] > -0.5403023
            collapsed = self.fall_time >= 0.3
            failed = in_pit | tilted | collapsed
            if not self.falls_only:
                if self.cross_terrain:
                    half_x = generator.num_rows * generator.size[0] / 2
                    failed |= (robot.root_pos_w[:, 0].abs() > half_x) | (robot.root_pos_w[:, 1].abs() > half_y)
                else:
                    failed |= left_course(self.env)
                timed_out = (self.env.episode_length_buf >= self.env.max_episode_length) & ~failed
        distance = (self.goal_w[:, :2] - robot.root_pos_w[:, :2]).norm(dim=-1)
        height_above_goal = robot.root_pos_w[:, 2] - self.goal_w[:, 2]
        on_goal_surface = (height_above_goal > 0.05) & (height_above_goal < 0.65)
        slow_enough = robot.root_lin_vel_w[:, :2].norm(dim=-1) < self.arrival_speed
        near = (
            on_goal_surface
            & (distance < self.arrival_radius)
            & slow_enough
            & self.goal_valid
            & ~self.skills.busy
            & ~failed
            & ~timed_out
        )
        self.arrival_steps = torch.where(near, self.arrival_steps + 1, 0)
        arrived = self.arrival_steps >= self.arrival_hold_steps  # 0.2 s debounce; no robot reset.
        progress = torch.where(failed | timed_out | ~self.goal_valid, 0.0, old_distance - distance)
        # Diagnostic navigation reward proposal. Scale once at the high-level step.
        progress_scale, arrival_bonus, failure_cost, step_cost = self.reward_scales
        reward = progress_scale * progress + arrival_bonus * arrived.float() - failure_cost * failed.float() - step_cost
        # Compute terminal state before either goal refresh or physical reset.
        terminal_observation = self.observe() if self.manual_reset else None
        self.goal_count += arrived.long()
        if arrived.any():
            self.sample_goals(arrived.nonzero().flatten())
        missing = (~self.goal_valid).nonzero().flatten()
        if len(missing):
            self.sample_goals(missing)
        done = failed | timed_out
        if self.manual_reset and done.any():
            ids = done.nonzero().flatten()
            if self.log_resets:
                import json

                for index in ids.tolist():
                    print(
                        "NAV_RESET "
                        + json.dumps(
                            {
                                "env": index,
                                "z": float(robot.root_pos_w[index, 2]),
                                "pit": bool(in_pit[index]),
                                "tilted": bool(tilted[index]),
                                "collapsed_or_base_contact": bool(collapsed[index]),
                                "bad_support_seconds": float(self.fall_time[index]),
                                "timeout": bool(timed_out[index]),
                            }
                        ),
                        flush=True,
                    )
            self.env._reset_idx(ids)
            self.env.scene.write_data_to_sim()
            self.env.sim.forward()
            self.skills.reset(ids)
            self.fall_time[ids] = 0
            self.route_heading[ids] = torch.rand(len(ids), device=self.env.device) * (2 * torch.pi)
            self.sample_goals(ids)
        if self.manual_reset and not done.any() and not arrived.any() and not len(missing):
            observation = terminal_observation
        else:
            observation = self.observe()
        return (
            observation,
            reward,
            failed | timed_out,
            {
                "terminal_observation": terminal_observation,
                "arrived": arrived,
                "failed": failed,
                "timeout": timed_out,
                "goals_completed": self.goal_count.clone(),
                "goal_valid": self.goal_valid.clone(),
            },
        )
