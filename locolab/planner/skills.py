"""Frozen skill inference adapter. Existing skill training configs are untouched.

This adapter is experimental: a single shared robot cannot simultaneously have
all three training-time actuator/self-collision configurations. It uses GoalGrid
joint limits and self-collision, but the base scene solver
iterations (4/1), versus GoalGrid 8/4. Cross-skill transfer requires measurement.
"""

from pathlib import Path

import torch

from .occupancy import yaw_rotate

POLICIES = (
    "multiskill_scratch80k/policy.pt",
    "highplatform_rough15k_batch196608/model_19998.pt",
    "goalgrid_scratch5k/policy.pt",
)


def load_frozen_policy(path, device):
    if path.name == "policy.pt":
        return torch.jit.load(str(path), map_location=device).eval()
    # The HighPlatform export is a local generated artifact, not tracked in Git.
    # Rebuild its actor directly from the retained checkpoint for portability.
    import yaml

    config = yaml.safe_load((path.parent / "params/agent.yaml").read_text())
    policy = config["policy"]
    if config["empirical_normalization"] or policy["actor_obs_normalization"] or policy["activation"] != "elu":
        raise ValueError("HighPlatform loader expects the retained unnormalized ELU MLP")
    dimensions = [53, *policy["actor_hidden_dims"], 16]
    layers = []
    for i, (left, right) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        layers.append(torch.nn.Linear(left, right))
        if i < len(dimensions) - 2:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = {
        key.removeprefix("actor."): value
        for key, value in checkpoint["model_state_dict"].items()
        if key.startswith("actor.")
    }
    actor.load_state_dict(state, strict=True)
    return actor.eval()


def decode_command(skill, normalized):
    # These are initial exploration limits, not certified capability envelopes.
    bounds = (
        ((-3.0, -0.5, -2.0, 0.22, 1.5, 0.0, 0.0, 0.0, 0.5, 0.0), (3.0, 0.5, 2.0, 0.46, 4.0, 0.5, 0.5, 0.5, 0.99, 0.15)),
        ((0.0, -0.3, -1.0), (1.5, 0.3, 1.0)),
        ((0.2, 0.15, 0.4, -0.15), (1.5, 0.25, 0.8, 0.15)),
    )
    lower, upper = (normalized.new_tensor(item) for item in bounds[skill])
    command = lower + (normalized[:, : len(lower)].clamp(-1, 1) + 1) * 0.5 * (upper - lower)
    if skill == 0:
        # Near-full stance maps to the exact rolling convention used in training.
        rolling = normalized[:, 8] > 0.8
        command[rolling, 8] = 1.0
        command[rolling, 9] = 0.0
    return command


class FrozenSkills:
    def __init__(self, env, root):
        self.env = env
        self.jump_always_active = False
        self.robot = env.scene["robot"]
        self.models = [load_frozen_policy(Path(root) / "deploy/policy/go2w" / path, env.device) for path in POLICIES]
        for model in self.models:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        self.legs, _ = self.robot.find_joints([".*_hip_.*", ".*_thigh_.*", ".*_calf_.*"])
        self.joints, _ = self.robot.find_joints([".*_hip_.*", ".*_thigh_.*", ".*_calf_.*", ".*_foot_.*"])
        self.phase = torch.zeros(env.num_envs, device=env.device)
        self.skill = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.timer = torch.zeros_like(self.phase)
        self.previous = torch.zeros(env.num_envs, 16, device=env.device)
        self.command = torch.zeros(env.num_envs, 10, device=env.device)
        self.jump_heading = torch.zeros(env.num_envs, 4, device=env.device)
        self.jump_heading[:, 0] = 1

    def reset(self, ids):
        self.phase[ids] = 0
        self.timer[ids] = 0
        self.skill[ids] = 0
        self.previous[ids] = 0
        self.command[ids] = 0
        self.command[ids, 3] = 0.34
        self.command[ids, 4] = 2.5
        self.command[ids, 8] = 1

    @property
    def busy(self):
        return (self.skill == 2) & (self.timer > 0)

    def allowed(self):
        allowed = torch.ones(self.env.num_envs, 3, dtype=torch.bool, device=self.env.device)
        allowed[self.busy, :2] = False
        return allowed

    def select(self, skill, command):
        # In-flight selection and landing targets are held, not overwritten by
        # the next high-level sample. Busy state is exposed to the Planner.
        available = ~self.busy
        for index in range(3):
            mask = available & (skill == index)
            decoded = decode_command(index, command[mask])
            self.command[mask] = 0
            self.command[mask, : decoded.shape[1]] = decoded
            self.skill[mask] = index
            if index == 2:
                self.timer[mask] = 1.8 if self.jump_always_active else 2.2  # Optional removal of the inactive recovery.
                self.jump_heading[mask] = self.robot.data.root_quat_w[mask]

    def proprio(self):
        data = self.robot.data
        return torch.cat(
            (
                data.root_lin_vel_b,
                data.root_ang_vel_b * 0.2,
                data.projected_gravity_b,
                data.joint_pos[:, self.legs] - data.default_joint_pos[:, self.legs],
                (data.joint_vel[:, self.joints] - data.default_joint_vel[:, self.joints]) * 0.05,
                self.previous,
                torch.nn.functional.one_hot(self.skill, 3).float(),
                (self.timer / 2.2)[:, None],
                self.busy.float()[:, None],
            ),
            -1,
        )

    @torch.no_grad()
    def act(self):
        from locolab.source.locomotion.mdp.wtw_math import contact_schedule

        data = self.robot.data
        prefix = torch.cat((data.root_ang_vel_b * 0.2, data.projected_gravity_b), -1)
        suffix = torch.cat(
            (
                data.joint_pos[:, self.legs] - data.default_joint_pos[:, self.legs],
                (data.joint_vel[:, self.joints] - data.default_joint_vel[:, self.joints]) * 0.05,
                self.previous,
            ),
            -1,
        )
        action = torch.zeros_like(self.previous)
        for index, model in enumerate(self.models):
            mask = self.skill == index
            if not mask.any():
                continue
            command = self.command[mask]
            if index == 0:
                clocks = contact_schedule(self.phase[mask], command)[2]
                scales = command.new_tensor([2.0, 2.0, 0.25, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
                obs = torch.cat((prefix[mask], command * scales, suffix[mask], clocks), -1)
            elif index == 1:
                obs = torch.cat((prefix[mask], command[:, :3], suffix[mask]), -1)
            else:
                velocity = torch.zeros(len(command), 3, device=command.device)
                velocity[:, 0] = command[:, 0]
                velocity = yaw_rotate(
                    yaw_rotate(velocity, self.jump_heading[mask]), data.root_quat_w[mask], inverse=True
                )
                velocity[:, 2] = 0  # Angular velocity command, not vertical velocity.
                active = (
                    torch.ones_like(self.timer[mask, None])
                    if self.jump_always_active
                    else (self.timer[mask] > 0.4).float()[:, None]
                )
                descriptors = command[:, 1:4] * active
                obs = torch.cat((prefix[mask], velocity, active, descriptors, suffix[mask]), -1)
            assert obs.shape[-1] == (64, 53, 57)[index]
            action[mask] = model(obs.clamp(-100, 100))
        return self._advance(action)

    def _advance(self, action):
        # Preserve normal tensor buffers even when callers use inference_mode.
        # Rebinding to inference tensors would break reset() outside that scope.
        self.phase.copy_(torch.remainder(self.phase + self.env.step_dt * self.command[:, 4] * (self.skill == 0), 1.0))
        self.timer.copy_((self.timer - self.env.step_dt).clamp(min=0))
        self.previous.copy_(action.clamp(-100, 100))
        return self.previous
