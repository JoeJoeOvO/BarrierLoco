"""Shared navigation backbone with categorical skill and conditional cmd heads.

The probability is p(skill|obs)*p(cmd_skill|skill,obs). Unselected command heads
must not contribute to PPO's log probability. Actions are squashed normals;
training stores the pre-tanh sample to avoid inverse-tanh numerical instability.
"""

import torch
from torch import nn
from torch.distributions import Categorical, Normal
from torch.nn import functional as F

SKILL_NAMES = ("multiskill", "highplatform", "goalgrid")
CMD_DIMS = (10, 3, 4)  # GoalGrid: speed, peak height, fixed target X, fixed target Y.


class NavigationActorCritic(nn.Module):
    def __init__(self, proprio_dim=58):
        super().__init__()
        self.occ_encoder = nn.Sequential(
            nn.Conv3d(2, 8, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv3d(8, 16, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(32 * 2 * 3 * 4, 128),
            nn.ELU(),
        )
        self.state_encoder = nn.Sequential(nn.Linear(proprio_dim + 3, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU())
        self.backbone = nn.Sequential(nn.Linear(192, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU())
        self.skill_head = nn.Linear(128, 3)
        self.cmd_heads = nn.ModuleList(nn.Linear(128, dim) for dim in CMD_DIMS)
        self.log_std = nn.ParameterList(nn.Parameter(torch.full((dim,), -1.0)) for dim in CMD_DIMS)
        self.value_head = nn.Linear(128, 1)

    def forward(self, observation):
        occ = self.occ_encoder(observation["occ"])
        state = self.state_encoder(torch.cat((observation["proprio"], observation["goal"]), -1))
        shared = self.backbone(torch.cat((occ, state), -1))
        return self.skill_head(shared), [head(shared) for head in self.cmd_heads], self.value_head(shared).squeeze(-1)

    def sample(self, observation, allowed=None, deterministic=False):
        logits, means, value = self(observation)
        if allowed is not None:
            if not allowed.any(-1).all():
                raise ValueError("Each environment requires at least one executable skill")
            logits = logits.masked_fill(~allowed, -torch.inf)
        categorical = Categorical(logits=logits)
        skill = logits.argmax(-1) if deterministic else categorical.sample()
        raw = logits.new_zeros((len(logits), max(CMD_DIMS)))
        for index, (mean, log_std) in enumerate(zip(means, self.log_std)):
            sample = mean if deterministic else Normal(mean, log_std.clamp(-5, 1).exp()).sample()
            mask = skill == index
            raw[mask, : CMD_DIMS[index]] = sample[mask]
        log_prob = self._log_prob(logits, means, skill, raw, observation["proprio"][:, -1] < 0.5)
        return dict(skill=skill, command=raw.tanh(), raw_command=raw, log_prob=log_prob, value=value)

    def _log_prob(self, logits, means, skill, raw, command_active):
        result = Categorical(logits=logits).log_prob(skill)
        for index, (mean, log_std) in enumerate(zip(means, self.log_std)):
            u = raw[:, : CMD_DIMS[index]]
            # Stable log(1-tanh(u)^2), including the squash Jacobian.
            log_jacobian = 2 * (torch.log(u.new_tensor(2.0)) - u - F.softplus(-2 * u))
            term = (Normal(mean, log_std.clamp(-5, 1).exp()).log_prob(u) - log_jacobian).sum(-1)
            result = result + torch.where((skill == index) & command_active, term, torch.zeros_like(term))
        return result

    def evaluate(self, observation, skill, raw_command, allowed=None, return_entropy=False):
        logits, means, value = self(observation)
        if allowed is not None:
            logits = logits.masked_fill(~allowed, -torch.inf)
        active = observation["proprio"][:, -1] < 0.5
        log_prob = self._log_prob(logits, means, skill, raw_command, active)
        if not return_entropy:
            return log_prob, value
        categorical = Categorical(logits=logits)
        entropy = categorical.entropy()
        for index, (mean, log_std) in enumerate(zip(means, self.log_std)):
            normal = Normal(mean, log_std.clamp(-5, 1).exp())
            u = normal.rsample()
            log_jacobian = 2 * (torch.log(u.new_tensor(2.0)) - u - F.softplus(-2 * u))
            conditional = (normal.entropy() + log_jacobian).sum(-1)
            entropy = entropy + categorical.probs[:, index] * conditional * active
        return log_prob, value, entropy

    @torch.no_grad()
    def initialize_navigation(self):
        """From-scratch prior: near-trained cmd settings, no terrain-based routing."""
        self.skill_head.weight.mul_(0.01)
        self.skill_head.bias.copy_(self.skill_head.bias.new_tensor([1.0, 0.0, -1.0]))
        # Normalized cmd defaults: trot, forward high-platform, moderate jump.
        defaults = (
            [0.0, 0.0, 0.0, 0.0, -0.2, 0.999, -0.999, -0.999, -0.999, 0.067],
            [0.333, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        )
        for head, command in zip(self.cmd_heads, defaults):
            head.weight.mul_(0.01)
            head.bias.copy_(torch.atanh(head.bias.new_tensor(command)))
