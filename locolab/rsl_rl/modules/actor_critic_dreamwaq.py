from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NoReturn

import torch
import torch.nn as nn
from rsl_rl.networks import MLP, EmpiricalNormalization
from tensordict import TensorDict
from torch.distributions import Normal

from .dreamwaq_estimator import DreamWaQVAE


class ActorCriticDreamWaQ(nn.Module):
    """DreamWaQ-style actor-critic with history encoder and latent terrain context."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [512, 256, 128],
        critic_hidden_dims: tuple[int] | list[int] = [512, 256, 128],
        estimator_hidden_dims: tuple[int] | list[int] = [128],
        context_decoder_hidden_dims: tuple[int] | list[int] = [64, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        history_obs_set: str = "policy_history",
        velocity_target_group: str = "dreamwaq_velocity",
        context_target_group: str = "policy",
        context_latent_dim: int = 16,
        vae_sigma_min: float = 0.0,
        vae_sigma_max: float = 5.0,
        reconstruction_mask_indices: Sequence[int] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticDreamWaQ.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.history_obs_set = history_obs_set
        self.velocity_target_group = velocity_target_group
        self.context_target_group = context_target_group
        self.context_latent_dim = context_latent_dim
        self.state_dependent_std = state_dependent_std
        self.noise_std_type = noise_std_type

        num_actor_obs = self._get_obs_dim(obs, self.obs_groups["policy"])
        num_history_obs = self._get_obs_dim(obs, self.obs_groups.get(self.history_obs_set, [self.history_obs_set]))
        num_critic_obs = self._get_obs_dim(obs, self.obs_groups["critic"])
        num_velocity_targets = obs[self.velocity_target_group].shape[-1]
        num_context_targets = obs[self.context_target_group].shape[-1]

        self.vae = DreamWaQVAE(
            history_obs_dim=num_history_obs,
            reconstruction_obs_dim=num_context_targets,
            velocity_dim=num_velocity_targets,
            latent_dim=context_latent_dim,
            encoder_hidden_dims=estimator_hidden_dims,
            decoder_hidden_dims=context_decoder_hidden_dims,
            activation=activation,
            sigma_min=vae_sigma_min,
            sigma_max=vae_sigma_max,
            reconstruction_mask_indices=reconstruction_mask_indices,
        )

        actor_input_dim = num_actor_obs + num_velocity_targets + context_latent_dim
        if self.state_dependent_std:
            self.actor = MLP(actor_input_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)

        print(f"DreamWaQ VAE: {self.vae}")
        print(f"DreamWaQ actor MLP: {self.actor}")
        print(f"DreamWaQ critic MLP: {self.critic}")

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
            self.actor_history_normalizer = EmpiricalNormalization(num_history_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()
            self.actor_history_normalizer = nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()

        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")

        self.distribution = None

        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        del kwargs
        actor_input = self._compute_actor_input(obs, sample_latent=self.training)
        self._update_distribution(actor_input)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_input = self._compute_actor_input(obs, sample_latent=False)
        if self.state_dependent_std:
            return self.actor(actor_input)[..., 0, :]
        else:
            return self.actor(actor_input)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        del kwargs
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.obs_groups["policy"]], dim=-1)

    def get_history_obs(self, obs: TensorDict) -> torch.Tensor:
        history_groups = self.obs_groups.get(self.history_obs_set, [self.history_obs_set])
        return torch.cat([obs[group] for group in history_groups], dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.obs_groups["critic"]], dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
            self.actor_history_normalizer.update(self.get_history_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def compute_dreamwaq_losses(
        self,
        obs: TensorDict,
        next_obs: TensorDict | None = None,
        kld_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        history_obs = self.actor_history_normalizer(self.get_history_obs(obs))
        # Reconstruction target is the *next* observation (paper Eq.7).
        # Fall back to current obs only when next_obs is unavailable.
        reconstruction_source = next_obs if next_obs is not None else obs
        context_target = reconstruction_source[self.context_target_group]
        velocity_target = obs[self.velocity_target_group]
        return self.vae.loss_fn(history_obs, context_target, velocity_target, kld_weight=kld_weight)

    def get_vae_constraint_info(self, obs: TensorDict) -> dict[str, torch.Tensor | float]:
        history_obs = self.actor_history_normalizer(self.get_history_obs(obs))
        return self.vae.get_constrained_latent_params(history_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        state_dict = dict(state_dict)
        remapped_state_dict: dict[str, torch.Tensor] = {}
        legacy_prefixes = {
            "estimator_encoder.": "vae.encoder.",
            "velocity_head.": "vae.vel_mu.",
            "context_mean_head.": "vae.latent_mu.",
            "context_logvar_head.": "vae.latent_logvar.",
            "latent_var.": "vae.latent_logvar.",
            "vel_var.": "vae.vel_logvar.",
            "vae.latent_var.": "vae.latent_logvar.",
            "vae.vel_var.": "vae.vel_logvar.",
        }
        remapped_legacy = False
        for key, value in state_dict.items():
            new_key = key
            for legacy_prefix, new_prefix in legacy_prefixes.items():
                if key.startswith(legacy_prefix):
                    new_key = key.replace(legacy_prefix, new_prefix, 1)
                    remapped_legacy = True
                    break
            remapped_state_dict[new_key] = value

        current_state_dict = super().state_dict()
        compatible_state_dict: dict[str, torch.Tensor] = {}
        dropped_keys: list[str] = []
        for key, value in remapped_state_dict.items():
            if key in current_state_dict and current_state_dict[key].shape == value.shape:
                compatible_state_dict[key] = value
            else:
                dropped_keys.append(key)

        if dropped_keys:
            print(f"DreamWaQ: Dropping {len(dropped_keys)} incompatible keys: {dropped_keys[:6]}...")

        load_result = super().load_state_dict(compatible_state_dict, strict=False)
        missing_count = len(load_result.missing_keys)
        unexpected_count = len(load_result.unexpected_keys)
        if missing_count or unexpected_count:
            print(
                "DreamWaQ: checkpoint loaded with missing/unexpected keys "
                f"(missing={missing_count}, unexpected={unexpected_count})."
            )

        # Treat as cold start only when weights were reshaped or renamed — in those
        # cases the optimizer state from the checkpoint no longer aligns with the
        # current parameters. Pure missing/unexpected keys from model evolution
        # still permit a genuine resume.
        return not (remapped_legacy or dropped_keys)

    def _compute_actor_input(self, obs: TensorDict, sample_latent: bool) -> torch.Tensor:
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        history_obs = self.actor_history_normalizer(self.get_history_obs(obs))
        if sample_latent:
            (context_sample, velocity_sample), _ = self.vae(history_obs)
        else:
            context_sample, velocity_sample = self.vae.inference(history_obs)

        # Detach VAE outputs so PPO policy gradients do not flow back into the
        # VAE encoder.  The VAE is trained by its own dedicated optimizer in
        # PPODreamWaQ._optimize_additional_modules.
        return torch.cat((actor_obs, velocity_sample.detach(), context_sample.detach()), dim=-1)

    def _update_distribution(self, actor_input: torch.Tensor) -> None:
        if self.state_dependent_std:
            mean_and_std = self.actor(actor_input)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")
        else:
            mean = self.actor(actor_input)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")
        self.distribution = Normal(mean, std)

    @staticmethod
    def _get_obs_dim(obs: TensorDict, groups: list[str]) -> int:
        obs_dim = 0
        for group in groups:
            assert len(obs[group].shape) == 2, "The DreamWaQ policy only supports 1D observations."
            obs_dim += obs[group].shape[-1]
        return obs_dim
