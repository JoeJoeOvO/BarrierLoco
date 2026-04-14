from __future__ import annotations

from typing import Any, NoReturn

import torch
import torch.nn as nn
from rsl_rl.networks import MLP, EmpiricalNormalization
from tensordict import TensorDict
from torch.distributions import Normal


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
        estimator_hidden_dims: tuple[int] | list[int] = [128, 64],
        context_decoder_hidden_dims: tuple[int] | list[int] = [64, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        history_obs_set: str = "policy_history",
        velocity_target_group: str = "dreamwaq_velocity",
        context_target_group: str = "policy",
        context_latent_dim: int = 16,
        velocity_loss_weight: float = 1.0,
        context_loss_weight: float = 1.0,
        beta_vae_kl_weight: float = 1.0,
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
        self.velocity_loss_weight = velocity_loss_weight
        self.context_loss_weight = context_loss_weight
        self.beta_vae_kl_weight = beta_vae_kl_weight
        self.state_dependent_std = state_dependent_std
        self.noise_std_type = noise_std_type

        num_actor_obs = self._get_obs_dim(obs, self.obs_groups["policy"])
        num_history_obs = self._get_obs_dim(obs, self.obs_groups.get(self.history_obs_set, [self.history_obs_set]))
        num_critic_obs = self._get_obs_dim(obs, self.obs_groups["critic"])
        num_velocity_targets = obs[self.velocity_target_group].shape[-1]
        num_context_targets = obs[self.context_target_group].shape[-1]

        self.estimator_encoder = MLP(
            input_dim=num_history_obs,
            output_dim=estimator_hidden_dims[-1],
            hidden_dims=estimator_hidden_dims[:-1] if len(estimator_hidden_dims) > 1 else [estimator_hidden_dims[0]],
            activation=activation,
        )
        self.velocity_head = nn.Linear(estimator_hidden_dims[-1], num_velocity_targets)
        self.context_mean_head = nn.Linear(estimator_hidden_dims[-1], context_latent_dim)
        self.context_logvar_head = nn.Linear(estimator_hidden_dims[-1], context_latent_dim)

        if len(context_decoder_hidden_dims) > 1:
            self.context_decoder_trunk = MLP(
                input_dim=context_latent_dim,
                output_dim=context_decoder_hidden_dims[-1],
                hidden_dims=context_decoder_hidden_dims[:-1],
                activation=activation,
            )
        else:
            self.context_decoder_trunk = MLP(
                input_dim=context_latent_dim,
                output_dim=context_decoder_hidden_dims[0],
                hidden_dims=[context_decoder_hidden_dims[0]],
                activation=activation,
            )
        self.obs_decoder_head = nn.Linear(context_decoder_hidden_dims[-1], num_context_targets)

        actor_input_dim = num_actor_obs + num_velocity_targets + context_latent_dim
        if self.state_dependent_std:
            self.actor = MLP(actor_input_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)

        print(f"DreamWaQ encoder: {self.estimator_encoder}")
        print(f"DreamWaQ decoder trunk: {self.context_decoder_trunk}")
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
        self.cached_velocity_estimate = None
        self.cached_context_sample = None
        self.cached_context_mean = None
        self.cached_context_logvar = None
        self.cached_obs_reconstruction = None

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

    def compute_auxiliary_losses(
        self,
        obs: TensorDict,
        next_obs: TensorDict | None = None,
        dones: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.cached_velocity_estimate is None:
            raise RuntimeError("DreamWaQ auxiliary losses requested before actor features were computed.")

        velocity_target = obs[self.velocity_target_group]
        context_target_obs = next_obs if next_obs is not None else obs
        context_target = context_target_obs[self.context_target_group]

        velocity_loss = nn.functional.mse_loss(self.cached_velocity_estimate, velocity_target)
        context_kl_loss = self._kl_loss(self.cached_context_mean, self.cached_context_logvar)

        if next_obs is not None and dones is not None:
            not_done = (1.0 - dones.float()).squeeze(-1)
            per_sample_recon = (self.cached_obs_reconstruction - context_target).pow(2).mean(dim=-1)
            n_valid = not_done.sum().clamp(min=1.0)
            obs_reconstruction_loss = (per_sample_recon * not_done).sum() / n_valid
        else:
            obs_reconstruction_loss = nn.functional.mse_loss(self.cached_obs_reconstruction, context_target)

        total_loss = self.velocity_loss_weight * velocity_loss + self.context_loss_weight * (
            obs_reconstruction_loss + self.beta_vae_kl_weight * context_kl_loss
        )

        return {
            "dreamwaq_total": total_loss,
            "dreamwaq_velocity": velocity_loss.detach(),
            "dreamwaq_obs_reconstruction": obs_reconstruction_loss.detach(),
            "dreamwaq_context_kl": context_kl_loss.detach(),
        }

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        state_dict = dict(state_dict)
        old_prefix = "context_decoder."
        if any(key.startswith(old_prefix) for key in state_dict):
            print("Migrating legacy context_decoder keys to context_decoder_trunk + decoder heads")
            migrated = {}
            for key, value in list(state_dict.items()):
                if key.startswith(old_prefix):
                    migrated[key.replace(old_prefix, "context_decoder_trunk.", 1)] = value
                    del state_dict[key]
            state_dict.update(migrated)
            strict = False
        if any(key.startswith("vel_decoder_head.") for key in state_dict):
            print("Dropping legacy vel_decoder_head keys")
            state_dict = {key: value for key, value in state_dict.items() if not key.startswith("vel_decoder_head.")}
            strict = False
        super().load_state_dict(state_dict, strict=strict)
        return True

    def _compute_actor_input(self, obs: TensorDict, sample_latent: bool) -> torch.Tensor:
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        history_obs = self.actor_history_normalizer(self.get_history_obs(obs))
        estimator_features = self.estimator_encoder(history_obs)

        velocity_estimate = self.velocity_head(estimator_features)
        context_mean = self.context_mean_head(estimator_features)
        context_logvar = self.context_logvar_head(estimator_features)
        context_sample = self._sample_context(context_mean, context_logvar, sample_latent)
        decoder_features = self.context_decoder_trunk(context_sample)
        obs_reconstruction = self.obs_decoder_head(decoder_features)

        self.cached_velocity_estimate = velocity_estimate
        self.cached_context_mean = context_mean
        self.cached_context_logvar = context_logvar
        self.cached_context_sample = context_sample
        self.cached_obs_reconstruction = obs_reconstruction

        return torch.cat((actor_obs, velocity_estimate, context_sample), dim=-1)

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

    @staticmethod
    def _sample_context(mean: torch.Tensor, logvar: torch.Tensor, sample_latent: bool) -> torch.Tensor:
        if sample_latent:
            std = torch.exp(0.5 * logvar)
            return mean + torch.randn_like(std) * std
        return mean

    @staticmethod
    def _kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1))
