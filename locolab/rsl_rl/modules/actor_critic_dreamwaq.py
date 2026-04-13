"""DreamWaQ actor-critic with Residual Value-Shaped Context Learning (RVCL).

Phase 1a implementation:
- Stop-gradient from actor → context encoder (actor consumes ``z.detach()``).
- Residual value shaping: ``V_prop + ΔV ≈ V_full``; the encoder's only critic-side gradient
  path is through ``ΔV``, so ``z`` is forced to encode the terrain-conditional value residual.
- EMA target encoder drives the latent transition consistency loss.
- Event-aware auxiliary supervision: encoder predicts next-step terrain height statistics and
  foot-contact binary pattern from the history latent.
- Velocity supervision retained from DreamWaQ.
- β-VAE proprio reconstruction removed; KL with free bits acts as a light regularizer only.
"""

from __future__ import annotations

import copy
from typing import Any, NoReturn

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.networks import MLP, EmpiricalNormalization
from tensordict import TensorDict
from torch.distributions import Normal


class ActorCriticDreamWaQ(nn.Module):
    """DreamWaQ-style actor-critic with history encoder, velocity estimator, and RVCL context latent."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        critic_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        estimator_hidden_dims: tuple[int] | list[int] = (128, 64),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        history_obs_set: str = "policy_history",
        velocity_target_group: str = "dreamwaq_velocity",
        context_latent_dim: int = 16,
        velocity_loss_weight: float = 1.0,
        # RVCL: observation group names (direct obs keys)
        proprio_critic_group: str = "proprio_critic",
        event_height_target_group: str = "event_height_target",
        event_contact_target_group: str = "event_contact_target",
        # RVCL: residual value shaping
        v_prop_hidden_dims: tuple[int] | list[int] = (256, 128),
        delta_v_hidden_dims: tuple[int] | list[int] = (128, 64),
        v_prop_loss_weight: float = 0.5,
        residual_loss_weight: float = 1.0,
        # RVCL: EMA latent dynamics
        ema_tau: float = 0.995,
        transition_hidden_dims: tuple[int] | list[int] = (128, 64),
        transition_loss_weight: float = 1.0,
        # RVCL: event-aware auxiliary supervision
        event_decoder_hidden_dims: tuple[int] | list[int] = (128, 64),
        event_height_loss_weight: float = 1.0,
        event_contact_loss_weight: float = 0.5,
        # RVCL: KL regularization
        beta_vae_kl_weight: float = 0.01,
        free_bits: float = 0.05,
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
        self.proprio_critic_group = proprio_critic_group
        self.event_height_target_group = event_height_target_group
        self.event_contact_target_group = event_contact_target_group

        self.context_latent_dim = context_latent_dim
        self.velocity_loss_weight = velocity_loss_weight
        self.v_prop_loss_weight = v_prop_loss_weight
        self.residual_loss_weight = residual_loss_weight
        self.transition_loss_weight = transition_loss_weight
        self.event_height_loss_weight = event_height_loss_weight
        self.event_contact_loss_weight = event_contact_loss_weight
        self.beta_vae_kl_weight = beta_vae_kl_weight
        self.free_bits = free_bits
        self.ema_tau = ema_tau
        self.state_dependent_std = state_dependent_std
        self.noise_std_type = noise_std_type

        # ----- observation dims -----
        num_actor_obs = self._get_obs_dim(obs, self.obs_groups["policy"])
        num_history_obs = self._get_obs_dim(obs, self.obs_groups.get(self.history_obs_set, [self.history_obs_set]))
        num_critic_obs = self._get_obs_dim(obs, self.obs_groups["critic"])
        num_velocity_targets = obs[self.velocity_target_group].shape[-1]
        num_proprio_critic = obs[self.proprio_critic_group].shape[-1]
        num_event_height = obs[self.event_height_target_group].shape[-1]
        num_event_contact = obs[self.event_contact_target_group].shape[-1]

        self.num_actions = num_actions
        self.num_event_height = num_event_height
        self.num_event_contact = num_event_contact

        # ----- velocity encoder (online, separate trunk) -----
        self.velocity_encoder = MLP(
            input_dim=num_history_obs,
            output_dim=estimator_hidden_dims[-1],
            hidden_dims=estimator_hidden_dims[:-1] if len(estimator_hidden_dims) > 1 else [estimator_hidden_dims[0]],
            activation=activation,
        )
        self.velocity_head = nn.Linear(estimator_hidden_dims[-1], num_velocity_targets)

        # ----- context encoder (online, separate trunk — gradient-isolated from velocity) -----
        self.context_encoder = MLP(
            input_dim=num_history_obs,
            output_dim=estimator_hidden_dims[-1],
            hidden_dims=estimator_hidden_dims[:-1] if len(estimator_hidden_dims) > 1 else [estimator_hidden_dims[0]],
            activation=activation,
        )
        self.context_mean_head = nn.Linear(estimator_hidden_dims[-1], context_latent_dim)
        self.context_logvar_head = nn.Linear(estimator_hidden_dims[-1], context_latent_dim)

        # ----- EMA target encoder (deep-copied after construction below) -----
        # Initialised to a copy of the context encoder + mean head. No gradients flow through it.
        self._target_context_encoder: MLP | None = None
        self._target_context_mean_head: nn.Linear | None = None

        # ----- latent transition model: (z_mean ⊕ a) → z_mean_next -----
        self.transition_model = MLP(
            input_dim=context_latent_dim + num_actions,
            output_dim=context_latent_dim,
            hidden_dims=list(transition_hidden_dims),
            activation=activation,
        )

        # ----- event-aware decoder: z_mean → shared trunk → (height, contact) heads -----
        event_trunk_hidden_dims = list(event_decoder_hidden_dims)
        if len(event_trunk_hidden_dims) == 0:
            raise ValueError("event_decoder_hidden_dims must be non-empty")
        self.event_decoder_trunk = MLP(
            input_dim=context_latent_dim,
            output_dim=event_trunk_hidden_dims[-1],
            hidden_dims=(
                event_trunk_hidden_dims[:-1] if len(event_trunk_hidden_dims) > 1 else [event_trunk_hidden_dims[0]]
            ),
            activation=activation,
        )
        event_trunk_out_dim = event_trunk_hidden_dims[-1]
        self.event_head_height = nn.Linear(event_trunk_out_dim, num_event_height)
        self.event_head_contact = nn.Linear(event_trunk_out_dim, num_event_contact)

        # ----- residual value heads -----
        # V_prop: proprio-critic obs → scalar value (no z, no exteroception)
        self.v_prop_head = MLP(
            input_dim=num_proprio_critic,
            output_dim=1,
            hidden_dims=list(v_prop_hidden_dims),
            activation=activation,
        )
        # ΔV: (proprio_critic ⊕ z) → scalar value residual. Encoder only gets critic-side gradient via this.
        self.delta_v_head = MLP(
            input_dim=num_proprio_critic + context_latent_dim,
            output_dim=1,
            hidden_dims=list(delta_v_hidden_dims),
            activation=activation,
        )

        # ----- actor / main critic (V_full) -----
        actor_input_dim = num_actor_obs + num_velocity_targets + context_latent_dim
        if self.state_dependent_std:
            self.actor = MLP(actor_input_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)

        print(f"DreamWaQ/RVCL velocity encoder: {self.velocity_encoder}")
        print(f"DreamWaQ/RVCL context encoder: {self.context_encoder}")
        print(f"DreamWaQ/RVCL transition: {self.transition_model}")
        print(f"DreamWaQ/RVCL event trunk: {self.event_decoder_trunk}")
        print(f"DreamWaQ/RVCL V_prop: {self.v_prop_head}")
        print(f"DreamWaQ/RVCL ΔV: {self.delta_v_head}")
        print(f"DreamWaQ/RVCL actor: {self.actor}")
        print(f"DreamWaQ/RVCL V_full critic: {self.critic}")

        # ----- normalisers -----
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
            self.proprio_critic_normalizer = EmpiricalNormalization(num_proprio_critic)
        else:
            self.critic_obs_normalizer = nn.Identity()
            self.proprio_critic_normalizer = nn.Identity()

        # ----- policy std -----
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

        # ----- cached encoder features (populated by _encode_context per forward pass) -----
        self.distribution = None
        self.cached_velocity_estimate: torch.Tensor | None = None
        self.cached_context_mean: torch.Tensor | None = None
        self.cached_context_logvar: torch.Tensor | None = None
        self.cached_context_sample: torch.Tensor | None = None

        Normal.set_default_validate_args(False)

        # Instantiate EMA target encoder as a frozen deep copy of the online encoder.
        self._initialize_target_encoder()

    def _initialize_target_encoder(self) -> None:
        self._target_context_encoder = copy.deepcopy(self.context_encoder)
        self._target_context_mean_head = copy.deepcopy(self.context_mean_head)
        for param in self._target_context_encoder.parameters():
            param.requires_grad_(False)
        for param in self._target_context_mean_head.parameters():
            param.requires_grad_(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def forward(self) -> NoReturn:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Distribution helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Obs helpers
    # ------------------------------------------------------------------

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[group] for group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_history_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[group] for group in self.obs_groups.get(self.history_obs_set, [self.history_obs_set])]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[group] for group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
            self.actor_history_normalizer.update(self.get_history_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))
            self.proprio_critic_normalizer.update(obs[self.proprio_critic_group])

    # ------------------------------------------------------------------
    # Encoder forward (shared by act / aux-loss)
    # ------------------------------------------------------------------

    def _encode_context(self, obs: TensorDict, sample_latent: bool) -> None:
        """Run both encoder trunks and populate cached latent features."""
        history_obs = self.actor_history_normalizer(self.get_history_obs(obs))
        # Velocity encoder — gradient flows only through velocity_loss and actor.
        vel_features = self.velocity_encoder(history_obs)
        self.cached_velocity_estimate = self.velocity_head(vel_features)
        # Context encoder — gradient-isolated from velocity; driven by ΔV, transition, event losses.
        ctx_features = self.context_encoder(history_obs)
        self.cached_context_mean = self.context_mean_head(ctx_features)
        self.cached_context_logvar = self.context_logvar_head(ctx_features)
        self.cached_context_sample = self._sample_context(
            self.cached_context_mean, self.cached_context_logvar, sample_latent
        )

    def _compute_actor_input(self, obs: TensorDict, sample_latent: bool) -> torch.Tensor:
        self._encode_context(obs, sample_latent=sample_latent)
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        # Stop-gradient from actor into encoder: actor sees ``z`` as a black-box feature.
        # Velocity estimate is NOT detached — keeps the velocity supervision signal tied to the actor.
        return torch.cat(
            (actor_obs, self.cached_velocity_estimate, self.cached_context_sample.detach()),
            dim=-1,
        )

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

    # ------------------------------------------------------------------
    # Target encoder (EMA)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        """Polyak update of the EMA target encoder. Call at the end of each PPO update()."""
        tau = self.ema_tau
        for p_online, p_target in zip(
            self.context_encoder.parameters(), self._target_context_encoder.parameters(), strict=True
        ):
            p_target.data.mul_(tau).add_(p_online.data, alpha=1.0 - tau)
        for p_online, p_target in zip(
            self.context_mean_head.parameters(), self._target_context_mean_head.parameters(), strict=True
        ):
            p_target.data.mul_(tau).add_(p_online.data, alpha=1.0 - tau)

    @torch.no_grad()
    def _encode_next_z_target(self, next_obs: TensorDict) -> torch.Tensor:
        """Encode next-step history with the EMA target context encoder → mean latent (stop-grad)."""
        next_history = self.actor_history_normalizer(self.get_history_obs(next_obs))
        next_features = self._target_context_encoder(next_history)
        return self._target_context_mean_head(next_features)

    # ------------------------------------------------------------------
    # Auxiliary losses
    # ------------------------------------------------------------------

    def compute_auxiliary_losses(
        self,
        obs: TensorDict,
        next_obs: TensorDict | None = None,
        dones: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        returns: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.cached_context_mean is None:
            raise RuntimeError("DreamWaQ auxiliary losses requested before encoder features were computed.")
        if next_obs is None:
            raise ValueError("RVCL auxiliary losses require `next_obs`.")
        if actions is None:
            raise ValueError("RVCL auxiliary losses require `actions` for the transition model.")

        # ---- terminal mask (strict; all representation-shaping losses use this) ----
        if dones is not None:
            not_done = (1.0 - dones.float()).squeeze(-1)
            valid = not_done.sum().clamp(min=1.0)
        else:
            not_done = None
            valid = None

        # ---- 1. velocity supervision (DreamWaQ baseline signal) ----
        velocity_target = obs[self.velocity_target_group]
        velocity_loss = F.mse_loss(self.cached_velocity_estimate, velocity_target)

        # ---- 2. latent transition consistency (EMA target, deterministic mean latent) ----
        target_next_z = self._encode_next_z_target(next_obs)  # no-grad, mean of target encoder
        trans_input = torch.cat([self.cached_context_mean, actions], dim=-1)
        pred_next_z = self.transition_model(trans_input)
        if not_done is not None:
            trans_per_sample = (pred_next_z - target_next_z).pow(2).mean(dim=-1)
            transition_loss = (trans_per_sample * not_done).sum() / valid
        else:
            transition_loss = F.mse_loss(pred_next_z, target_next_z)

        # ---- 3. event-aware auxiliary supervision ----
        event_features = self.event_decoder_trunk(self.cached_context_mean)
        pred_event_height = self.event_head_height(event_features)
        pred_event_contact_logits = self.event_head_contact(event_features)

        target_event_height = next_obs[self.event_height_target_group]
        target_event_contact = next_obs[self.event_contact_target_group]

        if not_done is not None:
            h_per_sample = (pred_event_height - target_event_height).pow(2).mean(dim=-1)
            event_height_loss = (h_per_sample * not_done).sum() / valid
            c_per_sample = F.binary_cross_entropy_with_logits(
                pred_event_contact_logits, target_event_contact, reduction="none"
            ).mean(dim=-1)
            event_contact_loss = (c_per_sample * not_done).sum() / valid
        else:
            event_height_loss = F.mse_loss(pred_event_height, target_event_height)
            event_contact_loss = F.binary_cross_entropy_with_logits(pred_event_contact_logits, target_event_contact)

        # ---- 4. residual value shaping ----
        proprio_critic = self.proprio_critic_normalizer(obs[self.proprio_critic_group])

        # V_full: main critic output, used only as a teacher (no-grad).
        with torch.no_grad():
            V_full_target = self.critic(self.critic_obs_normalizer(self.get_critic_obs(obs))).detach()

        # V_prop: proprio-only value. Updated via its own MSE against V_full. No gradient to encoder.
        V_prop_pred = self.v_prop_head(proprio_critic)
        if not_done is not None:
            v_prop_per_sample = (V_prop_pred - V_full_target).pow(2).mean(dim=-1)
            v_prop_loss = (v_prop_per_sample * not_done).sum() / valid
        else:
            v_prop_loss = F.mse_loss(V_prop_pred, V_full_target)
        V_prop_sg = V_prop_pred.detach()

        # ΔV: encoder's only critic-side gradient path. Use deterministic mean latent for representation shaping;
        # proprio_critic is detached so the delta_v_head / encoder cannot piggy-back on its gradient.
        delta_v_input = torch.cat([proprio_critic.detach(), self.cached_context_mean], dim=-1)
        delta_v_pred = self.delta_v_head(delta_v_input)
        if not_done is not None:
            residual_per_sample = (V_prop_sg + delta_v_pred - V_full_target).pow(2).mean(dim=-1)
            residual_loss = (residual_per_sample * not_done).sum() / valid
        else:
            residual_loss = F.mse_loss(V_prop_sg + delta_v_pred, V_full_target)

        # ---- 5. KL regularisation with free bits ----
        kl_per_dim = 0.5 * (
            self.cached_context_mean.pow(2) + self.cached_context_logvar.exp() - 1.0 - self.cached_context_logvar
        )  # (B, D)
        # Free bits: don't penalise dims whose batch-mean KL is below the free_bits budget.
        kl_per_dim_mean = kl_per_dim.mean(dim=0)  # (D,)
        kl_excess = (kl_per_dim_mean - self.free_bits).clamp(min=0.0)
        kl_loss = kl_excess.sum()
        kl_free_bits_active = (kl_per_dim_mean < self.free_bits).float().sum()

        # For logging only: raw KL (per sample sum over dims, mean over batch).
        kl_raw = kl_per_dim.sum(dim=-1).mean()

        # ---- total ----
        total_loss = (
            self.velocity_loss_weight * velocity_loss
            + self.transition_loss_weight * transition_loss
            + self.event_height_loss_weight * event_height_loss
            + self.event_contact_loss_weight * event_contact_loss
            + self.v_prop_loss_weight * v_prop_loss
            + self.residual_loss_weight * residual_loss
            + self.beta_vae_kl_weight * kl_loss
        )

        # For logging: residual diagnostics (shows how much ΔV fills the V_full-V_prop gap).
        with torch.no_grad():
            residual_gap = (V_full_target - V_prop_sg).abs().mean()
            residual_gap_after = (V_full_target - (V_prop_sg + delta_v_pred.detach())).abs().mean()

        return {
            "auxiliary_total": total_loss,
            "dreamwaq_velocity": velocity_loss.detach(),
            "dreamwaq_transition": transition_loss.detach(),
            "dreamwaq_event_height": event_height_loss.detach(),
            "dreamwaq_event_contact": event_contact_loss.detach(),
            "dreamwaq_v_prop": v_prop_loss.detach(),
            "dreamwaq_residual": residual_loss.detach(),
            "dreamwaq_residual_gap": residual_gap,
            "dreamwaq_residual_gap_after": residual_gap_after,
            "dreamwaq_context_kl": kl_raw.detach(),
            "dreamwaq_kl_free_bits_active": kl_free_bits_active.detach(),
        }

    # ------------------------------------------------------------------
    # State dict compatibility
    # ------------------------------------------------------------------

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        state_dict = dict(state_dict)
        # Drop legacy β-VAE decoder / vel decoder keys from older DreamWaQ checkpoints.
        legacy_prefixes = (
            "context_decoder.",
            "context_decoder_trunk.",
            "obs_decoder_head.",
            "vel_decoder_head.",
        )
        dropped = [k for k in state_dict if k.startswith(legacy_prefixes)]
        if dropped:
            print(f"Dropping {len(dropped)} legacy DreamWaQ keys from checkpoint: {dropped[:4]}...")
            for k in dropped:
                del state_dict[k]
            strict = False

        # Migrate shared-encoder checkpoint (estimator_encoder.*) → split-encoder layout.
        # Copy old shared weights into both velocity_encoder and context_encoder.
        shared_keys = [k for k in state_dict if k.startswith("estimator_encoder.")]
        if shared_keys:
            print(f"Migrating {len(shared_keys)} shared estimator_encoder keys → velocity_encoder + context_encoder")
            for k in shared_keys:
                suffix = k[len("estimator_encoder.") :]
                state_dict[f"velocity_encoder.{suffix}"] = state_dict[k].clone()
                state_dict[f"context_encoder.{suffix}"] = state_dict[k].clone()
                del state_dict[k]
            strict = False

        # Migrate old EMA target key prefix.
        old_target_keys = [k for k in state_dict if k.startswith("_target_estimator_encoder.")]
        if old_target_keys:
            for k in old_target_keys:
                suffix = k[len("_target_estimator_encoder.") :]
                state_dict[f"_target_context_encoder.{suffix}"] = state_dict[k]
                del state_dict[k]
            strict = False

        # RVCL modules may be absent from older checkpoints → relax strict loading.
        rvcl_prefixes = (
            "transition_model.",
            "event_decoder_trunk.",
            "event_head_height.",
            "event_head_contact.",
            "v_prop_head.",
            "delta_v_head.",
            "proprio_critic_normalizer.",
            "_target_context_encoder.",
            "_target_context_mean_head.",
        )
        if not any(k.startswith(rvcl_prefixes) for k in state_dict):
            print("Checkpoint lacks RVCL modules; loading with strict=False")
            strict = False

        super().load_state_dict(state_dict, strict=strict)
        # Re-sync the EMA target encoder to the newly-loaded online encoder if not in checkpoint.
        has_target = any(k.startswith("_target_context_encoder.") for k in state_dict)
        if not has_target:
            self._initialize_target_encoder()
        return True

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

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
        else:
            return mean
