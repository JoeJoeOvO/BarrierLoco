"""Dual-scale recurrent SLR-style actor-critic for blind locomotion.

This policy keeps the dual-timescale recurrent structure but changes the learning
signal on the slow latent from future-response regression to transition-metric
representation learning:

- Fast branch: per-step proprio dynamics and velocity estimation.
- Slow branch: gated lower-rate context latent that should capture hidden contact
  / terrain dynamics.
- Transition model: predicts future slow latent from ``(z_t, a_t)``.
- Triplet loss: prevents the slow latent from collapsing to a trivial code.
- Stop-gradient from actor to ``z``: the policy reads the latent as a feature,
  while critic/value learning and auxiliary losses shape the representation.
"""

from __future__ import annotations

from typing import Any, NoReturn

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict
from torch.distributions import Normal


class ActorCriticDSR(nn.Module):
    """Dual-scale recurrent policy with SLR-style transition and triplet losses."""

    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        critic_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        velocity_target_group: str = "dsr_velocity",
        response_target_group: str = "dsr_response_target",
        frontend_hidden_dims: tuple[int] | list[int] = (128, 64),
        fast_gru_hidden_dim: int = 64,
        slow_gru_hidden_dim: int = 64,
        dynamics_token_dim: int = 16,
        context_latent_dim: int = 16,
        context_head_hidden_dim: int = 32,
        gate_hidden_dim: int = 32,
        slow_update_interval: int = 2,
        transition_hidden_dims: tuple[int] | list[int] = (256, 128),
        transition_horizon: int = 2,
        velocity_loss_weight: float = 1.0,
        transition_loss_weight: float = 1.0,
        triplet_loss_weight: float = 1.0,
        triplet_margin: float = 1.0,
        triplet_num_negatives: int = 64,
        triplet_chunk_size: int = 2048,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticDSR.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.velocity_target_group = velocity_target_group
        self.response_target_group = response_target_group  # kept for env/config compatibility
        self.state_dependent_std = state_dependent_std
        self.noise_std_type = noise_std_type

        self.fast_gru_hidden_dim = fast_gru_hidden_dim
        self.slow_gru_hidden_dim = slow_gru_hidden_dim
        self.dynamics_token_dim = dynamics_token_dim
        self.context_latent_dim = context_latent_dim
        self.slow_update_interval = slow_update_interval
        self.transition_horizon = transition_horizon

        self.velocity_loss_weight = velocity_loss_weight
        self.transition_loss_weight = transition_loss_weight
        self.triplet_loss_weight = triplet_loss_weight
        self.triplet_margin = triplet_margin
        self.triplet_num_negatives = triplet_num_negatives
        self.triplet_chunk_size = triplet_chunk_size

        if self.slow_update_interval < 1:
            raise ValueError("slow_update_interval must be >= 1")
        if self.transition_horizon < 1:
            raise ValueError("transition_horizon must be >= 1")
        if self.triplet_num_negatives < 1:
            raise ValueError("triplet_num_negatives must be >= 1")
        if self.triplet_chunk_size < 1:
            raise ValueError("triplet_chunk_size must be >= 1")

        self._initial_batch_size = obs.batch_size[0] if len(obs.batch_size) > 0 else next(iter(obs.values())).shape[0]

        num_actor_obs = self._get_obs_dim(obs, self.obs_groups["policy"])
        num_critic_obs = self._get_obs_dim(obs, self.obs_groups["critic"])
        num_velocity_targets = obs[self.velocity_target_group].shape[-1]
        self.num_actions = num_actions

        frontend_dims = list(frontend_hidden_dims)
        self.frontend = MLP(
            input_dim=num_actor_obs,
            output_dim=frontend_dims[-1],
            hidden_dims=frontend_dims[:-1] if len(frontend_dims) > 1 else [frontend_dims[0]],
            activation=activation,
        )
        frontend_out_dim = frontend_dims[-1]

        self.gru_fast = nn.GRUCell(input_size=frontend_out_dim, hidden_size=fast_gru_hidden_dim)
        self.velocity_head = MLP(
            input_dim=fast_gru_hidden_dim,
            output_dim=num_velocity_targets,
            hidden_dims=[32],
            activation=activation,
        )
        self.dynamics_head = MLP(
            input_dim=fast_gru_hidden_dim,
            output_dim=dynamics_token_dim,
            hidden_dims=[32],
            activation=activation,
        )

        slow_input_dim = frontend_out_dim + num_velocity_targets + dynamics_token_dim
        self.gru_slow = nn.GRUCell(input_size=slow_input_dim, hidden_size=slow_gru_hidden_dim)
        self.gate_head = MLP(
            input_dim=dynamics_token_dim + slow_gru_hidden_dim,
            output_dim=1,
            hidden_dims=[gate_hidden_dim],
            activation=activation,
        )
        self.context_head = MLP(
            input_dim=slow_gru_hidden_dim,
            output_dim=context_latent_dim,
            hidden_dims=[context_head_hidden_dim],
            activation=activation,
        )
        self.transition_model = MLP(
            input_dim=context_latent_dim + num_actions,
            output_dim=context_latent_dim,
            hidden_dims=list(transition_hidden_dims),
            activation=activation,
        )

        actor_input_dim = num_actor_obs + num_velocity_targets + context_latent_dim
        critic_input_dim = num_critic_obs + context_latent_dim
        if self.state_dependent_std:
            self.actor = MLP(actor_input_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(critic_input_dim, 1, critic_hidden_dims, activation)

        print(f"DSR-SLR frontend: {self.frontend}")
        print(f"DSR-SLR GRU_fast: hidden={fast_gru_hidden_dim}")
        print(f"DSR-SLR GRU_slow: hidden={slow_gru_hidden_dim}, input={slow_input_dim}")
        print(f"DSR-SLR transition_model: {self.transition_model}")
        print(f"DSR-SLR actor: {self.actor}")
        print(f"DSR-SLR critic: {self.critic}")

        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()
        )

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

        self.hidden_fast: torch.Tensor | None = None
        self.hidden_slow: torch.Tensor | None = None
        self.slow_step_counter: torch.Tensor | None = None

        self.distribution = None
        self.cached_velocity_estimate: torch.Tensor | None = None
        self.cached_dynamics_token: torch.Tensor | None = None
        self.cached_context_latent: torch.Tensor | None = None
        self.cached_gate: torch.Tensor | None = None
        self.cached_slow_update_mask: torch.Tensor | None = None
        self._cached_masks: torch.Tensor | None = None
        self._context_cache_fresh = False

        Normal.set_default_validate_args(False)

    # ------------------------------------------------------------------
    # Hidden state management
    # ------------------------------------------------------------------

    def _init_hidden(self, batch_size: int, device: torch.device) -> None:
        if (
            self.hidden_fast is None
            or self.hidden_slow is None
            or self.slow_step_counter is None
            or self.hidden_fast.shape[0] != batch_size
            or self.hidden_slow.shape[0] != batch_size
            or self.slow_step_counter.shape[0] != batch_size
        ):
            self.hidden_fast = torch.zeros(batch_size, self.fast_gru_hidden_dim, device=device)
            self.hidden_slow = torch.zeros(batch_size, self.slow_gru_hidden_dim, device=device)
            self.slow_step_counter = torch.zeros(batch_size, dtype=torch.long, device=device)

    def get_hidden_states(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.hidden_fast is None or self.hidden_slow is None:
            self._init_hidden(self._initial_batch_size, next(self.parameters()).device)
        counter = self.slow_step_counter.to(self.hidden_fast.dtype).unsqueeze(-1)
        combined = torch.cat([self.hidden_fast, self.hidden_slow, counter], dim=-1)
        return (combined.unsqueeze(0), combined.unsqueeze(0))

    def _set_hidden_from_stored(self, hidden_state: torch.Tensor | None) -> None:
        if hidden_state is None:
            return
        combined = hidden_state.squeeze(0)
        split_idx = self.fast_gru_hidden_dim + self.slow_gru_hidden_dim
        self.hidden_fast = combined[:, : self.fast_gru_hidden_dim].contiguous()
        self.hidden_slow = combined[:, self.fast_gru_hidden_dim : split_idx].contiguous()
        if combined.shape[-1] > split_idx:
            self.slow_step_counter = combined[:, split_idx].round().to(dtype=torch.long, device=combined.device)
        else:
            self.slow_step_counter = torch.zeros(combined.shape[0], dtype=torch.long, device=combined.device)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        if dones is None or self.hidden_fast is None:
            return
        done_mask = dones.bool().squeeze(-1)
        if done_mask.any():
            self.hidden_fast[done_mask] = 0.0
            self.hidden_slow[done_mask] = 0.0
            self.slow_step_counter[done_mask] = 0
        self._context_cache_fresh = False

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

    # ------------------------------------------------------------------
    # Obs helpers
    # ------------------------------------------------------------------

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[group] for group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[group] for group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    # ------------------------------------------------------------------
    # Encoder and heads
    # ------------------------------------------------------------------

    def _step_encoder(
        self, actor_obs: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e_t = self.frontend(actor_obs)

        hidden_fast = self.gru_fast(e_t, self.hidden_fast)
        v_hat = self.velocity_head(hidden_fast)
        d_t = self.dynamics_head(hidden_fast)

        slow_input = torch.cat([e_t, v_hat.detach(), d_t.detach()], dim=-1)
        h_bar = self.gru_slow(slow_input, self.hidden_slow)

        gate_input = torch.cat([d_t.detach(), self.hidden_slow], dim=-1)
        g_t = torch.sigmoid(self.gate_head(gate_input))
        m_t = ((self.slow_step_counter % self.slow_update_interval) == 0).float().unsqueeze(-1)
        update_weight = m_t * g_t
        hidden_slow = (1.0 - update_weight) * self.hidden_slow + update_weight * h_bar
        slow_step_counter = self.slow_step_counter + 1

        if valid_mask is not None:
            valid_mask = valid_mask.to(device=actor_obs.device)
            hidden_mask = valid_mask.to(dtype=torch.bool).unsqueeze(-1)
            counter_mask = valid_mask.to(dtype=torch.bool)
            hidden_fast = torch.where(hidden_mask, hidden_fast, self.hidden_fast)
            hidden_slow = torch.where(hidden_mask, hidden_slow, self.hidden_slow)
            slow_step_counter = torch.where(counter_mask, slow_step_counter, self.slow_step_counter)

        self.hidden_fast = hidden_fast
        self.hidden_slow = hidden_slow
        self.slow_step_counter = slow_step_counter

        z_t = self.context_head(self.hidden_slow)
        return v_hat, d_t, z_t, g_t, m_t

    def _actor_forward(self, actor_obs: torch.Tensor, v_hat: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        actor_input = torch.cat([actor_obs, v_hat, z_t.detach()], dim=-1)
        return self.actor(actor_input)

    def _critic_forward(self, critic_obs: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        critic_input = torch.cat([critic_obs, z_t], dim=-1)
        return self.critic(critic_input)

    def _update_distribution(self, actor_output: torch.Tensor) -> None:
        if self.state_dependent_std:
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(actor_output, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(actor_output, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")
        else:
            mean = actor_output
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")
        self.distribution = Normal(mean, std)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        hidden_state = kwargs.get("hidden_state")
        masks = kwargs.get("masks")

        if hidden_state is not None:
            saved_fast = self.hidden_fast
            saved_slow = self.hidden_slow
            saved_counter = self.slow_step_counter
            self._set_hidden_from_stored(hidden_state)
            actor_output = self._forward_sequential(obs, masks)
            self.hidden_fast = saved_fast
            self.hidden_slow = saved_slow
            self.slow_step_counter = saved_counter
        else:
            actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
            self._init_hidden(actor_obs.shape[0], actor_obs.device)
            v_hat, d_t, z_t, gate, m_t = self._step_encoder(actor_obs)
            actor_output = self._actor_forward(actor_obs, v_hat, z_t)
            self.cached_velocity_estimate = v_hat
            self.cached_dynamics_token = d_t
            self.cached_context_latent = z_t
            self.cached_gate = gate
            self.cached_slow_update_mask = m_t
            self._cached_masks = None

        self._context_cache_fresh = True
        self._update_distribution(actor_output)
        return self.distribution.sample()

    def _forward_sequential(self, obs: TensorDict, masks: torch.Tensor | None) -> torch.Tensor:
        masks_bool = masks.to(dtype=torch.bool) if masks is not None else None
        actor_obs_seq = self.actor_obs_normalizer(self.get_actor_obs(obs))
        seq_len = actor_obs_seq.shape[0]

        all_outputs = []
        all_v_hat = []
        all_d_t = []
        all_z_t = []
        all_gate = []
        all_slow_masks = []

        for t in range(seq_len):
            obs_t = actor_obs_seq[t]
            mask_t = masks_bool[t] if masks_bool is not None else None
            v_hat, d_t, z_t, gate, slow_mask = self._step_encoder(obs_t, valid_mask=mask_t)
            actor_output_t = self._actor_forward(obs_t, v_hat, z_t)
            all_outputs.append(actor_output_t)
            all_v_hat.append(v_hat)
            all_d_t.append(d_t)
            all_z_t.append(z_t)
            all_gate.append(gate)
            all_slow_masks.append(slow_mask)

        self._cached_masks = masks_bool

        def _unpad_if_needed(sequence: list[torch.Tensor]) -> torch.Tensor:
            stacked = torch.stack(sequence, dim=0)
            if masks_bool is None:
                return stacked
            return unpad_trajectories(stacked, masks_bool)

        self.cached_velocity_estimate = _unpad_if_needed(all_v_hat)
        self.cached_dynamics_token = _unpad_if_needed(all_d_t)
        self.cached_context_latent = _unpad_if_needed(all_z_t)
        self.cached_gate = _unpad_if_needed(all_gate)
        self.cached_slow_update_mask = _unpad_if_needed(all_slow_masks)

        return _unpad_if_needed(all_outputs)

    def _encode_context_without_side_effects(
        self,
        obs: TensorDict,
        hidden_state: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        saved_fast = None if self.hidden_fast is None else self.hidden_fast.clone()
        saved_slow = None if self.hidden_slow is None else self.hidden_slow.clone()
        saved_counter = None if self.slow_step_counter is None else self.slow_step_counter.clone()

        if hidden_state is not None:
            self._set_hidden_from_stored(hidden_state)

        try:
            if obs.batch_dims >= 2:
                masks_bool = masks.to(dtype=torch.bool) if masks is not None else None
                actor_obs_seq = self.actor_obs_normalizer(self.get_actor_obs(obs))
                seq_len = actor_obs_seq.shape[0]
                all_z_t = []
                for t in range(seq_len):
                    obs_t = actor_obs_seq[t]
                    mask_t = masks_bool[t] if masks_bool is not None else None
                    _, _, z_t, _, _ = self._step_encoder(obs_t, valid_mask=mask_t)
                    all_z_t.append(z_t)
                z_seq = torch.stack(all_z_t, dim=0)
                if masks_bool is not None:
                    z_seq = unpad_trajectories(z_seq, masks_bool)
                return z_seq

            actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
            self._init_hidden(actor_obs.shape[0], actor_obs.device)
            _, _, z_t, _, _ = self._step_encoder(actor_obs)
            return z_t
        finally:
            self.hidden_fast = saved_fast
            self.hidden_slow = saved_slow
            self.slow_step_counter = saved_counter

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        self._init_hidden(actor_obs.shape[0], actor_obs.device)
        v_hat, _, z_t, _, _ = self._step_encoder(actor_obs)
        actor_output = self._actor_forward(actor_obs, v_hat, z_t)
        if self.state_dependent_std:
            return actor_output[..., 0, :]
        return actor_output

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        masks = kwargs.get("masks")
        hidden_state = kwargs.get("hidden_state")
        masks_bool = masks.to(dtype=torch.bool) if masks is not None else None

        if self._context_cache_fresh and self.cached_context_latent is not None:
            z_t = self.cached_context_latent
        else:
            z_t = self._encode_context_without_side_effects(obs, hidden_state=hidden_state, masks=masks)
        self._context_cache_fresh = False

        critic_obs = self.get_critic_obs(obs)
        if critic_obs.dim() == 3:
            orig_shape = critic_obs.shape
            critic_obs = self.critic_obs_normalizer(critic_obs.reshape(-1, orig_shape[-1])).reshape(orig_shape)
            if masks_bool is not None:
                critic_obs = unpad_trajectories(critic_obs, masks_bool)
        else:
            critic_obs = self.critic_obs_normalizer(critic_obs)

        return self._critic_forward(critic_obs, z_t)

    # ------------------------------------------------------------------
    # Auxiliary losses
    # ------------------------------------------------------------------

    def _unpad_obs(self, obs: TensorDict) -> TensorDict:
        masks = self._cached_masks
        if masks is None or obs.batch_dims < 2:
            return obs
        unpadded = {key: unpad_trajectories(obs[key], masks) for key in obs.keys()}
        return TensorDict(unpadded, batch_size=unpadded[next(iter(unpadded))].shape[:2])

    def compute_auxiliary_losses(
        self,
        obs: TensorDict,
        next_obs: TensorDict | None = None,
        dones: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        returns: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del returns
        if self.cached_context_latent is None or self.cached_velocity_estimate is None:
            raise RuntimeError("DSR auxiliary losses requested before encoder features were computed.")

        obs = self._unpad_obs(obs)
        velocity_target = obs[self.velocity_target_group]
        velocity_loss = F.mse_loss(self.cached_velocity_estimate, velocity_target)

        transition_loss = torch.tensor(0.0, device=velocity_target.device)
        triplet_loss = torch.tensor(0.0, device=velocity_target.device)
        if actions is not None:
            if self.cached_context_latent.dim() >= 3:
                transition_loss, triplet_loss = self._sequence_transition_triplet_losses(
                    self.cached_context_latent,
                    actions,
                    dones,
                )
            elif next_obs is not None:
                transition_loss, triplet_loss = self._single_step_transition_triplet_losses(
                    self.cached_context_latent,
                    next_obs,
                    actions,
                    dones,
                )

        total_loss = (
            self.velocity_loss_weight * velocity_loss
            + self.transition_loss_weight * transition_loss
            + self.triplet_loss_weight * triplet_loss
        )

        gate_mean = self.cached_gate.mean().detach() if self.cached_gate is not None else velocity_loss.new_tensor(0.0)
        slow_update_rate = (
            self.cached_slow_update_mask.mean().detach()
            if self.cached_slow_update_mask is not None
            else velocity_loss.new_tensor(0.0)
        )
        z_std = self.cached_context_latent.std(dim=0).mean().detach()

        return {
            "auxiliary_total": total_loss,
            "dsr_velocity": velocity_loss.detach(),
            "dsr_transition": transition_loss.detach(),
            "dsr_triplet": triplet_loss.detach(),
            "dsr_gate_mean": gate_mean,
            "dsr_slow_update_rate": slow_update_rate,
            "dsr_z_std": z_std,
        }

    def _sequence_transition_triplet_losses(
        self,
        z_seq: torch.Tensor,
        actions: torch.Tensor,
        dones: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        horizon = self.transition_horizon
        if z_seq.shape[0] <= horizon:
            zero = z_seq.new_tensor(0.0)
            return zero, zero

        valid_mask = self._build_horizon_valid_mask(dones, horizon)
        current_z = z_seq[:-horizon]
        target_z = z_seq[horizon:].detach()
        action_t = actions[:-horizon]

        if valid_mask is None:
            valid = torch.ones(current_z.shape[:2], dtype=torch.bool, device=current_z.device)
        else:
            valid = valid_mask[:-horizon].to(dtype=torch.bool, device=current_z.device)

        if not valid.any():
            zero = z_seq.new_tensor(0.0)
            return zero, zero

        current_z = current_z[valid]
        target_z = target_z[valid]
        action_t = action_t[valid]

        pred_next_z = self.transition_model(torch.cat([current_z, action_t], dim=-1))
        transition_loss = F.mse_loss(pred_next_z, target_z)
        triplet_loss = self._triplet_margin_loss(pred_next_z, target_z)
        return transition_loss, triplet_loss

    def _single_step_transition_triplet_losses(
        self,
        current_z: torch.Tensor,
        next_obs: TensorDict,
        actions: torch.Tensor,
        dones: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            next_hidden_state, _ = self.get_hidden_states()
            target_next_z = self._encode_context_without_side_effects(next_obs, hidden_state=next_hidden_state)

        pred_next_z = self.transition_model(torch.cat([current_z, actions], dim=-1))
        if dones is not None:
            valid = (1.0 - dones.float()).squeeze(-1).to(dtype=torch.bool, device=current_z.device)
            if valid.any():
                pred_next_z = pred_next_z[valid]
                target_next_z = target_next_z[valid]
            else:
                zero = current_z.new_tensor(0.0)
                return zero, zero

        transition_loss = F.mse_loss(pred_next_z, target_next_z)
        triplet_loss = self._triplet_margin_loss(pred_next_z, target_next_z)
        return transition_loss, triplet_loss

    def _triplet_margin_loss(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        num_samples = anchor.shape[0]
        if num_samples < 2:
            return anchor.new_tensor(0.0)

        positive_detached = positive.detach()
        pos_dist = torch.norm(anchor - positive_detached, dim=-1)
        num_negatives = min(self.triplet_num_negatives, num_samples - 1)
        if num_negatives < 1:
            return anchor.new_tensor(0.0)

        triplet_losses = []
        for start in range(0, num_samples, self.triplet_chunk_size):
            end = min(start + self.triplet_chunk_size, num_samples)
            chunk_anchor = anchor[start:end]
            chunk_pos_dist = pos_dist[start:end]
            chunk_size = end - start

            neg_idx = torch.randint(
                low=0,
                high=num_samples - 1,
                size=(chunk_size, num_negatives),
                device=anchor.device,
            )
            self_idx = torch.arange(start, end, device=anchor.device).unsqueeze(1)
            neg_idx = neg_idx + (neg_idx >= self_idx).long()

            negatives = positive_detached[neg_idx]
            neg_dist = torch.norm(chunk_anchor.unsqueeze(1) - negatives, dim=-1)
            hard_neg_dist = neg_dist.min(dim=1).values
            triplet_losses.append(F.relu(chunk_pos_dist - hard_neg_dist + self.triplet_margin))

        return torch.cat(triplet_losses, dim=0).mean()

    @staticmethod
    def _build_horizon_valid_mask(dones: torch.Tensor | None, horizon: int) -> torch.Tensor | None:
        if dones is None:
            return None

        not_done = (1.0 - dones.float()).squeeze(-1)
        if not_done.dim() < 2:
            if horizon > 1:
                return not_done * 0.0
            return not_done

        valid_mask = not_done.clone()
        for offset in range(1, horizon):
            shifted = torch.roll(not_done, shifts=-offset, dims=0)
            shifted[-offset:] = 0.0
            valid_mask = valid_mask * shifted

        valid_mask[-horizon:] = 0.0
        return valid_mask

    # ------------------------------------------------------------------
    # State dict compatibility
    # ------------------------------------------------------------------

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        state_dict = dict(state_dict)
        legacy_prefixes = (
            "context_encoder.",
            "context_mean_head.",
            "context_logvar_head.",
            "_target_context_encoder.",
            "_target_context_mean_head.",
            "fast_response_heads.",
            "slow_response_heads.",
            "fast_proj.",
            "slow_proj.",
            "actor_base.",
            "actor_ctx.",
            "beta_head.",
        )
        dropped = [k for k in state_dict if k.startswith(legacy_prefixes)]
        if dropped:
            print(f"DSR-SLR: Dropping {len(dropped)} incompatible keys from checkpoint: {dropped[:6]}...")
            for key in dropped:
                del state_dict[key]
            strict = False

        dsr_keys = {
            "frontend.",
            "gru_fast.",
            "gru_slow.",
            "velocity_head.",
            "dynamics_head.",
            "gate_head.",
            "context_head.",
            "transition_model.",
            "actor.",
            "critic.",
        }
        if not any(k.startswith(tuple(dsr_keys)) for k in state_dict):
            print("Checkpoint lacks DSR-SLR modules; loading with strict=False")
            strict = False

        current_state = self.state_dict()
        mismatched = [
            key for key, value in state_dict.items() if key in current_state and current_state[key].shape != value.shape
        ]
        if mismatched:
            print(f"DSR-SLR: Dropping {len(mismatched)} shape-mismatched keys: {mismatched[:6]}...")
            for key in mismatched:
                del state_dict[key]
            strict = False

        super().load_state_dict(state_dict, strict=strict)
        return True

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_obs_dim(obs: TensorDict, groups: list[str]) -> int:
        obs_dim = 0
        for group in groups:
            assert len(obs[group].shape) == 2, "The DSR policy only supports 1D observations."
            obs_dim += obs[group].shape[-1]
        return obs_dim
