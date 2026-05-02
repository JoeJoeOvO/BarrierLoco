# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from .ppo import PPO


class PPODreamWaQ(PPO):
    """PPO variant with a separately-optimized DreamWaQ VAE."""

    def __init__(self, *args, vae_learning_rate: float = 1.0e-3, kl_weight: float = 0.1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vae_learning_rate = vae_learning_rate
        self.kl_weight = kl_weight

        # Rebuild the PPO optimizer excluding VAE parameters so that the two
        # optimizers do not maintain conflicting Adam momentum states for the
        # same parameters.
        vae_param_ids = {id(p) for p in self.policy.vae.parameters()}
        non_vae_params = [p for p in self.policy.parameters() if id(p) not in vae_param_ids]
        self.optimizer = optim.Adam(non_vae_params, lr=self.learning_rate)

        self.vae_optimizer = optim.Adam(self.policy.vae.parameters(), lr=vae_learning_rate)

    def _compute_auxiliary_losses(
        self,
        obs_batch: TensorDict,
        next_obs_batch: TensorDict,
        dones_batch: torch.Tensor,
        actions_batch: torch.Tensor,
        returns_batch: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
        del obs_batch, next_obs_batch, dones_batch, actions_batch, returns_batch

    def _optimize_additional_modules(
        self,
        obs_batch: TensorDict,
        next_obs_batch: TensorDict,
        dones_batch: torch.Tensor,
        actions_batch: torch.Tensor,
        returns_batch: torch.Tensor,
    ) -> dict[str, float] | None:
        del actions_batch, returns_batch
        if not hasattr(self.policy, "compute_dreamwaq_losses"):
            return None

        self.optimizer.zero_grad()
        self.vae_optimizer.zero_grad()
        if self.rnd_optimizer:
            self.rnd_optimizer.zero_grad()
        if self.actor_decoder_optimizer:
            self.actor_decoder_optimizer.zero_grad()
        if self.critic_decoder_optimizer:
            self.critic_decoder_optimizer.zero_grad()

        vae_loss_dict = self.policy.compute_dreamwaq_losses(
            obs_batch, next_obs=next_obs_batch, kld_weight=self.kl_weight
        )
        valid = (dones_batch == 0).squeeze(-1)
        if not torch.any(valid):
            # No valid transitions in this minibatch — skip VAE update to avoid
            # training on post-reset next_obs that straddle episode boundaries.
            return {
                "dreamwaq_vae": 0.0,
                "dreamwaq_recons": 0.0,
                "dreamwaq_velocity": 0.0,
                "dreamwaq_kld": 0.0,
            }

        vae_loss = vae_loss_dict["loss"][valid].mean()
        recons_loss = vae_loss_dict["recons_loss"][valid].mean()
        velocity_loss = vae_loss_dict["vel_loss"][valid].mean()
        kld_loss = vae_loss_dict["kld_loss"][valid].mean()

        vae_loss.backward()
        if self.is_multi_gpu:
            self.reduce_parameters()
        nn.utils.clip_grad_norm_(self.policy.vae.parameters(), self.max_grad_norm)
        self.vae_optimizer.step()

        return {
            "dreamwaq_vae": vae_loss.item(),
            "dreamwaq_recons": recons_loss.item(),
            "dreamwaq_velocity": velocity_loss.item(),
            "dreamwaq_kld": kld_loss.item(),
        }

    def get_additional_checkpoint_state(self) -> dict[str, object]:
        return {"vae_optimizer_state_dict": self.vae_optimizer.state_dict()}

    def load_additional_checkpoint_state(self, loaded_dict: dict, load_optimizer: bool) -> None:
        if load_optimizer and "vae_optimizer_state_dict" in loaded_dict:
            self.vae_optimizer.load_state_dict(loaded_dict["vae_optimizer_state_dict"])
