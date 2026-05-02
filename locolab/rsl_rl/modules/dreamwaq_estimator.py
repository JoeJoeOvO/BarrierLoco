from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.networks import MLP


class DreamWaQVAE(nn.Module):
    """DreamWaQ VAE matching the GitHub repository's estimator semantics."""

    def __init__(
        self,
        history_obs_dim: int,
        reconstruction_obs_dim: int,
        velocity_dim: int,
        latent_dim: int,
        encoder_hidden_dims: Sequence[int] = (128,),
        decoder_hidden_dims: Sequence[int] = (64, 128),
        activation: str = "elu",
        sigma_min: float = 0.0,
        sigma_max: float = 5.0,
        reconstruction_mask_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__()

        self.history_obs_dim = history_obs_dim
        self.reconstruction_obs_dim = reconstruction_obs_dim
        self.velocity_dim = velocity_dim
        self.latent_dim = latent_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.reconstruction_mask_indices = tuple(reconstruction_mask_indices or ())

        encoded_dim = latent_dim * 4
        self.encoder = MLP(
            input_dim=history_obs_dim,
            output_dim=encoded_dim,
            hidden_dims=list(encoder_hidden_dims),
            activation=activation,
            last_activation=activation,
        )
        self.latent_mu = nn.Linear(encoded_dim, latent_dim)
        self.latent_logvar = nn.Linear(encoded_dim, latent_dim)
        self.vel_mu = nn.Linear(encoded_dim, velocity_dim)
        self.vel_logvar = nn.Linear(encoded_dim, velocity_dim)
        self.decoder = MLP(
            input_dim=latent_dim + velocity_dim,
            output_dim=reconstruction_obs_dim,
            hidden_dims=list(decoder_hidden_dims),
            activation=activation,
        )

    def encode(self, history_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(history_obs)
        latent_mu = self.latent_mu(encoded)
        latent_logvar = self._constrain_logvar(self.latent_logvar(encoded))
        vel_mu = self.vel_mu(encoded)
        vel_logvar = self._constrain_logvar(self.vel_logvar(encoded))
        return latent_mu, latent_logvar, vel_mu, vel_logvar

    def decode(self, latent: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat((latent, velocity), dim=-1))

    def forward(
        self, history_obs: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        latent_mu, latent_logvar, vel_mu, vel_logvar = self.encode(history_obs)
        latent = self.reparameterize(latent_mu, latent_logvar)
        velocity = self.reparameterize(vel_mu, vel_logvar)
        return (latent, velocity), (latent_mu, latent_logvar, vel_mu, vel_logvar)

    def inference(self, history_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent_mu, _, vel_mu, _ = self.encode(history_obs)
        return latent_mu, vel_mu

    def loss_fn(
        self,
        history_obs: torch.Tensor,
        next_obs: torch.Tensor,
        velocity_target: torch.Tensor,
        kld_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        (latent, velocity), (latent_mu, latent_logvar, _vel_mu, _vel_logvar) = self.forward(history_obs)

        reconstruction_target = next_obs
        if self.reconstruction_mask_indices:
            reconstruction_target = reconstruction_target.clone()
            reconstruction_target[..., list(self.reconstruction_mask_indices)] = 0.0

        reconstruction = self.decode(latent, velocity)
        reconstruction_loss = F.mse_loss(reconstruction, reconstruction_target, reduction="none").mean(dim=-1)
        velocity_loss = F.mse_loss(velocity, velocity_target, reduction="none").mean(dim=-1)
        kld_loss = -0.5 * torch.sum(1 + latent_logvar - latent_mu.pow(2) - latent_logvar.exp(), dim=-1)
        loss = reconstruction_loss + velocity_loss + kld_weight * kld_loss

        return {
            "loss": loss,
            "recons_loss": reconstruction_loss,
            "vel_loss": velocity_loss,
            "kld_loss": kld_loss,
        }

    def get_constrained_latent_params(self, history_obs: torch.Tensor) -> dict[str, torch.Tensor | float]:
        latent_mu, latent_logvar, vel_mu, vel_logvar = self.encode(history_obs)
        return {
            "latent_mu": latent_mu,
            "latent_sigma": torch.exp(0.5 * latent_logvar),
            "vel_mu": vel_mu,
            "vel_sigma": torch.exp(0.5 * vel_logvar),
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
        }

    def _constrain_logvar(self, logvar: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(0.5 * logvar)
        sigma = torch.clamp(sigma, min=self.sigma_min, max=self.sigma_max)
        return 2.0 * torch.log(sigma + 1.0e-8)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std
