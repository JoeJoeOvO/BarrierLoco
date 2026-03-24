from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.utils import resolve_nn_activation


class MLPDecoder(nn.Module):
    """MLP decoder with a latent bottleneck."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim: int = 32,
        hidden_dims: list[int] | None = None,
        activation: str = "elu",
    ) -> None:
        super().__init__()

        hidden_dims = hidden_dims if hidden_dims is not None else [128, 64]
        activation_fn = resolve_nn_activation(activation)

        self.encoder = nn.Identity() if latent_dim == input_dim else nn.Linear(input_dim, latent_dim)

        layers = []
        current_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(current_dim, hidden_dim), activation_fn])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class VAEDecoder(nn.Module):
    """VAE decoder that reconstructs targets from recurrent hidden states."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim: int = 32,
        hidden_dims: list[int] | None = None,
        activation: str = "elu",
    ) -> None:
        super().__init__()

        hidden_dims = hidden_dims if hidden_dims is not None else [128, 64]
        activation_fn = resolve_nn_activation(activation)

        self.mean_head = nn.Identity() if latent_dim == input_dim else nn.Linear(input_dim, latent_dim)
        self.logvar_head = nn.Linear(input_dim, latent_dim)

        layers = []
        current_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(current_dim, hidden_dim), activation_fn])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.mean_head(x)
        logvar = self.logvar_head(x)
        std = torch.exp(0.5 * logvar)
        z = mean + torch.randn_like(std) * std
        reconstruction = self.decoder(z)
        return reconstruction, mean, logvar

    @staticmethod
    def kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        return kl.sum(dim=-1).mean()


class DecoderCollection(nn.Module):
    """Collection of decoders attached to one recurrent branch."""

    def __init__(self, input_dim: int, decoder_configs: list[dict]) -> None:
        super().__init__()

        self.configs = decoder_configs
        self.decoders = nn.ModuleList()

        for config in decoder_configs:
            decoder_type = config["type"]
            output_dim = len(config["target_indices"])
            latent_dim = config.get("latent_dim", 32)
            hidden_dims = config.get("hidden_dims", [128, 64])
            activation = config.get("activation", "elu")

            if decoder_type == "mlp":
                decoder = MLPDecoder(input_dim, output_dim, latent_dim, hidden_dims, activation)
            elif decoder_type == "vae":
                decoder = VAEDecoder(input_dim, output_dim, latent_dim, hidden_dims, activation)
            else:
                raise ValueError(f"Unsupported decoder type: {decoder_type}")

            self.decoders.append(decoder)

    def forward_and_compute_losses(
        self,
        hidden_states: torch.Tensor,
        target_observations: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, float]]:
        total_loss = torch.tensor(0.0, device=hidden_states.device)
        losses_dict: dict[str, float] = {}

        for index, (decoder, config) in enumerate(zip(self.decoders, self.configs, strict=True)):
            target = target_observations[..., config["target_indices"]]
            loss_weight = config.get("loss_weight", 1.0)

            if config["type"] == "mlp":
                reconstruction = decoder(hidden_states)
                reconstruction_loss = nn.functional.mse_loss(reconstruction, target)
                decoder_loss = reconstruction_loss
                losses_dict[f"decoder_{index}_reconstruction_loss"] = reconstruction_loss.item()
            else:
                reconstruction, mean, logvar = decoder(hidden_states)
                reconstruction_loss = nn.functional.mse_loss(reconstruction, target)
                kl_weight = config.get("kl_weight", 1.0)
                kl_loss = decoder.kl_loss(mean, logvar)
                decoder_loss = reconstruction_loss + kl_weight * kl_loss
                losses_dict[f"decoder_{index}_reconstruction_loss"] = reconstruction_loss.item()
                losses_dict[f"decoder_{index}_kl_loss"] = kl_loss.item()
                losses_dict[f"decoder_{index}_weighted_kl_loss"] = (kl_weight * kl_loss).item()

            losses_dict[f"decoder_{index}_total_loss"] = decoder_loss.item()
            total_loss = total_loss + loss_weight * decoder_loss

        losses_dict["total_decoder_loss"] = total_loss.item()
        return {"loss": total_loss, "losses_dict": losses_dict}
