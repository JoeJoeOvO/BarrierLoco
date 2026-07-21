from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
MODULES_DIR = ROOT / "locolab/rsl_rl/modules"


class _TinyMLP(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | tuple[int, ...],
        activation: str,
        last_activation: str | None = None,
    ) -> None:
        del activation
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ELU()))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        if last_activation is not None:
            layers.append(nn.ELU())
        super().__init__(*layers)


class _IdentityNormalization(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        del size

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def update(self, value: torch.Tensor) -> None:
        del value


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_dreamwaq_modules(monkeypatch):
    rsl_rl_package = types.ModuleType("rsl_rl")
    rsl_rl_package.__path__ = []
    modules_package = types.ModuleType("rsl_rl.modules")
    modules_package.__path__ = [str(MODULES_DIR)]
    networks_module = types.ModuleType("rsl_rl.networks")
    networks_module.MLP = _TinyMLP
    networks_module.EmpiricalNormalization = _IdentityNormalization
    tensordict_module = types.ModuleType("tensordict")
    tensordict_module.TensorDict = dict

    monkeypatch.setitem(sys.modules, "rsl_rl", rsl_rl_package)
    monkeypatch.setitem(sys.modules, "rsl_rl.modules", modules_package)
    monkeypatch.setitem(sys.modules, "rsl_rl.networks", networks_module)
    monkeypatch.setitem(sys.modules, "tensordict", tensordict_module)

    estimator = _load_module("rsl_rl.modules.dreamwaq_estimator", MODULES_DIR / "dreamwaq_estimator.py")
    actor_critic = _load_module("rsl_rl.modules.actor_critic_dreamwaq", MODULES_DIR / "actor_critic_dreamwaq.py")
    return estimator, actor_critic


def test_dreamwaq_policy_mean_is_deterministic_during_training(monkeypatch) -> None:
    _, actor_critic = _load_dreamwaq_modules(monkeypatch)
    torch.manual_seed(7)
    observations = {
        "policy": torch.randn(6, 5),
        "policy_history": torch.randn(6, 20),
        "critic": torch.randn(6, 8),
        "dreamwaq_velocity": torch.randn(6, 3),
    }
    policy = actor_critic.ActorCriticDreamWaQ(
        obs=observations,
        obs_groups={"policy": ["policy"], "policy_history": ["policy_history"], "critic": ["critic"]},
        num_actions=4,
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
        estimator_hidden_dims=[8],
        context_decoder_hidden_dims=[8],
        context_latent_dim=4,
    )
    policy.train()

    policy.act(observations)
    first_mean = policy.action_mean.detach().clone()
    policy.act(observations)
    second_mean = policy.action_mean.detach().clone()

    torch.testing.assert_close(first_mean, second_mean, rtol=0.0, atol=0.0)


def test_dreamwaq_reconstruction_mask_excludes_dimensions_from_loss(monkeypatch) -> None:
    estimator, _ = _load_dreamwaq_modules(monkeypatch)
    vae = estimator.DreamWaQVAE(
        history_obs_dim=2,
        reconstruction_obs_dim=3,
        velocity_dim=1,
        latent_dim=1,
        encoder_hidden_dims=[],
        decoder_hidden_dims=[],
        reconstruction_mask_indices=[1],
    )
    with torch.no_grad():
        for parameter in vae.parameters():
            parameter.zero_()
        decoder_linear = next(module for module in vae.decoder.modules() if isinstance(module, nn.Linear))
        decoder_linear.bias.copy_(torch.tensor([1.0, 10.0, 3.0]))

    losses = vae.loss_fn(
        history_obs=torch.zeros(2, 2),
        next_obs=torch.zeros(2, 3),
        velocity_target=torch.zeros(2, 1),
        kld_weight=0.0,
    )

    torch.testing.assert_close(losses["recons_loss"], torch.full((2,), 5.0))
