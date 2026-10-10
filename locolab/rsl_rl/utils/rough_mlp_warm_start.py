"""Explicit, actor-only Go2W Rough (53 obs) -> binary Jump (54 obs) migration."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def warm_start_rough_mlp(policy, checkpoint_path: str, *, command_name: str = "jump_trigger") -> dict:
    """Insert a zero jump-command column without changing the initial actor.

    This is NOT a resume: critic, optimizer and iteration stay newly initialized.
    Source must be an unnormalized, log-std, 16-action Rough MLP. A scalar-std
    destination retains the original binary Jump network parameterization.
    Validate every tensor before mutating the destination model.
    """
    if (
        policy.is_recurrent
        or policy.actor_obs_normalization
        or policy.critic_obs_normalization
        or policy.state_dependent_std
        or policy.noise_std_type not in ("log", "scalar")
    ):
        raise ValueError("Rough warm start requires a plain, unnormalized ActorCritic MLP.")
    if command_name != "jump_trigger":
        raise ValueError("Unknown destination jump command semantics.")
    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    source = checkpoint["model_state_dict"]
    target = policy.state_dict()
    std_key = "log_std" if policy.noise_std_type == "log" else "std"
    expected_keys = (set(target) - {std_key}) | {"log_std"}
    if set(source) != expected_keys:
        raise ValueError("Checkpoint keys do not match plain Rough MLP (RNN/estimators are not supported).")
    if source["actor.0.weight"].shape[1] != 53 or target["actor.0.weight"].shape[1] != 54:
        raise ValueError("Expected source actor 53 and destination actor 54 observation columns.")
    if source["critic.0.weight"].shape[1] != 307:
        raise ValueError("Expected the validated Rough critic's 307-column observation contract.")
    if source["log_std"].shape != (16,) or target[std_key].shape != (16,):
        raise ValueError("Expected 16 Go2W actions with log standard deviation.")
    loaded = []
    for key, value in source.items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid/nonfinite checkpoint tensor: {key}")
        if key.startswith("critic."):
            continue
        if key == "log_std" and std_key == "std":
            value = value.exp()
            if not torch.isfinite(value).all() or not (value > 0).all():
                raise ValueError("Rough log_std cannot be converted to positive finite scalar std.")
            key = "std"
        if key == "actor.0.weight":
            if value.shape[0] != target[key].shape[0]:
                raise ValueError("Actor hidden architecture differs from Rough.")
            expanded = torch.zeros_like(target[key])
            expanded[:, :9] = value[:, :9].to(expanded)
            expanded[:, 10:] = value[:, 9:].to(expanded)
            target[key] = expanded
        else:
            if value.shape != target[key].shape:
                raise ValueError(f"Incompatible actor tensor: {key}")
            target[key] = value.to(target[key])
        loaded.append(key)
    policy.load_state_dict(target, strict=True)
    report = {
        "mode": "rough_mlp_actor_only",
        "source": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_iteration": checkpoint.get("iter"),
        "loaded_tensors": loaded,
        "command_name": command_name,
        "command_input_column": 9,
        "command_input_initialization": "zero",
        "noise_conversion": "log_std_to_std" if std_key == "std" else "none",
        "critic": "fresh",
        "optimizer": "fresh",
        "iteration": 0,
    }
    return report
