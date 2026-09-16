import ast
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_method(relative, class_name, method, namespace):
    tree = ast.parse((ROOT / relative).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method)
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, relative, "exec"), namespace)
    return namespace[method]


@pytest.mark.parametrize("peer_has_valid", [False, True])
def test_empty_local_vae_batch_joins_peer_update(monkeypatch, peer_has_valid):
    method = load_method(
        "locolab/rsl_rl/algorithms/ppo_dreamwaq.py",
        "PPODreamWaQ",
        "_optimize_additional_modules",
        {"torch": torch, "nn": torch.nn, "TensorDict": dict},
    )
    vae = torch.nn.Linear(1, 1, bias=False)
    vae.weight.data.fill_(1.0)
    optimizer = torch.optim.SGD(vae.parameters(), lr=0.1)
    calls = []

    def compute(*args, **kwargs):
        loss = vae(torch.ones(2, 1)).square().squeeze(-1)
        return dict(loss=loss, recons_loss=loss, vel_loss=loss, kld_loss=loss)

    def all_reduce(value, op):
        calls.append("valid_collective")
        value.fill_(int(peer_has_valid))

    def reduce():
        calls.append("gradient_collective")
        assert vae.weight.grad is not None
        assert vae.weight.grad.item() == 0.0
        vae.weight.grad.fill_(1.0)  # Mean gradient contributed by the other rank.

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    instance = types.SimpleNamespace(
        policy=types.SimpleNamespace(vae=vae, compute_dreamwaq_losses=compute),
        optimizer=optimizer,
        vae_optimizer=optimizer,
        rnd_optimizer=None,
        actor_decoder_optimizer=None,
        critic_decoder_optimizer=None,
        is_multi_gpu=True,
        reduce_parameters=reduce,
        kl_weight=0.1,
        max_grad_norm=10.0,
    )
    result = method(instance, {}, {}, torch.ones(2, 1), torch.empty(0), torch.empty(0))
    assert all(value == 0.0 for value in result.values())
    assert calls == (["valid_collective", "gradient_collective"] if peer_has_valid else ["valid_collective"])
    assert vae.weight.item() == pytest.approx(0.9 if peer_has_valid else 1.0)


def test_nonprimary_rank_cannot_write_checkpoint(tmp_path):
    method = load_method("locolab/rsl_rl/runners/on_policy_runner.py", "OnPolicyRunner", "save", {"torch": torch})
    target = tmp_path / "model.pt"
    method(types.SimpleNamespace(is_distributed=True, gpu_global_rank=1), str(target))
    assert not target.exists()
