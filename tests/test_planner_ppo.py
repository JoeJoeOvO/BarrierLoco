"""PPO correctness at reset boundaries and for mixed conditional actions."""

import torch

from locolab.planner.network import NavigationActorCritic
from locolab.planner.ppo import HybridPPO, generalized_advantage


def test_timeout_bootstraps_terminal_value_without_leaking_next_episode():
    reward = torch.tensor([[1.0], [100.0]])
    value = torch.tensor([[2.0], [3.0]])
    bootstrap = torch.tensor([[10.0], [4.0]])
    done = torch.tensor([[True], [False]])
    advantage, _ = generalized_advantage(reward, value, bootstrap, done, gamma=0.9, lam=0.95)
    torch.testing.assert_close(advantage[0], torch.tensor([8.0]))
    bootstrap[0] = 0.0  # physical failure: no value bootstrap
    advantage, _ = generalized_advantage(reward, value, bootstrap, done, gamma=0.9, lam=0.95)
    torch.testing.assert_close(advantage[0], torch.tensor([-1.0]))


def test_hybrid_ppo_updates_model_with_finite_gradients():
    torch.manual_seed(7)
    model = NavigationActorCritic()
    model.initialize_navigation()
    ppo = HybridPPO(model, epochs=2, minibatch_size=4)
    observation = {"occ": torch.zeros(4, 2, 16, 24, 32), "proprio": torch.zeros(4, 58), "goal": torch.randn(4, 3)}
    allowed = torch.ones(4, 3, dtype=torch.bool)
    with torch.no_grad():
        action = model.sample(observation, allowed)
    rollout = {key: value.unsqueeze(0).repeat(2, *([1] * value.ndim)) for key, value in observation.items()}
    rollout["occ"] = rollout["occ"].byte()
    for key in ("skill", "raw_command", "log_prob", "value"):
        rollout[key] = action[key].unsqueeze(0).repeat(2, *([1] * action[key].ndim))
    rollout["allowed"] = allowed[None].repeat(2, 1, 1)
    rollout["reward"] = torch.randn(2, 4)
    rollout["bootstrap"] = torch.zeros(2, 4)
    rollout["done"] = torch.ones(2, 4, dtype=torch.bool)
    before = model.value_head.weight.detach().clone()
    metrics = ppo.update(rollout)
    assert metrics["minibatches"] > 0
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert not torch.equal(before, model.value_head.weight)


def test_gae_continues_across_goal_refresh_without_reset():
    reward = torch.tensor([[5.0], [1.0]])
    value = torch.zeros(2, 1)
    advantage, _ = generalized_advantage(reward, value, value, torch.zeros(2, 1, dtype=torch.bool), gamma=0.9, lam=1.0)
    torch.testing.assert_close(advantage[:, 0], torch.tensor([5.9, 1.0]))
