"""PPO for categorical skills and their conditional squashed-Gaussian commands."""

import torch
from torch import nn


def generalized_advantage(rewards, values, bootstrap, done, gamma=0.99, lam=0.95):
    """Bootstrap timeout from terminal state, but never propagate across a reset.

    bootstrap is zero for physical failures and V(terminal_obs) for timeouts.
    This distinction is independent of the GAE trace mask (~done).
    """
    advantage = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * bootstrap[t] - values[t]
        running = delta + gamma * lam * (~done[t]).float() * running
        advantage[t] = running
    return advantage, advantage + values


class HybridPPO:
    def __init__(self, model, learning_rate=3e-4, epochs=4, minibatch_size=256):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.learning_rate = learning_rate

    def update(self, rollout):
        advantage, returns = generalized_advantage(
            rollout["reward"], rollout["value"], rollout["bootstrap"], rollout["done"]
        )
        advantage = advantage.flatten()
        advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp(min=1e-8)
        returns = returns.flatten()
        flat = {key: value.flatten(0, 1) for key, value in rollout.items()}
        size = len(advantage)
        totals = {key: 0.0 for key in ("policy_loss", "value_loss", "entropy", "kl", "clip_fraction", "grad_norm")}
        batches = 0
        stop = False
        for _ in range(self.epochs):
            for indices in torch.randperm(size, device=advantage.device).split(self.minibatch_size):
                obs = {name: flat[name][indices].float() for name in ("occ", "proprio", "goal")}
                log_prob, value, entropy = self.model.evaluate(
                    obs,
                    flat["skill"][indices],
                    flat["raw_command"][indices],
                    flat["allowed"][indices],
                    return_entropy=True,
                )
                log_ratio = log_prob - flat["log_prob"][indices]
                ratio = log_ratio.clamp(-20, 20).exp()
                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean()
                if kl > 0.05 and batches:
                    stop = True
                    break
                adv = advantage[indices]
                policy_loss = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
                old_value = flat["value"][indices]
                clipped = old_value + (value - old_value).clamp(-0.2, 0.2)
                value_loss = (
                    0.5
                    * torch.maximum((value - returns[indices]).square(), (clipped - returns[indices]).square()).mean()
                )
                loss = policy_loss + value_loss - 0.005 * entropy.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
                self.optimizer.step()
                metrics = (
                    policy_loss,
                    value_loss,
                    entropy.mean(),
                    kl,
                    ((ratio - 1).abs() > 0.2).float().mean(),
                    gradient,
                )
                for key, metric in zip(totals, metrics):
                    totals[key] += float(metric.detach())
                batches += 1
            if stop:
                break
        totals = {key: value / max(batches, 1) for key, value in totals.items()}
        # Adapt cautiously using measured policy change, not reward magnitude.
        if stop or totals["kl"] > 0.02:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        elif totals["kl"] < 0.005:
            self.learning_rate = min(3e-4, self.learning_rate * 1.1)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate
        totals.update(learning_rate=self.learning_rate, minibatches=batches)
        return totals
