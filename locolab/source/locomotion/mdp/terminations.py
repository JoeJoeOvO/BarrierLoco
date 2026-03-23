from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class _DelayedTerminationTerm(ManagerTermBase):
    """Base class for termination terms that require consecutive violations."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._violation_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._violation_steps.zero_()
        else:
            self._violation_steps[env_ids] = 0

    def _resolve_required_steps(self, delay_steps: int | None) -> int:
        if delay_steps is not None:
            return max(1, delay_steps)
        return 1

    def _apply_delay(self, violation_mask: torch.Tensor, required_steps: int) -> torch.Tensor:
        self._violation_steps = torch.where(
            violation_mask,
            self._violation_steps + 1,
            torch.zeros_like(self._violation_steps),
        )
        return self._violation_steps >= required_steps


class DelayedIllegalContact(_DelayedTerminationTerm):
    """Terminate only after contact threshold violations persist for a delay window."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        sensor_cfg: SceneEntityCfg,
        delay_steps: int | None = None,
    ) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        net_contact_forces = contact_sensor.data.net_forces_w_history
        violation_mask = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold,
            dim=1,
        )
        required_steps = self._resolve_required_steps(delay_steps=delay_steps)
        return self._apply_delay(violation_mask, required_steps)


class DelayedBadOrientation(_DelayedTerminationTerm):
    """Terminate only after orientation violations persist for a delay window."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        limit_angle: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        delay_steps: int | None = None,
    ) -> torch.Tensor:
        asset: RigidObject = env.scene[asset_cfg.name]
        violation_mask = torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle
        required_steps = self._resolve_required_steps(delay_steps=delay_steps)
        return self._apply_delay(violation_mask, required_steps)
