"""Reset and step an Isaac Lab task while checking all tensors are finite."""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch
from isaaclab.app import AppLauncher

PROJECT_ROOT = pathlib.Path(__file__).parents[3]
LOCOLAB_ROOT = pathlib.Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LOCOLAB_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
target_group = parser.add_mutually_exclusive_group(required=True)
target_group.add_argument("--task")
target_group.add_argument(
    "--validate-registered-go2w",
    action="store_true",
    help="Instantiate and validate every registered Go2W environment and runner config.",
)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=32)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
from rsl_rl.utils import assert_finite_tree  # noqa: E402

import locolab.source.locomotion.tasks  # noqa: E402, F401
from locolab.source.locomotion.utils.parser_cfg import parse_env_cfg  # noqa: E402


def _validate_registered_go2w_configs() -> None:
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    task_ids = sorted(spec.id for spec in gym.registry.values() if spec.id.startswith("OmniBot-Go2W-"))
    if not task_ids:
        raise RuntimeError("No registered Go2W tasks were found.")
    for task_id in task_ids:
        env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
        runner_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
        env_cfg.validate()
        runner_cfg.validate()
    print(f"CONFIG_SMOKE_OK prefix=OmniBot-Go2W- tasks={len(task_ids)}")


def main() -> None:
    env = None
    try:
        if args.validate_registered_go2w:
            _validate_registered_go2w_configs()
            return
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        env = gym.make(args.task, cfg=env_cfg)
        observations, _ = env.reset()
        assert_finite_tree(observations, "reset.observations")

        unwrapped = env.unwrapped
        actions = torch.zeros(
            (unwrapped.num_envs, unwrapped.action_manager.total_action_dim),
            device=unwrapped.device,
        )
        assert_finite_tree(actions, "actions")
        for step in range(args.steps):
            observations, rewards, terminated, truncated, _ = env.step(actions)
            assert_finite_tree(observations, f"step[{step}].observations")
            assert_finite_tree(rewards, f"step[{step}].rewards")
            assert_finite_tree(terminated, f"step[{step}].terminated")
            assert_finite_tree(truncated, f"step[{step}].truncated")
        print(f"ISAAC_SMOKE_OK task={args.task} envs={args.num_envs} steps={args.steps}")
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
