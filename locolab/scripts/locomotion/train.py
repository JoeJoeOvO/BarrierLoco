# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""


import os
import pathlib
import sys

import gymnasium as gym

# Add the project root to sys.path to allow importing locolab
# Structure: project_root/locolab/scripts/locomotion/train.py
# We need to add project_root to sys.path
project_root = pathlib.Path(__file__).parents[3]
locolab_root = pathlib.Path(__file__).parents[2]
logs_root = locolab_root / "logs"
sys.path.append(str(project_root))
# Prefer local rsl_rl package in this repo over any globally installed one.
sys.path.insert(0, str(locolab_root))

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

tasks = []
for task_spec in gym.registry.values():
    if "OmniBot" in task_spec.id and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)

import argparse

import argcomplete
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_terrain", type=str, default=None, help="Fix video camera on this curriculum terrain type.")
parser.add_argument(
    "--video_length",
    type=int,
    default=1000,
    help="Length of each recorded training clip in environment steps. Default is 1000 steps (~20s).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=500,
    help="Interval between recorded training clips in training iterations. Default is 500 iterations.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, choices=tasks, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--smoke-marker",
    action="store_true",
    help="Print a machine-checkable marker only after a smoke training run completes.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

# Only rank zero owns rendering, videos, and shared run artifacts.
is_primary = not args_cli.distributed or int(os.environ.get("RANK", "0")) == 0
record_video = args_cli.video and is_primary
if record_video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
try:
    import rsl_rl

    installed_version = getattr(rsl_rl, "__version__", "3.3.0")  # Default to local version
except ImportError:
    installed_version = "0.0.0"

if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

from datetime import datetime

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner  # TODO: Consider printing the experiment name in the terminal.

import locolab.source.locomotion.tasks  # noqa: F401
from locolab.source.locomotion.utils.video import TerrainVideoView, configure_robot_follow_view, configure_terrain_view

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    output_experiment_name = cli_args.get_output_experiment_name(agent_cfg)
    resume_experiment_name = cli_args.get_resume_experiment_name(agent_cfg)
    log_root_path = str(logs_root / "rsl_rl" / output_experiment_name)
    resume_root_path = str(logs_root / "rsl_rl" / resume_experiment_name)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.distributed:
        # torchrun provides a common rendezvous ID; launchers may supply a readable timestamp.
        log_dir = os.environ.get("OMNIBOT_RUN_TIMESTAMP") or os.environ.get("TORCHELASTIC_RUN_ID")
        if not log_dir or log_dir == "none" or pathlib.Path(log_dir).name != log_dir or log_dir in (".", ".."):
            raise ValueError("Distributed training requires a shared OMNIBOT_RUN_TIMESTAMP or torchrun rendezvous ID.")
        print(
            f"[DISTRIBUTED] rank={app_launcher.global_rank} local_rank={app_launcher.local_rank} "
            f"seed={env_cfg.seed} num_envs={env_cfg.scene.num_envs} video={record_video}"
        )
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    if record_video:
        if args_cli.video_terrain:
            index = configure_terrain_view(env_cfg, args_cli.video_terrain)
            print(f"[INFO] Video terrain: {args_cli.video_terrain}, env_index={index}, fixed tile view")
        else:
            configure_robot_follow_view(env_cfg.viewer)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if record_video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(resume_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    video_dir = None
    video_fps = None

    # wrap for video recording
    if record_video:
        if args_cli.video_terrain:
            env = TerrainVideoView(env)
        video_dir = os.path.join(log_dir, "videos", "train")
        video_fps = max(int(round(1.0 / env.unwrapped.step_dt)), 1)
        # Convert iteration-based interval to env-step-based interval for RecordVideo
        steps_per_iter = agent_cfg.num_steps_per_env
        step_interval = args_cli.video_interval * steps_per_iter
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step > 0 and step % step_interval == 0,
            "video_length": args_cli.video_length,
            "fps": video_fps,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env,
        cli_args.runner_cfg_to_dict(agent_cfg, validate_finite=args_cli.smoke_marker),
        log_dir=log_dir,
        device=agent_cfg.device,
    )
    if record_video:
        runner.logger.register_video_dir(
            video_dir,
            tag="Videos/train",
            fps=video_fps,
            steps_per_iter=agent_cfg.num_steps_per_env,
        )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    if is_primary:
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    if args_cli.smoke_marker and args_cli.distributed:
        parameters = torch.cat([p.detach().reshape(-1) for p in runner.alg.policy.parameters()])
        reference = parameters.clone()
        torch.distributed.broadcast(reference, src=0)
        torch.testing.assert_close(parameters, reference, rtol=0.0, atol=1.0e-6)
        print(f"DISTRIBUTED_PARAMETERS_OK rank={app_launcher.global_rank} seed={env_cfg.seed}")

    # close the simulator
    env.close()
    if record_video:
        runner.logger.flush_videos()
    runner.logger.close()
    if args_cli.smoke_marker:
        print(f"TRAINING_SMOKE_OK task={args_cli.task} iterations={agent_cfg.max_iterations}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
