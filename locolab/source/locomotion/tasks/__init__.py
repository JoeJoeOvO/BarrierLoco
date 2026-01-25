"""Locomotion tasks for OmniBot."""

import gymnasium as gym

gym.register(
    id="OmniBot-Go2W-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
