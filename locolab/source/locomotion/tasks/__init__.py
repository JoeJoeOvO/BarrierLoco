"""Locomotion tasks for OmniBot."""

import gymnasium as gym

# ==============================================================================
# Go2W Environments
# ==============================================================================

# Go2W Flat Terrain
gym.register(
    id="OmniBot-Go2W-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2W.flat_env_cfg:Go2WFlatEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.Go2W.flat_env_cfg:Go2WFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="OmniBot-Go2W-Flat-Rnn-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2W.flat_env_cfg:Go2WFlatEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.Go2W.flat_env_cfg:Go2WFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:Go2WFlatRnnDecoderPPORunnerCfg"
        ),
    },
)

# Go2W Rough Terrain
gym.register(
    id="OmniBot-Go2W-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2W.rough_env_cfg:Go2WRoughEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.Go2W.rough_env_cfg:Go2WRoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="OmniBot-Go2W-Rough-Rnn-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2W.rough_env_cfg:Go2WRoughEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.Go2W.rough_env_cfg:Go2WRoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:Go2WRoughRnnDecoderPPORunnerCfg"
        ),
    },
)

# Legacy alias (backward compatibility)
gym.register(
    id="OmniBot-Go2W-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2W.flat_env_cfg:Go2WFlatEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.Go2W.flat_env_cfg:Go2WFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
