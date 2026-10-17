"""Locomotion tasks for OmniBot."""

import gymnasium as gym

# ==============================================================================
# Go2W Environments
# ==============================================================================

# Go2W Flat Terrain
gym.register(
    id="OmniBot-Go2W-Gait-Curriculum-Flat-MLP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2W.gait_curriculum_env_cfg:Go2WGaitCurriculumEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.Go2W.gait_curriculum_env_cfg:Go2WGaitCurriculumPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:Go2WGaitCurriculumPPORunnerCfg"
        ),
    },
)

gym.register(
    id="OmniBot-Go2W-Gait-Flat-MLP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2W.gait_env_cfg:Go2WGaitEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.Go2W.gait_env_cfg:Go2WGaitPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:Go2WGaitMLPPPORunnerCfg",
    },
)

gym.register(
    id="OmniBot-Go2W-Flat-v0",
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

gym.register(
    id="OmniBot-Go2W-Flat-MLP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2W.flat_env_cfg:Go2WFlatEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.Go2W.flat_env_cfg:Go2WFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:Go2WFlatMLPPPORunnerCfg",
    },
)

gym.register(
    id="OmniBot-Go2W-Parkour-GoalGrid-MLP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2W.parkour_goal_grid_env_cfg:Go2WParkourGoalGridEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.Go2W.parkour_goal_grid_env_cfg:Go2WParkourGoalGridPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:Go2WParkourGoalGridMLPPPORunnerCfg"
        ),
    },
)

gym.register(
    id="OmniBot-Go2W-Handstand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2W.two_wheel_balance_env_cfg:Go2WHandstandEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.Go2W.two_wheel_balance_env_cfg:Go2WHandstandPlayEnvCfg",
        "rsl_rl_cfg_entry_point": ("locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:Go2WHandstandPPORunnerCfg"),
    },
)

gym.register(
    id="OmniBot-Go2W-Upright-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2W.two_wheel_balance_env_cfg:Go2WUprightEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.Go2W.two_wheel_balance_env_cfg:Go2WUprightPlayEnvCfg",
        "rsl_rl_cfg_entry_point": ("locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:Go2WUprightPPORunnerCfg"),
    },
)

gym.register(
    id="OmniBot-Go2W-Crawl-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2W.crawl_env_cfg:Go2WCrawlEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.Go2W.crawl_env_cfg:Go2WCrawlPlayEnvCfg",
        "rsl_rl_cfg_entry_point": ("locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:Go2WCrawlPPORunnerCfg"),
    },
)

gym.register(
    id="OmniBot-Go2W-Flat-DreamWaQ-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2W.dreamwaq_env_cfg:Go2WFlatDreamWaQEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.Go2W.dreamwaq_env_cfg:Go2WFlatDreamWaQPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:Go2WFlatDreamWaQPPORunnerCfg"
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
        "rsl_rl_cfg_entry_point": (
            "locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:Go2WRoughRnnDecoderPPORunnerCfg"
        ),
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

gym.register(
    id="OmniBot-Go2W-Rough-MLP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2W.rough_env_cfg:Go2WRoughEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.Go2W.rough_env_cfg:Go2WRoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:Go2WRoughMLPPPORunnerCfg",
    },
)

gym.register(
    id="OmniBot-Go2W-HighPlatform-MLP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2W.high_platform_env_cfg:Go2WHighPlatformEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.Go2W.high_platform_env_cfg:Go2WHighPlatformPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:Go2WHighPlatformMLPPPORunnerCfg"
        ),
    },
)

gym.register(
    id="OmniBot-Go2W-Rough-DreamWaQ-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2W.dreamwaq_env_cfg:Go2WRoughDreamWaQEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.Go2W.dreamwaq_env_cfg:Go2WRoughDreamWaQPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:Go2WRoughDreamWaQPPORunnerCfg"
        ),
    },
)

gym.register(
    id="OmniBot-Go2W-Recover-DreamWaQ-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2W.recover_env_cfg:Go2WRecoverDreamWaQEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.Go2W.recover_env_cfg:Go2WRecoverDreamWaQPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:Go2WRecoverDreamWaQPPORunnerCfg"
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
        "rsl_rl_cfg_entry_point": (
            "locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:Go2WFlatRnnDecoderPPORunnerCfg"
        ),
    },
)

# ==============================================================================
# WF_TRON1 Environments
# ==============================================================================

gym.register(
    id="OmniBot-WF-TRON1-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.WF_TRON1.flat_env_cfg:WFTRON1FlatEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.WF_TRON1.flat_env_cfg:WFTRON1FlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:WFTRON1FlatPPORunnerCfg",
    },
)

gym.register(
    id="OmniBot-WF-TRON1-Flat-MLP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.WF_TRON1.flat_env_cfg:WFTRON1FlatEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.WF_TRON1.flat_env_cfg:WFTRON1FlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:WFTRON1FlatPPORunnerCfg",
    },
)

gym.register(
    id="OmniBot-WF-TRON1-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.WF_TRON1.rough_env_cfg:WFTRON1RoughEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.WF_TRON1.rough_env_cfg:WFTRON1RoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:WFTRON1RoughPPORunnerCfg",
    },
)

gym.register(
    id="OmniBot-WF-TRON1-Rough-MLP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.WF_TRON1.rough_env_cfg:WFTRON1RoughEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.WF_TRON1.rough_env_cfg:WFTRON1RoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:WFTRON1RoughPPORunnerCfg",
    },
)

gym.register(
    id="OmniBot-WF-TRON1-Flat-DreamWaQ-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.WF_TRON1.dreamwaq_env_cfg:WFTRON1FlatDreamWaQEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.WF_TRON1.dreamwaq_env_cfg:WFTRON1FlatDreamWaQPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:WFTRON1FlatDreamWaQPPORunnerCfg"
        ),
    },
)

gym.register(
    id="OmniBot-WF-TRON1-Flat-Rnn-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.WF_TRON1.flat_env_cfg:WFTRON1FlatEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.WF_TRON1.flat_env_cfg:WFTRON1FlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:WFTRON1FlatRnnDecoderPPORunnerCfg"
        ),
    },
)

gym.register(
    id="OmniBot-WF-TRON1-Rough-DreamWaQ-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.WF_TRON1.dreamwaq_env_cfg:WFTRON1RoughDreamWaQEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.WF_TRON1.dreamwaq_env_cfg:WFTRON1RoughDreamWaQPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:WFTRON1RoughDreamWaQPPORunnerCfg"
        ),
    },
)

gym.register(
    id="OmniBot-WF-TRON1-Rough-Rnn-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.WF_TRON1.rough_env_cfg:WFTRON1RoughEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.WF_TRON1.rough_env_cfg:WFTRON1RoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "locolab.source.locomotion.tasks.agents.rsl_rl_ppo_cfg:WFTRON1RoughRnnDecoderPPORunnerCfg"
        ),
    },
)

# Scene task; navigation uses the dedicated train_planner.py runner rather than RSL entry points.
gym.register(
    id="BarrierLoco-Go2W-PlannerTerrain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.Go2W.planner_terrain_env_cfg:Go2WPlannerTerrainEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.Go2W.planner_terrain_env_cfg:Go2WPlannerTerrainPlayEnvCfg",
    },
)
