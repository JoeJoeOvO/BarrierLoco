from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "locolab/source/locomotion/tasks/__init__.py"
RUNNERS = ROOT / "locolab/source/locomotion/tasks/agents/rsl_rl_ppo_cfg.py"


def test_only_supported_go2w_networks_are_registered() -> None:
    source = TASKS.read_text()
    runner_source = RUNNERS.read_text()
    expected_classes = {"ActorCritic", "ActorCriticDreamWaQ", "ActorCriticRecurrent"}
    assert runner_source.count('class_name: str = "ActorCritic') == len(expected_classes)
    for class_name in expected_classes:
        assert f'class_name: str = "{class_name}"' in runner_source
    assert "-DSR-" not in source
    assert "ActorCriticDSR" not in runner_source


def test_dreamwaq_go2w_configs_use_log_std_and_mask_unpredictable_targets() -> None:
    source = RUNNERS.read_text()
    go2w_dreamwaq = source[source.index("class Go2WDreamWaQPPORunnerCfg") :]
    assert 'policy.noise_std_type = "log"' in go2w_dreamwaq
    expected_mask = "[6, 7, 8, *range(37, 53)]"
    assert go2w_dreamwaq.count(expected_mask) == 2


def test_recent_rough_network_configs_preserve_training_and_checkpoint_contracts() -> None:
    source = RUNNERS.read_text()
    rough_mlp = source[source.index("class Go2WRoughMLPPPORunnerCfg") :]
    recurrent = source[
        source.index("class RecurrentDecoderPPORunnerCfg") : source.index("class Go2WRecurrentDecoderPPORunnerCfg")
    ]
    go2w_recurrent = source[source.index("class Go2WRecurrentDecoderPPORunnerCfg") :]
    rough_rnn = source[source.index("class Go2WRoughRnnDecoderPPORunnerCfg") :]
    dreamwaq = source[source.index("class DreamWaQPPORunnerCfg") : source.index("class Go2WDreamWaQPPORunnerCfg")]
    go2w_dreamwaq = source[source.index("class Go2WDreamWaQPPORunnerCfg") :]

    assert "self.algorithm.learning_rate = 3.0e-4" in rough_mlp[:500]
    assert "learning_rate=1.0e-4" in recurrent
    assert "algorithm.decoder_cfg.actor.learning_rate = 3.0e-4" in go2w_recurrent[:1000]
    assert "algorithm.decoder_cfg.critic.learning_rate = 3.0e-4" in go2w_recurrent[:1000]
    assert "policy.rnn_hidden_dim = 256" in rough_rnn[:700]
    assert "algorithm.decoder_cfg.critic = None" in rough_rnn[:700]
    assert "learning_rate=1.0e-3" in dreamwaq
    assert "vae_learning_rate=1.0e-3" in dreamwaq
    assert "algorithm.learning_rate = 3.0e-4" in go2w_dreamwaq[:1000]
    assert "algorithm.vae_learning_rate = 3.0e-4" in go2w_dreamwaq[:1000]


def test_go2w_network_overrides_do_not_change_shared_wf_defaults() -> None:
    source = RUNNERS.read_text()
    recurrent = source[
        source.index("class RecurrentDecoderPPORunnerCfg") : source.index("class Go2WRecurrentDecoderPPORunnerCfg")
    ]
    dreamwaq = source[source.index("class DreamWaQPPORunnerCfg") : source.index("class Go2WDreamWaQPPORunnerCfg")]

    assert "empirical_normalization = True" in recurrent
    assert "learning_rate=1.0e-4" in recurrent
    assert "empirical_normalization = False" in dreamwaq
    assert "actor_obs_normalization=True" in dreamwaq
    assert "critic_obs_normalization=True" in dreamwaq
    assert 'noise_std_type="log"' not in dreamwaq
