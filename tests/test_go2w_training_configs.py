from __future__ import annotations

import ast
import runpy
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "locolab/source/locomotion/tasks/__init__.py"
RUNNERS = ROOT / "locolab/source/locomotion/tasks/agents/rsl_rl_ppo_cfg.py"


def test_registered_go2w_tasks_resolve_standard_training_configs(monkeypatch) -> None:
    registrations = {}
    gym = ModuleType("gymnasium")

    def register(*, id, **spec):
        assert id not in registrations
        registrations[id] = spec

    gym.register = register
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    runpy.run_path(str(TASKS), run_name="locolab.source.locomotion.tasks")

    go2w_tasks = {task_id: spec for task_id, spec in registrations.items() if task_id.startswith("OmniBot-Go2W-")}
    assert "OmniBot-Go2W-Gait-Flat-MLP-v0" in go2w_tasks
    assert "OmniBot-Go2W-MultiSkill-Flat-MLP-v0" not in go2w_tasks
    for task_id, spec in go2w_tasks.items():
        assert set(spec["kwargs"]) == {
            "env_cfg_entry_point",
            "play_env_cfg_entry_point",
            "rsl_rl_cfg_entry_point",
        }, task_id
        for entry_point in spec["kwargs"].values():
            module_name, class_name = entry_point.split(":")
            path = ROOT.joinpath(*module_name.split(".")).with_suffix(".py")
            assert path.is_file(), (task_id, entry_point)
            classes = {node.name for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef)}
            assert class_name in classes, (task_id, entry_point)


def test_gait_runner_and_showcase_use_the_gait_names() -> None:
    runner = next(
        node
        for node in ast.parse(RUNNERS.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "Go2WGaitMLPPPORunnerCfg"
    )
    assert 'self.experiment_name = "omnibot_go2w_gait_flat_mlp_v0"' in ast.get_source_segment(
        RUNNERS.read_text(), runner
    )
    showcase = ROOT / "locolab/scripts/locomotion/record_wtw_showcase.py"
    task_args = [
        node
        for node in ast.walk(ast.parse(showcase.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--task"
    ]
    assert len(task_args) == 1
    default = next(keyword.value for keyword in task_args[0].keywords if keyword.arg == "default")
    assert ast.literal_eval(default) == "OmniBot-Go2W-Gait-Flat-MLP-v0"


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
    rough_mlp = source[
        source.index("class Go2WRoughMLPPPORunnerCfg") : source.index("class Go2WHighPlatformMLPPPORunnerCfg")
    ]
    recurrent = source[
        source.index("class RecurrentDecoderPPORunnerCfg") : source.index("class Go2WRecurrentDecoderPPORunnerCfg")
    ]
    go2w_recurrent = source[source.index("class Go2WRecurrentDecoderPPORunnerCfg") :]
    rough_rnn = source[source.index("class Go2WRoughRnnDecoderPPORunnerCfg") :]
    dreamwaq = source[source.index("class DreamWaQPPORunnerCfg") : source.index("class Go2WDreamWaQPPORunnerCfg")]
    go2w_dreamwaq = source[source.index("class Go2WDreamWaQPPORunnerCfg") :]

    assert "self.algorithm.learning_rate = 3.0e-4" in rough_mlp[:500]
    assert 'self.policy.noise_std_type = "log"' in rough_mlp
    assert "self.algorithm.entropy_coef = 0.008" in rough_mlp
    rough_dreamwaq = source[
        source.index("class Go2WRoughDreamWaQPPORunnerCfg") : source.index("class Go2WRecoverDreamWaQPPORunnerCfg")
    ]
    assert "self.algorithm.entropy_coef = 0.008" in rough_dreamwaq
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
