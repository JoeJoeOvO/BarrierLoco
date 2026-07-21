from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CLI_ARGS = ROOT / "locolab/scripts/locomotion/cli_args.py"


def _load_cli_args():
    spec = importlib.util.spec_from_file_location("runner_cli_args_under_test", CLI_ARGS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resume_experiment_is_independent_from_output_experiment() -> None:
    cli_args = _load_cli_args()
    cfg = SimpleNamespace(
        experiment_name="output-tree",
        resume_experiment_name="source-tree",
        to_dict=lambda: {
            "experiment_name": "output-tree",
            "resume_experiment_name": "source-tree",
            "seed": 42,
        },
    )

    assert cli_args.get_output_experiment_name(cfg) == "output-tree"
    assert cli_args.get_resume_experiment_name(cfg) == "source-tree"
    assert cli_args.runner_cfg_to_dict(cfg) == {"experiment_name": "output-tree", "seed": 42}
    assert cli_args.runner_cfg_to_dict(cfg, validate_finite=True) == {
        "experiment_name": "output-tree",
        "seed": 42,
        "validate_finite": True,
    }


def test_resume_experiment_falls_back_to_output_tree() -> None:
    cli_args = _load_cli_args()
    cfg = SimpleNamespace(experiment_name="single-tree", resume_experiment_name=None)

    assert cli_args.get_resume_experiment_name(cfg) == "single-tree"


def test_cli_overrides_output_experiment_and_non_distributed_runner_device() -> None:
    cli_args = _load_cli_args()
    cfg = SimpleNamespace(
        seed=1,
        resume=False,
        load_run=".*",
        load_checkpoint="model_.*.pt",
        experiment_name="old-tree",
        resume_experiment_name=None,
        run_name="",
        logger="tensorboard",
        wandb_group=None,
        device="cuda:0",
    )
    args = Namespace(
        seed=None,
        resume=False,
        load_run=None,
        checkpoint=None,
        experiment_name="new-tree",
        resume_experiment_name=None,
        run_name=None,
        logger=None,
        log_project_name=None,
        wandb_entity=None,
        wandb_group=None,
        task="OmniBot-Go2W-Rough-MLP-v0",
        device="cuda:2",
    )

    cli_args.update_rsl_rl_cfg(cfg, args)

    assert cfg.experiment_name == "new-tree"
    assert cfg.device == "cuda:2"
