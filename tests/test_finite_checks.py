from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
FINITE_CHECKS = ROOT / "locolab/rsl_rl/utils/finite.py"


def _load_finite_checks():
    spec = importlib.util.spec_from_file_location("rsl_rl_finite_checks_under_test", FINITE_CHECKS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_assert_finite_tree_accepts_nested_training_values() -> None:
    finite = _load_finite_checks()
    finite.assert_finite_tree(
        {
            "observations": {"policy": torch.tensor([[1.0, 2.0]])},
            "actions": torch.zeros(1, 2),
            "losses": {"value": 0.25, "auxiliary": [torch.tensor(0.5)]},
        },
        "training",
    )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), torch.tensor([0.0, float("-inf")])])
def test_assert_finite_tree_rejects_non_finite_values(invalid) -> None:
    finite = _load_finite_checks()
    with pytest.raises(FloatingPointError, match=r"training\.loss"):
        finite.assert_finite_tree({"loss": invalid}, "training")
