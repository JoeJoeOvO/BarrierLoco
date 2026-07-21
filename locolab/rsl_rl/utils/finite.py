"""Finite-value guards used by short training and rollout smoke tests."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import torch


def assert_finite_tree(value: Any, name: str) -> None:
    """Raise when a nested tensor/numeric structure contains NaN or Inf."""
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{name} contains NaN or Inf.")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite_tree(item, f"{name}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            assert_finite_tree(item, f"{name}[{index}]")
        return
    if isinstance(value, Real) and not math.isfinite(float(value)):
        raise FloatingPointError(f"{name} contains NaN or Inf.")
