"""Ensure inspection never assigns obstacles to rolling lanes or drops motions."""

import importlib.util
from pathlib import Path

import torch
from test_parkour_reference_bank import CORE


def test_sweep_covers_all_reference_speed_pairs_on_matching_tiles():
    path = Path(__file__).resolve().parents[1] / "locolab/source/locomotion/utils/parkour_sweep.py"
    spec = importlib.util.spec_from_file_location("parkour_sweep", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bank = CORE.ParkourReferenceBank(CORE.GOAL_GRID_BANK_PATH)
    motions, rolling = bank.layout(torch.arange(100), 3)
    cases = module.inspection_cases(motions.tolist(), rolling.tolist())
    assert len(cases) == 120
    assert {(m, s) for m, s, _ in cases} == {(m, s) for m in range(30) for s in (0.2, 0.5, 1.0, 1.5)}
    for motion, _, index in cases:
        assert not rolling[index] and motions[index] == motion
