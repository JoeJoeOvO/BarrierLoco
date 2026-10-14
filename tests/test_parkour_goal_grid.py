"""Goal/phase alignment and balanced sampling contracts for variable-size banks."""

import copy
import gzip
import importlib.util
import itertools
import json
from pathlib import Path

import pytest
import torch
from test_parkour_reference_bank import CORE

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def builder(monkeypatch):
    scripts = ROOT / "locolab/scripts/locomotion"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "grid_builder_under_test", scripts / "build_parkour_goal_grid_bank.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grid_entries():
    return [
        dict(id=f"grid_{i}", requested_goal=dict(height=h, landing_x=x, landing_y=y))
        for i, (h, x, y) in enumerate(itertools.product((0.15, 0.20, 0.25), (0.4, 0.6, 0.8), (-0.15, 0, 0.15)))
    ]


def test_full_grid_has_independent_height_and_xy_coverage(builder):
    coverage = builder.coverage_report(grid_entries())
    assert coverage["training_ready"]
    assert coverage["accepted_per_height"] == {"0.15": 9, "0.2": 9, "0.25": 9}
    assert len(coverage["xy_cells_with_multiple_heights"]) == 9


def test_one_height_or_unpaired_goals_do_not_enable_training(builder):
    assert not builder.coverage_report(grid_entries()[:9])["training_ready"]
    sparse = [e for e in grid_entries() if e["requested_goal"]["landing_x"] == 0.6]
    assert not builder.coverage_report(sparse)["training_ready"]


def test_balanced_layout_keeps_rolling_on_flat_geometry(builder):
    schedule, weights = builder.balanced_sampling([{}] * 7 + grid_entries())
    assert sum(weights[:7]) == pytest.approx(0.25)
    assert [sum(weights[7 + i * 9 : 7 + (i + 1) * 9]) for i in range(3)] == pytest.approx([0.25] * 3)
    ids = torch.arange(len(schedule) * 10)
    flat = (0, 1, 2, *range(7, 34))
    motion, rolling = CORE.environment_layout(ids, 3, schedule, flat)
    assert rolling.float().mean() == pytest.approx(0.3)
    assert set(motion[rolling].tolist()).issubset(flat)
    counts = torch.bincount(motion[~rolling], minlength=34).float()
    torch.testing.assert_close(counts / counts.sum(), torch.tensor(weights))
    assert set(motion[~rolling].tolist()) == set(range(34))


def test_height_balance_survives_rejected_xy_cells(builder):
    selected = [e for i, e in enumerate(grid_entries()) if i not in (0, 1, 10, 22)]
    schedule, weights = builder.balanced_sampling([{}] * 7 + selected)
    for height in (0.15, 0.20, 0.25):
        ids = [i + 7 for i, e in enumerate(selected) if e["requested_goal"]["height"] == height]
        assert sum(weights[i] for i in ids) == pytest.approx(0.25)
    assert set(schedule) == set(range(len(selected) + 7))


def test_flat_resampling_never_moves_obstacles_or_rolling_lanes():
    bank = CORE.ParkourReferenceBank()
    motion, rolling = bank.layout(torch.arange(10000), 3)
    old = motion.clone()
    torch.manual_seed(23)
    sampled = bank.resample_flat_motions(motion, rolling)
    fixed = ~bank.is_flat[motion] | rolling
    torch.testing.assert_close(sampled[fixed], old[fixed])
    assert bank.is_flat[sampled[~fixed]].all()
    assert (sampled[~fixed] != old[~fixed]).any()
    torch.testing.assert_close(motion, old)


def test_unaccepted_new_reference_cannot_enter_a_grid_bank(tmp_path):
    with gzip.open(CORE.BANK_PATH, "rt") as stream:
        payload = json.load(stream)
    import hashlib

    payload["schema_version"] = 2
    payload["base_bank_sha256"] = hashlib.sha256(CORE.BANK_PATH.read_bytes()).hexdigest()
    extra = copy.deepcopy(payload["entries"][0])
    extra["id"] = "new_unaccepted_goal"
    payload["entries"].append(extra)
    p = tmp_path / "unaccepted.json.gz"
    p.write_bytes(gzip.compress(json.dumps(payload).encode()))
    with pytest.raises(ValueError, match="independently accepted"):
        CORE.ParkourReferenceBank(p)


def test_baseline_bank_keeps_original_shape_and_layout():
    bank = CORE.ParkourReferenceBank()
    assert bank.descriptors.shape == (7, 3)
    ids = torch.arange(100)
    old = CORE.environment_layout(ids, 3)
    actual = bank.layout(ids, 3)
    for a, b in zip(old, actual):
        torch.testing.assert_close(a, b)
