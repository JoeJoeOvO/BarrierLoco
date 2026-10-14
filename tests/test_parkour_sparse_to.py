"""Sparse TO proposals preserve inactive modes and valid physical control schedules."""

import pytest
import torch
from test_parkour_leap_to import PARKOUR


def test_sparse_candidates_are_local_bounded_and_do_not_mutate_anchor():
    mean, std, low, high, mask = PARKOUR.parameter_distribution()
    anchor = mean.repeat(3, 1)
    saved = anchor.clone()
    active = mask.repeat(3, 1)
    proposals = PARKOUR.sparse_control_proposals(anchor, active, std.repeat(3, 1), 96, 7, low, high)
    assert proposals.shape == (3, 96, 84)
    assert torch.isfinite(proposals).all()
    assert (proposals >= low).all() and (proposals <= high).all()
    torch.testing.assert_close(anchor, saved)
    assert ((proposals - anchor[:, None]).abs() > 1e-8).sum(-1).max() <= 4
    torch.testing.assert_close(proposals[..., ~mask], anchor[:, None, ~mask].expand_as(proposals[..., ~mask]))
    assert (PARKOUR.knot_schedule(proposals.reshape(-1, 84)).diff() > 0).all()


def test_sparse_coordinates_cover_all_enabled_variables_across_iterations():
    mean, std, low, high, mask = PARKOUR.parameter_distribution()
    covered = torch.zeros(84, dtype=torch.bool)
    for iteration in range(8):
        p = PARKOUR.sparse_control_proposals(mean[None], mask[None], std[None], 96, iteration, low, high)
        covered |= ((p[0] - mean).abs() > 1e-8).any(0)
    torch.testing.assert_close(covered, mask)


def test_paired_drive_changes_keep_fold_and_recovery_timestamps():
    mean, std, low, high, mask = PARKOUR.parameter_distribution()
    p = PARKOUR.sparse_control_proposals(mean[None], mask[None], std[None], 96, 2, low, high)
    rows = torch.arange(48)
    selected = 48 + rows[rows % 3 == 2]
    knots = PARKOUR.knot_schedule(p[0, selected])
    original = PARKOUR.knot_schedule(mean[None])
    torch.testing.assert_close(knots[:, :, 3:], original[:, :, 3:].expand_as(knots[:, :, 3:]), rtol=0, atol=1e-6)


@pytest.mark.parametrize("count", [0, 1, 3])
def test_invalid_local_population_rejected(count):
    mean, std, low, high, mask = PARKOUR.parameter_distribution()
    with pytest.raises(ValueError, match="four proposals"):
        PARKOUR.sparse_control_proposals(mean[None], mask[None], std[None], count, 0, low, high)
