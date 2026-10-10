import torch

from locolab.source.locomotion.utils.platform_evaluation import WHEEL_RADIUS, PlatformProgress


def pose(x=2.25, z=0.6, y=0.0):
    root = torch.tensor([[x, y, z + 0.35]])
    wheels = torch.tensor(
        [[[x + dx, y + dy, z + WHEEL_RADIUS] for dx, dy in [(0.25, 0.2), (0.25, -0.2), (-0.25, 0.2), (-0.25, -0.2)]]]
    )
    return root, wheels, torch.full((1, 4), 50.0), torch.tensor([-1.0])


def update(tracker, state, count, terminated=False, truncated=False):
    for i in range(count):
        tracker.update(*state, torch.tensor([terminated]), torch.tensor([truncated]), (i + 1) * 0.02)


def test_requires_mount_then_stable_landing():
    tracker = PlatformProgress(torch.tensor([0.6]))
    update(tracker, pose(), 4)
    assert not tracker.mounted.item()
    update(tracker, pose(), 1)
    assert tracker.mounted.item()
    update(tracker, pose(x=3.5, z=0.0), 9)
    assert not tracker.passed.item()
    update(tracker, pose(x=3.5, z=0.0), 1)
    assert tracker.passed.item()
    tracker.finish()


def test_bypass_does_not_count():
    tracker = PlatformProgress(torch.tensor([0.6]))
    update(tracker, pose(x=3.5, z=0.0), 20)
    assert not tracker.mounted.item() and not tracker.passed.item()
    tracker.finish()
    assert tracker.timed_out.item()


def test_no_success_after_reset_or_corridor_exit():
    for reason in ["termination", "timeout", "corridor"]:
        tracker = PlatformProgress(torch.tensor([0.6]))
        update(
            tracker,
            pose(y=0.8 if reason == "corridor" else 0.0),
            1,
            terminated=reason == "termination",
            truncated=reason == "timeout",
        )
        update(tracker, pose(), 10)
        update(tracker, pose(x=3.5, z=0.0), 20)
        assert not tracker.mounted.item() and not tracker.passed.item()
        tracker.finish()


def test_wrong_height_or_unsupported_airborne_is_not_mount():
    for state in [pose(z=0.4), pose(z=0.8), (*pose()[:2], torch.zeros((1, 4)), pose()[3])]:
        tracker = PlatformProgress(torch.tensor([0.6]))
        update(tracker, state, 20)
        assert not tracker.mounted.item()


def test_only_front_wheels_on_platform_is_not_mount():
    tracker = PlatformProgress(torch.tensor([0.6]))
    update(tracker, pose(x=1.55), 20)
    assert not tracker.mounted.item()
