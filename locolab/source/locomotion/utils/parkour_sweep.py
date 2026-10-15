"""Deterministic inspection cases; keep each motion on its own collision tile."""


def inspection_cases(motion_ids, locomotion_only):
    """One jump environment per reference, at each of four supported speeds.

    Begin with 1 m/s, matching reference generation. No independent lateral/yaw
    commands or invented height/landing combinations are introduced.
    """
    if len(motion_ids) != len(locomotion_only):
        raise ValueError("Motion IDs and rolling masks must have equal lengths")
    selected = {}
    for index, (motion, rolling) in enumerate(zip(motion_ids, locomotion_only)):
        if not rolling:
            selected.setdefault(motion, index)
    return [(motion, speed, selected[motion]) for speed in (1.0, 0.5, 0.2, 1.5) for motion in sorted(selected)]
