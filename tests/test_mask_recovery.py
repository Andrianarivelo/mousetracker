"""Tests for watershed-based and manual mask recovery."""

import numpy as np

from app.core.mask_recovery import split_mask_by_polygon, watershed_split


def _ellipse_mask(shape: tuple[int, int], center: tuple[int, int], rx: int, ry: int) -> np.ndarray:
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    cx, cy = center
    return (((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2) <= 1.0


def test_user_requested_watershed_can_force_two_way_split() -> None:
    """Manual split should still separate an elongated merged blob into two parts."""
    mask = _ellipse_mask((140, 220), (110, 70), 78, 30)
    total_area = int(mask.sum())

    splits = watershed_split(
        mask,
        min_area=int(total_area * 0.22),
        max_area=int(total_area * 0.65),
        expected_parts=2,
        allow_relaxed_area=True,
    )

    assert len(splits) == 2
    areas = sorted(int(part.sum()) for part in splits)
    assert areas[0] > int(total_area * 0.35)
    assert areas[1] > int(total_area * 0.35)
    assert int(np.logical_and(splits[0], splits[1]).sum()) == 0


def test_watershed_split_keeps_multi_peak_behavior_for_recovery() -> None:
    """Automatic recovery should still split a dumbbell-like merged mask."""
    shape = (160, 200)
    left = _ellipse_mask(shape, (70, 80), 28, 22)
    right = _ellipse_mask(shape, (130, 80), 28, 22)
    bridge = np.zeros(shape, dtype=bool)
    bridge[68:92, 85:115] = True
    mask = left | right | bridge

    splits = watershed_split(
        mask,
        min_area=1200,
        max_area=2600,
    )

    assert len(splits) == 2
    assert all(1200 <= int(part.sum()) <= 2600 for part in splits)


def test_split_mask_by_polygon_separates_selected_region() -> None:
    shape = (120, 180)
    left = _ellipse_mask(shape, (55, 60), 32, 24)
    right = _ellipse_mask(shape, (120, 60), 32, 24)
    bridge = np.zeros(shape, dtype=bool)
    bridge[50:70, 78:100] = True
    merged = left | right | bridge

    polygon = [(10, 20), (95, 20), (95, 100), (10, 100)]
    parts = split_mask_by_polygon(merged, polygon, min_area=800)

    assert len(parts) == 2
    left_overlap = int(np.logical_and(parts[0], left).sum())
    right_overlap = int(np.logical_and(parts[1], right).sum())
    assert left_overlap > 1500
    assert right_overlap > 1500
    assert int(np.logical_and(parts[0], parts[1]).sum()) == 0


def test_split_mask_by_polygon_rejects_tiny_selection() -> None:
    mask = _ellipse_mask((100, 100), (50, 50), 24, 18)
    tiny_polygon = [(48, 48), (52, 48), (52, 52), (48, 52)]

    parts = split_mask_by_polygon(mask, tiny_polygon, min_area=200)

    assert parts == []
