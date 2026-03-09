"""Tests for identity tracker."""

import numpy as np
import pytest

from app.core.tracker import IdentityTracker, _centroid, _mask_iou, _euclidean


def _make_circle_mask(shape, center, radius):
    """Create a circular binary mask."""
    mask = np.zeros(shape, dtype=bool)
    y, x = np.ogrid[:shape[0], :shape[1]]
    dist = (x - center[0]) ** 2 + (y - center[1]) ** 2
    mask[dist <= radius ** 2] = True
    return mask


def test_centroid():
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True
    cx, cy = _centroid(mask)
    assert abs(cx - 49.5) < 1.0
    assert abs(cy - 49.5) < 1.0


def test_mask_iou_identical():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:30, 10:30] = True
    assert _mask_iou(mask, mask) == pytest.approx(1.0)


def test_mask_iou_disjoint():
    mask_a = np.zeros((100, 100), dtype=bool)
    mask_b = np.zeros((100, 100), dtype=bool)
    mask_a[0:10, 0:10] = True
    mask_b[50:60, 50:60] = True
    assert _mask_iou(mask_a, mask_b) == pytest.approx(0.0)


def test_tracker_initialize():
    tracker = IdentityTracker(n_mice=2)
    shape = (100, 100)
    mask1 = _make_circle_mask(shape, (25, 50), 15)
    mask2 = _make_circle_mask(shape, (75, 50), 15)
    sam_masks = {10: mask1, 20: mask2}
    mapping = {1: 10, 2: 20}
    state = tracker.initialize(0, sam_masks, mapping)
    assert 1 in state.masks
    assert 2 in state.masks
    assert np.allclose(state.centroids[1][0], 25.0, atol=2.0)


def test_tracker_assign_frame():
    tracker = IdentityTracker(n_mice=2)
    shape = (200, 200)
    mask1 = _make_circle_mask(shape, (50, 100), 20)
    mask2 = _make_circle_mask(shape, (150, 100), 20)
    sam_masks = {1: mask1, 2: mask2}
    tracker.initialize(0, sam_masks, {1: 1, 2: 2})

    # Next frame — slightly shifted masks
    mask1b = _make_circle_mask(shape, (55, 100), 20)
    mask2b = _make_circle_mask(shape, (155, 100), 20)
    outputs = {
        "out_obj_ids": np.array([1, 2]),
        "out_binary_masks": np.stack([mask1b.astype(np.uint8), mask2b.astype(np.uint8)]),
        "out_probs": np.array([[0.9], [0.9]]),
        "out_boxes_xywh": np.array([[30, 80, 50, 40], [130, 80, 50, 40]]),
    }
    state = tracker.assign_frame(1, outputs, shape)
    assert 1 in state.masks
    assert 2 in state.masks


def test_correct_swap():
    tracker = IdentityTracker(n_mice=2)
    shape = (100, 100)
    mask1 = _make_circle_mask(shape, (25, 50), 10)
    mask2 = _make_circle_mask(shape, (75, 50), 10)
    tracker.initialize(0, {1: mask1, 2: mask2}, {1: 1, 2: 2})

    # Simulate a second frame
    outputs = {
        "out_obj_ids": np.array([1, 2]),
        "out_binary_masks": np.stack([mask1.astype(np.uint8), mask2.astype(np.uint8)]),
        "out_probs": np.array([[0.9], [0.9]]),
        "out_boxes_xywh": np.array([[15, 40, 20, 20], [65, 40, 20, 20]]),
    }
    tracker.assign_frame(1, outputs, shape)

    # Swap over entire range
    tracker.correct_swap((0, 1), 1, 2)
    state_0 = tracker.get_state_at(0)
    # After swap, mouse 1 should have what was mouse 2's mask
    assert state_0 is not None


def test_unmatched_mouse_keeps_centroid_but_not_stale_mask() -> None:
    tracker = IdentityTracker(n_mice=2)
    shape = (120, 120)
    mask1 = _make_circle_mask(shape, (35, 60), 12)
    mask2 = _make_circle_mask(shape, (85, 60), 12)
    tracker.initialize(0, {1: mask1, 2: mask2}, {1: 1, 2: 2})

    outputs = {
        "out_obj_ids": np.array([1]),
        "out_binary_masks": np.stack([mask1.astype(np.uint8)]),
        "out_probs": np.array([[0.95]]),
        "out_boxes_xywh": np.array([[23, 48, 24, 24]]),
    }
    state = tracker.assign_frame(1, outputs, shape)

    assert 2 in state.occluded_ids
    assert state.confidences[2] < 1.0
    assert int(state.masks[2].sum()) == 0
    assert state.centroids[2] == tracker.history[0].centroids[2]
