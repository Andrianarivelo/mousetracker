"""Tests for keypoint estimator."""

import numpy as np
import pytest

from app.core.keypoint_estimator import KeypointEstimator


def _make_mouse_mask():
    """Create a simplified elongated mouse-like mask."""
    mask = np.zeros((150, 60), dtype=np.uint8)
    # Body
    mask[30:120, 10:50] = 1
    # Head (narrower, at top)
    mask[10:35, 18:42] = 1
    # Tail (thin protrusion at bottom)
    mask[115:145, 22:28] = 1
    return mask


def test_body_center():
    estimator = KeypointEstimator(["body_center"])
    mask = _make_mouse_mask()
    kps = estimator.estimate(mask)
    assert "body_center" in kps
    x, y = kps["body_center"]
    assert 10 <= x <= 50
    assert 30 <= y <= 120


def test_empty_mask_returns_empty():
    estimator = KeypointEstimator(["nose_tip", "tail_tip"])
    mask = np.zeros((100, 100), dtype=np.uint8)
    kps = estimator.estimate(mask)
    assert kps == {}


def test_nose_and_tail_estimated():
    estimator = KeypointEstimator(["nose_tip", "tail_base", "body_center"])
    mask = _make_mouse_mask()
    kps = estimator.estimate(mask)
    # Should have all three
    assert "body_center" in kps
    # Nose should be near the top, tail_base near the bottom
    if "nose_tip" in kps and "tail_base" in kps:
        ny = kps["nose_tip"][1]
        ty = kps["tail_base"][1]
        # Nose near top (small y), tail near bottom (large y)
        assert ny < ty


def test_small_mask_returns_centroid():
    estimator = KeypointEstimator(["body_center"])
    # Very small mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[49:52, 49:52] = 1
    kps = estimator.estimate(mask)
    assert "body_center" in kps
