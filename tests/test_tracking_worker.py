"""Tests for TrackingWorker mask comparisons."""

import numpy as np

from app.workers.tracking_worker import _mask_dicts_equal


def test_mask_dicts_equal_handles_numpy_masks_without_ambiguity() -> None:
    left = {
        1: np.array([[True, False], [False, True]], dtype=bool),
        2: np.array([[False, True], [True, False]], dtype=bool),
    }
    right = {
        1: np.array([[True, False], [False, True]], dtype=bool),
        2: np.array([[False, True], [True, False]], dtype=bool),
    }

    assert _mask_dicts_equal(left, right) is True


def test_mask_dicts_equal_detects_mask_changes() -> None:
    left = {1: np.array([[True, False], [False, True]], dtype=bool)}
    right = {1: np.array([[True, True], [False, True]], dtype=bool)}

    assert _mask_dicts_equal(left, right) is False
