"""Tests for video I/O utilities."""

import numpy as np
import pytest

from app.core.video_io import compose_mask_overlay


def test_mask_overlay_basic():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :] = [128, 128, 128]  # gray

    mask = np.zeros((100, 100), dtype=bool)
    mask[20:60, 20:60] = True

    identity_colors = {1: (0, 200, 255)}
    result = compose_mask_overlay(frame, {1: mask}, identity_colors, alpha=0.4)

    assert result.shape == (100, 100, 3)
    assert result.dtype == np.uint8

    # Pixels inside mask should be blended
    inside = result[30, 30]
    # 0.6 * 128 + 0.4 * 0 = 76.8 for R
    # 0.6 * 128 + 0.4 * 200 = 156.8 for G
    # 0.6 * 128 + 0.4 * 255 = 178.8 for B
    assert abs(int(inside[0]) - 77) <= 2
    assert abs(int(inside[1]) - 157) <= 2
    assert abs(int(inside[2]) - 179) <= 2


def test_mask_overlay_outside_unchanged():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :] = [128, 128, 128]
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:60, 20:60] = True

    result = compose_mask_overlay(frame, {1: mask}, {1: (0, 200, 255)}, alpha=0.4)
    # Outside mask should be unchanged
    outside = result[10, 10]
    assert all(outside == 128)


def test_mask_overlay_multiple_mice():
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 100
    mask1 = np.zeros((100, 100), dtype=bool)
    mask1[:50, :] = True
    mask2 = np.zeros((100, 100), dtype=bool)
    mask2[50:, :] = True

    colors = {1: (255, 0, 0), 2: (0, 0, 255)}
    result = compose_mask_overlay(frame, {1: mask1, 2: mask2}, colors, alpha=0.5)

    # Top half should be reddish
    assert result[25, 50, 0] > result[25, 50, 2]
    # Bottom half should be bluish
    assert result[75, 50, 2] > result[75, 50, 0]
