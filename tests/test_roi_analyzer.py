"""Tests for ROI analyzer."""

import pytest
from app.core.roi_analyzer import ROIAnalyzer, _point_in_polygon, _compute_bout_metrics


def test_point_in_polygon_inside():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert _point_in_polygon(5, 5, square)


def test_point_in_polygon_outside():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert not _point_in_polygon(15, 5, square)


def test_rectangle_roi():
    analyzer = ROIAnalyzer()
    analyzer.add_roi("box", "rectangle", [(0, 0, 100, 100)])
    assert analyzer.rois["box"].contains_point(50, 50)
    assert not analyzer.rois["box"].contains_point(150, 50)


def test_circle_roi():
    analyzer = ROIAnalyzer()
    analyzer.add_roi("circle", "circle", [(50, 50, 30)])
    assert analyzer.rois["circle"].contains_point(50, 50)
    assert not analyzer.rois["circle"].contains_point(90, 90)


def test_analyze_returns_dataframe():
    analyzer = ROIAnalyzer()
    analyzer.add_roi("zone", "rectangle", [(0, 0, 100, 100)])
    # Mouse 1 spends first 10 frames in zone, next 10 out
    trajs = {
        1: [(i, 50.0, 50.0) for i in range(10)] + [(i + 10, 150.0, 150.0) for i in range(10)]
    }
    df = analyzer.analyze(trajs, fps=25.0)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["mouse_id"] == 1
    assert row["n_entries"] == 1
    assert row["total_time_s"] == pytest.approx(10 / 25.0, rel=0.01)


def test_bout_metrics_multiple_entries():
    # In/out/in/out pattern
    occupancy = [True] * 5 + [False] * 3 + [True] * 4 + [False] * 2
    metrics = _compute_bout_metrics(occupancy, fps=10.0, total_frames=len(occupancy))
    assert metrics["n_entries"] == 2
    assert metrics["n_exits"] == 2
    assert metrics["total_time_s"] == pytest.approx(0.9, rel=0.01)
