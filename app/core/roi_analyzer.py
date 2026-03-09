"""ROI time-in-zone analysis."""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ROIDefinition:
    name: str
    roi_type: str  # "polygon", "circle", "rectangle"
    # For polygon: list of (x, y) vertices
    # For circle: [(cx, cy, radius)]
    # For rectangle: [(x1, y1, x2, y2)]
    data: list[Any] = field(default_factory=list)
    color: tuple[int, int, int] = (255, 255, 0)

    def contains_point(self, x: float, y: float) -> bool:
        """Return True if point (x, y) is inside this ROI."""
        if self.roi_type == "polygon":
            return _point_in_polygon(x, y, self.data)
        elif self.roi_type == "circle":
            cx, cy, r = self.data[0]
            return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        elif self.roi_type == "rectangle":
            x1, y1, x2, y2 = self.data[0]
            return x1 <= x <= x2 and y1 <= y <= y2
        return False

    def contains_mask(self, mask: np.ndarray) -> bool:
        """Return True if the mask centroid is inside this ROI."""
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return False
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        return self.contains_point(cx, cy)


def _point_in_polygon(x: float, y: float, vertices: list[tuple]) -> bool:
    """Ray casting algorithm for point-in-polygon test."""
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


class ROIAnalyzer:
    """
    Analyze time spent in user-drawn regions of interest.

    ROI types: polygon, circle, rectangle
    Metrics: time_in_zone, n_entries, n_exits, first_entry_latency,
             mean_bout_duration, fraction_of_total_time
    """

    def __init__(self) -> None:
        self.rois: dict[str, ROIDefinition] = {}

    def add_roi(
        self,
        name: str,
        roi_type: str,
        data: list[Any],
        color: tuple[int, int, int] = (255, 255, 0),
    ) -> None:
        """Add or replace an ROI definition."""
        self.rois[name] = ROIDefinition(
            name=name, roi_type=roi_type, data=data, color=color
        )
        logger.info(f"Added ROI '{name}' ({roi_type})")

    def remove_roi(self, name: str) -> None:
        self.rois.pop(name, None)

    def clear(self) -> None:
        self.rois.clear()

    def analyze(
        self,
        trajectories: dict[int, list[tuple[int, float, float]]],
        fps: float,
    ) -> pd.DataFrame:
        """
        Compute per-mouse per-ROI metrics.

        Args:
            trajectories: {mouse_id: [(frame_idx, cx, cy), ...]}
            fps: Video frame rate.

        Returns:
            DataFrame with columns:
              mouse_id, roi_name, total_time_s, n_entries, n_exits,
              first_entry_latency_s, mean_bout_duration_s, fraction_of_time
        """
        if not trajectories or not self.rois:
            return pd.DataFrame()

        records = []
        for mouse_id, traj in trajectories.items():
            if not traj:
                continue
            total_frames = len(traj)
            for roi_name, roi in self.rois.items():
                occupancy = [roi.contains_point(cx, cy) for _, cx, cy in traj]
                metrics = _compute_bout_metrics(occupancy, fps, total_frames)
                records.append(
                    dict(
                        mouse_id=mouse_id,
                        roi_name=roi_name,
                        **metrics,
                    )
                )

        return pd.DataFrame(records)

    def get_occupancy_per_frame(
        self,
        trajectories: dict[int, list[tuple[int, float, float]]],
    ) -> dict[str, dict[int, list[bool]]]:
        """
        Return {roi_name: {mouse_id: [bool per frame]}} occupancy arrays.
        Useful for CSV export.
        """
        result: dict[str, dict] = {}
        for roi_name, roi in self.rois.items():
            result[roi_name] = {}
            for mouse_id, traj in trajectories.items():
                result[roi_name][mouse_id] = [
                    roi.contains_point(cx, cy) for _, cx, cy in traj
                ]
        return result


def _compute_bout_metrics(
    occupancy: list[bool],
    fps: float,
    total_frames: int,
) -> dict:
    """Compute bout-level statistics from a binary occupancy sequence."""
    n = len(occupancy)
    if n == 0:
        return dict(
            total_time_s=0.0, n_entries=0, n_exits=0,
            first_entry_latency_s=-1.0, mean_bout_duration_s=0.0,
            fraction_of_time=0.0,
        )

    in_zone = False
    n_entries = 0
    n_exits = 0
    bout_start = -1
    bout_durations: list[float] = []
    first_entry_frame = -1

    for i, occ in enumerate(occupancy):
        if occ and not in_zone:
            # Entry
            n_entries += 1
            in_zone = True
            bout_start = i
            if first_entry_frame < 0:
                first_entry_frame = i
        elif not occ and in_zone:
            # Exit
            n_exits += 1
            in_zone = False
            duration = (i - bout_start) / fps
            bout_durations.append(duration)

    # Close any open bout at end
    if in_zone and bout_start >= 0:
        duration = (n - bout_start) / fps
        bout_durations.append(duration)

    total_time_s = sum(bout_durations)
    mean_bout = float(np.mean(bout_durations)) if bout_durations else 0.0
    latency = first_entry_frame / fps if first_entry_frame >= 0 else -1.0
    fraction = total_time_s / (total_frames / fps) if fps > 0 and total_frames > 0 else 0.0

    return dict(
        total_time_s=round(total_time_s, 3),
        n_entries=n_entries,
        n_exits=n_exits,
        first_entry_latency_s=round(latency, 3),
        mean_bout_duration_s=round(mean_bout, 3),
        fraction_of_time=round(fraction, 4),
    )
