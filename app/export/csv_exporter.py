"""CSV export: trajectory + keypoints + ROI occupancy."""

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def export_csv(
    tracker_history,
    output_path: str,
    fps: float,
    keypoints_by_frame: Optional[dict] = None,
    roi_occupancy: Optional[dict] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> str:
    """
    Export trajectories to CSV.

    Columns: frame, timestamp_s, mouse_id, centroid_x, centroid_y,
             bbox_x1, bbox_y1, bbox_x2, bbox_y2,
             [kp_name_x, kp_name_y, ...],
             [roi_name_occupied, ...]

    Returns output path.
    """
    records = []
    n_frames = len(tracker_history)

    for i, state in enumerate(tracker_history):
        fi = state.frame_idx
        ts = fi / fps if fps > 0 else 0.0

        for mouse_id in sorted(state.masks.keys()):
            cx, cy = state.centroids.get(mouse_id, (0.0, 0.0))
            x1, y1, x2, y2 = state.bboxes.get(mouse_id, (0, 0, 0, 0))

            row = dict(
                frame=fi,
                timestamp_s=round(ts, 4),
                mouse_id=mouse_id,
                centroid_x=round(cx, 2),
                centroid_y=round(cy, 2),
                bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
                confidence=round(state.confidences.get(mouse_id, 0.0), 3),
            )

            # Keypoints
            if keypoints_by_frame:
                kps = keypoints_by_frame.get(fi, {}).get(mouse_id, {})
                for kp_name, (kx, ky) in kps.items():
                    row[f"{kp_name}_x"] = round(kx, 2)
                    row[f"{kp_name}_y"] = round(ky, 2)

            # ROI occupancy
            if roi_occupancy:
                for roi_name, mouse_occ in roi_occupancy.items():
                    occ_list = mouse_occ.get(mouse_id, [])
                    # Find this frame's occupancy
                    occ_val = False
                    if occ_list:
                        # occ_list indexed by trajectory order, not frame_idx
                        # Use frame index if available
                        occ_val = occ_list[i] if i < len(occ_list) else False
                    row[f"roi_{roi_name}"] = int(occ_val)

            records.append(row)

        if progress_callback and n_frames > 0:
            progress_callback(int((i + 1) * 100 / n_frames))

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    logger.info(f"CSV exported: {output_path} ({len(records)} rows)")
    return output_path
