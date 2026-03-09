"""Overlay video export with colored masks, bounding boxes, and keypoints."""

import logging
from typing import Callable, Optional

import cv2
import numpy as np

from app.config import IDENTITY_COLORS, IDENTITY_NAMES
from app.core.video_io import compose_mask_overlay, draw_bboxes, draw_keypoints

logger = logging.getLogger(__name__)


def export_video(
    video_path: str,
    tracker_history,
    output_path: str,
    draw_masks: bool = True,
    draw_bbox: bool = False,
    draw_kps: bool = False,
    draw_labels: bool = True,
    mask_alpha: float = 0.4,
    keypoints_by_frame: Optional[dict] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
    start_frame: int = 0,
    end_frame: int = -1,
) -> str:
    """
    Render original video with overlays and write to output_path.

    Args:
        start_frame: First frame to include (default 0).
        end_frame:   Exclusive end frame (-1 = full video).

    Returns output path.
    """
    # Build frame_idx → state lookup, plus sorted list for nearest-frame fallback
    state_map = {s.frame_idx: s for s in tracker_history}
    _tracked_frames = sorted(state_map.keys())

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end_frame < 0:
        end_frame = total_frames
    n_frames = end_frame - start_frame

    # Seek to start_frame if needed
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    try:
        frame_idx = start_frame
        written = 0
        while frame_idx < end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            composite = frame_rgb.copy()

            state = state_map.get(frame_idx)
            # Nearest-frame fallback when frame_skip > 1 leaves gaps
            if state is None and _tracked_frames:
                import bisect
                pos = bisect.bisect_right(_tracked_frames, frame_idx) - 1
                if pos >= 0:
                    state = state_map[_tracked_frames[pos]]
            if state:
                if draw_masks and state.masks:
                    composite = compose_mask_overlay(
                        composite, state.masks, IDENTITY_COLORS, mask_alpha
                    )
                if draw_bbox and state.bboxes:
                    labels = {mid: IDENTITY_NAMES.get(mid, f"M{mid}") for mid in state.bboxes}
                    composite = draw_bboxes(composite, state.bboxes, IDENTITY_COLORS, labels)
                if draw_labels and state.centroids:
                    for mid, (cx, cy) in state.centroids.items():
                        label = IDENTITY_NAMES.get(mid, f"M{mid}")
                        color = IDENTITY_COLORS.get(mid, (255, 255, 255))
                        bgr = (color[2], color[1], color[0])
                        cv2.putText(
                            composite, label,
                            (int(cx) + 6, int(cy) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr, 2, cv2.LINE_AA,
                        )
                if draw_kps and keypoints_by_frame:
                    fkps = keypoints_by_frame.get(frame_idx, {})
                    if fkps:
                        composite = draw_keypoints(composite, fkps, IDENTITY_COLORS)

            writer.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            frame_idx += 1
            written += 1

            if progress_callback and n_frames > 0:
                progress_callback(min(99, int(written * 100 / n_frames)))

    finally:
        cap.release()
        writer.release()

    if progress_callback:
        progress_callback(100)

    logger.info(f"Video exported: {output_path} ({written} frames)")
    return output_path
