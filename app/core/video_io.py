"""OpenCV video I/O utilities."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    codec: str = ""

    @property
    def name(self) -> str:
        return Path(self.path).name


class VideoReader:
    """Thread-safe OpenCV video reader."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._cap: Optional[cv2.VideoCapture] = None
        self.info: Optional[VideoInfo] = None
        self._open()

    def _open(self) -> None:
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {self.path}")
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        self.info = VideoInfo(
            path=self.path,
            width=w,
            height=h,
            fps=fps,
            frame_count=n,
            duration_s=n / fps if fps > 0 else 0,
            codec=codec,
        )
        logger.info(f"Opened {self.path}: {w}x{h} @ {fps:.2f} fps, {n} frames")

    def read_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Read a single frame (BGR → RGB). Returns None on failure."""
        if self._cap is None:
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self._cap.read()
        if not ok:
            logger.warning(f"Failed to read frame {frame_idx} from {self.path}")
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def iter_frames(
        self,
        start: int = 0,
        end: int = -1,
        step: int = 1,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (frame_idx, rgb_frame) for the given range."""
        if self._cap is None:
            return
        n = self.info.frame_count if self.info else 0
        end = n if end < 0 else min(end, n)
        for idx in range(start, end, step):
            frame = self.read_frame(idx)
            if frame is not None:
                yield idx, frame

    def sample_frames(self, n_samples: int) -> list[tuple[int, np.ndarray]]:
        """Return N evenly spaced (idx, rgb_frame) tuples from the video."""
        total = self.info.frame_count if self.info else 0
        if total == 0 or n_samples <= 0:
            return []
        indices = [
            int(i * total / n_samples) for i in range(n_samples)
        ]
        result = []
        for idx in indices:
            frame = self.read_frame(idx)
            if frame is not None:
                result.append((idx, frame))
        return result

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def get_video_info(path: str) -> VideoInfo:
    """Read video metadata without keeping the file open."""
    with VideoReader(path) as r:
        return r.info


def compose_mask_overlay(
    frame_rgb: np.ndarray,
    masks: dict[int, np.ndarray],
    identity_colors: dict[int, tuple[int, int, int]],
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Composite colored semi-transparent mask overlays onto an RGB frame.

    Args:
        frame_rgb: HxWx3 uint8 RGB frame.
        masks: {mouse_id: binary_mask (HxW bool/uint8)}.
        identity_colors: {mouse_id: (R, G, B)}.
        alpha: Opacity of the mask color (0=transparent, 1=solid).

    Returns:
        Composite RGB image (HxWx3 uint8).
    """
    overlay = frame_rgb.copy().astype(np.float32)
    for mouse_id, mask in masks.items():
        color = identity_colors.get(mouse_id)
        if color is None:
            continue
        color_arr = np.array(color, dtype=np.float32)
        mask_bool = mask > 0
        overlay[mask_bool] = (
            (1 - alpha) * overlay[mask_bool] + alpha * color_arr
        )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def draw_keypoints(
    frame_rgb: np.ndarray,
    keypoints_by_id: dict[int, dict[str, tuple[float, float]]],
    identity_colors: dict[int, tuple[int, int, int]],
    radius: int = 4,
    keypoint_colors: Optional[dict[str, tuple[int, int, int]]] = None,
    show_labels: bool = True,
) -> np.ndarray:
    """Draw keypoint dots with per-keypoint colors and labels."""
    out = frame_rgb.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    thickness = 1
    for mouse_id, kps in keypoints_by_id.items():
        fallback = identity_colors.get(mouse_id, (255, 255, 255))
        for name, (x, y) in kps.items():
            # Use per-keypoint color if available, else entity color
            if keypoint_colors and name in keypoint_colors:
                color = keypoint_colors[name]
            else:
                color = fallback
            bgr = (color[2], color[1], color[0])
            ix, iy = int(x), int(y)
            cv2.circle(out, (ix, iy), radius, bgr, -1)
            cv2.circle(out, (ix, iy), radius + 1, (0, 0, 0), 1)
            if show_labels:
                label = name.replace("_", " ")
                # Shadow + text
                tx, ty = ix + radius + 2, iy + 3
                cv2.putText(out, label, (tx + 1, ty + 1), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
                cv2.putText(out, label, (tx, ty), font, font_scale, bgr, thickness, cv2.LINE_AA)
    return out


def draw_entity_labels(
    frame_rgb: np.ndarray,
    centroids: dict[int, tuple[float, float]],
    confidences: dict[int, float],
    identity_colors: dict[int, tuple[int, int, int]],
    names: Optional[dict[int, str]] = None,
    active_entity_id: Optional[int] = None,
) -> np.ndarray:
    """
    Draw entity name + confidence score centred on each tracked mask.

    Each label is drawn with a dark shadow for readability on any background.
    """
    out = frame_rgb.copy()
    for mouse_id, (cx, cy) in centroids.items():
        color = identity_colors.get(mouse_id, (255, 255, 255))
        bgr = (color[2], color[1], color[0])
        name = (names or {}).get(mouse_id, f"M{mouse_id}")
        conf = confidences.get(mouse_id, 0.0)
        label = f"{name}  {conf:.0%}"
        x, y = int(cx), int(cy)
        if active_entity_id == mouse_id:
            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                1,
            )
            top_left = (x - 6, y - text_h - 8)
            bottom_right = (x + text_w + 6, y + baseline + 6)
            cv2.rectangle(out, top_left, bottom_right, (0, 0, 0), -1)
            cv2.rectangle(out, top_left, bottom_right, bgr, 2)
        # Shadow (black, thick)
        cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 3, cv2.LINE_AA)
        # Foreground (entity colour)
        cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, bgr, 1, cv2.LINE_AA)
    return out


def draw_bboxes(
    frame_rgb: np.ndarray,
    bboxes: dict[int, tuple[int, int, int, int]],
    identity_colors: dict[int, tuple[int, int, int]],
    labels: Optional[dict[int, str]] = None,
) -> np.ndarray:
    """Draw bounding boxes and optional ID labels on a frame."""
    out = frame_rgb.copy()
    for mouse_id, (x1, y1, x2, y2) in bboxes.items():
        color = identity_colors.get(mouse_id, (255, 255, 255))
        bgr = (color[2], color[1], color[0])
        cv2.rectangle(out, (x1, y1), (x2, y2), bgr, 2)
        label = labels.get(mouse_id, f"M{mouse_id}") if labels else f"M{mouse_id}"
        cv2.putText(
            out, label, (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2, cv2.LINE_AA,
        )
    return out
