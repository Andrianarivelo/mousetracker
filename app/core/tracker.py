"""Identity tracking with Hungarian algorithm across frames."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.optimize

from app.config import TRACKING_COST_WEIGHTS, ID_SWITCH_COST_THRESHOLD

logger = logging.getLogger(__name__)

# Confidence threshold to accept a frame as "reliable" for state anchoring
RELIABLE_CONF_THRESHOLD = 0.40
# Frames a mouse must be absent before entering OCCLUDED state
OCCLUSION_ENTER_FRAMES = 5
# Velocity spike threshold multiplier (MAD-based, used in detect_velocity_swaps)
VELOCITY_SPIKE_K = 4.0


@dataclass
class TrackState:
    """Snapshot of identity assignments for a single frame."""
    frame_idx: int
    # {mouse_id: binary_mask (HxW bool)}
    masks: dict[int, np.ndarray] = field(default_factory=dict)
    # {mouse_id: (cx, cy)}
    centroids: dict[int, tuple[float, float]] = field(default_factory=dict)
    # {mouse_id: (x1,y1,x2,y2)}
    bboxes: dict[int, tuple[int, int, int, int]] = field(default_factory=dict)
    # {mouse_id: confidence 0-1}
    confidences: dict[int, float] = field(default_factory=dict)
    # matching cost (higher = less confident)
    match_cost: float = 0.0
    # mouse IDs currently in occluded state (absent, position carried forward)
    occluded_ids: set = field(default_factory=set)


class IdentityTracker:
    """
    Maintains consistent mouse IDs across frames using Hungarian matching.

    Inputs: raw SAM3 object IDs (arbitrary integers per frame) →
    Outputs: stable mouse IDs (1, 2, 3, …) across the full video.

    Features:
      - Confidence-gated reliable state: anchors matching only on high-confidence frames.
      - Occlusion carry-forward: absent mice keep last known position/mask at low conf.
      - Per-mouse appearance histogram: grayscale intensity for tie-breaking.
      - Velocity-spike post-processing: flags anomalous frame-to-frame jumps.
      - Trajectory smoothing: Savitzky-Golay filter applied post-tracking.
    """

    def __init__(
        self,
        n_mice: int,
        weights: Optional[dict[str, float]] = None,
    ) -> None:
        self.n_mice = n_mice
        self.weights = weights or TRACKING_COST_WEIGHTS
        self.history: list[TrackState] = []
        # {mouse_id: (R,G,B)} — set externally
        self.id_colors: dict[int, tuple[int, int, int]] = {}
        # Frames flagged as likely ID switches
        self.swap_log: list[int] = []
        # Last frame state (always updated)
        self._last_state: Optional[TrackState] = None
        # Only updated when all non-occluded confidences ≥ RELIABLE_CONF_THRESHOLD
        self._last_reliable_state: Optional[TrackState] = None
        # Consecutive frames each mouse has been absent
        self._absent_frames: dict[int, int] = {}
        # Per-mouse occlusion state: "visible" | "occluded"
        self._occlusion_states: dict[int, str] = {}
        # Per-mouse grayscale appearance histogram (64 bins), updated on reliable frames
        self._appearance_histograms: dict[int, np.ndarray] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def initialize(
        self,
        frame_idx: int,
        sam_masks: dict[int, np.ndarray],
        mouse_id_to_sam_id: dict[int, int],
    ) -> TrackState:
        """
        Set the initial identity assignment from user annotation.

        Args:
            sam_masks: {sam_obj_id: binary_mask}
            mouse_id_to_sam_id: {mouse_id (1..N): sam_obj_id}
        """
        state = TrackState(frame_idx=frame_idx)
        for mouse_id, sam_id in mouse_id_to_sam_id.items():
            mask = sam_masks.get(sam_id)
            if mask is None:
                continue
            state.masks[mouse_id] = mask
            state.centroids[mouse_id] = _centroid(mask)
            state.confidences[mouse_id] = 1.0
            self._absent_frames[mouse_id] = 0
            self._occlusion_states[mouse_id] = "visible"
        self._last_state = state
        self._last_reliable_state = state
        self.history.append(state)
        return state

    def reinitialize_at_keyframe(
        self,
        frame_idx: int,
        mouse_masks: dict[int, np.ndarray],
    ) -> None:
        """
        Re-anchor the tracker at a user-annotated keyframe without clearing history.

        Called at multi-segment boundaries so subsequent matching is relative to
        the known-good assignment at this frame.

        Args:
            frame_idx: Absolute frame index of the keyframe.
            mouse_masks: {mouse_id: binary_mask} from user annotation.
        """
        state = TrackState(frame_idx=frame_idx)
        for mouse_id, mask in mouse_masks.items():
            if mask is None or not mask.any():
                continue
            state.masks[mouse_id] = mask
            state.centroids[mouse_id] = _centroid(mask)
            state.confidences[mouse_id] = 1.0
            self._absent_frames[mouse_id] = 0
            self._occlusion_states[mouse_id] = "visible"
        self._last_state = state
        self._last_reliable_state = state
        logger.info(f"Tracker re-anchored at keyframe {frame_idx}")

    def assign_frame(
        self,
        frame_idx: int,
        sam_outputs: dict,
        frame_shape: tuple[int, int],
        mask_filter_fn=None,
        frame_rgb: Optional[np.ndarray] = None,
    ) -> TrackState:
        """
        Assign identities to SAM3 outputs for a new frame.

        Args:
            sam_outputs: Raw outputs dict from SAM3Engine.
            frame_shape: (H, W) of the frame.
            mask_filter_fn: Optional filter applied to masks before matching.
            frame_rgb: Optional HxWx3 uint8 frame for appearance histogram matching.

        Returns:
            TrackState with stable mouse IDs.
        """
        import cv2
        obj_ids = sam_outputs.get("out_obj_ids", np.array([]))
        raw_masks_arr = sam_outputs.get("out_binary_masks", np.array([]))
        boxes_arr = sam_outputs.get("out_boxes_xywh", np.array([]))
        probs_arr = sam_outputs.get("out_probs", np.array([]))

        h, w = frame_shape
        curr_sam_masks: dict[int, np.ndarray] = {}
        curr_bboxes: dict[int, tuple[int, int, int, int]] = {}
        curr_probs: dict[int, float] = {}

        for i, sid in enumerate(obj_ids):
            sid = int(sid)
            if i < len(raw_masks_arr):
                mask = raw_masks_arr[i].astype(np.uint8) * 255
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                curr_sam_masks[sid] = (mask > 0)
            if i < len(boxes_arr):
                x, y, bw, bh = boxes_arr[i]
                curr_bboxes[sid] = (int(x), int(y), int(x + bw), int(y + bh))
            if i < len(probs_arr):
                curr_probs[sid] = float(np.mean(probs_arr[i]))

        if mask_filter_fn is not None and curr_sam_masks:
            curr_sam_masks = mask_filter_fn(curr_sam_masks)

        state = TrackState(frame_idx=frame_idx)

        if not curr_sam_masks:
            # No detections: carry forward last known state at low/zero confidence
            if self._last_state:
                for mid in self._last_state.masks:
                    state.masks[mid] = np.zeros(frame_shape, dtype=bool)
                    state.centroids[mid] = self._last_state.centroids.get(mid, (0.0, 0.0))
                    state.confidences[mid] = 0.0
                    state.occluded_ids.add(mid)
                    self._absent_frames[mid] = self._absent_frames.get(mid, 0) + 1
            self._last_state = state
            self.history.append(state)
            return state

        if self._last_state is None or not self._last_state.masks:
            # No history: assign by detection order
            for i, (sid, mask) in enumerate(curr_sam_masks.items()):
                mouse_id = i + 1
                if mouse_id > self.n_mice:
                    break
                state.masks[mouse_id] = mask
                state.centroids[mouse_id] = _centroid(mask)
                state.bboxes[mouse_id] = curr_bboxes.get(sid, (0, 0, 0, 0))
                state.confidences[mouse_id] = curr_probs.get(sid, 1.0)
                self._absent_frames[mouse_id] = 0
                self._occlusion_states[mouse_id] = "visible"
            self._last_state = state
            self._last_reliable_state = state
            self.history.append(state)
            return state

        # ── Pre-compute per-sam-id appearance histograms for this frame ───────
        gray: Optional[np.ndarray] = None
        curr_hists: dict[int, np.ndarray] = {}
        if frame_rgb is not None:
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            for sid, mask in curr_sam_masks.items():
                curr_hists[sid] = _mask_histogram(gray, mask)

        # ── Hungarian matching ─────────────────────────────────────────────────
        # Prefer reliable state when available — more stable reference
        ref_state = self._last_reliable_state or self._last_state
        prev_ids = sorted(ref_state.masks.keys())
        curr_sam_ids = list(curr_sam_masks.keys())

        cost = self._build_cost_matrix(
            prev_ids, curr_sam_ids, ref_state, curr_sam_masks, curr_hists,
        )
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)

        assignment: dict[int, int] = {}
        for r, c in zip(row_ind, col_ind):
            assignment[prev_ids[r]] = curr_sam_ids[c]

        matched_mouse_ids: set[int] = set(assignment.keys())
        max_cost = 0.0
        for mouse_id, sam_id in assignment.items():
            mask = curr_sam_masks[sam_id]
            state.masks[mouse_id] = mask
            state.centroids[mouse_id] = _centroid(mask)
            state.bboxes[mouse_id] = curr_bboxes.get(sam_id, (0, 0, 0, 0))
            state.confidences[mouse_id] = curr_probs.get(sam_id, 1.0)
            r = prev_ids.index(mouse_id)
            c = curr_sam_ids.index(sam_id)
            max_cost = max(max_cost, cost[r, c])
            self._absent_frames[mouse_id] = 0
            self._occlusion_states[mouse_id] = "visible"

        # ── Occlusion carry-forward for unmatched mice ─────────────────────────
        for mouse_id in prev_ids:
            if mouse_id in matched_mouse_ids:
                continue
            absent = self._absent_frames.get(mouse_id, 0) + 1
            self._absent_frames[mouse_id] = absent
            if absent >= OCCLUSION_ENTER_FRAMES:
                self._occlusion_states[mouse_id] = "occluded"

            # Confidence decays linearly to 0 over OCCLUSION_ENTER_FRAMES frames
            conf = max(0.0, 1.0 - absent / max(OCCLUSION_ENTER_FRAMES, 1))
            last = self._last_state
            last_cent = last.centroids.get(mouse_id, (0.0, 0.0)) if last else (0.0, 0.0)
            # Keep the centroid for continuity, but do not paint a stale mask
            # onto the current frame when the mouse was not actually detected.
            state.masks[mouse_id] = np.zeros(frame_shape, dtype=bool)
            state.centroids[mouse_id] = last_cent
            state.confidences[mouse_id] = conf
            state.occluded_ids.add(mouse_id)
            logger.debug(
                "Mouse %d absent %d frame(s) — %s",
                mouse_id, absent,
                "OCCLUDED" if absent >= OCCLUSION_ENTER_FRAMES else "entering",
            )

        state.match_cost = max_cost

        # Flag ID switch only when all mice are visible
        if max_cost > ID_SWITCH_COST_THRESHOLD and not state.occluded_ids:
            self.swap_log.append(frame_idx)
            logger.debug(f"Potential ID switch at frame {frame_idx} (cost={max_cost:.3f})")

        # ── Update reliable state when all visible mice are high-confidence ────
        self._last_state = state
        visible_confs = [
            c for mid, c in state.confidences.items()
            if mid not in state.occluded_ids
        ]
        if visible_confs and min(visible_confs) >= RELIABLE_CONF_THRESHOLD:
            self._last_reliable_state = state
            # Update appearance histograms via EMA (α=0.1)
            if gray is not None:
                for mouse_id, sam_id in assignment.items():
                    hist = curr_hists.get(sam_id)
                    if hist is not None:
                        prev_hist = self._appearance_histograms.get(mouse_id)
                        if prev_hist is None:
                            self._appearance_histograms[mouse_id] = hist.copy()
                        else:
                            self._appearance_histograms[mouse_id] = (
                                0.9 * prev_hist + 0.1 * hist
                            )

        self.history.append(state)
        return state

    def detect_velocity_swaps(
        self,
        fps: float = 25.0,
        k: float = VELOCITY_SPIKE_K,
    ) -> list[int]:
        """
        Post-processing: flag frames where centroid velocity > k * MAD.

        Adds newly found spike frames to swap_log (deduplicated, sorted).
        Returns the list of newly detected spike frames.
        """
        trajs = self.get_trajectories()
        spike_frames: list[int] = []
        existing = set(self.swap_log)

        for mid, traj in trajs.items():
            if len(traj) < 3:
                continue
            frames = np.array([t[0] for t in traj])
            xs = np.array([t[1] for t in traj])
            ys = np.array([t[2] for t in traj])

            dt = np.diff(frames).clip(1)
            speed = np.sqrt((np.diff(xs) / dt) ** 2 + (np.diff(ys) / dt) ** 2)
            if len(speed) == 0:
                continue

            median = np.median(speed)
            mad = np.median(np.abs(speed - median)) + 1e-6
            threshold = median + k * mad

            for idx in np.where(speed > threshold)[0]:
                f = int(frames[idx + 1])
                if f not in existing:
                    spike_frames.append(f)
                    existing.add(f)
                    logger.debug(
                        "Velocity spike: mouse %d frame %d "
                        "(speed=%.1f > threshold=%.1f)",
                        mid, f, speed[idx], threshold,
                    )

        self.swap_log = sorted(existing)
        return spike_frames

    def smooth_trajectories(
        self, window_length: int = 11, polyorder: int = 3
    ) -> None:
        """
        In-place Savitzky-Golay smoothing of centroid trajectories in history.

        Modifies TrackState.centroids. Safe to call once after tracking is complete.
        """
        try:
            from scipy.signal import savgol_filter
        except ImportError:
            logger.warning("scipy.signal not available — trajectory smoothing skipped")
            return

        trajs = self.get_trajectories()
        smoothed_any = False
        for mid, traj in trajs.items():
            if len(traj) < window_length:
                continue
            frames = [t[0] for t in traj]
            xs = np.array([t[1] for t in traj])
            ys = np.array([t[2] for t in traj])

            n = len(traj)
            wl = window_length if window_length % 2 == 1 else window_length + 1
            wl = min(wl, n if n % 2 == 1 else n - 1)
            if wl <= polyorder:
                continue

            xs_s = savgol_filter(xs, wl, polyorder)
            ys_s = savgol_filter(ys, wl, polyorder)

            frame_to_smooth: dict[int, tuple[float, float]] = {
                f: (float(xs_s[i]), float(ys_s[i])) for i, f in enumerate(frames)
            }
            for state in self.history:
                if state.frame_idx in frame_to_smooth:
                    state.centroids[mid] = frame_to_smooth[state.frame_idx]
            smoothed_any = True

        if smoothed_any:
            logger.info(
                "Trajectory smoothing complete (window=%d, poly=%d)",
                window_length, polyorder,
            )

    def correct_swap(
        self,
        frame_range: tuple[int, int],
        id_a: int,
        id_b: int,
    ) -> None:
        """Manually swap two mouse IDs over a range of frames."""
        start, end = frame_range
        for state in self.history:
            if start <= state.frame_idx <= end:
                for d in (state.masks, state.centroids, state.bboxes, state.confidences):
                    a_val = d.get(id_a)
                    b_val = d.get(id_b)
                    if a_val is not None:
                        d[id_b] = a_val
                    if b_val is not None:
                        d[id_a] = b_val
        logger.info(f"Corrected swap of Mouse {id_a} ↔ Mouse {id_b} over frames {start}-{end}")

    def replace_masks(
        self,
        frame_idx: int,
        corrected_masks: dict[int, np.ndarray],
    ) -> TrackState:
        """
        Replace masks in an existing TrackState and recompute centroids.

        Used by the recovery pipeline when watershed/CC finds better masks.
        Returns the updated TrackState (also updates last_state/reliable_state
        if they point to this frame).
        """
        state = self.get_state_at(frame_idx)
        if state is None:
            # Build a new state if none exists yet
            state = TrackState(frame_idx=frame_idx)
            self.history.append(state)

        for eid, mask in corrected_masks.items():
            state.masks[eid] = mask
            state.centroids[eid] = _centroid(mask)

        if self._last_state and self._last_state.frame_idx == frame_idx:
            self._last_state = state
        if self._last_reliable_state and self._last_reliable_state.frame_idx == frame_idx:
            self._last_reliable_state = state

        return state

    def get_state_at(self, frame_idx: int) -> Optional[TrackState]:
        """Return the TrackState for a specific frame (if it exists)."""
        for s in reversed(self.history):
            if s.frame_idx == frame_idx:
                return s
        return None

    def get_trajectories(self) -> dict[int, list[tuple[int, float, float]]]:
        """Return {mouse_id: [(frame_idx, cx, cy), ...]} for all tracked frames."""
        trajs: dict[int, list] = {}
        for state in self.history:
            for mid, (cx, cy) in state.centroids.items():
                trajs.setdefault(mid, []).append((state.frame_idx, cx, cy))
        return trajs

    def reset(self) -> None:
        """Clear all tracking history."""
        self.history.clear()
        self.swap_log.clear()
        self._last_state = None
        self._last_reliable_state = None
        self._absent_frames.clear()
        self._occlusion_states.clear()
        self._appearance_histograms.clear()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_cost_matrix(
        self,
        prev_mouse_ids: list[int],
        curr_sam_ids: list[int],
        prev_state: TrackState,
        curr_masks: dict[int, np.ndarray],
        curr_hists: dict[int, np.ndarray],
    ) -> np.ndarray:
        n_prev = len(prev_mouse_ids)
        n_curr = len(curr_sam_ids)
        C = np.zeros((n_prev, n_curr), dtype=np.float64)

        w_d = self.weights.get("centroid_distance", 0.4)
        w_iou = self.weights.get("mask_iou", 0.35)
        w_app = self.weights.get("appearance", 0.25)

        for i, pmid in enumerate(prev_mouse_ids):
            prev_mask = prev_state.masks.get(pmid)
            prev_cent = prev_state.centroids.get(pmid, (0.0, 0.0))
            prev_hist = self._appearance_histograms.get(pmid)
            if prev_mask is None:
                C[i, :] = 1.0
                continue

            h, w = prev_mask.shape
            diag = float(np.sqrt(h * h + w * w)) + 1e-6
            # Slightly penalise occluded mice so re-emergent detections closer
            # in space are preferred
            occ_bias = 0.15 if self._occlusion_states.get(pmid) == "occluded" else 0.0

            for j, csid in enumerate(curr_sam_ids):
                curr_mask = curr_masks.get(csid)
                if curr_mask is None:
                    C[i, j] = 1.0
                    continue
                curr_cent = _centroid(curr_mask)
                curr_hist = curr_hists.get(csid)

                d = _euclidean(prev_cent, curr_cent) / diag
                iou = _mask_iou(prev_mask, curr_mask)

                if prev_hist is not None and curr_hist is not None:
                    app = 1.0 - _histogram_intersection(prev_hist, curr_hist)
                else:
                    app = _appearance_distance(prev_mask, curr_mask)

                C[i, j] = w_d * d + w_iou * (1.0 - iou) + w_app * app + occ_bias

        return C


# ── Module-level helpers ───────────────────────────────────────────────────────

def _centroid(mask: np.ndarray) -> tuple[float, float]:
    """Return (cx, cy) centroid of a boolean mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0.0, 0.0)
    return (float(np.mean(xs)), float(np.mean(ys)))


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection over Union of two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def _mask_histogram(gray: np.ndarray, mask: np.ndarray, bins: int = 64) -> np.ndarray:
    """Normalized grayscale histogram of pixels inside mask (64 bins)."""
    pixels = gray[mask > 0]
    if len(pixels) == 0:
        return np.zeros(bins, dtype=np.float32)
    hist, _ = np.histogram(pixels, bins=bins, range=(0, 256))
    total = hist.sum()
    return hist.astype(np.float32) / max(total, 1)


def _histogram_intersection(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Histogram intersection similarity (Swain & Ballard 1991).
    Returns [0, 1]: 1 = identical distributions, 0 = no overlap.
    """
    return float(np.minimum(h1, h2).sum())


def _appearance_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Fallback appearance metric: area ratio difference.
    Returns 0 (same size) to 1 (completely different sizes).
    """
    area_a = max(1, mask_a.sum())
    area_b = max(1, mask_b.sum())
    ratio = min(area_a, area_b) / max(area_a, area_b)
    return 1.0 - ratio
