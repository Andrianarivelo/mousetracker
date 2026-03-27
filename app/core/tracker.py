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
# Split a merged detection only when it is clearly larger than one animal.
MERGE_AREA_RATIO_THRESHOLD = 1.35
# Expand the candidate search box a bit so near-edge merges are still caught.
MERGE_BBOX_MARGIN_PX = 24.0


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
    # mouse IDs carried by a synthetic split of a merged detection
    merge_resolved_ids: set = field(default_factory=set)


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

        # Some SAM3 builds return saturated confidence (all 1.0). In that case,
        # use geometric matching quality as the confidence signal instead.
        has_informative_probs = self._has_informative_probs(curr_probs)

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
                raw_prob = curr_probs.get(sid) if has_informative_probs else None
                state.confidences[mouse_id] = (
                    self._clip01(raw_prob) if raw_prob is not None else 0.95
                )
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
        # Prefer the last reliable state while tracking is stable, but once a
        # merge or occlusion happens switch to the freshest state so the split
        # seeds keep moving with the merged cluster.
        ref_state = self._select_reference_state() or self._last_state
        curr_sam_masks, curr_bboxes, curr_probs, synthetic_sam_ids = (
            self._augment_merged_masks(
                frame_idx,
                curr_sam_masks,
                curr_bboxes,
                curr_probs,
                ref_state,
            )
        )
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
            r = prev_ids.index(mouse_id)
            c = curr_sam_ids.index(sam_id)
            match_cost = float(cost[r, c])
            max_cost = max(max_cost, match_cost)
            geom_conf = self._cost_to_confidence(match_cost)
            raw_prob = curr_probs.get(sam_id) if has_informative_probs else None
            if raw_prob is not None:
                # Blend model score with assignment quality for stability.
                conf = 0.7 * self._clip01(raw_prob) + 0.3 * geom_conf
            else:
                conf = geom_conf
            state.confidences[mouse_id] = self._clip01(conf)
            if sam_id in synthetic_sam_ids:
                state.merge_resolved_ids.add(mouse_id)
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
        if (
            max_cost > ID_SWITCH_COST_THRESHOLD
            and not state.occluded_ids
            and not state.merge_resolved_ids
        ):
            self.swap_log.append(frame_idx)
            logger.debug(f"Potential ID switch at frame {frame_idx} (cost={max_cost:.3f})")

        # ── Update reliable state when all visible mice are high-confidence ────
        self._last_state = state
        visible_confs = [
            c for mid, c in state.confidences.items()
            if mid not in state.occluded_ids
        ]
        if (
            visible_confs
            and min(visible_confs) >= RELIABLE_CONF_THRESHOLD
            and not state.occluded_ids
            and not state.merge_resolved_ids
        ):
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

    @staticmethod
    def _clip01(value: Optional[float]) -> float:
        if value is None:
            return 0.0
        return float(max(0.0, min(1.0, float(value))))

    @staticmethod
    def _has_informative_probs(curr_probs: dict[int, float]) -> bool:
        if not curr_probs:
            return False
        vals = np.asarray(list(curr_probs.values()), dtype=np.float32)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return False
        spread = float(vals.max() - vals.min())
        mean_val = float(vals.mean())
        # Treat "all ~1.0" as non-informative saturation.
        return spread > 1e-3 or mean_val < 0.995

    @staticmethod
    def _cost_to_confidence(cost: float) -> float:
        # Cost 0 -> 1.0, cost 1 -> ~0.37, higher costs decay smoothly.
        return float(np.exp(-max(0.0, float(cost))))

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

    def _select_reference_state(self) -> Optional[TrackState]:
        """Pick the best reference state for the next frame assignment."""
        if self._last_state is None:
            return self._last_reliable_state
        if self._last_reliable_state is None:
            return self._last_state
        if self._last_state.occluded_ids or self._last_state.merge_resolved_ids:
            return self._last_state
        return self._last_reliable_state

    def _reference_mask(
        self,
        ref_state: TrackState,
        mouse_id: int,
    ) -> Optional[np.ndarray]:
        """Use the freshest non-empty mask, falling back to the last reliable one."""
        mask = ref_state.masks.get(mouse_id)
        if mask is not None and np.any(mask):
            return mask
        if self._last_reliable_state is None:
            return mask
        reliable_mask = self._last_reliable_state.masks.get(mouse_id)
        if reliable_mask is not None and np.any(reliable_mask):
            return reliable_mask
        return mask

    def _reference_centroid(
        self,
        ref_state: TrackState,
        mouse_id: int,
    ) -> tuple[float, float]:
        """Use the freshest centroid, with reliable-state fallback when needed."""
        if mouse_id in ref_state.centroids:
            return ref_state.centroids[mouse_id]
        if self._last_reliable_state is not None:
            return self._last_reliable_state.centroids.get(mouse_id, (0.0, 0.0))
        return (0.0, 0.0)

    def _augment_merged_masks(
        self,
        frame_idx: int,
        curr_sam_masks: dict[int, np.ndarray],
        curr_bboxes: dict[int, tuple[int, int, int, int]],
        curr_probs: dict[int, float],
        ref_state: TrackState,
    ) -> tuple[
        dict[int, np.ndarray],
        dict[int, tuple[int, int, int, int]],
        dict[int, float],
        set[int],
    ]:
        """
        Replace a single merged blob with cheap geometry-only pseudo-splits.

        This avoids extra SAM calls. It only runs when the current frame has
        fewer detections than tracked identities, so the normal fast path is
        unchanged.
        """
        prev_ids = sorted(ref_state.masks.keys())
        missing = max(0, len(prev_ids) - len(curr_sam_masks))
        if missing <= 0 or len(curr_sam_masks) >= len(prev_ids):
            return curr_sam_masks, curr_bboxes, curr_probs, set()

        split_masks = dict(curr_sam_masks)
        split_bboxes = dict(curr_bboxes)
        split_probs = dict(curr_probs)
        synthetic_sam_ids: set[int] = set()
        claimed_mouse_ids: set[int] = set()
        next_sam_id = -1

        for sam_id, mask in sorted(
            curr_sam_masks.items(),
            key=lambda item: int(item[1].sum()),
            reverse=True,
        ):
            if missing <= 0 or sam_id not in split_masks:
                continue

            candidate_mouse_ids = self._candidate_merge_mouse_ids(
                mask,
                ref_state,
                claimed_mouse_ids,
            )
            target_parts = min(len(candidate_mouse_ids), missing + 1)
            if target_parts < 2:
                continue

            candidate_mouse_ids = candidate_mouse_ids[:target_parts]
            if not self._should_split_merged_mask(mask, candidate_mouse_ids, ref_state):
                continue

            pseudo_masks = self._split_mask_from_reference(
                mask,
                candidate_mouse_ids,
                ref_state,
            )
            if len(pseudo_masks) != target_parts:
                continue

            logger.debug(
                "Frame %d: split merged detection %d into %d pseudo-masks for mice %s",
                frame_idx,
                sam_id,
                target_parts,
                candidate_mouse_ids,
            )

            source_prob = split_probs.pop(sam_id, 1.0)
            split_masks.pop(sam_id, None)
            split_bboxes.pop(sam_id, None)
            for pseudo_mask in pseudo_masks:
                pseudo_id = next_sam_id
                next_sam_id -= 1
                split_masks[pseudo_id] = pseudo_mask
                split_bboxes[pseudo_id] = _mask_bbox(pseudo_mask)
                split_probs[pseudo_id] = source_prob
                synthetic_sam_ids.add(pseudo_id)

            claimed_mouse_ids.update(candidate_mouse_ids)
            missing -= target_parts - 1

        return split_masks, split_bboxes, split_probs, synthetic_sam_ids

    def _candidate_merge_mouse_ids(
        self,
        mask: np.ndarray,
        ref_state: TrackState,
        claimed_mouse_ids: set[int],
    ) -> list[int]:
        """Find tracked identities that are plausibly inside a merged mask."""
        if not np.any(mask):
            return []

        mask_centroid = _centroid(mask)
        x1, y1, x2, y2 = _mask_bbox(mask)
        bbox_diag = float(np.hypot(max(1, x2 - x1), max(1, y2 - y1)))
        distance_limit = max(MERGE_BBOX_MARGIN_PX, 0.6 * bbox_diag)

        candidates: list[tuple[int, int, float]] = []
        for mouse_id in sorted(ref_state.masks.keys()):
            if mouse_id in claimed_mouse_ids:
                continue

            prev_mask = self._reference_mask(ref_state, mouse_id)
            prev_centroid = self._reference_centroid(ref_state, mouse_id)
            overlap = 0
            if prev_mask is not None and np.shape(prev_mask) == np.shape(mask):
                overlap = int(np.logical_and(mask, prev_mask).sum())

            inside = _point_in_bbox(prev_centroid, (x1, y1, x2, y2), MERGE_BBOX_MARGIN_PX)
            distance = _euclidean(prev_centroid, mask_centroid)
            if overlap <= 0 and not inside and distance > distance_limit:
                continue

            candidates.append((mouse_id, overlap, distance))

        candidates.sort(key=lambda item: (-item[1], item[2], item[0]))
        return [mouse_id for mouse_id, _, _ in candidates]

    def _should_split_merged_mask(
        self,
        mask: np.ndarray,
        mouse_ids: list[int],
        ref_state: TrackState,
    ) -> bool:
        """Guard against over-splitting a normal single-animal mask."""
        if len(mouse_ids) < 2:
            return False

        mask_area = int(mask.sum())
        reference_areas = []
        for mouse_id in mouse_ids:
            ref_mask = self._reference_mask(ref_state, mouse_id)
            if ref_mask is None:
                continue
            area = int(ref_mask.sum())
            if area > 0:
                reference_areas.append(area)

        if len(reference_areas) < 2:
            return False

        largest_area = max(reference_areas)
        combined_area = sum(reference_areas)
        return (
            mask_area >= int(largest_area * MERGE_AREA_RATIO_THRESHOLD)
            or mask_area >= int(0.6 * combined_area)
        )

    def _split_mask_from_reference(
        self,
        mask: np.ndarray,
        mouse_ids: list[int],
        ref_state: TrackState,
    ) -> list[np.ndarray]:
        """Split a merged mask by nearest shifted reference centroid."""
        if len(mouse_ids) < 2 or not np.any(mask):
            return []

        current_centroid = np.array(_centroid(mask), dtype=np.float32)
        reference_centroids = np.array(
            [self._reference_centroid(ref_state, mouse_id) for mouse_id in mouse_ids],
            dtype=np.float32,
        )
        if reference_centroids.size == 0:
            return []

        shifted_centroids = reference_centroids.copy()
        shifted_centroids += current_centroid - reference_centroids.mean(axis=0)

        ys, xs = np.where(mask)
        if len(xs) < len(mouse_ids):
            return []

        for idx in range(len(shifted_centroids)):
            shifted_centroids[idx] = _snap_point_to_mask(
                mask,
                shifted_centroids[idx],
            )

        pixels = np.column_stack((xs, ys)).astype(np.float32)
        distances = np.sum(
            (pixels[:, None, :] - shifted_centroids[None, :, :]) ** 2,
            axis=2,
        )
        labels = np.argmin(distances, axis=1)

        min_piece_area = max(16, int(mask.sum() / max(8, len(mouse_ids) * 4)))
        result: list[np.ndarray] = []
        for label_idx in range(len(mouse_ids)):
            submask = np.zeros(mask.shape, dtype=bool)
            submask[ys, xs] = labels == label_idx
            if int(submask.sum()) < min_piece_area:
                return []
            result.append(submask)

        return result

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
            prev_mask = self._reference_mask(prev_state, pmid)
            prev_cent = self._reference_centroid(prev_state, pmid)
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

def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return the tight bounding box of a boolean mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _point_in_bbox(
    point: tuple[float, float],
    bbox: tuple[int, int, int, int],
    margin: float = 0.0,
) -> bool:
    """Return True when a point falls inside a bounding box plus margin."""
    x, y = point
    x1, y1, x2, y2 = bbox
    return (
        x >= x1 - margin
        and x <= x2 + margin
        and y >= y1 - margin
        and y <= y2 + margin
    )


def _snap_point_to_mask(mask: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Move a seed point onto the nearest foreground pixel if needed."""
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    h, w = mask.shape
    x = min(max(x, 0), max(w - 1, 0))
    y = min(max(y, 0), max(h - 1, 0))
    if h > 0 and w > 0 and mask[y, x]:
        return np.array([x, y], dtype=np.float32)

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.array([x, y], dtype=np.float32)

    coords = np.column_stack((xs, ys)).astype(np.float32)
    deltas = coords - np.array([x, y], dtype=np.float32)
    nearest = coords[np.argmin(np.sum(deltas * deltas, axis=1))]
    return nearest.astype(np.float32)


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
