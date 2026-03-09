"""QThread worker for running SAM3 tracking in the background."""

import logging
import time
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from app.config import AUTO_PROMPTS
from app.core.mask_recovery import SizeValidator, recover_masks
from app.core.tracker import IdentityTracker, TrackState

logger = logging.getLogger(__name__)

# Videos longer than this many frames are processed in chunks to avoid
# SAM3 loading the entire video into GPU memory at once.
# 1500 frames ≈ 50 s @ 30 fps.  Adjust lower if you hit OOM.
CHUNK_THRESHOLD = 1500
CHUNK_SIZE = 1500


def _mask_dicts_equal(
    left: dict[int, np.ndarray],
    right: dict[int, np.ndarray],
) -> bool:
    """Compare mask dictionaries without triggering NumPy truth-value errors."""
    if left.keys() != right.keys():
        return False

    for key in left:
        left_mask = left[key]
        right_mask = right[key]
        if left_mask is right_mask:
            continue
        if left_mask is None or right_mask is None:
            return False
        if np.shape(left_mask) != np.shape(right_mask):
            return False
        if not np.array_equal(left_mask, right_mask):
            return False

    return True


class TrackingWorker(QThread):
    """
    Runs SAM3 propagation + identity tracking in a background thread.

    Modes (selected automatically):
      1. **Multi-segment** — when the user has annotated multiple keyframes.
         Each segment between adjacent keyframes gets its own SAM3 session;
         the tracker is re-anchored at every keyframe so identity is preserved
         across the full video without single-session drift.
      2. **Chunked** — for long videos (> CHUNK_THRESHOLD frames) without
         user keyframes; auto-splits at CHUNK_SIZE intervals via ffmpeg.
      3. **Single-session** — for shorter videos without user keyframes.

    Signals:
        progress(int, float):      (percent 0-100, eta_seconds)
        status(str):               Human-readable status line
        frame_result(int, object): (frame_idx, TrackState) — emitted every N frames
        chunk_complete(int):       Last absolute frame_idx of the completed chunk
        finished(bool, str):       (success, error_message)
        model_loading():           Emitted while SAM3 model loads
    """

    progress = Signal(int, float)       # percent, eta_s
    status = Signal(str)                # status message
    frame_result = Signal(int, object)  # frame_idx, TrackState
    chunk_complete = Signal(int)        # last absolute frame_idx of a chunk
    finished = Signal(bool, str)        # success, error
    model_loading = Signal()

    def __init__(
        self,
        engine,                        # SAM3Engine instance
        tracker: IdentityTracker,
        video_path: str,
        frame_count: int,
        frame_shape: tuple[int, int],
        fps: float = 30.0,
        initial_masks: Optional[dict[int, np.ndarray]] = None,
        start_frame: int = 0,
        update_every: int = 30,
        frame_skip: int = 1,
        max_area_frac: float = 0.40,
        max_edge_frac: float = 0.28,
        use_area_filter: bool = True,
        use_edge_filter: bool = True,
        bypass_filters: bool = False,
        # {frame_idx: {mouse_id: mask}} for multi-segment re-anchoring.
        # Must be sorted by frame_idx; frame 0 is the first keyframe.
        keyframes: Optional[list[tuple[int, dict[int, np.ndarray]]]] = None,
        chunk_size: int = CHUNK_SIZE,
        size_validator: Optional[SizeValidator] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self.tracker = tracker
        self.video_path = video_path
        self.frame_count = frame_count
        self.frame_shape = frame_shape
        self.fps = fps
        self.initial_masks: dict[int, np.ndarray] = initial_masks or {}
        self.start_frame = start_frame
        self.update_every = update_every
        self.frame_skip = max(1, frame_skip)
        self.max_area_frac = max_area_frac
        self.max_edge_frac = max_edge_frac
        self.use_area_filter = use_area_filter
        self.use_edge_filter = use_edge_filter
        self.bypass_filters = bypass_filters
        # Sorted list of (frame_idx, {mouse_id: mask}) — all user-annotated frames
        self.keyframes: list[tuple[int, dict[int, np.ndarray]]] = sorted(
            keyframes or [], key=lambda kv: kv[0]
        )
        self.chunk_size = max(100, chunk_size)
        self.size_validator = size_validator
        self._abort = False
        self._recovery_count = 0  # frames where recovery was triggered
        self._reprompt_count = 0  # frames fixed by adaptive re-prompting
        self._failed_frames: list[int] = []  # frames needing adaptive re-prompt

    def abort(self) -> None:
        """Signal the worker to stop at the next opportunity."""
        self._abort = True
        logger.info("Tracking abort requested")

    def run(self) -> None:
        try:
            self._run_tracking()
        except Exception as e:
            logger.exception(f"Tracking worker error: {e}")
            self.finished.emit(False, str(e))

    def _run_tracking(self) -> None:
        if not self.engine.is_loaded():
            self.model_loading.emit()
            self.status.emit("Loading SAM3 model…")
            self.engine.load_model()

        remaining = max(1, self.frame_count - self.start_frame)

        # Multi-segment mode: user has annotated ≥ 2 keyframes
        if len(self.keyframes) >= 2:
            self._run_multi_segment(remaining)
        elif remaining > self.chunk_size:
            self._run_chunked(remaining)
        else:
            self._run_single_session(remaining)

        # Adaptive re-prompting for frames that failed watershed/CC recovery
        if self._failed_frames and not self._abort:
            self._adaptive_reprompt()

        if self._recovery_count or self._reprompt_count:
            logger.info(
                "Recovery stats: %d watershed/CC, %d adaptive re-prompt",
                self._recovery_count, self._reprompt_count,
            )

        self.progress.emit(100, 0.0)
        self.finished.emit(True, "")

    # ── Multi-segment path (user keyframes) ───────────────────────────────────

    def _run_multi_segment(self, remaining: int) -> None:
        """
        Process each interval between consecutive user keyframes as a separate
        SAM3 session.  At the start of each segment the tracker is re-anchored
        from the user's known-good annotation.
        """
        n_segs = len(self.keyframes) - 1
        frames_done = 0
        t_start = time.time()
        last_abs_frame = self.start_frame

        for seg_idx in range(n_segs):
            if self._abort:
                break

            kf_start_idx, kf_start_masks = self.keyframes[seg_idx]
            kf_end_idx, _ = self.keyframes[seg_idx + 1]

            seg_start = max(kf_start_idx, self.start_frame)
            seg_end = min(kf_end_idx, self.frame_count)

            if seg_start >= seg_end:
                continue

            self.status.emit(
                f"Segment {seg_idx + 1}/{n_segs} — "
                f"frames {seg_start}–{seg_end - 1}…"
            )
            logger.info(
                "Multi-segment %d/%d: frames %d–%d",
                seg_idx + 1, n_segs, seg_start, seg_end - 1,
            )

            # Re-anchor tracker at the segment's start keyframe
            self.tracker.reinitialize_at_keyframe(kf_start_idx, kf_start_masks)

            # Convert {mouse_id: mask} → {sam_obj_id: mask} using mouse_id as obj_id
            seed_masks = {mid: mask for mid, mask in kf_start_masks.items()}

            def seg_status_cb(sf, ef):
                pass  # already emitted above

            for result in self.engine.propagate_segment(
                video_path=self.video_path,
                fps=self.fps,
                frame_shape=self.frame_shape,
                start_frame=seg_start,
                end_frame=seg_end,
                initial_masks=seed_masks,
                frame_skip=self.frame_skip,
                status_callback=seg_status_cb,
            ):
                if self._abort:
                    break
                frames_done = self._process_result(result, frames_done, remaining, t_start)
                last_abs_frame = result["frame_index"]

            if not self._abort:
                self.chunk_complete.emit(last_abs_frame)
                # Between segments: re-prompt any failed frames from this segment
                if self._failed_frames:
                    self._adaptive_reprompt()

        # Handle tail beyond last keyframe (extend the final segment to end of video)
        if not self._abort and self.keyframes:
            kf_last_idx, kf_last_masks = self.keyframes[-1]
            tail_start = kf_last_idx
            tail_end = self.frame_count
            if tail_start < tail_end:
                self.status.emit(
                    f"Final segment — frames {tail_start}–{tail_end - 1}…"
                )
                logger.info("Final tail segment: frames %d–%d", tail_start, tail_end - 1)
                self.tracker.reinitialize_at_keyframe(kf_last_idx, kf_last_masks)
                seed_masks = {mid: mask for mid, mask in kf_last_masks.items()}

                for result in self.engine.propagate_segment(
                    video_path=self.video_path,
                    fps=self.fps,
                    frame_shape=self.frame_shape,
                    start_frame=tail_start,
                    end_frame=tail_end,
                    initial_masks=seed_masks,
                    frame_skip=self.frame_skip,
                ):
                    if self._abort:
                        break
                    frames_done = self._process_result(result, frames_done, remaining, t_start)
                    last_abs_frame = result["frame_index"]

                if not self._abort:
                    self.chunk_complete.emit(last_abs_frame)

        logger.info(f"Multi-segment tracking complete: {frames_done} frames")

    # ── Single-session path (short videos, no keyframes) ──────────────────────

    def _run_single_session(self, remaining: int) -> None:
        self.status.emit("Loading video into SAM3 — please wait…")
        self.engine.start_session(self.video_path)

        # Seed from initial masks (from the user's prompt/assignment step).
        # Without this, SAM3 has no prompts and propagation produces nothing.
        if self.initial_masks:
            self.engine._ensure_cached_frame_outputs(all_frames=True)
            for obj_id, mask in self.initial_masks.items():
                self.engine.add_mask_prompt(
                    self.start_frame, mask, obj_id, self.frame_shape,
                )
            self.status.emit("Seeded SAM3 — propagating…")

        frames_done = 0
        t_start = time.time()

        for result in self.engine.propagate(
            direction="forward",
            start_frame=self.start_frame,
        ):
            if self._abort:
                break
            frames_done = self._process_result(result, frames_done, remaining, t_start)

        logger.info(f"Single-session tracking complete: {frames_done} frames")

    # ── Chunked path (long videos, no keyframes) ───────────────────────────────

    def _run_chunked(self, remaining: int) -> None:
        frames_done = 0
        t_start = time.time()

        prev_chunk_last: list[int] = []

        def chunk_cb(chunk_idx, n_chunks, start_f, end_f):
            if chunk_idx > 0 and prev_chunk_last:
                self.chunk_complete.emit(prev_chunk_last[0])
            prev_chunk_last.clear()
            prev_chunk_last.append(end_f - 1)
            msg = (
                f"Chunk {chunk_idx + 1}/{n_chunks} — "
                f"loading frames {start_f}–{end_f - 1} into SAM3…"
            )
            self.status.emit(msg)
            logger.info(msg)

        last_abs_frame = self.start_frame
        for result in self.engine.propagate_chunked(
            video_path=self.video_path,
            fps=self.fps,
            frame_count=self.frame_count,
            frame_shape=self.frame_shape,
            initial_masks=self.initial_masks,
            chunk_frames=self.chunk_size,
            start_frame=self.start_frame,
            chunk_status_callback=chunk_cb,
            frame_skip=self.frame_skip,
        ):
            if self._abort:
                break
            frames_done = self._process_result(result, frames_done, remaining, t_start)
            last_abs_frame = result["frame_index"]

        if not self._abort:
            self.chunk_complete.emit(last_abs_frame)

        logger.info(f"Chunked tracking complete: {frames_done} frames")

    # ── Adaptive re-prompting (between chunks/segments) ────────────────────────

    def _adaptive_reprompt(self) -> None:
        """
        Re-prompt failed frames with alternative text prompts.

        Called between chunks/segments when the engine has no active session.
        For each failed frame, tries up to 5 prompts from AUTO_PROMPTS; if any
        produces masks passing size validation, updates the tracker state.
        """
        if not self._failed_frames or not self.size_validator:
            return
        if self._abort:
            return

        n = len(self._failed_frames)
        self.status.emit(f"Adaptive re-prompting {n} frame(s)…")
        logger.info("Adaptive re-prompting %d failed frame(s)", n)

        # Close any lingering session
        if self.engine.session_id is not None:
            self.engine.close_session()

        fixed = 0
        for frame_idx in list(self._failed_frames):
            if self._abort:
                break

            state = self.tracker.get_state_at(frame_idx)
            if state is None:
                continue

            best_masks = None
            best_prompt = None

            for prompt in AUTO_PROMPTS[:5]:
                try:
                    self.engine.start_session_on_frame(
                        self.video_path, frame_index=frame_idx,
                    )
                    outputs = self.engine.add_text_prompt(0, prompt)
                    candidate = self.engine.outputs_to_masks(
                        outputs, self.frame_shape,
                    )
                    self.engine.close_session()
                except Exception as e:
                    logger.debug("Re-prompt '%s' on frame %d failed: %s", prompt, frame_idx, e)
                    if self.engine.session_id is not None:
                        self.engine.close_session()
                    continue

                if not candidate:
                    continue

                # Match candidates to entities by area proximity
                matched = self._match_reprompt_masks(candidate, state)
                if not matched:
                    continue

                ok, _ = self.size_validator.validate(matched)
                if all(ok.values()):
                    best_masks = matched
                    best_prompt = prompt
                    break  # found valid masks

            if best_masks is not None:
                self.tracker.replace_masks(frame_idx, best_masks)
                fixed += 1
                logger.info(
                    "Frame %d: adaptive re-prompt '%s' succeeded",
                    frame_idx, best_prompt,
                )

        self._reprompt_count += fixed
        self._failed_frames.clear()
        if fixed:
            logger.info("Adaptive re-prompting fixed %d/%d frame(s)", fixed, n)

    def _match_reprompt_masks(
        self,
        candidate_masks: dict[int, "np.ndarray"],
        state: TrackState,
    ) -> dict[int, "np.ndarray"]:
        """
        Match re-prompted candidate masks to existing entity IDs by area
        similarity to the size validator's reference areas.

        Returns {entity_id: mask} for matched candidates, or empty dict
        if no viable matching is found.
        """
        if not self.size_validator or not candidate_masks or not state.masks:
            return {}

        entity_ids = sorted(state.masks.keys())
        cand_ids = list(candidate_masks.keys())

        if len(cand_ids) < len(entity_ids):
            return {}  # fewer candidates than entities

        # Build cost matrix: area difference from reference
        import scipy.optimize
        n_ent = len(entity_ids)
        n_cand = len(cand_ids)
        cost = np.zeros((n_ent, n_cand), dtype=np.float64)

        for i, eid in enumerate(entity_ids):
            ref = self.size_validator.ref_areas.get(eid, 0)
            if ref <= 0:
                continue
            for j, cid in enumerate(cand_ids):
                area = int(candidate_masks[cid].sum())
                cost[i, j] = abs(area - ref) / max(ref, 1)

        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)
        result: dict[int, np.ndarray] = {}
        for r, c in zip(row_ind, col_ind):
            result[entity_ids[r]] = candidate_masks[cand_ids[c]]

        return result

    # ── Shared per-frame logic ─────────────────────────────────────────────────

    def _build_filter_fn(self):
        """Build the mask filter function from current settings."""
        if self.bypass_filters:
            return None
        return lambda masks: self.engine.filter_masks(
            masks, self.frame_shape,
            max_area_frac=self.max_area_frac,
            max_edge_frac=self.max_edge_frac,
            use_area_filter=self.use_area_filter,
            use_edge_filter=self.use_edge_filter,
            max_detections=self.tracker.n_mice,
        )

    def _process_result(
        self,
        result: dict,
        frames_done: int,
        remaining: int,
        t_start: float,
    ) -> int:
        frame_idx = result["frame_index"]
        outputs = result["outputs"]

        filter_fn = self._build_filter_fn()
        state = self.tracker.assign_frame(
            frame_idx, outputs, self.frame_shape, mask_filter_fn=filter_fn
        )

        # Size validation + recovery (watershed / connected components)
        if self.size_validator and self.size_validator.any_reference() and state.masks:
            ok, reasons = self.size_validator.validate(state.masks)
            if not all(ok.values()):
                bad = {eid: r for eid, r in reasons.items() if r}
                logger.info(
                    "Frame %d: size mismatch %s — running recovery",
                    frame_idx, bad,
                )
                corrected = recover_masks(state.masks, self.size_validator)
                if not _mask_dicts_equal(corrected, state.masks):
                    self._recovery_count += 1
                    state = self.tracker.replace_masks(frame_idx, corrected)
                else:
                    # Watershed/CC failed — queue for adaptive re-prompting
                    self._failed_frames.append(frame_idx)

        frames_done += 1

        elapsed = time.time() - t_start
        fps_proc = frames_done / max(elapsed, 1e-6)
        eta_s = (remaining - frames_done) / max(fps_proc, 1e-6)
        percent = min(99, int(frames_done * 100 / remaining))

        self.progress.emit(percent, eta_s)
        self.frame_result.emit(frame_idx, state)

        return frames_done
