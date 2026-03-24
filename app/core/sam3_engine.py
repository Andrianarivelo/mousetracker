"""SAM3 video predictor wrapper for mouse segmentation."""

import logging
import os
import tempfile
from contextlib import nullcontext
from typing import Generator, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SAM3Engine:
    """
    Wraps SAM3 video predictor for mouse segmentation.

    Lifecycle:
      1. start_session(video_path) → session_id
      2. add_text_prompt / add_point_prompt / add_mask_prompt
      3. propagate() → generator of per-frame results
      4. close_session()
    """

    def __init__(self) -> None:
        self.predictor = None
        self.session_id: Optional[str] = None
        self._loaded = False
        self._tmp_frame_jpeg: Optional[str] = None

    def load_model(self) -> None:
        """Load the SAM3 model (call once, may take a while)."""
        if self._loaded:
            return
        from app.config import get_sam3_checkpoint
        ckpt = get_sam3_checkpoint()
        if ckpt:
            logger.info(f"Loading SAM3 model from local checkpoint: {ckpt}")
        else:
            logger.info("No local checkpoint found — will attempt HuggingFace download")
        from sam3.model_builder import build_sam3_video_predictor
        # Passing checkpoint_path prevents HuggingFace download (see model_builder.py)
        try:
            self.predictor = build_sam3_video_predictor(checkpoint_path=ckpt)
        except Exception as error:
            if ckpt is None:
                raise RuntimeError(
                    "No local SAM3 checkpoint found.\n"
                    "Expected order: C:\\sam3_pt\\sam3.pt, then "
                    "D:\\Analysis\\sam3-weights\\sam3.pt, then user selection.\n"
                    "You can also set SAM3_CHECKPOINT_PATH before launching.\n"
                    f"Original error: {error}"
                ) from error
            raise
        self._loaded = True
        logger.info("SAM3 model loaded.")

    def is_loaded(self) -> bool:
        return self._loaded

    # ── Session management ────────────────────────────────────────────────────

    def start_session(self, video_path: str) -> str:
        """Load video into SAM3 and return session_id."""
        self._ensure_loaded()
        if self.session_id is not None:
            self.close_session()
        logger.info(f"Starting SAM3 session for: {video_path}")
        resp = self._handle_request(
            dict(type="start_session", resource_path=video_path)
        )
        self.session_id = resp["session_id"]
        logger.info(f"Session started: {self.session_id}")
        return self.session_id

    def start_session_on_frame(
        self,
        video_path: str,
        frame_index: int = 0,
    ) -> str:
        """
        Start a SAM3 session on a *single extracted frame* rather than the
        full video.  This is near-instant regardless of video length and is
        the correct approach for the interactive text-prompt / identity-setup
        step.

        SAM3 accepts a JPEG path as resource_path and treats it as a 1-frame
        video (see io_utils.load_image_as_single_frame_video).

        Returns:
            session_id for the temporary single-frame session.
        """
        import cv2

        self._ensure_loaded()

        # Extract the frame
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = cap.read()
        cap.release()
        if not ok:
            raise IOError(f"Cannot read frame {frame_index} from {video_path}")

        # Save as temp JPEG
        with tempfile.NamedTemporaryFile(
            suffix=".jpg", delete=False, prefix="mt_frame_"
        ) as tf:
            tmp_jpeg = tf.name
        cv2.imwrite(tmp_jpeg, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if self.session_id is not None:
            self.close_session()

        logger.info(
            f"Starting single-frame SAM3 session: frame {frame_index} → {tmp_jpeg}"
        )
        resp = self._handle_request(
            dict(type="start_session", resource_path=tmp_jpeg)
        )
        self.session_id = resp["session_id"]
        self._tmp_frame_jpeg = tmp_jpeg  # remember for cleanup
        logger.info(f"Session started: {self.session_id}")
        return self.session_id

    def close_session(self) -> None:
        """Close the current session and free resources."""
        if self.session_id and self.predictor:
            try:
                self._handle_request(
                    dict(type="close_session", session_id=self.session_id)
                )
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
            finally:
                self.session_id = None
                # Clean up any temp frame JPEG
                tmp = self._tmp_frame_jpeg
                if tmp:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
                    self._tmp_frame_jpeg = None

    def unload_model(self) -> None:
        """Release the loaded predictor so another process can use GPU memory."""
        self.close_session()
        self.predictor = None
        self._loaded = False
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def reset_session(self) -> None:
        """Reset tracked objects without closing the session."""
        if self.session_id and self.predictor:
            self._handle_request(
                dict(type="reset_session", session_id=self.session_id)
            )

    # ── Prompting ─────────────────────────────────────────────────────────────

    def add_text_prompt(self, frame_index: int, text: str) -> dict:
        """Add a text prompt (e.g., 'mouse') to detect objects on a frame."""
        self._ensure_session()
        resp = self._handle_request(
            dict(
                type="add_prompt",
                session_id=self.session_id,
                frame_index=frame_index,
                text=text,
            )
        )
        outputs = resp.get("outputs", {})
        logger.debug(
            f"Text prompt '{text}' on frame {frame_index}: "
            f"{len(outputs.get('out_obj_ids', []))} objects detected"
        )
        return outputs

    def add_point_prompt(
        self,
        frame_index: int,
        points: list[list[float]],
        point_labels: list[int],
        obj_id: Optional[int] = None,
        frame_shape: Optional[tuple[int, int]] = None,
    ) -> dict:
        """
        Add click prompts for a specific object.

        Args:
            points: [[x, y], ...] in pixel coordinates.
            point_labels: [1=foreground, 0=background, ...].
            obj_id: Object ID to refine (None = new object).
            frame_shape: (H, W) — required to normalize pixel coords to 0-1
                         for SAM3's tracker (rel_coordinates=True).
        """
        self._ensure_session()

        # SAM3 tracker expects points in normalized 0-1 coordinates
        # (rel_coordinates=True is the API default, which multiplies by image_size).
        norm_points = points
        if frame_shape is not None:
            h, w = frame_shape
            norm_points = [[x / w, y / h] for x, y in points]

        request = dict(
            type="add_prompt",
            session_id=self.session_id,
            frame_index=frame_index,
            points=norm_points,
            point_labels=point_labels,
        )
        if obj_id is not None:
            request["obj_id"] = obj_id
        resp = self._handle_request(request)
        return resp.get("outputs", {})

    def add_bbox_prompt(
        self,
        frame_index: int,
        bbox_xywh: list[float],
        obj_id: Optional[int] = None,
    ) -> dict:
        """Add a bounding-box prompt [x, y, w, h]."""
        self._ensure_session()
        request = dict(
            type="add_prompt",
            session_id=self.session_id,
            frame_index=frame_index,
            bounding_boxes=[bbox_xywh],
            bounding_box_labels=[1],
        )
        if obj_id is not None:
            request["obj_id"] = obj_id
        resp = self._handle_request(request)
        return resp.get("outputs", {})

    def remove_object(self, obj_id: int) -> None:
        """Remove a tracked object from the session."""
        self._ensure_session()
        self._handle_request(
            dict(type="remove_object", session_id=self.session_id, obj_id=obj_id)
        )

    # ── Propagation ───────────────────────────────────────────────────────────

    def propagate(
        self,
        direction: str = "forward",
        start_frame: Optional[int] = None,
        max_frames: Optional[int] = None,
    ) -> Generator[dict, None, None]:
        """
        Propagate tracking across frames. Yields per-frame dicts:
            {
                "frame_index": int,
                "outputs": {
                    "out_obj_ids": np.ndarray,
                    "out_binary_masks": np.ndarray,  # (N, H, W) bool
                    "out_probs": np.ndarray,
                    "out_boxes_xywh": np.ndarray,
                }
            }

        Args:
            direction: "forward", "backward", or "both".
            start_frame: Frame index to start from (None = beginning).
            max_frames: Max frames to track (None = all).
        """
        self._ensure_session()
        request = dict(
            type="propagate_in_video",
            session_id=self.session_id,
            propagation_direction=direction,
        )
        if start_frame is not None:
            request["start_frame_index"] = start_frame
        if max_frames is not None:
            request["max_frame_num_to_track"] = max_frames

        # SAM3 partial tracking paths expect cache slots to exist for each frame.
        self._ensure_cached_frame_outputs(all_frames=True)

        logger.info(
            f"Starting propagation: direction={direction}, "
            f"start={start_frame}, max={max_frames}"
        )
        yield from self._handle_stream_request(request)

    # ── Chunked propagation (for long videos) ────────────────────────────────

    def propagate_chunked(
        self,
        video_path: str,
        fps: float,
        frame_count: int,
        frame_shape: tuple[int, int],
        initial_masks: dict[int, np.ndarray],
        chunk_frames: int = 1000,
        start_frame: int = 0,
        chunk_status_callback=None,
        frame_skip: int = 1,
    ) -> Generator[dict, None, None]:
        """
        Track a long video by splitting it into overlapping chunks processed
        with separate SAM3 sessions. Each chunk is trimmed with ffmpeg so that
        SAM3 only loads ~chunk_frames frames at a time.

        At chunk boundaries the final masks from chunk N become mask prompts
        for chunk N+1, preserving identity across the seam.

        Yields the same dict format as propagate():
            {"frame_index": int (absolute), "outputs": dict}

        Args:
            video_path:    Path to original video.
            fps:           Video frame rate.
            frame_count:   Total number of frames.
            frame_shape:   (H, W) of original video.
            initial_masks: {sam_obj_id: binary_mask} from the user prompt on
                           the first chunk.  Used only for chunk 0.
            chunk_frames:  Frames per chunk (default 1000 ≈ 33 s @ 30 fps).
            start_frame:   Absolute frame index to begin from.
            chunk_status_callback: Optional fn(chunk_idx, n_chunks, start_f, end_f)
                           called at the start of each chunk.
            frame_skip:    Process only every Nth frame (default 1 = all frames).
                           Higher values give proportional speedup at the cost of
                           temporal resolution. Results are duplicated to fill gaps.
        """
        import subprocess
        import tempfile
        import os

        chunk_size = chunk_frames
        total = frame_count - start_frame
        n_chunks = max(1, (total + chunk_size - 1) // chunk_size)

        # Current masks to seed the next chunk: {sam_obj_id → binary_mask}
        carry_masks: dict[int, np.ndarray] = dict(initial_masks)

        for chunk_idx in range(n_chunks):
            chunk_start_abs = start_frame + chunk_idx * chunk_size
            chunk_end_abs = min(chunk_start_abs + chunk_size, frame_count)
            chunk_len = chunk_end_abs - chunk_start_abs

            if chunk_status_callback:
                chunk_status_callback(chunk_idx, n_chunks, chunk_start_abs, chunk_end_abs)

            logger.info(
                f"Chunk {chunk_idx + 1}/{n_chunks}: "
                f"frames {chunk_start_abs}–{chunk_end_abs - 1}"
            )

            # ── Trim chunk to a temp .mp4 ──────────────────────────────────
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".mp4", delete=False, prefix="mt_chunk_"
                ) as tf:
                    tmp_path = tf.name

                start_s = chunk_start_abs / fps
                duration_s = chunk_len / fps
                skip = max(1, int(frame_skip))
                target_fps = fps / skip
                from app.core.preprocessing import detect_gpu_codec
                codec = detect_gpu_codec()
                # -ss AFTER -i = output seeking (frame-accurate but slower).
                # Input seeking (-ss before -i) seeks to nearest keyframe,
                # which can be off by several frames, misaligning the seed mask.
                cmd = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-ss", f"{start_s:.6f}",
                    "-t", f"{duration_s:.6f}",
                    "-vf", f"fps={target_fps:.6f}",
                    "-c:v", codec, "-crf", "18",
                    "-an",       # drop audio
                    "-loglevel", "error",
                    tmp_path,
                ]
                subprocess.run(cmd, check=True, capture_output=True)

                # ── Open SAM3 session on the chunk clip ───────────────────
                self.start_session(tmp_path)

                # Ensure SAM3's cache keys exist for this chunk before prompts.
                self._ensure_cached_frame_outputs(all_frames=True)

                # ── Seed with masks from previous chunk ───────────────────
                for obj_id, mask in carry_masks.items():
                    # Resize carry mask if needed (chunk clip is same resolution)
                    self.add_mask_prompt(0, mask, obj_id, frame_shape)

                # ── Propagate the chunk ───────────────────────────────────
                last_outputs: Optional[dict] = None
                for result in self.propagate(direction="forward", start_frame=0):
                    local_frame = result["frame_index"]
                    abs_frame = chunk_start_abs + local_frame * skip
                    last_outputs = result["outputs"]
                    if abs_frame < chunk_end_abs:
                        yield {"frame_index": abs_frame, "outputs": last_outputs}

                # ── Carry forward last frame masks for next chunk seed ────
                if last_outputs is not None:
                    carry_masks = self.outputs_to_masks(last_outputs, frame_shape)

                self.close_session()

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    def propagate_segment(
        self,
        video_path: str,
        fps: float,
        frame_shape: tuple[int, int],
        start_frame: int,
        end_frame: int,
        initial_masks: dict[int, np.ndarray],
        frame_skip: int = 1,
        status_callback=None,
    ) -> Generator[dict, None, None]:
        """
        Track a single explicit video segment in isolation.

        Opens a fresh SAM3 session for [start_frame, end_frame), seeds it with
        initial_masks (keyed by SAM obj ID), propagates forward, then cleans up.

        Yields: {"frame_index": int (absolute), "outputs": dict}

        Args:
            video_path:     Path to original video.
            fps:            Video frame rate.
            frame_shape:    (H, W) of the original video.
            start_frame:    First frame to include (absolute index).
            end_frame:      One past the last frame to include (exclusive).
            initial_masks:  {sam_obj_id: binary_mask} to seed the session.
            frame_skip:     Process every Nth frame; results duplicated to fill gaps.
            status_callback: Optional fn(start_frame, end_frame) called before processing.
        """
        import subprocess
        import tempfile
        import os
        from app.core.preprocessing import detect_gpu_codec

        if status_callback:
            status_callback(start_frame, end_frame)

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".mp4", delete=False, prefix="mt_seg_"
            ) as tf:
                tmp_path = tf.name

            skip = max(1, int(frame_skip))
            target_fps = fps / skip
            start_s = start_frame / fps
            duration_s = (end_frame - start_frame) / fps
            codec = detect_gpu_codec()

            # -ss AFTER -i = frame-accurate output seeking
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-ss", f"{start_s:.6f}",
                "-t", f"{duration_s:.6f}",
                "-vf", f"fps={target_fps:.6f}",
                "-c:v", codec, "-crf", "18",
                "-an", "-loglevel", "error",
                tmp_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            self.start_session(tmp_path)
            self._ensure_cached_frame_outputs(all_frames=True)

            for obj_id, mask in initial_masks.items():
                self.add_mask_prompt(0, mask, obj_id, frame_shape)

            for result in self.propagate(direction="forward", start_frame=0):
                local_frame = result["frame_index"]
                abs_frame = start_frame + local_frame * skip
                if abs_frame < end_frame:
                    yield {"frame_index": abs_frame, "outputs": result["outputs"]}

            self.close_session()

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def add_mask_prompt(
        self,
        frame_index: int,
        mask: np.ndarray,
        obj_id: int,
        frame_shape: Optional[tuple[int, int]] = None,
    ) -> dict:
        """
        Add a binary mask as a prompt for a specific object.

        Prefer passing the full binary mask to SAM3's tracker so the
        initial track is anchored to the entire segmented object.
        Older backends without tracker-mask prompting fall back to the
        sampled-point approximation for compatibility.
        """
        self._ensure_session()

        if mask is None:
            return {}

        mask_array = np.asarray(mask)
        if mask_array.ndim > 2:
            mask_array = np.squeeze(mask_array)
        if mask_array.ndim != 2:
            raise ValueError(f"Expected a 2D mask, got shape {mask_array.shape}")

        mask_array = mask_array > 0
        if not mask_array.any():
            return {}

        if frame_shape is not None and mask_array.shape != frame_shape:
            import cv2

            fh, fw = frame_shape
            mask_array = cv2.resize(
                mask_array.astype(np.uint8),
                (fw, fh),
                interpolation=cv2.INTER_NEAREST,
            ) > 0

        if not mask_array.any():
            return {}

        self._ensure_cached_frame_outputs(frame_index=frame_index)

        try:
            outputs = self._add_tracker_mask_prompt(
                frame_index=frame_index,
                mask=mask_array,
                obj_id=obj_id,
            )
            if outputs is not None:
                return outputs
        except AssertionError as error:
            if "No cached outputs found" in str(error):
                logger.info(
                    "Priming tracker cache before mask prompt (frame=%d, obj_id=%d).",
                    frame_index,
                    obj_id,
                )
                self._prime_tracker_cache(frame_index)
                outputs = self._add_tracker_mask_prompt(
                    frame_index=frame_index,
                    mask=mask_array,
                    obj_id=obj_id,
                )
                if outputs is not None:
                    return outputs
            else:
                raise

        return self._add_mask_prompt_via_points(
            frame_index=frame_index,
            mask=mask_array,
            obj_id=obj_id,
            frame_shape=frame_shape,
        )

        points, labels = _sample_prompt_points(mask, fw, fh)
        if not points:
            return {}

        logger.debug(
            "add_mask_prompt: obj %d, frame %d — %d positive, %d negative points",
            obj_id, frame_index,
            sum(1 for l in labels if l == 1),
            sum(1 for l in labels if l == 0),
        )

        self._ensure_cached_frame_outputs(frame_index=frame_index)

        request = dict(
            type="add_prompt",
            session_id=self.session_id,
            frame_index=frame_index,
            obj_id=obj_id,
            points=points,
            point_labels=labels,
        )
        try:
            resp = self._handle_request(request)
        except AssertionError as error:
            if "No cached outputs found" in str(error):
                logger.info(
                    "Priming tracker cache before mask prompt (frame=%d, obj_id=%d).",
                    frame_index,
                    obj_id,
                )
                self._prime_tracker_cache(frame_index)
                resp = self._handle_request(request)
            else:
                raise
        return resp.get("outputs", {})

    # ── Utilities ─────────────────────────────────────────────────────────────

    def outputs_to_masks(
        self, outputs: dict, frame_shape: tuple[int, int]
    ) -> dict[int, np.ndarray]:
        """
        Convert raw SAM3 outputs to {obj_id: binary_mask (HxW bool)}.

        SAM3 returns masks at a potentially different resolution; we resize
        to frame_shape (H, W).
        """
        import cv2
        obj_ids = outputs.get("out_obj_ids", np.array([]))
        masks = outputs.get("out_binary_masks", np.array([]))
        result: dict[int, np.ndarray] = {}
        h, w = frame_shape
        for i, oid in enumerate(obj_ids):
            if i >= len(masks):
                break
            mask = masks[i].astype(np.uint8) * 255
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            result[int(oid)] = (mask > 0)
        return result

    def outputs_to_bboxes(self, outputs: dict) -> dict[int, tuple[int, int, int, int]]:
        """Convert raw outputs to {obj_id: (x1, y1, x2, y2)} bboxes."""
        obj_ids = outputs.get("out_obj_ids", np.array([]))
        boxes = outputs.get("out_boxes_xywh", np.array([]))
        result: dict[int, tuple[int, int, int, int]] = {}
        for i, oid in enumerate(obj_ids):
            if i >= len(boxes):
                break
            x, y, bw, bh = boxes[i]
            result[int(oid)] = (int(x), int(y), int(x + bw), int(y + bh))
        return result

    def filter_masks(
        self,
        masks: dict[int, np.ndarray],
        frame_shape: tuple[int, int],
        min_area_px: int = 0,
        max_area_frac: float = 0.40,
        max_edge_frac: float = 0.28,
        use_area_filter: bool = True,
        use_edge_filter: bool = True,
        max_detections: int = 0,
    ) -> dict[int, np.ndarray]:
        """
        Strip cage-wall blobs and tiny artifacts from masks.

        Strategy (per SAM object):
          1. Split the mask into connected components (8-connectivity).
          2. For each component, apply three tests — fail any → discard that blob:
               a. area < adaptive_min_area  (noise / tiny artefact)
               b. area / frame_area > max_area_frac  (whole background)
               c. (if use_edge_filter) fraction of the blob's pixels that lie
                  within the frame border zone > max_edge_frac (cage wall bar)
          3. Reassemble surviving components into the output mask.
          4. If no components survive, the object is dropped entirely.

        min_area_px: Explicit minimum area threshold. If 0 (default), uses an
            adaptive minimum of 0.15% of total frame area (~1380 px on 1280x720).
            This prevents tiny spots on bedding from being detected as mice.

        max_detections: if > 0, keep only the N *largest* surviving masks.
            After background/cage-wall blobs are removed by the area and edge
            filters, mice are the *largest* remaining objects. Tiny artifact
            spots that pass the minimum-area threshold are smaller than mice,
            so largest-N selection preferentially returns the animals.
        """
        import cv2

        h, w = frame_shape
        total_px = h * w

        # Adaptive minimum area: 0.15% of frame area.
        # For 1280x720 = 921600 px → min 1382 px (≈37x37 square).
        # For  640x480 = 307200 px → min  460 px (≈21x21 square).
        # Rejects tiny bedding spots that SAM3 sometimes detects.
        effective_min_area = min_area_px if min_area_px > 0 else max(200, int(total_px * 0.0015))

        # Narrow margin for edge detection: catches blobs 1-2 px off the exact
        # edge without being large enough to incorrectly penalise mice near corners.
        # ~1.5 % of shorter dimension, minimum 2 px (≈5–10 px for typical videos).
        margin = max(2, int(min(h, w) * 0.015))
        filtered: dict[int, np.ndarray] = {}

        for obj_id, mask in masks.items():
            mask_u8 = mask.astype(np.uint8)

            # Connected-component labelling (cv2 is always available)
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask_u8, connectivity=8
            )

            kept = np.zeros((h, w), dtype=bool)
            n_kept = 0

            for comp in range(1, n_labels):  # 0 = background
                area = int(stats[comp, cv2.CC_STAT_AREA])

                if area < effective_min_area:
                    continue
                if use_area_filter and area / total_px > max_area_frac:
                    continue

                blob = labels == comp

                if use_edge_filter:
                    # For each frame edge, compute what fraction of that edge's
                    # length the blob covers (within the narrow margin).
                    # This is "edge-line coverage": a cage bar running along the
                    # left side will cover nearly 100 % of the left edge; a mouse
                    # near the edge covers a small fraction of it.
                    top_cov   = float(blob[:margin, :].any(axis=0).sum()) / w
                    bot_cov   = float(blob[-margin:, :].any(axis=0).sum()) / w
                    left_cov  = float(blob[:, :margin].any(axis=1).sum()) / h
                    right_cov = float(blob[:, -margin:].any(axis=1).sum()) / h
                    if max(top_cov, bot_cov, left_cov, right_cov) > max_edge_frac:
                        logger.debug(
                            "filter_masks: obj %d component dropped "
                            "(edge cov top=%.2f bot=%.2f left=%.2f right=%.2f)",
                            obj_id, top_cov, bot_cov, left_cov, right_cov,
                        )
                        continue

                kept |= blob
                n_kept += 1

            if n_kept > 0:
                filtered[obj_id] = kept
            else:
                logger.debug(
                    "filter_masks: obj %d fully rejected (all %d components failed)",
                    obj_id, n_labels - 1,
                )

        n_in, n_out = len(masks), len(filtered)
        if n_in != n_out:
            logger.info(
                "filter_masks: kept %d/%d masks (removed %d cage-wall objects)",
                n_out, n_in, n_in - n_out,
            )

        # If max_detections is set, keep only the N *largest* surviving masks.
        # After cage-wall and background blobs are removed by the area/edge
        # filters, mice are the biggest remaining objects.  Tiny artifact spots
        # that SAM3 sometimes detects are smaller than a mouse and get dropped.
        if max_detections > 0 and len(filtered) > max_detections:
            sorted_by_area = sorted(filtered.items(), key=lambda kv: int(kv[1].sum()), reverse=True)
            filtered = dict(sorted_by_area[:max_detections])
            logger.info(
                "filter_masks: selected %d largest mask(s) from %d survivors",
                max_detections, n_out,
            )

        return filtered

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _cuda_autocast_context(self):
        """
        Return CUDA autocast context when available, otherwise a no-op context.
        SAM3 may internally produce bfloat16 features; keeping requests inside
        autocast avoids dtype mismatch crashes during prompt refinement.
        """
        try:
            import torch

            if torch.cuda.is_available():
                return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        except Exception:
            pass
        return nullcontext()

    @staticmethod
    def _is_bfloat16_mismatch_error(error: Exception) -> bool:
        text = str(error)
        return (
            "BFloat16" in text
            and "bias type" in text
            and "should be the same" in text
        )

    def _handle_request(self, request: dict) -> dict:
        self._ensure_loaded()
        try:
            with self._cuda_autocast_context():
                return self.predictor.handle_request(request)
        except RuntimeError as error:
            if self._is_bfloat16_mismatch_error(error):
                logger.warning(
                    "SAM3 dtype mismatch detected (%s). Retrying without autocast.",
                    error,
                )
                return self.predictor.handle_request(request)
            raise

    def _handle_stream_request(self, request: dict) -> Generator[dict, None, None]:
        self._ensure_loaded()
        with self._cuda_autocast_context():
            yield from self.predictor.handle_stream_request(request)

    def _prime_tracker_cache(self, frame_index: int = 0) -> None:
        # Ensure SAM3 has cache slots required by instance point prompts.
        self._ensure_cached_frame_outputs(all_frames=True)
        self._ensure_cached_frame_outputs(frame_index=frame_index)

    def _add_tracker_mask_prompt(
        self,
        frame_index: int,
        mask: np.ndarray,
        obj_id: int,
    ) -> Optional[dict]:
        state = self._get_inference_state()
        model = getattr(self.predictor, "model", None) if self.predictor else None
        tracker = getattr(model, "tracker", None) if model is not None else None
        if state is None or model is None or tracker is None:
            return None
        if not hasattr(tracker, "add_new_mask"):
            return None

        required_methods = (
            "_initialize_metadata",
            "_get_gpu_id_by_obj_id",
            "_prepare_backbone_feats",
            "_assign_new_det_to_gpus",
            "_init_new_tracker_state",
            "add_action_history",
            "_get_tracker_inference_states_by_obj_ids",
            "_build_tracker_output",
            "_cache_frame_outputs",
            "_postprocess_output",
        )
        if any(not hasattr(model, name) for name in required_methods):
            return None

        state.setdefault("tracker_inference_states", [])
        tracker_metadata = state.setdefault("tracker_metadata", {})
        if tracker_metadata == {}:
            tracker_metadata.update(model._initialize_metadata())

        def _run() -> dict:
            import torch

            obj_rank = model._get_gpu_id_by_obj_id(state, obj_id)
            model._prepare_backbone_feats(state, frame_index, reverse=False)

            if obj_rank is None:
                num_prev_obj = int(np.sum(tracker_metadata["num_obj_per_gpu"]))
                max_num_objects = int(getattr(model, "max_num_objects", num_prev_obj + 1))
                if num_prev_obj >= max_num_objects:
                    logger.warning(
                        "add_mask_prompt: cannot add obj_id=%d because tracker already has %d objects",
                        obj_id,
                        num_prev_obj,
                    )
                    return {}

                new_det_gpu_ids = model._assign_new_det_to_gpus(
                    new_det_num=1,
                    prev_workload_per_gpu=tracker_metadata["num_obj_per_gpu"],
                )
                obj_rank = int(new_det_gpu_ids[0])
                if getattr(model, "rank", 0) == obj_rank:
                    tracker_state = model._init_new_tracker_state(state)
                    state["tracker_inference_states"].append(tracker_state)

                tracker_metadata["obj_ids_per_gpu"][obj_rank] = np.concatenate(
                    [
                        tracker_metadata["obj_ids_per_gpu"][obj_rank],
                        np.array([obj_id], dtype=np.int64),
                    ]
                )
                tracker_metadata["num_obj_per_gpu"][obj_rank] = len(
                    tracker_metadata["obj_ids_per_gpu"][obj_rank]
                )
                tracker_metadata["obj_ids_all_gpu"] = np.concatenate(
                    tracker_metadata["obj_ids_per_gpu"]
                )
                tracker_metadata["max_obj_id"] = max(
                    int(tracker_metadata["max_obj_id"]),
                    int(obj_id),
                )
                model.add_action_history(
                    state,
                    "add",
                    frame_idx=frame_index,
                    obj_ids=[obj_id],
                )
            else:
                obj_rank = int(obj_rank)
                tracker_states = model._get_tracker_inference_states_by_obj_ids(
                    state,
                    [obj_id],
                )
                assert len(tracker_states) == 1, (
                    f"Expected exactly one tracker state for obj_id={obj_id}, "
                    f"found {len(tracker_states)}"
                )
                tracker_state = tracker_states[0]
                model.add_action_history(
                    state,
                    "refine",
                    frame_idx=frame_index,
                    obj_ids=[obj_id],
                )

            tracker_metadata["obj_id_to_score"][obj_id] = 1.0
            tracker_metadata["obj_id_to_tracker_score_frame_wise"][frame_index][obj_id] = 1.0

            if getattr(model, "rank", 0) == 0:
                rank0_metadata = tracker_metadata.get("rank0_metadata", {})
                if "removed_obj_ids" in rank0_metadata:
                    rank0_metadata["removed_obj_ids"].discard(obj_id)
                if "suppressed_obj_ids" in rank0_metadata:
                    for frame_id in rank0_metadata["suppressed_obj_ids"]:
                        rank0_metadata["suppressed_obj_ids"][frame_id].discard(obj_id)
                if "masklet_confirmation" in rank0_metadata:
                    obj_ids_all_gpu = tracker_metadata["obj_ids_all_gpu"]
                    obj_indices = np.where(obj_ids_all_gpu == obj_id)[0]
                    if len(obj_indices) > 0:
                        obj_idx = int(obj_indices[0])
                        confirmations = rank0_metadata["masklet_confirmation"]
                        if obj_idx < len(confirmations["status"]):
                            confirmations["status"][obj_idx] = 1
                            confirmations["consecutive_det_num"][obj_idx] = getattr(
                                model,
                                "masklet_confirmation_consecutive_det_thresh",
                                confirmations["consecutive_det_num"][obj_idx],
                            )

            if getattr(model, "rank", 0) != obj_rank:
                return {}

            mask_tensor = torch.as_tensor(mask, dtype=torch.float32)
            _, obj_ids, _, video_res_masks = tracker.add_new_mask(
                inference_state=tracker_state,
                frame_idx=frame_index,
                obj_id=obj_id,
                mask=mask_tensor,
                add_mask_to_memory=False,
            )
            tracker.propagate_in_video_preflight(tracker_state, run_mem_encoder=True)

            new_mask_data = None
            if len(obj_ids) > 0:
                try:
                    new_mask_data = (video_res_masks[obj_ids.index(obj_id)] > 0.0).to(
                        torch.bool
                    )
                except (ValueError, IndexError):
                    new_mask_data = None

            if getattr(model, "world_size", 1) > 1:
                data_list = [new_mask_data.cpu() if new_mask_data is not None else None]
                model.broadcast_python_obj_cpu(data_list, src=obj_rank)
                if data_list[0] is not None:
                    new_mask_data = data_list[0].to(model.device)

            if getattr(model, "rank", 0) != 0:
                return {}

            override = {obj_id: new_mask_data} if new_mask_data is not None else None
            obj_id_to_mask = model._build_tracker_output(state, frame_index, override)
            suppressed_obj_ids = tracker_metadata["rank0_metadata"]["suppressed_obj_ids"][
                frame_index
            ]
            out = {
                "obj_id_to_mask": obj_id_to_mask,
                "obj_id_to_score": tracker_metadata["obj_id_to_score"],
                "obj_id_to_tracker_score": tracker_metadata[
                    "obj_id_to_tracker_score_frame_wise"
                ][frame_index],
            }
            model._cache_frame_outputs(
                state,
                frame_index,
                obj_id_to_mask,
                suppressed_obj_ids=suppressed_obj_ids,
            )
            return model._postprocess_output(
                state,
                out,
                suppressed_obj_ids=suppressed_obj_ids,
            )

        try:
            import torch

            with torch.inference_mode():
                with self._cuda_autocast_context():
                    return _run()
        except RuntimeError as error:
            if self._is_bfloat16_mismatch_error(error):
                logger.warning(
                    "SAM3 dtype mismatch detected during mask prompt (%s). Retrying without autocast.",
                    error,
                )
                with torch.inference_mode():
                    return _run()
            raise

    def _add_mask_prompt_via_points(
        self,
        frame_index: int,
        mask: np.ndarray,
        obj_id: int,
        frame_shape: Optional[tuple[int, int]] = None,
    ) -> dict:
        if frame_shape is not None:
            fh, fw = frame_shape
        else:
            fh, fw = mask.shape[:2]

        points, labels = _sample_prompt_points(mask, fw, fh)
        if not points:
            return {}

        logger.debug(
            "add_mask_prompt fallback: obj %d, frame %d via %d positive and %d negative points",
            obj_id,
            frame_index,
            sum(1 for l in labels if l == 1),
            sum(1 for l in labels if l == 0),
        )

        resp = self._handle_request(
            dict(
                type="add_prompt",
                session_id=self.session_id,
                frame_index=frame_index,
                obj_id=obj_id,
                points=points,
                point_labels=labels,
            )
        )
        return resp.get("outputs", {})

    def _get_inference_state(self) -> Optional[dict]:
        if not self.predictor or not self.session_id:
            return None
        get_session = getattr(self.predictor, "_get_session", None)
        if get_session is None:
            return None
        try:
            session = get_session(self.session_id)
        except Exception:
            return None
        if not isinstance(session, dict):
            return None
        state = session.get("state")
        return state if isinstance(state, dict) else None

    def _ensure_cached_frame_outputs(
        self,
        frame_index: Optional[int] = None,
        all_frames: bool = False,
    ) -> None:
        """
        Populate missing cache keys to satisfy SAM3 partial propagation assertions.
        """
        state = self._get_inference_state()
        if state is None:
            return

        cache = state.get("cached_frame_outputs")
        if not isinstance(cache, dict):
            cache = {}
            state["cached_frame_outputs"] = cache

        if all_frames:
            num_frames = int(state.get("num_frames", 0) or 0)
            for idx in range(num_frames):
                cache.setdefault(idx, {})
            return

        if frame_index is not None and frame_index >= 0:
            cache.setdefault(int(frame_index), {})

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load_model()

    def _ensure_session(self) -> None:
        self._ensure_loaded()
        if self.session_id is None:
            raise RuntimeError("No active SAM3 session. Call start_session() first.")


# ── Module-level helpers ──────────────────────────────────────────────────────

def _sample_prompt_points(
    mask: np.ndarray,
    img_w: int,
    img_h: int,
    n_positive: int = 5,
    n_negative: int = 3,
) -> tuple[list[list[float]], list[int]]:
    """
    Sample normalized (0-1) prompt points from a binary mask.

    Returns (points, labels) where each point is [x_norm, y_norm]:
      - n_positive interior points (label=1): centroid + spread samples
      - n_negative exterior points (label=0): just outside the mask boundary

    These give SAM3's tracker a strong signal about what IS the object
    and what IS NOT (background/bedding).
    """
    import cv2

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [], []

    points: list[list[float]] = []
    labels: list[int] = []

    # ── Positive points (inside the mask) ─────────────────────────
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    points.append([cx / img_w, cy / img_h])
    labels.append(1)

    # Additional spread points via distance-transform sampling:
    # pick points that are deep inside the mask (far from edges).
    mask_u8 = mask.astype(np.uint8) * 255
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    # Flatten and pick top-distance pixels, then subsample evenly
    flat_indices = np.argsort(dist.ravel())[::-1]
    # Only consider pixels with dist > 0 (inside the mask)
    top_k = min(len(flat_indices), max(100, n_positive * 20))
    candidates = flat_indices[:top_k]
    candidates = candidates[dist.ravel()[candidates] > 0]

    if len(candidates) > 1:
        # Subsample evenly from the sorted candidates
        step = max(1, len(candidates) // (n_positive - 1))
        for i in range(1, n_positive):
            idx = min(i * step, len(candidates) - 1)
            py, px = np.unravel_index(candidates[idx], mask.shape)
            points.append([float(px) / img_w, float(py) / img_h])
            labels.append(1)

    # ── Negative points (just outside the mask boundary) ──────────
    # Dilate the mask and sample from the dilation ring
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(5, img_w // 40), max(5, img_h // 40)),
    )
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    border_ring = (dilated > 0) & (mask_u8 == 0)
    neg_ys, neg_xs = np.where(border_ring)

    if len(neg_xs) >= n_negative:
        # Sample evenly spaced negative points around the boundary
        step = max(1, len(neg_xs) // n_negative)
        for i in range(n_negative):
            idx = min(i * step, len(neg_xs) - 1)
            points.append([float(neg_xs[idx]) / img_w, float(neg_ys[idx]) / img_h])
            labels.append(0)
    elif len(neg_xs) > 0:
        for nx, ny in zip(neg_xs, neg_ys):
            points.append([float(nx) / img_w, float(ny) / img_h])
            labels.append(0)

    # Clamp all points to [0, 1]
    for pt in points:
        pt[0] = max(0.0, min(1.0, pt[0]))
        pt[1] = max(0.0, min(1.0, pt[1]))

    return points, labels
