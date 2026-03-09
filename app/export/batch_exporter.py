"""Batch processing pipeline for multiple videos."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class BatchJob:
    video_path: str
    n_mice: int
    text_prompt: str = "mouse"
    export_csv: bool = True
    export_h5: bool = False
    export_video: bool = False
    output_dir: str = ""
    status: str = "pending"  # pending | running | done | error
    error: str = ""


class BatchExporter:
    """
    Queue multiple video jobs and process them sequentially.
    Designed to be run in a QThread or subprocess.
    """

    def __init__(self) -> None:
        self.jobs: list[BatchJob] = []

    def add_job(self, job: BatchJob) -> None:
        self.jobs.append(job)

    def clear(self) -> None:
        self.jobs.clear()

    def run(
        self,
        engine,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> list[BatchJob]:
        """
        Process all pending jobs.

        Args:
            engine: SAM3Engine instance.
            progress_callback: fn(percent, status_message).

        Returns updated list of jobs with status set.
        """
        from app.core.tracker import IdentityTracker
        from app.core.video_io import VideoReader
        from app.export.csv_exporter import export_csv
        from app.export.h5_exporter import export_h5
        from app.export.video_exporter import export_video

        n_total = len(self.jobs)
        for i, job in enumerate(self.jobs):
            if job.status == "done":
                continue
            job.status = "running"
            status = f"Processing {Path(job.video_path).name} ({i+1}/{n_total})"
            if progress_callback:
                progress_callback(int(i * 100 / n_total), status)
            try:
                _process_single(job, engine, export_csv, export_h5, export_video)
                job.status = "done"
            except Exception as e:
                job.status = "error"
                job.error = str(e)
                logger.error(f"Batch job failed ({job.video_path}): {e}")

        if progress_callback:
            progress_callback(100, "Batch complete")
        return self.jobs


def _process_single(job, engine, export_csv_fn, export_h5_fn, export_video_fn) -> None:
    """Process a single batch job (blocking)."""
    from app.core.tracker import IdentityTracker
    from app.core.video_io import VideoReader

    out_dir = Path(job.output_dir or Path(job.video_path).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(job.video_path).stem

    reader = VideoReader(job.video_path)
    info = reader.info
    frame_shape = (info.height, info.width)
    reader.close()

    tracker = IdentityTracker(n_mice=job.n_mice)

    # Start session and add text prompt on frame 0
    engine.start_session(job.video_path)
    outputs = engine.add_text_prompt(0, job.text_prompt)

    # Initialize tracker with SAM IDs in order
    obj_ids = list(outputs.get("out_obj_ids", []))
    sam_masks = engine.outputs_to_masks(outputs, frame_shape)
    mapping = {i + 1: int(oid) for i, oid in enumerate(obj_ids[:job.n_mice])}
    tracker.initialize(0, sam_masks, mapping)

    # Propagate
    for result in engine.propagate(direction="forward", start_frame=0):
        tracker.assign_frame(result["frame_index"], result["outputs"], frame_shape)

    # Export
    if job.export_csv:
        export_csv_fn(
            tracker.history,
            str(out_dir / f"{stem}_tracking.csv"),
            info.fps,
        )
    if job.export_h5:
        export_h5_fn(
            tracker.history,
            str(out_dir / f"{stem}_masks.h5"),
            frame_shape,
            info.fps,
            job.video_path,
        )
    if job.export_video:
        export_video_fn(
            job.video_path,
            tracker.history,
            str(out_dir / f"{stem}_overlay.mp4"),
        )

    engine.close_session()
