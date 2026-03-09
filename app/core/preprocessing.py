"""ffmpeg-based video preprocessing (crop, resize, time window)."""

import logging
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level cache for GPU codec probe result
_cached_codec: Optional[str] = None


def detect_gpu_codec() -> str:
    """
    Return 'h264_nvenc' if NVENC hardware encoding is available, else 'libx264'.
    Result is cached after the first call (single ffmpeg probe per process).
    """
    global _cached_codec
    if _cached_codec is not None:
        return _cached_codec
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "color=black:size=16x16:rate=1",
                "-t", "0.1",
                "-c:v", "h264_nvenc",
                "-f", "null", "-",
            ],
            capture_output=True,
            timeout=10,
        )
        _cached_codec = "h264_nvenc" if result.returncode == 0 else "libx264"
    except Exception:
        _cached_codec = "libx264"
    logger.info(f"ffmpeg video codec selected: {_cached_codec}")
    return _cached_codec


@dataclass
class PreprocessingConfig:
    """Configuration for ffmpeg preprocessing."""
    output_path: str = ""
    crop_x: Optional[int] = None
    crop_y: Optional[int] = None
    crop_w: Optional[int] = None
    crop_h: Optional[int] = None
    resize_w: Optional[int] = None
    resize_h: Optional[int] = None
    start_time_s: Optional[float] = None
    end_time_s: Optional[float] = None
    video_codec: str = "libx264"
    crf: int = 18

    def has_crop(self) -> bool:
        return all(
            v is not None
            for v in [self.crop_x, self.crop_y, self.crop_w, self.crop_h]
        )

    def has_resize(self) -> bool:
        return self.resize_w is not None or self.resize_h is not None

    def has_time_window(self) -> bool:
        return self.start_time_s is not None or self.end_time_s is not None


def build_ffmpeg_command(
    input_path: str,
    config: PreprocessingConfig,
    with_progress: bool = False,
) -> list[str]:
    """Build the ffmpeg command for preprocessing."""
    cmd = ["ffmpeg", "-y"]

    if config.start_time_s is not None:
        cmd += ["-ss", str(config.start_time_s)]

    cmd += ["-i", input_path]

    if config.end_time_s is not None:
        if config.start_time_s is not None:
            duration = config.end_time_s - config.start_time_s
        else:
            duration = config.end_time_s
        cmd += ["-t", str(duration)]

    # Build vf filter chain
    filters = []
    if config.has_crop():
        filters.append(
            f"crop={config.crop_w}:{config.crop_h}:{config.crop_x}:{config.crop_y}"
        )
    if config.has_resize():
        w = config.resize_w or -2
        h = config.resize_h or -2
        filters.append(f"scale={w}:{h}")

    if filters:
        cmd += ["-vf", ",".join(filters)]

    codec = config.video_codec if config.video_codec != "libx264" else detect_gpu_codec()
    cmd += ["-c:v", codec, "-crf", str(config.crf), "-an"]

    if with_progress:
        cmd += ["-progress", "pipe:1"]

    cmd.append(config.output_path)
    return cmd


def preprocess_video(
    input_path: str,
    config: PreprocessingConfig,
    progress_callback=None,
    total_duration_s: float = 0.0,
) -> str:
    """
    Run ffmpeg preprocessing. Returns output path.

    Args:
        progress_callback: Optional callable(percent: int) for progress.
        total_duration_s:  Video duration for progress calculation.
                           Required for real-time progress; ignored otherwise.
    """
    if not config.output_path:
        p = Path(input_path)
        config.output_path = str(p.parent / f"{p.stem}_processed{p.suffix}")

    use_progress = bool(progress_callback and total_duration_s > 0)
    cmd = build_ffmpeg_command(input_path, config, with_progress=use_progress)
    logger.info(f"Running ffmpeg: {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stderr_lines: list[str] = []

        def _drain_stderr() -> None:
            for line in proc.stderr:
                stderr_lines.append(line)

        err_thread = threading.Thread(target=_drain_stderr, daemon=True)
        err_thread.start()

        if use_progress:
            # ffmpeg writes key=value progress lines to stdout via -progress pipe:1
            time_re = re.compile(r"^out_time=(\d+):(\d+):([\d.]+)")
            for line in proc.stdout:
                m = time_re.match(line.strip())
                if m:
                    elapsed = (
                        int(m.group(1)) * 3600
                        + int(m.group(2)) * 60
                        + float(m.group(3))
                    )
                    pct = min(99, int(elapsed / total_duration_s * 100))
                    progress_callback(pct)

        proc.wait()
        err_thread.join()

        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{''.join(stderr_lines[-20:])}")
        logger.info(f"ffmpeg output: {config.output_path}")
        return config.output_path
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg and add it to PATH."
        )


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is installed and accessible."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
