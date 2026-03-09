"""QThread worker for ffmpeg preprocessing."""

import logging
from PySide6.QtCore import QThread, Signal

from app.core.preprocessing import PreprocessingConfig, preprocess_video

logger = logging.getLogger(__name__)


class PreprocessingWorker(QThread):
    """
    Runs ffmpeg preprocessing in the background.

    Signals:
        progress(int):         Percent complete (0–99 during run, 100 on finish).
        finished(bool, str):   (success, output_path or error_message)
    """

    progress = Signal(int)
    finished = Signal(bool, str)

    def __init__(
        self,
        input_path: str,
        config: PreprocessingConfig,
        total_duration_s: float = 0.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.input_path = input_path
        self.config = config
        self.total_duration_s = total_duration_s

    def run(self) -> None:
        try:
            output = preprocess_video(
                self.input_path,
                self.config,
                progress_callback=self.progress.emit,
                total_duration_s=self.total_duration_s,
            )
            self.progress.emit(100)
            self.finished.emit(True, output)
        except Exception as e:
            logger.exception(f"Preprocessing failed: {e}")
            self.finished.emit(False, str(e))
