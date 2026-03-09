"""QThread worker for export operations."""

import logging
from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


class ExportWorker(QThread):
    """
    Runs export operations in the background.

    Signals:
        progress(int):       0-100 percent
        finished(bool, str): (success, error_message or output_path)
    """

    progress = Signal(int)
    finished = Signal(bool, str)

    def __init__(self, export_fn, *args, parent=None, **kwargs) -> None:
        super().__init__(parent)
        self._export_fn = export_fn
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            result = self._export_fn(
                *self._args,
                progress_callback=lambda p: self.progress.emit(p),
                **self._kwargs,
            )
            self.finished.emit(True, str(result or ""))
        except Exception as e:
            logger.exception(f"Export failed: {e}")
            self.finished.emit(False, str(e))
