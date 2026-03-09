"""Progress bar + ETA display widget."""

import logging
from typing import Optional

from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)


class ProgressWidget(QWidget):
    """Shows a progress bar, percentage, and estimated time remaining."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        row = QHBoxLayout()
        self.lbl_status = QLabel("Idle")
        self.lbl_status.setStyleSheet("color: #6c7086; font-size: 11px;")
        row.addWidget(self.lbl_status)
        row.addStretch()
        self.lbl_eta = QLabel("")
        self.lbl_eta.setStyleSheet("color: #6c7086; font-size: 11px;")
        row.addWidget(self.lbl_eta)
        layout.addLayout(row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(18)
        layout.addWidget(self.progress_bar)

    def update_progress(self, percent: int, status: str = "", eta_s: float = -1) -> None:
        """Update progress display.

        Args:
            percent: 0-100 progress value.
            status: Short status string.
            eta_s: Estimated seconds remaining (-1 to hide).
        """
        self.progress_bar.setValue(percent)
        if status:
            self.lbl_status.setText(status)
        if eta_s >= 0:
            m, s = divmod(int(eta_s), 60)
            self.lbl_eta.setText(f"ETA {m}:{s:02d}")
        else:
            self.lbl_eta.setText("")

    def reset(self, status: str = "Idle") -> None:
        self.progress_bar.setValue(0)
        self.lbl_status.setText(status)
        self.lbl_eta.setText("")

    def set_indeterminate(self, status: str = "Working…") -> None:
        self.progress_bar.setMaximum(0)
        self.lbl_status.setText(status)

    def set_determinate(self) -> None:
        self.progress_bar.setMaximum(100)
