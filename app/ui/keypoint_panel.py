"""Keypoint selection checkboxes."""

import logging
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.config import ALL_KEYPOINTS, DEFAULT_KEYPOINTS, KEYPOINT_COLORS

logger = logging.getLogger(__name__)


class KeypointPanel(QWidget):
    """
    Checkbox list of available keypoints.

    Signals:
        selection_changed(list[str]): Emitted when checkbox states change.
        estimate_requested():         User clicked "Estimate Keypoints".
    """

    selection_changed = Signal(list)
    estimate_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._checkboxes: dict[str, QCheckBox] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        lbl = QLabel("Keypoints")
        lbl.setStyleSheet("font-weight: bold; color: #cba6f7;")
        layout.addWidget(lbl)

        # Scroll area for checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        container = QWidget()
        cb_layout = QVBoxLayout(container)
        cb_layout.setSpacing(2)
        cb_layout.setContentsMargins(0, 0, 0, 0)

        for kp in ALL_KEYPOINTS:
            row = QHBoxLayout()
            row.setSpacing(4)
            row.setContentsMargins(0, 0, 0, 0)
            # Color swatch
            r, g, b = KEYPOINT_COLORS.get(kp, (200, 200, 200))
            swatch = QLabel("")
            swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(
                f"background: rgb({r},{g},{b}); border-radius: 2px; border: 1px solid #585b70;"
            )
            row.addWidget(swatch)
            cb = QCheckBox(kp.replace("_", " ").title())
            cb.setChecked(kp in DEFAULT_KEYPOINTS)
            cb.stateChanged.connect(self._on_changed)
            row.addWidget(cb)
            row.addStretch()
            cb_layout.addLayout(row)
            self._checkboxes[kp] = cb

        cb_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

        # Select all / none row
        row = QHBoxLayout()
        btn_all = QPushButton("All")
        btn_all.setObjectName("secondary_button")
        btn_all.clicked.connect(self._select_all)
        row.addWidget(btn_all)
        btn_none = QPushButton("None")
        btn_none.setObjectName("secondary_button")
        btn_none.clicked.connect(self._select_none)
        row.addWidget(btn_none)
        layout.addLayout(row)

        self.btn_estimate = QPushButton("Estimate Keypoints")
        self.btn_estimate.clicked.connect(self.estimate_requested)
        layout.addWidget(self.btn_estimate)

    def _on_changed(self) -> None:
        self.selection_changed.emit(self.selected_keypoints())

    def _select_all(self) -> None:
        for cb in self._checkboxes.values():
            cb.setChecked(True)

    def _select_none(self) -> None:
        for cb in self._checkboxes.values():
            cb.setChecked(False)

    def selected_keypoints(self) -> list[str]:
        return [kp for kp, cb in self._checkboxes.items() if cb.isChecked()]
