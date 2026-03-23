"""Filter panel — confidence, sample frames, area/edge filters.

Displayed as a right-side activity panel in the sidebar.
"""

import logging
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.config import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_SAMPLE_FRAMES

logger = logging.getLogger(__name__)


def _help_label(tip: str) -> QLabel:
    lbl = QLabel("?")
    lbl.setToolTip(tip)
    lbl.setFixedSize(16, 16)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet(
        "QLabel { color: #a6adc8; border: 1px solid #585b70; border-radius: 8px;"
        " font-size: 10px; font-weight: bold; background: #313244; }"
        "QLabel:hover { color: #cba6f7; border-color: #cba6f7; }"
    )
    return lbl


class FilterPanel(QWidget):
    """SAM3 filter settings: confidence, sample frames, area/edge filters."""

    filter_changed = Signal()  # emitted whenever any filter value changes

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # ── Confidence ─────────────────────────────────────────────────────────
        conf_row = QHBoxLayout()
        conf_row.setSpacing(4)
        conf_row.addWidget(QLabel("Confidence:"))
        self.slider_confidence = QSlider(Qt.Horizontal)
        self.slider_confidence.setRange(0, 100)
        self.slider_confidence.setValue(int(DEFAULT_CONFIDENCE_THRESHOLD * 100))
        self.lbl_confidence = QLabel(f"{DEFAULT_CONFIDENCE_THRESHOLD:.2f}")
        self.lbl_confidence.setFixedWidth(32)
        self.lbl_confidence.setStyleSheet("color: #cdd6f4; font-weight: bold;")
        self.slider_confidence.valueChanged.connect(
            lambda v: self.lbl_confidence.setText(f"{v / 100:.2f}")
        )
        conf_row.addWidget(self.slider_confidence, 1)
        conf_row.addWidget(self.lbl_confidence)
        conf_row.addWidget(_help_label(
            "Detection confidence threshold (0.0–1.0).\n"
            "Lower = more detections. Default: 0.50."
        ))
        lay.addLayout(conf_row)

        # ── Sample frames ──────────────────────────────────────────────────────
        sf_row = QHBoxLayout()
        sf_row.setSpacing(4)
        sf_row.addWidget(QLabel("Sample frames:"))
        self.spin_sample_frames = QSpinBox()
        self.spin_sample_frames.setRange(1, 20)
        self.spin_sample_frames.setValue(DEFAULT_SAMPLE_FRAMES)
        self.spin_sample_frames.setFixedWidth(48)
        sf_row.addWidget(self.spin_sample_frames)
        sf_row.addStretch()
        sf_row.addWidget(_help_label(
            "Frames SAM3 uses to calibrate.\n3–10 recommended."
        ))
        lay.addLayout(sf_row)

        # ── Raw SAM bypass ─────────────────────────────────────────────────────
        self.chk_raw_sam = QCheckBox("Raw SAM (no filters)")
        self.chk_raw_sam.setChecked(False)
        self.chk_raw_sam.setToolTip(
            "Pass SAM3 masks directly — no area, edge, or size filtering."
        )
        lay.addWidget(self.chk_raw_sam)

        # ── Area filter ────────────────────────────────────────────────────────
        area_row = QHBoxLayout()
        area_row.setSpacing(4)
        self.chk_area_filter = QCheckBox("Max area:")
        self.chk_area_filter.setChecked(False)
        self.chk_area_filter.setToolTip("Reject masks covering more than this % of the frame")
        area_row.addWidget(self.chk_area_filter)
        self.slider_max_area = QSlider(Qt.Horizontal)
        self.slider_max_area.setRange(1, 100)
        self.slider_max_area.setValue(40)
        self.slider_max_area.setEnabled(False)
        self.lbl_max_area = QLabel("40%")
        self.lbl_max_area.setFixedWidth(30)
        self.lbl_max_area.setEnabled(False)
        self.lbl_max_area.setStyleSheet("color: #cdd6f4; font-weight: bold;")
        self.slider_max_area.valueChanged.connect(
            lambda v: self.lbl_max_area.setText(f"{v}%")
        )
        self.slider_max_area.valueChanged.connect(lambda: self.filter_changed.emit())
        self.chk_area_filter.toggled.connect(self.slider_max_area.setEnabled)
        self.chk_area_filter.toggled.connect(self.lbl_max_area.setEnabled)
        self.chk_area_filter.toggled.connect(lambda: self.filter_changed.emit())
        area_row.addWidget(self.slider_max_area, 1)
        area_row.addWidget(self.lbl_max_area)
        lay.addLayout(area_row)

        # ── Edge filter ────────────────────────────────────────────────────────
        edge_row = QHBoxLayout()
        edge_row.setSpacing(4)
        self.chk_edge_filter = QCheckBox("Max edge:")
        self.chk_edge_filter.setChecked(False)
        self.chk_edge_filter.setToolTip("Reject masks hugging the frame border")
        edge_row.addWidget(self.chk_edge_filter)
        self.slider_max_edge = QSlider(Qt.Horizontal)
        self.slider_max_edge.setRange(1, 50)
        self.slider_max_edge.setValue(28)
        self.slider_max_edge.setEnabled(False)
        self.lbl_max_edge = QLabel("28%")
        self.lbl_max_edge.setFixedWidth(30)
        self.lbl_max_edge.setEnabled(False)
        self.lbl_max_edge.setStyleSheet("color: #cdd6f4; font-weight: bold;")
        self.slider_max_edge.valueChanged.connect(
            lambda v: self.lbl_max_edge.setText(f"{v}%")
        )
        self.slider_max_edge.valueChanged.connect(lambda: self.filter_changed.emit())
        self.chk_edge_filter.toggled.connect(self.slider_max_edge.setEnabled)
        self.chk_edge_filter.toggled.connect(self.lbl_max_edge.setEnabled)
        self.chk_edge_filter.toggled.connect(lambda: self.filter_changed.emit())
        edge_row.addWidget(self.slider_max_edge, 1)
        edge_row.addWidget(self.lbl_max_edge)
        lay.addLayout(edge_row)

        # ── Wire raw-SAM bypass ────────────────────────────────────────────────
        def _on_raw_toggled(raw: bool) -> None:
            self.chk_area_filter.setEnabled(not raw)
            self.chk_edge_filter.setEnabled(not raw)
            area_on = not raw and self.chk_area_filter.isChecked()
            edge_on = not raw and self.chk_edge_filter.isChecked()
            self.slider_max_area.setEnabled(area_on)
            self.lbl_max_area.setEnabled(area_on)
            self.slider_max_edge.setEnabled(edge_on)
            self.lbl_max_edge.setEnabled(edge_on)
            self.filter_changed.emit()

        self.chk_raw_sam.toggled.connect(_on_raw_toggled)

        lay.addStretch()

    # ── Accessors ─────────────────────────────────────────────────────────────

    def confidence_threshold(self) -> float:
        return self.slider_confidence.value() / 100.0

    def sample_frames(self) -> int:
        return self.spin_sample_frames.value()

    def max_area_frac(self) -> float:
        return self.slider_max_area.value() / 100.0

    def area_filter_enabled(self) -> bool:
        return self.chk_area_filter.isChecked()

    def max_edge_frac(self) -> float:
        return self.slider_max_edge.value() / 100.0

    def edge_filter_enabled(self) -> bool:
        return self.chk_edge_filter.isChecked()

    def raw_sam_enabled(self) -> bool:
        return self.chk_raw_sam.isChecked()
