"""ffmpeg crop/resize/trim controls."""

import logging
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.core.preprocessing import PreprocessingConfig

logger = logging.getLogger(__name__)


class PreprocessingPanel(QWidget):
    """
    Controls for ffmpeg-based video preprocessing.

    Signals:
        preprocess_requested(PreprocessingConfig): User clicked Preprocess.
    """

    preprocess_requested = Signal(object)   # PreprocessingConfig
    crop_draw_requested = Signal(bool)      # True = enter draw mode
    crop_rect_changed = Signal(int, int, int, int)  # x, y, w, h

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._crop_aspect: float = 0.0    # w/h; 0 = unconstrained
        self._resize_aspect: float = 0.0
        self._updating: bool = False      # reentrancy guard
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ── Time window ───────────────────────────────────────────────────────
        time_group = QGroupBox("Time Window")
        time_form = QFormLayout(time_group)

        self.chk_time = QCheckBox("Enable")
        time_form.addRow("", self.chk_time)

        self.spin_start = QDoubleSpinBox()
        self.spin_start.setRange(0.0, 99999.0)
        self.spin_start.setDecimals(2)
        self.spin_start.setSuffix(" s")
        time_form.addRow("Start:", self.spin_start)

        self.spin_end = QDoubleSpinBox()
        self.spin_end.setRange(0.0, 99999.0)
        self.spin_end.setDecimals(2)
        self.spin_end.setSuffix(" s")
        time_form.addRow("End:", self.spin_end)

        layout.addWidget(time_group)

        # ── Crop ──────────────────────────────────────────────────────────────
        crop_group = QGroupBox("Crop")
        crop_form = QFormLayout(crop_group)

        self.chk_crop = QCheckBox("Enable")
        crop_form.addRow("", self.chk_crop)

        self.spin_crop_x = QSpinBox()
        self.spin_crop_x.setRange(0, 9999)
        crop_form.addRow("X offset:", self.spin_crop_x)

        self.spin_crop_y = QSpinBox()
        self.spin_crop_y.setRange(0, 9999)
        crop_form.addRow("Y offset:", self.spin_crop_y)

        self.chk_crop_lock = QCheckBox("Lock AR")
        self.chk_crop_lock.setToolTip("Lock crop aspect ratio")
        crop_form.addRow("", self.chk_crop_lock)

        self.spin_crop_w = QSpinBox()
        self.spin_crop_w.setRange(1, 9999)
        self.spin_crop_w.setValue(640)
        crop_form.addRow("Width:", self.spin_crop_w)

        self.spin_crop_h = QSpinBox()
        self.spin_crop_h.setRange(1, 9999)
        self.spin_crop_h.setValue(480)
        crop_form.addRow("Height:", self.spin_crop_h)

        self.spin_crop_w.valueChanged.connect(self._on_crop_w_changed)
        self.spin_crop_h.valueChanged.connect(self._on_crop_h_changed)
        self.spin_crop_x.valueChanged.connect(self._emit_crop_rect)
        self.spin_crop_y.valueChanged.connect(self._emit_crop_rect)
        self.spin_crop_w.valueChanged.connect(self._emit_crop_rect)
        self.spin_crop_h.valueChanged.connect(self._emit_crop_rect)
        self.chk_crop.stateChanged.connect(self._emit_crop_rect)

        # Draw-crop button
        self.btn_draw_crop = QPushButton("Draw Crop")
        self.btn_draw_crop.setToolTip(
            "Click to enter crop-draw mode, then click two corners on the video"
        )
        self.btn_draw_crop.setCheckable(True)
        self.btn_draw_crop.clicked.connect(self._on_draw_crop_clicked)
        crop_form.addRow("", self.btn_draw_crop)

        layout.addWidget(crop_group)

        # ── Resize ────────────────────────────────────────────────────────────
        resize_group = QGroupBox("Resize")
        resize_form = QFormLayout(resize_group)

        self.chk_resize = QCheckBox("Enable")
        resize_form.addRow("", self.chk_resize)

        self.chk_resize_lock = QCheckBox("Lock AR")
        self.chk_resize_lock.setToolTip("Lock resize aspect ratio")
        resize_form.addRow("", self.chk_resize_lock)

        self.spin_resize_w = QSpinBox()
        self.spin_resize_w.setRange(64, 9999)
        self.spin_resize_w.setValue(1280)
        resize_form.addRow("Width:", self.spin_resize_w)

        self.spin_resize_h = QSpinBox()
        self.spin_resize_h.setRange(64, 9999)
        self.spin_resize_h.setValue(720)
        resize_form.addRow("Height:", self.spin_resize_h)

        self.spin_resize_w.valueChanged.connect(self._on_resize_w_changed)
        self.spin_resize_h.valueChanged.connect(self._on_resize_h_changed)

        layout.addWidget(resize_group)

        # ── Apply ─────────────────────────────────────────────────────────────
        self.btn_apply = QPushButton("Preprocess Video")
        self.btn_apply.setToolTip("Run ffmpeg preprocessing")
        self.btn_apply.clicked.connect(self._on_apply)
        layout.addWidget(self.btn_apply)

        layout.addStretch()

    # ── Aspect-ratio lock helpers ──────────────────────────────────────────────

    def _on_crop_w_changed(self, w: int) -> None:
        if self._updating or not self.chk_crop_lock.isChecked() or self._crop_aspect <= 0:
            return
        self._updating = True
        self.spin_crop_h.setValue(max(1, round(w / self._crop_aspect)))
        self._updating = False

    def _on_crop_h_changed(self, h: int) -> None:
        if self._updating or not self.chk_crop_lock.isChecked() or self._crop_aspect <= 0:
            return
        self._updating = True
        self.spin_crop_w.setValue(max(1, round(h * self._crop_aspect)))
        self._updating = False

    def _on_resize_w_changed(self, w: int) -> None:
        if self._updating or not self.chk_resize_lock.isChecked() or self._resize_aspect <= 0:
            return
        self._updating = True
        self.spin_resize_h.setValue(max(64, round(w / self._resize_aspect)))
        self._updating = False

    def _on_resize_h_changed(self, h: int) -> None:
        if self._updating or not self.chk_resize_lock.isChecked() or self._resize_aspect <= 0:
            return
        self._updating = True
        self.spin_resize_w.setValue(max(64, round(h * self._resize_aspect)))
        self._updating = False

    def _on_draw_crop_clicked(self, checked: bool) -> None:
        self.chk_crop.setChecked(True)
        self.crop_draw_requested.emit(checked)

    def _emit_crop_rect(self) -> None:
        """Emit crop_rect_changed so the viewer can show a live overlay."""
        if self.chk_crop.isChecked():
            self.crop_rect_changed.emit(
                self.spin_crop_x.value(),
                self.spin_crop_y.value(),
                self.spin_crop_w.value(),
                self.spin_crop_h.value(),
            )
        else:
            self.crop_rect_changed.emit(0, 0, 0, 0)

    def fill_crop_from_rect(self, x: int, y: int, w: int, h: int) -> None:
        """Fill crop spinboxes from a drawn rectangle (from VideoViewer)."""
        self.btn_draw_crop.setChecked(False)
        self._updating = True
        self.spin_crop_x.setValue(x)
        self.spin_crop_y.setValue(y)
        self.spin_crop_w.setValue(max(1, w))
        self.spin_crop_h.setValue(max(1, h))
        self._updating = False
        self.chk_crop.setChecked(True)

    def _on_apply(self) -> None:
        config = self.build_config()
        self.preprocess_requested.emit(config)

    def build_config(self) -> PreprocessingConfig:
        config = PreprocessingConfig()
        if self.chk_time.isChecked():
            config.start_time_s = self.spin_start.value()
            config.end_time_s = self.spin_end.value()
        if self.chk_crop.isChecked():
            config.crop_x = self.spin_crop_x.value()
            config.crop_y = self.spin_crop_y.value()
            config.crop_w = self.spin_crop_w.value()
            config.crop_h = self.spin_crop_h.value()
        if self.chk_resize.isChecked():
            config.resize_w = self.spin_resize_w.value()
            config.resize_h = self.spin_resize_h.value()
        return config

    def populate_from_video(self, width: int, height: int, duration: float) -> None:
        """Pre-fill controls from loaded video info."""
        self.spin_crop_w.setValue(width)
        self.spin_crop_h.setValue(height)
        self.spin_end.setValue(duration)
        self.spin_resize_w.setValue(width)
        self.spin_resize_h.setValue(height)
        if height > 0:
            ar = width / height
            self._crop_aspect = ar
            self._resize_aspect = ar
