"""Frame scrubber / timeline widget with playback controls."""

import logging
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QStyle,
    QStyleOptionSlider,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class _MarkerSlider(QSlider):
    """
    QSlider subclass that paints colored tick marks for:
      - ID-switch frames (red)
      - Swap-range from marker (cyan)
      - Swap-range to marker (orange)
    """

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self._id_switch_frames: list[int] = []
        self._flag_from: Optional[int] = None
        self._flag_to: Optional[int] = None

    def set_id_switch_frames(self, frames: list[int]) -> None:
        self._id_switch_frames = frames
        self.update()

    def set_flag_from(self, frame: Optional[int]) -> None:
        self._flag_from = frame
        self.update()

    def set_flag_to(self, frame: Optional[int]) -> None:
        self._flag_to = frame
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        max_val = self.maximum()
        if max_val <= 0:
            return

        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        groove_rect = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self
        )
        gx = groove_rect.x()
        gw = groove_rect.width()
        gy = groove_rect.y()
        gh = groove_rect.height()

        def frame_to_x(f: int) -> int:
            return gx + int(f / max_val * gw)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)

        # ID-switch markers (red, thin)
        if self._id_switch_frames:
            pen = QPen(QColor("#f38ba8"), 1)
            painter.setPen(pen)
            for f in self._id_switch_frames:
                x = frame_to_x(f)
                painter.drawLine(x, gy, x, gy + gh)

        # Swap-range highlight (semi-transparent cyan band)
        if self._flag_from is not None and self._flag_to is not None:
            x1 = frame_to_x(min(self._flag_from, self._flag_to))
            x2 = frame_to_x(max(self._flag_from, self._flag_to))
            if x2 > x1:
                band_color = QColor("#89dceb")
                band_color.setAlpha(40)
                painter.fillRect(x1, gy, x2 - x1, gh, band_color)

        # Flag-from marker (cyan tick, taller)
        if self._flag_from is not None:
            pen = QPen(QColor("#89dceb"), 2)
            painter.setPen(pen)
            x = frame_to_x(self._flag_from)
            painter.drawLine(x, gy - 3, x, gy + gh + 3)

        # Flag-to marker (orange tick, taller)
        if self._flag_to is not None:
            pen = QPen(QColor("#fab387"), 2)
            painter.setPen(pen)
            x = frame_to_x(self._flag_to)
            painter.drawLine(x, gy - 3, x, gy + gh + 3)

        painter.end()


class TimelineWidget(QWidget):
    """
    Horizontal timeline with:
      - Frame slider (scrubbing)
      - Play / Pause button
      - Frame counter label
      - Timestamp label
      - Optional ID-switch markers (painted as red ticks)

    Signals:
        frame_changed(int): Emitted when the user scrubs or playback advances.
        playback_toggled(bool): Emitted when play/pause state changes.
    """

    frame_changed = Signal(int)
    playback_toggled = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._frame_count = 0
        self._fps = 25.0
        self._playing = False
        self._id_switch_frames: list[int] = []
        self._flag_from: Optional[int] = None  # swap range start
        self._flag_to: Optional[int] = None    # swap range end
        self._setup_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Controls row
        controls = QHBoxLayout()
        controls.setSpacing(6)

        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedWidth(36)
        self.btn_play.setToolTip("Play / Pause")
        self.btn_play.clicked.connect(self.toggle_playback)
        controls.addWidget(self.btn_play)

        self.btn_prev = QPushButton("◀")
        self.btn_prev.setFixedWidth(30)
        self.btn_prev.setToolTip("Previous frame")
        self.btn_prev.clicked.connect(self._step_backward)
        controls.addWidget(self.btn_prev)

        self.btn_next = QPushButton("▶|")
        self.btn_next.setFixedWidth(30)
        self.btn_next.setToolTip("Next frame")
        self.btn_next.clicked.connect(self._step_forward)
        controls.addWidget(self.btn_next)

        self.lbl_frame = QLabel("0 / 0")
        self.lbl_frame.setMinimumWidth(80)
        self.lbl_frame.setAlignment(Qt.AlignCenter)
        controls.addWidget(self.lbl_frame)

        self.lbl_time = QLabel("0.00 s")
        self.lbl_time.setMinimumWidth(70)
        self.lbl_time.setAlignment(Qt.AlignCenter)
        controls.addWidget(self.lbl_time)

        controls.addStretch()
        layout.addLayout(controls)

        # Slider row — custom subclass paints marker ticks
        self.slider = _MarkerSlider(Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_video(self, frame_count: int, fps: float) -> None:
        """Configure the timeline for a loaded video."""
        self._frame_count = frame_count
        self._fps = fps if fps > 0 else 25.0
        self.slider.setMaximum(max(0, frame_count - 1))
        self.slider.setValue(0)
        self.slider.setEnabled(frame_count > 0)
        self._update_labels(0)
        self.stop_playback()

    def set_frame(self, frame_idx: int, emit: bool = False) -> None:
        """Jump to a specific frame."""
        if emit:
            self.slider.setValue(frame_idx)
        else:
            self.slider.blockSignals(True)
            self.slider.setValue(frame_idx)
            self.slider.blockSignals(False)
            self._update_labels(frame_idx)

    def current_frame(self) -> int:
        return self.slider.value()

    def set_id_switch_markers(self, frame_indices: list[int]) -> None:
        """Store frame indices where ID switches were detected (painted as red ticks)."""
        self._id_switch_frames = frame_indices
        self.slider.set_id_switch_frames(frame_indices)

    def set_swap_flags(self, from_frame: Optional[int], to_frame: Optional[int]) -> None:
        """Set the cyan/orange flag markers that define the swap correction range."""
        self._flag_from = from_frame
        self._flag_to = to_frame
        self.slider.set_flag_from(from_frame)
        self.slider.set_flag_to(to_frame)

    def toggle_playback(self) -> None:
        if self._playing:
            self.stop_playback()
        else:
            self.start_playback()

    def start_playback(self) -> None:
        if self._frame_count == 0:
            return
        self._playing = True
        self.btn_play.setText("⏸")
        interval = max(1, int(1000 / self._fps))
        self._timer.start(interval)
        self.playback_toggled.emit(True)

    def stop_playback(self) -> None:
        self._playing = False
        self._timer.stop()
        self.btn_play.setText("▶")
        self.playback_toggled.emit(False)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _on_slider_changed(self, value: int) -> None:
        self._update_labels(value)
        self.frame_changed.emit(value)

    def _update_labels(self, frame_idx: int) -> None:
        self.lbl_frame.setText(f"{frame_idx} / {max(0, self._frame_count - 1)}")
        ts = frame_idx / self._fps if self._fps > 0 else 0
        self.lbl_time.setText(f"{ts:.2f} s")

    def _advance_frame(self) -> None:
        cur = self.slider.value()
        if cur >= self._frame_count - 1:
            self.stop_playback()
            return
        self.slider.setValue(cur + 1)

    def _step_forward(self) -> None:
        cur = self.slider.value()
        if cur < self._frame_count - 1:
            self.slider.setValue(cur + 1)

    def _step_backward(self) -> None:
        cur = self.slider.value()
        if cur > 0:
            self.slider.setValue(cur - 1)
