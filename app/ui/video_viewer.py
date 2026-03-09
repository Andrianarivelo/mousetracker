"""pyqtgraph ImageView wrapper for video frame display."""

import logging
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout

logger = logging.getLogger(__name__)


class _CropViewBox(pg.ViewBox):
    """
    ViewBox subclass that supports a two-click crop-draw mode.

    In crop-draw mode:
      - First left click  → records start corner.
      - Second left click → finalises and emits crop_drawn(x, y, w, h).
      - Live preview is handled by VideoViewer via sigMouseMoved on the scene.
    """

    crop_drawn = Signal(int, int, int, int)  # x, y, w, h in image coords
    lasso_started = Signal(int, int)
    lasso_moved = Signal(int, int)
    lasso_finished = Signal(int, int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._crop_mode = False
        self._crop_start: Optional[tuple[int, int]] = None
        self._just_drew = False  # suppress click-through after second corner
        self._lasso_drag_active = False

    def set_crop_mode(self, enabled: bool) -> None:
        self._crop_mode = enabled
        self._crop_start = None
        self._just_drew = False

    def mousePressEvent(self, ev):
        if not self._crop_mode or ev.button() != Qt.LeftButton:
            super().mousePressEvent(ev)
            return
        ev.accept()
        pos = self.mapSceneToView(ev.scenePos())
        x, y = int(pos.x()), int(pos.y())
        if self._crop_start is None:
            self._crop_start = (x, y)
        else:
            x0, y0 = self._crop_start
            rx = min(x0, x)
            ry = min(y0, y)
            rw = abs(x - x0)
            rh = abs(y - y0)
            self._crop_start = None
            self._crop_mode = False
            self._just_drew = True
            if rw > 4 and rh > 4:
                self.crop_drawn.emit(rx, ry, rw, rh)

    def mouseDragEvent(self, ev, axis=None):
        if self._crop_mode or ev.button() != Qt.LeftButton:
            super().mouseDragEvent(ev, axis=axis)
            return

        ctrl_down = bool(ev.modifiers() & Qt.ControlModifier)
        if ev.isStart():
            self._lasso_drag_active = ctrl_down
        elif not self._lasso_drag_active:
            super().mouseDragEvent(ev, axis=axis)
            return

        if not self._lasso_drag_active:
            super().mouseDragEvent(ev, axis=axis)
            return

        ev.accept()
        pos = self.mapSceneToView(ev.scenePos())
        x, y = int(pos.x()), int(pos.y())

        if ev.isStart():
            self.lasso_started.emit(x, y)
        else:
            self.lasso_moved.emit(x, y)

        if ev.isFinish():
            self.lasso_finished.emit(x, y)
            self._lasso_drag_active = False


class VideoViewer(QWidget):
    """
    Wraps a pyqtgraph ImageView for displaying video frames with mask overlays.

    Extra features:
        - Crop-draw mode: two-click rectangle defines the crop region.
        - Crop rect overlay: draws a dashed rectangle on the image data space.

    Signals:
        click_point(int, int):          Left-click in image coordinates.
        right_click_point(int, int):    Right-click in image coordinates.
        crop_drawn(int, int, int, int): Crop finalised (x, y, w, h).
    """

    click_point = Signal(int, int)
    right_click_point = Signal(int, int)
    crop_drawn = Signal(int, int, int, int)
    lasso_started = Signal(int, int)
    lasso_moved = Signal(int, int)
    lasso_finished = Signal(int, int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()
        self._current_frame: Optional[np.ndarray] = None
        self._crop_rect: Optional[tuple[int, int, int, int]] = None

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

        self._view_box = _CropViewBox()
        self.image_view = pg.ImageView(view=self._view_box)
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.histogram.hide()
        self.image_view.getView().setBackgroundColor("#1e1e2e")

        layout.addWidget(self.image_view)

        # Crop rectangle overlay using a non-interactive ROI in data space
        self._crop_roi = pg.RectROI(
            pos=[0, 0], size=[1, 1],
            movable=False, rotatable=False, resizable=False,
            pen=pg.mkPen(color=(249, 226, 175), width=2, style=Qt.DashLine),
        )
        for h in list(self._crop_roi.handles):
            self._crop_roi.removeHandle(h["item"])
        self._crop_roi.setVisible(False)
        self.image_view.getView().addItem(self._crop_roi)

        # Scene signals
        scene = self.image_view.getView().scene()
        scene.sigMouseClicked.connect(self._on_scene_click)
        scene.sigMouseMoved.connect(self._on_scene_mouse_moved)  # live crop preview

        # Crop finalised
        self._view_box.crop_drawn.connect(self._on_crop_drawn)
        self._view_box.lasso_started.connect(self._on_lasso_started)
        self._view_box.lasso_moved.connect(self._on_lasso_moved)
        self._view_box.lasso_finished.connect(self._on_lasso_finished)

    # ── Frame display ────────────────────────────────────────────────────────

    def display_frame(self, frame_rgb: np.ndarray) -> None:
        """Display an RGB frame (HxWx3 uint8)."""
        self._current_frame = frame_rgb
        self.image_view.setImage(
            frame_rgb,
            autoRange=False,
            autoLevels=False,
            autoHistogramRange=False,
            levels=(0, 255),
        )

    def display_first_frame(self, frame_rgb: np.ndarray) -> None:
        """Display and auto-range on the first frame."""
        self._current_frame = frame_rgb
        self.image_view.setImage(
            frame_rgb,
            autoRange=True,
            autoLevels=False,
            autoHistogramRange=False,
            levels=(0, 255),
        )

    def clear(self) -> None:
        """Clear the display."""
        self._current_frame = None
        self.image_view.clear()

    # ── Crop-draw mode ───────────────────────────────────────────────────────

    def start_crop_draw_mode(self) -> None:
        """
        Enter two-click crop-draw mode.
        First click sets the start corner, second click finalises the rect.
        """
        self._view_box.set_crop_mode(True)

    def stop_crop_draw_mode(self) -> None:
        """Exit crop-draw mode without finalising."""
        self._view_box.set_crop_mode(False)
        if self._crop_rect is None:
            self._crop_roi.setVisible(False)

    def set_crop_rect(self, x: int, y: int, w: int, h: int) -> None:
        """Show a persistent crop-rect overlay (driven by spinbox values)."""
        if w > 0 and h > 0:
            self._crop_rect = (x, y, w, h)
            self._crop_roi.setPos([x, y])
            self._crop_roi.setSize([w, h])
            self._crop_roi.setVisible(True)
        else:
            self._crop_rect = None
            self._crop_roi.setVisible(False)

    def clear_crop_rect(self) -> None:
        self._crop_rect = None
        self._crop_roi.setVisible(False)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _on_scene_mouse_moved(self, scene_pos) -> None:
        """Update live crop preview while the user moves between two clicks."""
        if not self._view_box._crop_mode or self._view_box._crop_start is None:
            return
        view = self.image_view.getView()
        data_pos = view.mapSceneToView(scene_pos)
        x, y = int(data_pos.x()), int(data_pos.y())
        x0, y0 = self._view_box._crop_start
        rx, ry = min(x0, x), min(y0, y)
        rw, rh = abs(x - x0), abs(y - y0)
        if rw > 0 and rh > 0:
            self._crop_roi.setPos([rx, ry])
            self._crop_roi.setSize([rw, rh])
            self._crop_roi.setVisible(True)

    def _on_crop_drawn(self, x: int, y: int, w: int, h: int) -> None:
        """Crop rect finalised — persist overlay and re-emit to main_window."""
        self._crop_rect = (x, y, w, h)
        self._crop_roi.setPos([x, y])
        self._crop_roi.setSize([w, h])
        self._crop_roi.setVisible(True)
        self.crop_drawn.emit(x, y, w, h)

    def _clip_point_to_frame(self, x: int, y: int) -> Optional[tuple[int, int]]:
        if self._current_frame is None:
            return None
        h, w = self._current_frame.shape[:2]
        if w <= 0 or h <= 0:
            return None
        return max(0, min(x, w - 1)), max(0, min(y, h - 1))

    def _on_lasso_started(self, x: int, y: int) -> None:
        point = self._clip_point_to_frame(x, y)
        if point is not None:
            self.lasso_started.emit(*point)

    def _on_lasso_moved(self, x: int, y: int) -> None:
        point = self._clip_point_to_frame(x, y)
        if point is not None:
            self.lasso_moved.emit(*point)

    def _on_lasso_finished(self, x: int, y: int) -> None:
        point = self._clip_point_to_frame(x, y)
        if point is not None:
            self.lasso_finished.emit(*point)

    def _on_scene_click(self, event) -> None:
        """Convert scene mouse click to image coordinates and emit signal."""
        if self._view_box._crop_mode:
            return  # handled in _CropViewBox.mousePressEvent
        if self._view_box._just_drew:
            self._view_box._just_drew = False
            return  # swallow the click-through after the second corner
        if event.button() not in (Qt.LeftButton, Qt.RightButton):
            return
        view = self.image_view.getView()
        pos = view.mapSceneToView(event.scenePos())
        x, y = int(pos.x()), int(pos.y())
        if self._current_frame is not None:
            h, w = self._current_frame.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                if event.button() == Qt.LeftButton:
                    self.click_point.emit(x, y)
                else:
                    self.right_click_point.emit(x, y)

    def set_assignment_cursor(self, enabled: bool) -> None:
        """Show crosshair cursor while click-to-assign is active."""
        view = self.image_view.getView()
        if enabled:
            view.setCursor(Qt.CrossCursor)
        else:
            view.unsetCursor()

    @property
    def view_box(self) -> pg.ViewBox:
        return self.image_view.getView()
