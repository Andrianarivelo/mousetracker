"""ROI drawing overlay on the video viewer."""

import logging
import math
from enum import Enum
from typing import Optional

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF, QFont
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.roi_analyzer import ROIAnalyzer, ROIDefinition

logger = logging.getLogger(__name__)


class DrawMode(Enum):
    NONE = "none"
    POLYGON = "polygon"
    CIRCLE = "circle"
    RECTANGLE = "rectangle"


class ROIPanel(QWidget):
    """
    Panel for managing ROI definitions.

    Signals:
        rois_changed(): Emitted whenever ROIs are added/removed.
        draw_mode_changed(str): Emitted when drawing mode changes.
    """

    rois_changed = Signal()
    draw_mode_changed = Signal(str)

    def __init__(self, roi_analyzer: ROIAnalyzer, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.analyzer = roi_analyzer
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        lbl = QLabel("ROI Analysis")
        lbl.setStyleSheet("font-weight: bold; color: #cba6f7;")
        layout.addWidget(lbl)

        # Mode selector
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Draw:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Polygon", "Circle", "Rectangle"])
        self.combo_mode.currentTextChanged.connect(
            lambda t: self.draw_mode_changed.emit(t.lower())
        )
        mode_row.addWidget(self.combo_mode)
        layout.addLayout(mode_row)

        # ROI name
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.edit_roi_name = QLineEdit("Zone 1")
        name_row.addWidget(self.edit_roi_name)
        layout.addLayout(name_row)

        # Draw / Cancel buttons
        btn_row = QHBoxLayout()
        self.btn_start_draw = QPushButton("Start Drawing")
        self.btn_start_draw.clicked.connect(self._start_drawing)
        btn_row.addWidget(self.btn_start_draw)
        self.btn_finish_draw = QPushButton("Finish")
        self.btn_finish_draw.setObjectName("secondary_button")
        self.btn_finish_draw.setEnabled(False)
        self.btn_finish_draw.clicked.connect(self._finish_drawing)
        btn_row.addWidget(self.btn_finish_draw)
        layout.addLayout(btn_row)

        # ROI list
        self.roi_list = QListWidget()
        self.roi_list.setMaximumHeight(100)
        layout.addWidget(self.roi_list)

        # Remove button
        self.btn_remove = QPushButton("Remove Selected ROI")
        self.btn_remove.setObjectName("danger_button")
        self.btn_remove.clicked.connect(self._remove_roi)
        layout.addWidget(self.btn_remove)

        layout.addStretch()

    def _start_drawing(self) -> None:
        mode = self.combo_mode.currentText().lower()
        self.btn_start_draw.setEnabled(False)
        self.btn_finish_draw.setEnabled(True)
        self.draw_mode_changed.emit(mode)

    def _finish_drawing(self) -> None:
        self.btn_start_draw.setEnabled(True)
        self.btn_finish_draw.setEnabled(False)
        self.draw_mode_changed.emit("none")

    def _remove_roi(self) -> None:
        item = self.roi_list.currentItem()
        if item:
            name = item.text()
            self.analyzer.remove_roi(name)
            self.roi_list.takeItem(self.roi_list.row(item))
            self.rois_changed.emit()

    def add_roi_to_list(self, name: str) -> None:
        """Add a newly created ROI to the display list."""
        item = QListWidgetItem(name)
        self.roi_list.addItem(item)
        self.rois_changed.emit()
        self._finish_drawing()

    def current_mode(self) -> DrawMode:
        txt = self.combo_mode.currentText().lower()
        return DrawMode(txt)

    def current_roi_name(self) -> str:
        return self.edit_roi_name.text().strip() or "ROI"


class ROIOverlayManager:
    """
    Manages in-progress ROI drawing state.
    Called by the main window when the viewer receives clicks.
    """

    def __init__(self, roi_panel: ROIPanel, roi_analyzer: ROIAnalyzer) -> None:
        self.panel = roi_panel
        self.analyzer = roi_analyzer
        self._mode = DrawMode.NONE
        self._points: list[tuple[float, float]] = []
        self._circle_center: Optional[tuple[float, float]] = None

        roi_panel.draw_mode_changed.connect(self._on_mode_changed)

    def _on_mode_changed(self, mode_str: str) -> None:
        try:
            self._mode = DrawMode(mode_str)
        except ValueError:
            self._mode = DrawMode.NONE
        self._points.clear()
        self._circle_center = None

    def handle_click(self, x: float, y: float) -> Optional[str]:
        """
        Handle a click on the viewer during drawing mode.

        Returns the name of a completed ROI (if one was just finished), else None.
        """
        if self._mode == DrawMode.NONE:
            return None

        if self._mode == DrawMode.POLYGON:
            self._points.append((x, y))
            # 3+ points: double-click needed to close, but for simplicity
            # we'll finish on right-click (handled separately) or Finish button
            return None

        elif self._mode == DrawMode.RECTANGLE:
            self._points.append((x, y))
            if len(self._points) == 2:
                return self._finish_rectangle()

        elif self._mode == DrawMode.CIRCLE:
            if self._circle_center is None:
                self._circle_center = (x, y)
            else:
                return self._finish_circle(x, y)

        return None

    def handle_right_click(self, x: float, y: float) -> Optional[str]:
        """Right-click closes a polygon."""
        if self._mode == DrawMode.POLYGON and len(self._points) >= 3:
            return self._finish_polygon()
        return None

    def _finish_polygon(self) -> str:
        name = self.panel.current_roi_name()
        self.analyzer.add_roi(name, "polygon", list(self._points))
        self.panel.add_roi_to_list(name)
        self._points.clear()
        return name

    def _finish_rectangle(self) -> str:
        name = self.panel.current_roi_name()
        x1, y1 = self._points[0]
        x2, y2 = self._points[1]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        self.analyzer.add_roi(name, "rectangle", [(x1, y1, x2, y2)])
        self.panel.add_roi_to_list(name)
        self._points.clear()
        return name

    def _finish_circle(self, x2: float, y2: float) -> str:
        name = self.panel.current_roi_name()
        cx, cy = self._circle_center
        r = math.sqrt((x2 - cx) ** 2 + (y2 - cy) ** 2)
        self.analyzer.add_roi(name, "circle", [(cx, cy, r)])
        self.panel.add_roi_to_list(name)
        self._circle_center = None
        return name

    def get_preview_points(self) -> list[tuple[float, float]]:
        """Return current in-progress polygon points for preview rendering."""
        return list(self._points)

    def get_circle_center(self) -> Optional[tuple[float, float]]:
        return self._circle_center

    @property
    def is_drawing(self) -> bool:
        return self._mode != DrawMode.NONE


def draw_rois_on_frame(
    frame_rgb: np.ndarray,
    analyzer: ROIAnalyzer,
) -> np.ndarray:
    """Draw all ROIs as colored outlines on a frame."""
    import cv2
    out = frame_rgb.copy()
    for roi_name, roi in analyzer.rois.items():
        color = roi.color
        bgr = (color[2], color[1], color[0])

        if roi.roi_type == "polygon" and roi.data:
            pts = np.array([(int(x), int(y)) for x, y in roi.data], dtype=np.int32)
            cv2.polylines(out, [pts], isClosed=True, color=bgr, thickness=2)
            # Transparent fill
            overlay = out.copy()
            cv2.fillPoly(overlay, [pts], bgr)
            cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)
            # Label
            cx = int(np.mean([p[0] for p in roi.data]))
            cy = int(np.mean([p[1] for p in roi.data]))
            cv2.putText(out, roi_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)

        elif roi.roi_type == "circle" and roi.data:
            cx, cy, r = roi.data[0]
            cv2.circle(out, (int(cx), int(cy)), int(r), bgr, 2)
            cv2.putText(out, roi_name, (int(cx - r), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)

        elif roi.roi_type == "rectangle" and roi.data:
            x1, y1, x2, y2 = roi.data[0]
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), bgr, 2)
            cv2.putText(out, roi_name, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)

    return out
