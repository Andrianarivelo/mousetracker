"""DockArea setup and dock creation/toggle for MouseTracker Pro."""

import logging
from typing import Optional

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QMainWindow, QWidget

logger = logging.getLogger(__name__)

# Dock names — used as identifiers
DOCK_FILES = "Files"
DOCK_VIEWER = "Video Viewer"
DOCK_IDENTITY = "Identity"
DOCK_KEYPOINTS = "Keypoints"
DOCK_ROI = "ROI"
DOCK_SETTINGS = "Settings"
DOCK_PREPROCESS = "Preprocess"
DOCK_EXAMPLES = "Examples"


class DockManager:
    """
    Creates and manages all DockWidgets for MouseTracker Pro.

    Uses PySide6 QDockWidget rather than pyqtgraph DockArea so that
    docks integrate naturally with QMainWindow menus and resizing.
    """

    def __init__(self, main_window: QMainWindow) -> None:
        self._mw = main_window
        self._docks: dict[str, QDockWidget] = {}

    def create_dock(
        self,
        name: str,
        widget: QWidget,
        area: Qt.DockWidgetArea = Qt.LeftDockWidgetArea,
        min_width: int = 0,
        min_height: int = 0,
        features: Optional[QDockWidget.DockWidgetFeature] = None,
    ) -> QDockWidget:
        """Create a QDockWidget and add it to the main window."""
        dock = QDockWidget(name, self._mw)
        dock.setWidget(widget)
        dock.setObjectName(f"dock_{name.lower().replace(' ', '_')}")
        dock.setAllowedAreas(
            Qt.LeftDockWidgetArea
            | Qt.RightDockWidgetArea
            | Qt.BottomDockWidgetArea
            | Qt.TopDockWidgetArea
        )
        if features is None:
            features = QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        dock.setFeatures(features)
        if min_width > 0:
            widget.setMinimumWidth(min_width)
        if min_height > 0:
            widget.setMinimumHeight(min_height)
        self._mw.addDockWidget(area, dock)
        self._docks[name] = dock
        return dock

    def get_dock(self, name: str) -> Optional[QDockWidget]:
        return self._docks.get(name)

    def toggle_dock(self, name: str) -> None:
        dock = self._docks.get(name)
        if dock:
            dock.setVisible(not dock.isVisible())

    def tabify_docks(self, *names: str) -> None:
        """Tabify a sequence of docks together."""
        docks = [self._docks[n] for n in names if n in self._docks]
        for i in range(1, len(docks)):
            self._mw.tabifyDockWidget(docks[i - 1], docks[i])
        if docks:
            docks[0].raise_()

    def all_docks(self) -> dict[str, QDockWidget]:
        return dict(self._docks)
