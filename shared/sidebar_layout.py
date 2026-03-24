"""Simple sidebar + central layout used by MainWindow."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


@dataclass
class _Activity:
    key: str
    button: QToolButton
    panel: QWidget


class SidebarLayout(QWidget):
    """Container widget with a central area and an activity side panel."""

    def __init__(self, bar_position: str = "right", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bar_position = str(bar_position or "right").strip().lower()
        self._activities: dict[str, _Activity] = {}
        self._current_key: str | None = None

        self._root = QHBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 0)
        self._root.setSpacing(0)

        self._central_host = QWidget(self)
        self._central_layout = QVBoxLayout(self._central_host)
        self._central_layout.setContentsMargins(0, 0, 0, 0)
        self._central_layout.setSpacing(0)

        self._panel_stack = QStackedWidget(self)
        self._panel_stack.setMinimumWidth(320)
        self._panel_stack.setMaximumWidth(460)

        self._button_bar = QWidget(self)
        self._button_layout = QVBoxLayout(self._button_bar)
        self._button_layout.setContentsMargins(8, 8, 8, 8)
        self._button_layout.setSpacing(6)

        self._button_layout.addStretch(1)
        self._bar_wrapper = QWidget(self)
        self._bar_wrapper_layout = QVBoxLayout(self._bar_wrapper)
        self._bar_wrapper_layout.setContentsMargins(0, 0, 0, 0)
        self._bar_wrapper_layout.setSpacing(0)
        self._bar_wrapper_layout.addWidget(self._panel_stack, 1)
        self._bar_wrapper_layout.addWidget(self._button_bar, 0)

        if self._bar_position == "left":
            self._root.addWidget(self._bar_wrapper, 0)
            self._root.addWidget(self._central_host, 1)
        else:
            self._root.addWidget(self._central_host, 1)
            self._root.addWidget(self._bar_wrapper, 0)

    def set_central(self, widget: QWidget) -> None:
        while self._central_layout.count():
            item = self._central_layout.takeAt(0)
            if item and item.widget():
                old = item.widget()
                old.setParent(None)
        self._central_layout.addWidget(widget, 1)

    def add_bar_separator(self) -> None:
        sep = QFrame(self._button_bar)
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        self._button_layout.insertWidget(self._button_layout.count() - 1, sep)

    def add_activity(
        self,
        key: str,
        icon_name: str,
        short_label: str,
        title: str,
        panel: QWidget,
    ) -> None:
        if not key or panel is None:
            return

        button = QToolButton(self._button_bar)
        button.setText(short_label or key)
        button.setToolTip(title or short_label or key)
        button.setCheckable(True)
        button.setAutoExclusive(True)
        button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        button.setProperty("icon_name", icon_name)

        # Provide a visible title if panel has none.
        container = QWidget(self._panel_stack)
        panel_layout = QVBoxLayout(container)
        panel_layout.setContentsMargins(8, 8, 8, 8)
        panel_layout.setSpacing(8)
        if getattr(panel, "windowTitle", None):
            name = panel.windowTitle()
        else:
            name = ""
        if not name and title:
            header = QLabel(str(title), container)
            header.setObjectName("sidebar_panel_title")
            panel_layout.addWidget(header)
        panel_layout.addWidget(panel, 1)

        self._panel_stack.addWidget(container)
        self._button_layout.insertWidget(self._button_layout.count() - 1, button)

        self._activities[key] = _Activity(key=key, button=button, panel=container)
        button.clicked.connect(lambda _checked=False, k=key: self.show_activity(k))

        if self._current_key is None:
            self.show_activity(key)

    def show_activity(self, key: str) -> None:
        activity = self._activities.get(key)
        if activity is None:
            return
        self._panel_stack.setCurrentWidget(activity.panel)
        activity.button.setChecked(True)
        self._current_key = key

