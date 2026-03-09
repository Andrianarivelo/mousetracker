"""Tests for VideoViewer lasso interactions."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication

from app.ui.video_viewer import VideoViewer, _CropViewBox


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _FakeDragEvent:
    def __init__(
        self,
        x: int,
        y: int,
        *,
        start: bool = False,
        finish: bool = False,
    ) -> None:
        self._point = QPointF(float(x), float(y))
        self._start = start
        self._finish = finish
        self.accepted = False

    def button(self):
        return Qt.LeftButton

    def modifiers(self):
        return Qt.ControlModifier

    def scenePos(self):
        return self._point

    def accept(self) -> None:
        self.accepted = True

    def isStart(self) -> bool:
        return self._start

    def isFinish(self) -> bool:
        return self._finish


def test_crop_view_box_emits_ctrl_lasso_path() -> None:
    _app()
    box = _CropViewBox()
    box.mapSceneToView = lambda point: point
    events: list[tuple[str, int, int]] = []
    box.lasso_started.connect(lambda x, y: events.append(("start", x, y)))
    box.lasso_moved.connect(lambda x, y: events.append(("move", x, y)))
    box.lasso_finished.connect(lambda x, y: events.append(("finish", x, y)))

    box.mouseDragEvent(_FakeDragEvent(10, 12, start=True))
    box.mouseDragEvent(_FakeDragEvent(16, 20))
    box.mouseDragEvent(_FakeDragEvent(22, 28, finish=True))

    assert events == [
        ("start", 10, 12),
        ("move", 16, 20),
        ("move", 22, 28),
        ("finish", 22, 28),
    ]


def test_video_viewer_clips_lasso_points_to_frame_bounds() -> None:
    _app()
    viewer = VideoViewer()
    viewer.display_first_frame(np.zeros((12, 20, 3), dtype=np.uint8))
    points: list[tuple[int, int]] = []
    viewer.lasso_moved.connect(lambda x, y: points.append((x, y)))

    viewer._on_lasso_moved(-4, 99)

    assert points == [(0, 11)]
