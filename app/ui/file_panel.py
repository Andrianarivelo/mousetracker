"""Drag-drop zone + folder browser for video file loading."""

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm"}


class DropZone(QLabel):
    """A label that accepts video file drops."""

    files_dropped = Signal(list)  # list[str]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setText("Drop video(s)\nor folder here")
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setMinimumHeight(80)
        self.setStyleSheet(
            "border: 2px dashed #45475a; border-radius: 8px; color: #6c7086; "
            "font-size: 12px; padding: 8px; background: #313244;"
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(
                "border: 2px dashed #7c3aed; border-radius: 8px; color: #cdd6f4; "
                "font-size: 12px; padding: 8px; background: #313244;"
            )

    def dragLeaveEvent(self, event) -> None:
        self._reset_style()

    def dropEvent(self, event: QDropEvent) -> None:
        self._reset_style()
        paths = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).is_dir():
                paths.extend(_collect_videos_from_dir(path))
            elif Path(path).suffix.lower() in VIDEO_EXTENSIONS:
                paths.append(path)
        if paths:
            self.files_dropped.emit(paths)

    def _reset_style(self) -> None:
        self.setStyleSheet(
            "border: 2px dashed #45475a; border-radius: 8px; color: #6c7086; "
            "font-size: 12px; padding: 8px; background: #313244;"
        )


def _collect_videos_from_dir(directory: str) -> list[str]:
    """Collect all video files in a directory (non-recursive)."""
    return [
        str(p)
        for p in Path(directory).iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]


class FilePanel(QWidget):
    """
    Panel for loading video files.

    Signals:
        video_selected(str): Emitted when a video is chosen for tracking.
        videos_loaded(list[str]): Emitted when files are added to the queue.
    """

    video_selected = Signal(str)
    videos_loaded = Signal(list)
    batch_track_requested = Signal()  # user wants to track all listed videos

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._video_paths: list[str] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Drop zone
        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self._add_videos)
        layout.addWidget(self.drop_zone)

        # Browse button
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.setObjectName("secondary_button")
        self.btn_browse.clicked.connect(self._browse_files)
        layout.addWidget(self.btn_browse)

        # Video list
        self.video_list = QListWidget()
        self.video_list.setMinimumHeight(100)
        self.video_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.video_list)

        # Action buttons
        btn_row = QHBoxLayout()
        self.btn_open = QPushButton("Open Selected")
        self.btn_open.clicked.connect(self._open_selected)
        btn_row.addWidget(self.btn_open)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setObjectName("secondary_button")
        self.btn_clear.clicked.connect(self._clear_list)
        btn_row.addWidget(self.btn_clear)
        layout.addLayout(btn_row)

        # Batch track
        self.btn_batch_track = QPushButton("Batch Track All")
        self.btn_batch_track.setToolTip(
            "Auto-segment and track all listed videos sequentially,\n"
            "exporting CSV for each one."
        )
        self.btn_batch_track.clicked.connect(self.batch_track_requested)
        layout.addWidget(self.btn_batch_track)

        # Info label
        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("color: #6c7086; font-size: 11px;")
        layout.addWidget(self.lbl_info)

        layout.addStretch()

    def _browse_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v *.webm);;All Files (*)",
        )
        if paths:
            self._add_videos(paths)

    def _add_videos(self, paths: list[str]) -> None:
        added = []
        for p in paths:
            if p not in self._video_paths:
                self._video_paths.append(p)
                item = QListWidgetItem(Path(p).name)
                item.setData(Qt.UserRole, p)
                item.setToolTip(p)
                self.video_list.addItem(item)
                added.append(p)
        if added:
            self.videos_loaded.emit(added)
        logger.info(f"Added {len(added)} video(s) to the list")

    def _open_selected(self) -> None:
        item = self.video_list.currentItem()
        if item:
            path = item.data(Qt.UserRole)
            self.video_selected.emit(path)

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        self.video_selected.emit(path)

    def _clear_list(self) -> None:
        self._video_paths.clear()
        self.video_list.clear()
        self.lbl_info.setText("")

    def all_video_paths(self) -> list[str]:
        """Return all video paths currently in the list."""
        return list(self._video_paths)

    def set_video_info(self, info_text: str) -> None:
        """Display info about the currently loaded video."""
        self.lbl_info.setText(info_text)

    def set_batch_progress(self, idx: int, total: int, status: str) -> None:
        """Update the batch track button text with progress."""
        if idx < total:
            self.btn_batch_track.setText(f"Batch: {idx + 1}/{total} — {status}")
            self.btn_batch_track.setEnabled(False)
        else:
            self.btn_batch_track.setText("Batch Track All")
            self.btn_batch_track.setEnabled(True)
