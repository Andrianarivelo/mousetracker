"""Segmentation examples panel: frame list, thumbnails, prompt actions, and dataset management."""

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ExamplesPanel(QWidget):
    """
    Stores and manages segmentation examples selected by the user.

    Each example is keyed by frame index and can display a thumbnail.
    Also provides YOLO dataset management controls.
    """

    frame_selected = Signal(int)
    remove_requested = Signal(int)
    reprompt_requested = Signal(int)
    split_merged_requested = Signal()
    clear_requested = Signal()

    # Capture current tracked frame as a new example
    capture_frame_requested = Signal()

    # Dataset signals
    build_dataset_requested = Signal(str)           # dataset_dir
    add_to_dataset_requested = Signal(str, str)     # dataset_dir, split
    load_labels_requested = Signal(str)             # dataset_dir
    fine_tune_requested = Signal(str)               # dataset_dir

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._items_by_frame: dict[int, QListWidgetItem] = {}
        self._label_order: list[int] = []  # frames in order they were labeled
        self._sort_by_label_order: bool = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        title = QLabel("Segmentation Examples")
        title.setObjectName("title_label")
        layout.addWidget(title)

        hint = QLabel(
            "Select a frame to jump/re-prompt/remove. "
            "Tip: left-click adds foreground, right-click adds background to refine merged blobs."
        )
        hint.setObjectName("subtitle_label")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # Sort toggle
        sort_row = QHBoxLayout()
        sort_row.setSpacing(4)
        sort_row.addWidget(QLabel("Sort:"))
        self.btn_sort_frame = QPushButton("Frame #")
        self.btn_sort_frame.setObjectName("secondary_button")
        self.btn_sort_frame.setFixedHeight(20)
        self.btn_sort_frame.setCheckable(True)
        self.btn_sort_frame.setChecked(True)
        self.btn_sort_frame.setToolTip("Sort examples by frame number (ascending)")
        self.btn_sort_frame.clicked.connect(lambda: self._set_sort_mode(False))
        sort_row.addWidget(self.btn_sort_frame)
        self.btn_sort_labeled = QPushButton("Last Labeled")
        self.btn_sort_labeled.setObjectName("secondary_button")
        self.btn_sort_labeled.setFixedHeight(20)
        self.btn_sort_labeled.setCheckable(True)
        self.btn_sort_labeled.setChecked(False)
        self.btn_sort_labeled.setToolTip("Sort examples by labeling order (most recent at bottom)")
        self.btn_sort_labeled.clicked.connect(lambda: self._set_sort_mode(True))
        sort_row.addWidget(self.btn_sort_labeled)
        sort_row.addStretch()
        layout.addLayout(sort_row)

        self.list_examples = QListWidget()
        self.list_examples.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_examples.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.list_examples.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.list_examples, stretch=1)

        row_actions = QHBoxLayout()
        row_actions.setSpacing(4)
        self.btn_jump = QPushButton("Jump")
        self.btn_jump.setObjectName("secondary_button")
        self.btn_jump.clicked.connect(self._emit_jump)
        row_actions.addWidget(self.btn_jump)

        self.btn_reprompt = QPushButton("Re-prompt")
        self.btn_reprompt.setObjectName("secondary_button")
        self.btn_reprompt.clicked.connect(self._emit_reprompt)
        row_actions.addWidget(self.btn_reprompt)

        self.btn_remove = QPushButton("Remove")
        self.btn_remove.setObjectName("secondary_button")
        self.btn_remove.clicked.connect(self._emit_remove)
        row_actions.addWidget(self.btn_remove)
        layout.addLayout(row_actions)

        self.btn_split = QPushButton("Split Merged")
        self.btn_split.setObjectName("secondary_button")
        self.btn_split.setToolTip(
            "Run watershed / connected-component split on the current frame's masks.\n"
            "Use when two animals are merged into a single mask."
        )
        self.btn_split.clicked.connect(self.split_merged_requested)
        layout.addWidget(self.btn_split)

        self.btn_capture = QPushButton("Capture Current Frame")
        self.btn_capture.setObjectName("accent_button")
        self.btn_capture.setToolTip(
            "Add the current video frame (with its tracked masks) to the example set.\n"
            "Use this after tracking to capture good frames for the YOLO dataset."
        )
        self.btn_capture.clicked.connect(self.capture_frame_requested)
        layout.addWidget(self.btn_capture)

        self.btn_clear = QPushButton("Clear Examples")
        self.btn_clear.setObjectName("danger_button")
        self.btn_clear.clicked.connect(self.clear_requested)
        layout.addWidget(self.btn_clear)

        # ── Dataset management section ────────────────────────────────────────
        sep = QLabel("")
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: #444;")
        layout.addWidget(sep)

        ds_title = QLabel("Dataset & Training")
        ds_title.setObjectName("title_label")
        layout.addWidget(ds_title)

        # Dataset directory picker
        dir_row = QHBoxLayout()
        dir_row.setSpacing(4)
        self.txt_dataset_dir = QLineEdit()
        self.txt_dataset_dir.setPlaceholderText("Dataset directory…")
        dir_row.addWidget(self.txt_dataset_dir, stretch=1)
        self.btn_browse_dir = QPushButton("…")
        self.btn_browse_dir.setFixedWidth(28)
        self.btn_browse_dir.setToolTip("Browse for dataset directory")
        self.btn_browse_dir.clicked.connect(self._browse_dataset_dir)
        dir_row.addWidget(self.btn_browse_dir)
        layout.addLayout(dir_row)

        # Action buttons row
        ds_row = QHBoxLayout()
        ds_row.setSpacing(4)
        self.btn_build_ds = QPushButton("Build")
        self.btn_build_ds.setToolTip("Create YOLO dataset directory structure")
        self.btn_build_ds.setObjectName("secondary_button")
        self.btn_build_ds.clicked.connect(self._on_build_dataset)
        ds_row.addWidget(self.btn_build_ds)

        self.btn_add_ds = QPushButton("Add to Dataset")
        self.btn_add_ds.setToolTip("Export annotated frames to the dataset (train split)")
        self.btn_add_ds.setObjectName("accent_button")
        self.btn_add_ds.clicked.connect(self._on_add_to_dataset)
        ds_row.addWidget(self.btn_add_ds)

        self.btn_load_labels = QPushButton("Load Labels")
        self.btn_load_labels.setToolTip("Load YOLO labels for the current video back into the app")
        self.btn_load_labels.setObjectName("secondary_button")
        self.btn_load_labels.clicked.connect(self._on_load_labels)
        ds_row.addWidget(self.btn_load_labels)
        layout.addLayout(ds_row)

        self.btn_fine_tune = QPushButton("Fine-Tune SAM3...")
        self.btn_fine_tune.setObjectName("accent_button")
        self.btn_fine_tune.setToolTip(
            "Prepare the current dataset for SAM3 fine-tuning and open training settings."
        )
        self.btn_fine_tune.clicked.connect(self._on_fine_tune)
        layout.addWidget(self.btn_fine_tune)

        # Val split toggle
        split_row = QHBoxLayout()
        split_row.setSpacing(4)
        self.btn_add_val = QPushButton("Add to Val")
        self.btn_add_val.setToolTip("Export annotated frames to the val split instead of train")
        self.btn_add_val.setObjectName("secondary_button")
        self.btn_add_val.clicked.connect(self._on_add_to_val)
        split_row.addWidget(self.btn_add_val)
        split_row.addStretch()
        layout.addLayout(split_row)

        # Status label
        self.lbl_dataset_status = QLabel("")
        self.lbl_dataset_status.setObjectName("subtitle_label")
        self.lbl_dataset_status.setWordWrap(True)
        layout.addWidget(self.lbl_dataset_status)

        self._update_button_state()

    # ── Example list API ──────────────────────────────────────────────────────

    def _frame_label(self, frame_idx: int, note: str = "") -> str:
        label = f"Frame {frame_idx:06d}"
        if note:
            label = f"{label} | {note}"
        return label

    def upsert_example(
        self,
        frame_idx: int,
        thumbnail: Optional[QPixmap] = None,
        note: str = "",
    ) -> None:
        item = self._items_by_frame.get(frame_idx)
        label = self._frame_label(frame_idx, note)

        # Track labeling order (move to end if already present)
        if frame_idx in self._label_order:
            self._label_order.remove(frame_idx)
        self._label_order.append(frame_idx)

        if item is None:
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, int(frame_idx))
            self._items_by_frame[frame_idx] = item
            if self._sort_by_label_order:
                # Append at end (most recent)
                self.list_examples.addItem(item)
            else:
                # Insert sorted by frame index
                insert_row = self.list_examples.count()
                for row in range(self.list_examples.count()):
                    row_item = self.list_examples.item(row)
                    row_frame = int(row_item.data(Qt.ItemDataRole.UserRole))
                    if frame_idx < row_frame:
                        insert_row = row
                        break
                self.list_examples.insertItem(insert_row, item)
        else:
            item.setText(label)
            # In label-order mode, move item to bottom
            if self._sort_by_label_order:
                row = self.list_examples.row(item)
                if row >= 0:
                    self.list_examples.takeItem(row)
                    self.list_examples.addItem(item)
                    self._items_by_frame[frame_idx] = item

        if thumbnail is not None:
            item.setIcon(QIcon(thumbnail))
        # Auto-select the just-upserted item
        self.list_examples.setCurrentItem(item)
        self._update_button_state()

    def set_example_note(self, frame_idx: int, note: str) -> None:
        item = self._items_by_frame.get(frame_idx)
        if item is None:
            return
        item.setText(self._frame_label(frame_idx, note))

    def remove_example(self, frame_idx: int) -> None:
        item = self._items_by_frame.pop(frame_idx, None)
        if item is None:
            return
        row = self.list_examples.row(item)
        if row >= 0:
            self.list_examples.takeItem(row)
        if frame_idx in self._label_order:
            self._label_order.remove(frame_idx)
        self._update_button_state()

    def clear_examples(self) -> None:
        self._items_by_frame.clear()
        self._label_order.clear()
        self.list_examples.clear()
        self._update_button_state()

    def selected_frame(self) -> Optional[int]:
        item = self.list_examples.currentItem()
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        if value is None:
            return None
        return int(value)

    def frames(self) -> list[int]:
        return sorted(self._items_by_frame.keys())

    def dataset_dir(self) -> str:
        return self.txt_dataset_dir.text().strip()

    def set_dataset_status(self, text: str) -> None:
        self.lbl_dataset_status.setText(text)

    def set_split_mode_active(self, active: bool) -> None:
        if active:
            self.btn_split.setText("Cancel Split")
            self.btn_split.setToolTip(
                "Cancel the guided polygon split.\n"
                "While active: left-click to place vertices, right-click to apply the split.\n"
                "Shortcut: hold Ctrl and drag in the viewer for a freehand lasso split."
            )
        else:
            self.btn_split.setText("Split Merged")
            self.btn_split.setToolTip(
                "Draw a polygon around one part of a merged blob on the current frame.\n"
                "Right-click closes the polygon and splits the mask into inside vs outside.\n"
                "Shortcut: hold Ctrl and drag in the viewer for a freehand lasso split."
            )

    # ── Signal emitters ───────────────────────────────────────────────────────

    def _emit_jump(self) -> None:
        frame = self.selected_frame()
        if frame is not None:
            self.frame_selected.emit(frame)

    def _emit_reprompt(self) -> None:
        frame = self.selected_frame()
        if frame is not None:
            self.reprompt_requested.emit(frame)

    def _emit_remove(self) -> None:
        frame = self.selected_frame()
        if frame is not None:
            self.remove_requested.emit(frame)

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        value = item.data(Qt.ItemDataRole.UserRole)
        if value is not None:
            self.frame_selected.emit(int(value))

    def _on_selection_changed(self) -> None:
        """Single-click on a frame: update buttons and auto-jump."""
        self._update_button_state()
        frame = self.selected_frame()
        if frame is not None:
            self.frame_selected.emit(frame)

    def _browse_dataset_dir(self) -> None:
        current = self.txt_dataset_dir.text().strip()
        folder = QFileDialog.getExistingDirectory(
            self, "Select Dataset Directory", current or ""
        )
        if folder:
            self.txt_dataset_dir.setText(folder)

    def _on_build_dataset(self) -> None:
        ds_dir = self.dataset_dir()
        if ds_dir:
            self.build_dataset_requested.emit(ds_dir)

    def _on_add_to_dataset(self) -> None:
        ds_dir = self.dataset_dir()
        if ds_dir:
            self.add_to_dataset_requested.emit(ds_dir, "train")

    def _on_add_to_val(self) -> None:
        ds_dir = self.dataset_dir()
        if ds_dir:
            self.add_to_dataset_requested.emit(ds_dir, "val")

    def _on_load_labels(self) -> None:
        ds_dir = self.dataset_dir()
        if ds_dir:
            self.load_labels_requested.emit(ds_dir)

    def _on_fine_tune(self) -> None:
        ds_dir = self.dataset_dir()
        if ds_dir:
            self.fine_tune_requested.emit(ds_dir)

    def _set_sort_mode(self, by_label: bool) -> None:
        """Toggle between frame-number and label-order sorting."""
        self._sort_by_label_order = by_label
        self.btn_sort_frame.setChecked(not by_label)
        self.btn_sort_labeled.setChecked(by_label)
        self._rebuild_list_order()

    def _rebuild_list_order(self) -> None:
        """Re-order the list widget items according to the current sort mode."""
        prev_frame = self.selected_frame()
        # Block signals to avoid firing selection_changed during rebuild
        self.list_examples.blockSignals(True)

        # Take all items out (without deleting them)
        while self.list_examples.count():
            self.list_examples.takeItem(0)

        if self._sort_by_label_order:
            ordered_frames = list(self._label_order)
            # Include any frames not yet in _label_order (shouldn't happen, but safe)
            for f in sorted(self._items_by_frame.keys()):
                if f not in ordered_frames:
                    ordered_frames.append(f)
        else:
            ordered_frames = sorted(self._items_by_frame.keys())

        for frame_idx in ordered_frames:
            item = self._items_by_frame.get(frame_idx)
            if item is not None:
                self.list_examples.addItem(item)

        self.list_examples.blockSignals(False)

        # Restore previous selection
        if prev_frame is not None and prev_frame in self._items_by_frame:
            self.list_examples.setCurrentItem(self._items_by_frame[prev_frame])

    def _update_button_state(self) -> None:
        has_selection = self.selected_frame() is not None
        self.btn_jump.setEnabled(has_selection)
        self.btn_reprompt.setEnabled(has_selection)
        self.btn_remove.setEnabled(has_selection)
        self.btn_clear.setEnabled(len(self._items_by_frame) > 0)
