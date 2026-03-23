"""Compact action bar — prompt, segment/track buttons, export.

Pinned below the video viewer for quick access to the most-used controls.
Laid out in two horizontal rows for readability.
"""

import logging
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.config import (
    AUTO_PROMPTS,
    DEFAULT_TEXT_PROMPT,
    MASK_ALPHA,
    MAX_MICE,
)

logger = logging.getLogger(__name__)


class ActionBar(QWidget):
    """Compact two-row action bar with prompt, segment/track, and export controls."""

    segment_requested = Signal()
    paint_toggled = Signal(bool)
    paint_add_size_changed = Signal(int)
    paint_erase_size_changed = Signal(int)
    undo_requested = Signal()
    redo_requested = Signal()
    track_requested = Signal()
    free_track_requested = Signal()
    stop_requested = Signal()
    prompt_changed = Signal(str)
    export_csv_requested = Signal()
    export_h5_requested = Signal()
    export_video_requested = Signal()
    shortcuts_changed = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._tracking_active = False
        self._paint_mode_active = False
        self._setup_ui()
        self._sync_paint_controls()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 2, 4, 2)
        root.setSpacing(2)

        # ── Row 1: Prompt + main action buttons ────────────────────────────────
        row1 = QHBoxLayout()
        row1.setSpacing(4)

        lbl_prompt = QLabel("Prompt:")
        lbl_prompt.setStyleSheet("color: #a6adc8; font-size: 11px;")
        row1.addWidget(lbl_prompt)

        self.edit_prompt = QLineEdit(DEFAULT_TEXT_PROMPT)
        self.edit_prompt.setToolTip("SAM3 text prompt for detecting objects")
        self.edit_prompt.setMinimumWidth(120)
        self.edit_prompt.textChanged.connect(self.prompt_changed)
        row1.addWidget(self.edit_prompt, 1)

        # Hidden default-prompt store
        self.edit_default_prompt = QLineEdit(DEFAULT_TEXT_PROMPT)
        self.edit_default_prompt.hide()

        self.btn_reset_prompt = QPushButton("Reset")
        self.btn_reset_prompt.setObjectName("secondary_button")
        self.btn_reset_prompt.setFixedHeight(22)
        self.btn_reset_prompt.setFixedWidth(42)
        self.btn_reset_prompt.setToolTip("Reset prompt to default")
        self.btn_reset_prompt.clicked.connect(
            lambda: self.edit_prompt.setText(self.edit_default_prompt.text().strip())
        )
        row1.addWidget(self.btn_reset_prompt)

        self.btn_set_default = QPushButton("Set Default")
        self.btn_set_default.setObjectName("secondary_button")
        self.btn_set_default.setFixedHeight(22)
        self.btn_set_default.setToolTip("Save current prompt text as the new default")
        self.btn_set_default.clicked.connect(
            lambda: self.edit_default_prompt.setText(self.edit_prompt.text().strip())
        )
        row1.addWidget(self.btn_set_default)

        # Quick-prompt chips (first 4 to save space)
        for p in AUTO_PROMPTS[:4]:
            btn = QPushButton(p)
            btn.setObjectName("secondary_button")
            btn.setFixedHeight(22)
            btn.clicked.connect(lambda checked, text=p: self.edit_prompt.setText(text))
            row1.addWidget(btn)

        # Separator
        sep1 = QLabel("|")
        sep1.setStyleSheet("color: #45475a; margin: 0 2px;")
        row1.addWidget(sep1)

        self.btn_segment = QPushButton("Segment [S]")
        self.btn_segment.setToolTip("Run SAM3 segmentation on the current frame (shortcut: S)")
        self.btn_segment.setFixedHeight(24)
        self.btn_segment.setStyleSheet(
            "QPushButton { background: #7c3aed; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 2px 10px; }"
            "QPushButton:hover { background: #6d28d9; }"
            "QPushButton:disabled { background: #45475a; color: #6c7086; }"
        )
        self.btn_segment.clicked.connect(self.segment_requested)
        row1.addWidget(self.btn_segment)

        self.btn_undo = QPushButton("Undo (Ctrl+Z)")
        self.btn_undo.setFixedHeight(24)
        self.btn_undo.setObjectName("secondary_button")
        self.btn_undo.setToolTip("Undo the last SAM3 edit on the current frame")
        self.btn_undo.setEnabled(False)
        self.btn_undo.clicked.connect(self.undo_requested)
        row1.addWidget(self.btn_undo)

        self.btn_redo = QPushButton("Redo (Ctrl+Y)")
        self.btn_redo.setFixedHeight(24)
        self.btn_redo.setObjectName("secondary_button")
        self.btn_redo.setToolTip("Redo the last undone SAM3 edit on the current frame")
        self.btn_redo.setEnabled(False)
        self.btn_redo.clicked.connect(self.redo_requested)
        row1.addWidget(self.btn_redo)

        self.btn_paint = QPushButton("Paint [P]")
        self.btn_paint.setFixedHeight(24)
        self.btn_paint.setObjectName("secondary_button")
        self.btn_paint.setCheckable(True)
        self.btn_paint.setToolTip(
            "Toggle paint mode for selected entity. P=paint add, C=paint erase."
        )
        self.btn_paint.toggled.connect(self.paint_toggled)
        row1.addWidget(self.btn_paint)

        self.lbl_paint_add = QLabel("+")
        self.lbl_paint_add.setToolTip("Paint add brush size (pixels)")
        row1.addWidget(self.lbl_paint_add)

        self.spin_paint_add = QSpinBox()
        self.spin_paint_add.setRange(1, 80)
        self.spin_paint_add.setValue(8)
        self.spin_paint_add.setFixedWidth(54)
        self.spin_paint_add.setSuffix(" px")
        self.spin_paint_add.setToolTip("Paint add brush size (pixels)")
        self.spin_paint_add.valueChanged.connect(self.paint_add_size_changed)
        row1.addWidget(self.spin_paint_add)

        self.lbl_paint_erase = QLabel("-")
        self.lbl_paint_erase.setToolTip("Paint erase brush size (pixels)")
        row1.addWidget(self.lbl_paint_erase)

        self.spin_paint_erase = QSpinBox()
        self.spin_paint_erase.setRange(1, 80)
        self.spin_paint_erase.setValue(4)
        self.spin_paint_erase.setFixedWidth(54)
        self.spin_paint_erase.setSuffix(" px")
        self.spin_paint_erase.setToolTip("Paint erase brush size (pixels)")
        self.spin_paint_erase.valueChanged.connect(self.paint_erase_size_changed)
        row1.addWidget(self.spin_paint_erase)

        self.btn_track = QPushButton("▶ Track")
        self.btn_track.setFixedHeight(24)
        self.btn_track.setToolTip("Propagate tracking through the video")
        self.btn_track.clicked.connect(self.track_requested)
        row1.addWidget(self.btn_track)

        self.btn_stop = QPushButton("■ Stop")
        self.btn_stop.setObjectName("danger_button")
        self.btn_stop.setFixedHeight(24)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_requested)
        row1.addWidget(self.btn_stop)

        self.btn_free_track = QPushButton("Free Track")
        self.btn_free_track.setFixedHeight(24)
        self.btn_free_track.setToolTip(
            "Auto-segment with text prompt and track the full video"
        )
        self.btn_free_track.clicked.connect(self.free_track_requested)
        row1.addWidget(self.btn_free_track)

        root.addLayout(row1)

        # ── Row 2: Options + export ────────────────────────────────────────────
        row2 = QHBoxLayout()
        row2.setSpacing(4)

        self.chk_adaptive_reprompt = QCheckBox("Adaptive")
        self.chk_adaptive_reprompt.setChecked(False)
        self.chk_adaptive_reprompt.setToolTip(
            "Retry failed tracking frames with alternate text prompts"
        )
        row2.addWidget(self.chk_adaptive_reprompt)

        row2.addWidget(QLabel("Skip:"))
        self.spin_frame_skip = QSpinBox()
        self.spin_frame_skip.setRange(1, 10)
        self.spin_frame_skip.setValue(1)
        self.spin_frame_skip.setFixedWidth(44)
        self.spin_frame_skip.setToolTip("Process every Nth frame (1=all)")
        row2.addWidget(self.spin_frame_skip)

        self.chk_show_masks = QCheckBox("Mask")
        self.chk_show_masks.setChecked(True)
        self.chk_show_masks.setToolTip("Show/hide segmentation masks")
        row2.addWidget(self.chk_show_masks)

        row2.addWidget(QLabel("Opacity:"))
        self.spin_mask_alpha = QSpinBox()
        self.spin_mask_alpha.setRange(5, 90)
        self.spin_mask_alpha.setValue(int(round(MASK_ALPHA * 100)))
        self.spin_mask_alpha.setSingleStep(5)
        self.spin_mask_alpha.setSuffix("%")
        self.spin_mask_alpha.setFixedWidth(64)
        self.spin_mask_alpha.setToolTip("Mask overlay opacity")
        row2.addWidget(self.spin_mask_alpha)
        self.chk_show_masks.toggled.connect(self.spin_mask_alpha.setEnabled)

        self.chk_show_labels = QCheckBox("ID")
        self.chk_show_labels.setChecked(True)
        self.chk_show_labels.setToolTip("Draw entity name + confidence")
        row2.addWidget(self.chk_show_labels)

        self.chk_show_names = QCheckBox("Names")
        self.chk_show_names.setChecked(True)
        self.chk_show_names.setToolTip("Use custom mouse names in labels")
        row2.addWidget(self.chk_show_names)

        self.chk_show_bbox = QCheckBox("BBox")
        self.chk_show_bbox.setChecked(False)
        self.chk_show_bbox.setToolTip("Draw bounding boxes")
        row2.addWidget(self.chk_show_bbox)

        row2.addWidget(QLabel("Seg:"))
        self.spin_segment_size = QSpinBox()
        self.spin_segment_size.setRange(100, 50000)
        self.spin_segment_size.setValue(1500)
        self.spin_segment_size.setSingleStep(100)
        self.spin_segment_size.setSuffix(" fr")
        self.spin_segment_size.setFixedWidth(82)
        self.spin_segment_size.setToolTip("Frames per segment for long-video processing")
        row2.addWidget(self.spin_segment_size)

        self.chk_time_window = QCheckBox("Window:")
        self.chk_time_window.setToolTip("Restrict tracking/export to a time sub-range")
        row2.addWidget(self.chk_time_window)
        self.spin_tw_start = QDoubleSpinBox()
        self.spin_tw_start.setRange(0, 99999)
        self.spin_tw_start.setDecimals(1)
        self.spin_tw_start.setSuffix(" s")
        self.spin_tw_start.setFixedWidth(68)
        self.spin_tw_start.setEnabled(False)
        row2.addWidget(self.spin_tw_start)
        row2.addWidget(QLabel("–"))
        self.spin_tw_end = QDoubleSpinBox()
        self.spin_tw_end.setRange(0, 99999)
        self.spin_tw_end.setDecimals(1)
        self.spin_tw_end.setSuffix(" s")
        self.spin_tw_end.setFixedWidth(68)
        self.spin_tw_end.setEnabled(False)
        row2.addWidget(self.spin_tw_end)
        self.chk_time_window.toggled.connect(self.spin_tw_start.setEnabled)
        self.chk_time_window.toggled.connect(self.spin_tw_end.setEnabled)

        # Separator
        sep2 = QLabel("|")
        sep2.setStyleSheet("color: #45475a; margin: 0 2px;")
        row2.addWidget(sep2)

        self.btn_csv = QPushButton("CSV")
        self.btn_csv.setFixedHeight(22)
        self.btn_csv.setObjectName("secondary_button")
        self.btn_csv.setToolTip("Export trajectories as CSV")
        self.btn_csv.clicked.connect(self.export_csv_requested)
        row2.addWidget(self.btn_csv)

        self.btn_h5 = QPushButton("HDF5")
        self.btn_h5.setFixedHeight(22)
        self.btn_h5.setObjectName("secondary_button")
        self.btn_h5.setToolTip("Export masks and keypoints as HDF5")
        self.btn_h5.clicked.connect(self.export_h5_requested)
        row2.addWidget(self.btn_h5)

        self.btn_video = QPushButton("Video")
        self.btn_video.setFixedHeight(22)
        self.btn_video.setObjectName("secondary_button")
        self.btn_video.setToolTip("Export annotated overlay video")
        self.btn_video.clicked.connect(self.export_video_requested)
        row2.addWidget(self.btn_video)

        self.btn_shortcuts = QPushButton("⚙")
        self.btn_shortcuts.setObjectName("secondary_button")
        self.btn_shortcuts.setFixedSize(24, 22)
        self.btn_shortcuts.setToolTip("Configure keyboard shortcuts")
        self.btn_shortcuts.clicked.connect(self._open_shortcuts_dialog)
        row2.addWidget(self.btn_shortcuts)

        row2.addStretch()
        root.addLayout(row2)

        # ── Shortcut edits (hidden — live inside the dialog) ──────────────────
        self._default_keys = ["1", "2", "3", "4", "5", "6"]
        self._shortcut_edits: list[QLineEdit] = []
        for i in range(MAX_MICE):
            edit = QLineEdit(self._default_keys[i])
            edit.setMaxLength(1)
            self._shortcut_edits.append(edit)

    # ── Shortcuts dialog ──────────────────────────────────────────────────────

    def _open_shortcuts_dialog(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Keyboard Shortcuts")
        dlg.setMinimumWidth(220)
        lay = QVBoxLayout(dlg)
        lay.setSpacing(6)

        lay.addWidget(QLabel("Key to press for each entity:"))

        edits: list[QLineEdit] = []
        for i in range(MAX_MICE):
            row = QHBoxLayout()
            row.setSpacing(4)
            row.addWidget(QLabel(f"Entity {i + 1}:"))
            edit = QLineEdit(self._shortcut_edits[i].text())
            edit.setFixedWidth(36)
            edit.setMaxLength(1)
            edit.setAlignment(Qt.AlignCenter)
            edits.append(edit)
            row.addWidget(edit)
            row.addStretch()
            lay.addLayout(row)

        btn_row = QHBoxLayout()
        btn_ok = QPushButton("Apply")
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setObjectName("secondary_button")
        btn_cancel.clicked.connect(dlg.reject)
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        lay.addLayout(btn_row)

        if dlg.exec() == QDialog.Accepted:
            for i, edit in enumerate(edits):
                self._shortcut_edits[i].setText(edit.text())
            self._on_apply_shortcuts()

    # ── Accessors ─────────────────────────────────────────────────────────────

    def text_prompt(self) -> str:
        return self.edit_prompt.text().strip() or self.default_prompt()

    def default_prompt(self) -> str:
        return self.edit_default_prompt.text().strip() or DEFAULT_TEXT_PROMPT

    def frame_skip(self) -> int:
        return self.spin_frame_skip.value()

    def segment_size(self) -> int:
        return self.spin_segment_size.value()

    def adaptive_reprompt_enabled(self) -> bool:
        return self.chk_adaptive_reprompt.isChecked()

    def time_window_enabled(self) -> bool:
        return self.chk_time_window.isChecked()

    def time_window(self) -> tuple[float, float]:
        return self.spin_tw_start.value(), self.spin_tw_end.value()

    def set_video_duration(self, duration_s: float) -> None:
        self.spin_tw_end.setMaximum(duration_s)
        self.spin_tw_start.setMaximum(duration_s)
        self.spin_tw_end.setValue(duration_s)

    def paint_add_size(self) -> int:
        return int(self.spin_paint_add.value())

    def paint_erase_size(self) -> int:
        return int(self.spin_paint_erase.value())

    def masks_visible(self) -> bool:
        return bool(self.chk_show_masks.isChecked())

    def mask_alpha(self) -> float:
        return max(0.05, min(0.90, float(self.spin_mask_alpha.value()) / 100.0))

    def set_tracking(self, active: bool) -> None:
        self._tracking_active = bool(active)
        self.btn_track.setEnabled(not self._tracking_active)
        self.btn_free_track.setEnabled(not self._tracking_active)
        self.btn_stop.setEnabled(self._tracking_active)
        self._sync_paint_controls()

    def set_undo_redo_enabled(self, can_undo: bool, can_redo: bool) -> None:
        self.btn_undo.setEnabled(bool(can_undo))
        self.btn_redo.setEnabled(bool(can_redo))

    def set_paint_mode(self, active: bool, *, add_mode: bool = True) -> None:
        self._paint_mode_active = bool(active)
        self.btn_paint.blockSignals(True)
        self.btn_paint.setChecked(bool(active))
        self.btn_paint.blockSignals(False)
        if not active:
            self.btn_paint.setText("Paint [P]")
        else:
            self.btn_paint.setText("Paint + [P]" if add_mode else "Paint - [C]")
        self._sync_paint_controls()

    def _sync_paint_controls(self) -> None:
        can_segment = (not self._tracking_active) and (not self._paint_mode_active)
        self.btn_segment.setEnabled(can_segment)
        self.btn_paint.setEnabled(not self._tracking_active)
        show_paint_tools = self._paint_mode_active
        paint_widgets = (
            self.lbl_paint_add,
            self.spin_paint_add,
            self.lbl_paint_erase,
            self.spin_paint_erase,
        )
        for widget in paint_widgets:
            widget.setVisible(show_paint_tools)
            widget.setEnabled(show_paint_tools and (not self._tracking_active))

    def get_shortcut_names(self) -> dict[int, str]:
        return {
            i + 1: edit.text().strip() or self._default_keys[i]
            for i, edit in enumerate(self._shortcut_edits)
        }

    def _on_apply_shortcuts(self) -> None:
        names = self.get_shortcut_names()
        self.shortcuts_changed.emit(names)
