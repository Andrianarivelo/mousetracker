"""SAM3 prompt tuning, threshold sliders, and tracking settings."""

import logging
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.config import (
    AUTO_PROMPTS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_SAMPLE_FRAMES,
    DEFAULT_TEXT_PROMPT,
    MAX_MICE,
)

logger = logging.getLogger(__name__)


def _help_label(tip: str) -> "QLabel":
    """Return a small '?' badge that shows `tip` as a tooltip on hover."""
    lbl = QLabel("?")
    lbl.setToolTip(tip)
    lbl.setFixedSize(16, 16)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet(
        "QLabel { color: #a6adc8; border: 1px solid #585b70; border-radius: 8px;"
        " font-size: 10px; font-weight: bold; background: #313244; }"
        "QLabel:hover { color: #cba6f7; border-color: #cba6f7; }"
    )
    return lbl


class SettingsPanel(QWidget):
    """
    Controls panel laid out in three horizontal columns:
      1. Segmentation — prompt, confidence, filters
      2. Actions      — Segment, Track, Stop, skip, time window
      3. Export       — CSV, HDF5, Video, Shortcuts

    Signals:
        segment_requested():     User clicked Segment (or pressed S)
        track_requested():       User clicked Track
        stop_requested():        User clicked Stop
        prompt_changed(str):     Text prompt changed
        export_csv_requested():
        export_h5_requested():
        export_video_requested():
        shortcuts_changed(dict): Key-name mapping updated
    """

    segment_requested = Signal()
    track_requested = Signal()
    free_track_requested = Signal()
    stop_requested = Signal()
    prompt_changed = Signal(str)
    export_csv_requested = Signal()
    export_h5_requested = Signal()
    export_video_requested = Signal()
    shortcuts_changed = Signal(dict)  # {1: "1", 2: "2", ...} display names

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 2, 4, 2)
        root.setSpacing(6)

        # ── Column 1: Segmentation settings ──────────────────────────────────
        seg_group = QGroupBox("Segmentation")
        seg_inner = QVBoxLayout(seg_group)
        seg_inner.setSpacing(2)
        seg_inner.setContentsMargins(6, 4, 6, 2)

        # Prompt row — with Set Default button
        prompt_row = QHBoxLayout()
        prompt_row.setSpacing(4)
        prompt_row.addWidget(QLabel("Prompt:"))
        self.edit_prompt = QLineEdit(DEFAULT_TEXT_PROMPT)
        self.edit_prompt.setToolTip("SAM3 text prompt for detecting objects")
        self.edit_prompt.textChanged.connect(self.prompt_changed)
        prompt_row.addWidget(self.edit_prompt)
        # Hidden default-prompt store (user can change via Set Default)
        self.edit_default_prompt = QLineEdit(DEFAULT_TEXT_PROMPT)
        self.edit_default_prompt.hide()
        self.btn_reset_prompt = QPushButton("Reset")
        self.btn_reset_prompt.setObjectName("secondary_button")
        self.btn_reset_prompt.setFixedHeight(20)
        self.btn_reset_prompt.setFixedWidth(42)
        self.btn_reset_prompt.setToolTip("Reset prompt to default")
        self.btn_reset_prompt.clicked.connect(
            lambda: self.edit_prompt.setText(self.edit_default_prompt.text().strip())
        )
        prompt_row.addWidget(self.btn_reset_prompt)
        self.btn_set_default = QPushButton("Set Default")
        self.btn_set_default.setObjectName("secondary_button")
        self.btn_set_default.setFixedHeight(20)
        self.btn_set_default.setToolTip(
            "Save current prompt text as the new default.\n"
            "The default is used when you click Reset or open a new session."
        )
        self.btn_set_default.clicked.connect(
            lambda: self.edit_default_prompt.setText(self.edit_prompt.text().strip())
        )
        prompt_row.addWidget(self.btn_set_default)
        prompt_row.addWidget(_help_label(
            "Text description of the object to detect.\n"
            "Examples: 'black mouse', 'dark mouse on white bedding'.\n"
            "Click a quick-prompt chip below to apply it instantly.\n"
            "Use Set Default to save the current prompt as baseline.\n"
            "Use Reset to restore the saved default."
        ))
        seg_inner.addLayout(prompt_row)

        # Quick-prompt chips
        chips_row = QHBoxLayout()
        chips_row.setSpacing(3)
        for p in AUTO_PROMPTS[:6]:
            btn = QPushButton(p)
            btn.setObjectName("secondary_button")
            btn.setMaximumHeight(20)
            btn.clicked.connect(lambda checked, text=p: self.edit_prompt.setText(text))
            chips_row.addWidget(btn)
        chips_row.addStretch()
        seg_inner.addLayout(chips_row)

        # Confidence + Sample frames on one row
        conf_sf_row = QHBoxLayout()
        conf_sf_row.setSpacing(4)
        conf_sf_row.addWidget(QLabel("Conf:"))
        self.slider_confidence = QSlider(Qt.Horizontal)
        self.slider_confidence.setRange(0, 100)
        self.slider_confidence.setValue(int(DEFAULT_CONFIDENCE_THRESHOLD * 100))
        self.lbl_confidence = QLabel(f"{DEFAULT_CONFIDENCE_THRESHOLD:.2f}")
        self.lbl_confidence.setFixedWidth(32)
        self.lbl_confidence.setStyleSheet("color: #cdd6f4; font-weight: bold;")
        self.slider_confidence.valueChanged.connect(
            lambda v: self.lbl_confidence.setText(f"{v / 100:.2f}")
        )
        conf_sf_row.addWidget(self.slider_confidence)
        conf_sf_row.addWidget(self.lbl_confidence)
        conf_sf_row.addWidget(QLabel("Samples:"))
        self.spin_sample_frames = QSpinBox()
        self.spin_sample_frames.setRange(1, 20)
        self.spin_sample_frames.setValue(DEFAULT_SAMPLE_FRAMES)
        self.spin_sample_frames.setFixedWidth(48)
        conf_sf_row.addWidget(self.spin_sample_frames)
        conf_sf_row.addWidget(_help_label(
            "Conf: Detection confidence threshold (0.0–1.0).\n"
            "Lower = more detections. Default: 0.50.\n\n"
            "Samples: frames SAM3 uses to calibrate. 3–10 recommended."
        ))
        seg_inner.addLayout(conf_sf_row)

        # Raw SAM bypass + area filter + edge filter — two compact rows
        raw_row = QHBoxLayout()
        raw_row.setSpacing(4)
        self.chk_raw_sam = QCheckBox("Raw SAM (no filters)")
        self.chk_raw_sam.setChecked(False)
        self.chk_raw_sam.setToolTip(
            "Pass SAM3 masks directly — no area, edge, or size filtering."
        )
        raw_row.addWidget(self.chk_raw_sam)
        raw_row.addStretch()
        seg_inner.addLayout(raw_row)

        # Area + Edge on one row
        filter_row = QHBoxLayout()
        filter_row.setSpacing(4)

        self.chk_area_filter = QCheckBox("Area:")
        self.chk_area_filter.setChecked(False)
        self.chk_area_filter.setToolTip("Reject masks covering more than this % of the frame")
        filter_row.addWidget(self.chk_area_filter)
        self.slider_max_area = QSlider(Qt.Horizontal)
        self.slider_max_area.setRange(1, 100)
        self.slider_max_area.setValue(40)
        self.slider_max_area.setEnabled(False)
        self.slider_max_area.setMaximumWidth(80)
        self.lbl_max_area = QLabel("40%")
        self.lbl_max_area.setFixedWidth(30)
        self.lbl_max_area.setEnabled(False)
        self.lbl_max_area.setStyleSheet("color: #cdd6f4; font-weight: bold;")
        self.slider_max_area.valueChanged.connect(
            lambda v: self.lbl_max_area.setText(f"{v}%")
        )
        self.chk_area_filter.toggled.connect(self.slider_max_area.setEnabled)
        self.chk_area_filter.toggled.connect(self.lbl_max_area.setEnabled)
        filter_row.addWidget(self.slider_max_area)
        filter_row.addWidget(self.lbl_max_area)

        self.chk_edge_filter = QCheckBox("Edge:")
        self.chk_edge_filter.setChecked(False)
        self.chk_edge_filter.setToolTip("Reject masks hugging the frame border")
        filter_row.addWidget(self.chk_edge_filter)
        self.slider_max_edge = QSlider(Qt.Horizontal)
        self.slider_max_edge.setRange(1, 50)
        self.slider_max_edge.setValue(28)
        self.slider_max_edge.setEnabled(False)
        self.slider_max_edge.setMaximumWidth(80)
        self.lbl_max_edge = QLabel("28%")
        self.lbl_max_edge.setFixedWidth(30)
        self.lbl_max_edge.setEnabled(False)
        self.lbl_max_edge.setStyleSheet("color: #cdd6f4; font-weight: bold;")
        self.slider_max_edge.valueChanged.connect(
            lambda v: self.lbl_max_edge.setText(f"{v}%")
        )
        self.chk_edge_filter.toggled.connect(self.slider_max_edge.setEnabled)
        self.chk_edge_filter.toggled.connect(self.lbl_max_edge.setEnabled)
        filter_row.addWidget(self.slider_max_edge)
        filter_row.addWidget(self.lbl_max_edge)
        filter_row.addWidget(_help_label(
            "Area: rejects masks covering > N% of the frame.\n"
            "Edge: rejects masks hugging the border zone.\n"
            "Both OFF by default. Enable if false detections appear."
        ))

        seg_inner.addLayout(filter_row)

        # Wire raw-SAM bypass
        def _on_raw_toggled(raw: bool) -> None:
            self.chk_area_filter.setEnabled(not raw)
            self.chk_edge_filter.setEnabled(not raw)
            area_on = not raw and self.chk_area_filter.isChecked()
            edge_on = not raw and self.chk_edge_filter.isChecked()
            self.slider_max_area.setEnabled(area_on)
            self.lbl_max_area.setEnabled(area_on)
            self.slider_max_edge.setEnabled(edge_on)
            self.lbl_max_edge.setEnabled(edge_on)

        self.chk_raw_sam.toggled.connect(_on_raw_toggled)

        seg_group.setMinimumWidth(280)
        root.addWidget(seg_group, stretch=3)

        # ── Column 2: Actions ─────────────────────────────────────────────────
        action_group = QGroupBox("Actions")
        action_inner = QVBoxLayout(action_group)
        action_inner.setSpacing(3)
        action_inner.setContentsMargins(6, 4, 6, 2)

        self.btn_segment = QPushButton("Segment  [S]")
        self.btn_segment.setToolTip("Run SAM3 segmentation on the current frame (shortcut: S)")
        self.btn_segment.setFixedHeight(26)
        self.btn_segment.clicked.connect(self.segment_requested)
        action_inner.addWidget(self.btn_segment)

        track_row = QHBoxLayout()
        track_row.setSpacing(4)
        self.btn_track = QPushButton("▶  Track")
        self.btn_track.setFixedHeight(26)
        self.btn_track.setToolTip("Propagate tracking through the video")
        self.btn_track.clicked.connect(self.track_requested)
        track_row.addWidget(self.btn_track)

        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setObjectName("danger_button")
        self.btn_stop.setFixedHeight(26)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_requested)
        track_row.addWidget(self.btn_stop)
        action_inner.addLayout(track_row)

        self.btn_free_track = QPushButton("Free Track")
        self.btn_free_track.setFixedHeight(26)
        self.btn_free_track.setToolTip(
            "Auto-segment with text prompt and track the full video — no manual ID assignment needed"
        )
        self.btn_free_track.clicked.connect(self.free_track_requested)
        action_inner.addWidget(self.btn_free_track)

        self.chk_adaptive_reprompt = QCheckBox("Adaptive re-prompt")
        self.chk_adaptive_reprompt.setChecked(False)
        self.chk_adaptive_reprompt.setToolTip(
            "Retry failed tracking frames with alternate text prompts.\n"
            "Can recover some bad frames, but slows tracking down noticeably."
        )
        action_inner.addWidget(self.chk_adaptive_reprompt)

        # Skip + overlay options on one row
        skip_row = QHBoxLayout()
        skip_row.setSpacing(4)
        skip_row.addWidget(QLabel("Skip:"))
        self.spin_frame_skip = QSpinBox()
        self.spin_frame_skip.setRange(1, 10)
        self.spin_frame_skip.setValue(1)
        self.spin_frame_skip.setFixedWidth(48)
        self.spin_frame_skip.setToolTip(
            "Process every Nth frame during tracking (1=all, 2=2× faster)"
        )
        skip_row.addWidget(self.spin_frame_skip)
        self.chk_show_labels = QCheckBox("ID")
        self.chk_show_labels.setChecked(True)
        self.chk_show_labels.setToolTip("Draw entity name + confidence")
        skip_row.addWidget(self.chk_show_labels)
        self.chk_show_bbox = QCheckBox("BBox")
        self.chk_show_bbox.setChecked(False)
        self.chk_show_bbox.setToolTip("Draw bounding boxes")
        skip_row.addWidget(self.chk_show_bbox)
        skip_row.addStretch()
        action_inner.addLayout(skip_row)

        # Segment size + Time window on one row
        seg_tw_row = QHBoxLayout()
        seg_tw_row.setSpacing(4)
        seg_tw_row.addWidget(QLabel("Seg:"))
        self.spin_segment_size = QSpinBox()
        self.spin_segment_size.setRange(100, 50000)
        self.spin_segment_size.setValue(1500)
        self.spin_segment_size.setSingleStep(100)
        self.spin_segment_size.setSuffix(" fr")
        self.spin_segment_size.setFixedWidth(82)
        self.spin_segment_size.setToolTip(
            "Frames per segment for long-video processing"
        )
        seg_tw_row.addWidget(self.spin_segment_size)
        self.chk_time_window = QCheckBox("Window:")
        self.chk_time_window.setToolTip(
            "Restrict tracking and export to a time sub-range (seconds)."
        )
        seg_tw_row.addWidget(self.chk_time_window)
        self.spin_tw_start = QDoubleSpinBox()
        self.spin_tw_start.setRange(0, 99999)
        self.spin_tw_start.setDecimals(1)
        self.spin_tw_start.setSuffix(" s")
        self.spin_tw_start.setFixedWidth(72)
        self.spin_tw_start.setEnabled(False)
        seg_tw_row.addWidget(self.spin_tw_start)
        seg_tw_row.addWidget(QLabel("–"))
        self.spin_tw_end = QDoubleSpinBox()
        self.spin_tw_end.setRange(0, 99999)
        self.spin_tw_end.setDecimals(1)
        self.spin_tw_end.setSuffix(" s")
        self.spin_tw_end.setFixedWidth(72)
        self.spin_tw_end.setEnabled(False)
        seg_tw_row.addWidget(self.spin_tw_end)
        self.chk_time_window.toggled.connect(self.spin_tw_start.setEnabled)
        self.chk_time_window.toggled.connect(self.spin_tw_end.setEnabled)
        seg_tw_row.addStretch()
        action_inner.addLayout(seg_tw_row)

        action_inner.addStretch()
        action_group.setMinimumWidth(165)
        root.addWidget(action_group, stretch=2)

        # ── Column 3: Export + Shortcuts button ───────────────────────────────
        export_group = QGroupBox("Export")
        export_inner = QVBoxLayout(export_group)
        export_inner.setSpacing(3)
        export_inner.setContentsMargins(6, 4, 6, 2)

        self.btn_csv = QPushButton("CSV")
        self.btn_csv.setFixedHeight(26)
        self.btn_csv.setToolTip("Export trajectories as CSV")
        self.btn_csv.clicked.connect(self.export_csv_requested)
        export_inner.addWidget(self.btn_csv)

        self.btn_h5 = QPushButton("HDF5")
        self.btn_h5.setFixedHeight(26)
        self.btn_h5.setToolTip("Export masks and keypoints as HDF5")
        self.btn_h5.clicked.connect(self.export_h5_requested)
        export_inner.addWidget(self.btn_h5)

        self.btn_video = QPushButton("Overlay Video")
        self.btn_video.setFixedHeight(26)
        self.btn_video.setToolTip("Export annotated overlay video")
        self.btn_video.clicked.connect(self.export_video_requested)
        export_inner.addWidget(self.btn_video)

        # Shortcuts button — opens a popup dialog
        self.btn_shortcuts = QPushButton("Shortcuts…")
        self.btn_shortcuts.setObjectName("secondary_button")
        self.btn_shortcuts.setFixedHeight(24)
        self.btn_shortcuts.setToolTip("Configure keyboard shortcuts for entity selection")
        self.btn_shortcuts.clicked.connect(self._open_shortcuts_dialog)
        export_inner.addWidget(self.btn_shortcuts)

        export_inner.addStretch()
        export_group.setMinimumWidth(110)
        root.addWidget(export_group, stretch=1)

        # ── Shortcut edits (hidden — live inside the dialog) ─────────────────
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

    def confidence_threshold(self) -> float:
        return self.slider_confidence.value() / 100.0

    def sample_frames(self) -> int:
        return self.spin_sample_frames.value()

    def frame_skip(self) -> int:
        """Frames to skip during tracking (1 = all frames)."""
        return self.spin_frame_skip.value()

    def max_area_frac(self) -> float:
        """Maximum mask area as fraction of frame (0.01–1.00)."""
        return self.slider_max_area.value() / 100.0

    def area_filter_enabled(self) -> bool:
        """True when the area filter is active."""
        return self.chk_area_filter.isChecked()

    def max_edge_frac(self) -> float:
        """Maximum border-zone fraction (0.01–0.50)."""
        return self.slider_max_edge.value() / 100.0

    def edge_filter_enabled(self) -> bool:
        """True when the edge-contact filter is active."""
        return self.chk_edge_filter.isChecked()

    def raw_sam_enabled(self) -> bool:
        """True when all custom filtering is bypassed."""
        return self.chk_raw_sam.isChecked()

    def segment_size(self) -> int:
        """Frames per chunk/segment for long-video processing."""
        return self.spin_segment_size.value()

    def adaptive_reprompt_enabled(self) -> bool:
        """True when failed frames should be retried with alternate prompts."""
        return self.chk_adaptive_reprompt.isChecked()

    def time_window_enabled(self) -> bool:
        return self.chk_time_window.isChecked()

    def time_window(self) -> tuple[float, float]:
        """Return (start_s, end_s). Only meaningful when time_window_enabled()."""
        return self.spin_tw_start.value(), self.spin_tw_end.value()

    def set_video_duration(self, duration_s: float) -> None:
        """Update the time-window end spinbox to match the loaded video."""
        self.spin_tw_end.setMaximum(duration_s)
        self.spin_tw_start.setMaximum(duration_s)
        self.spin_tw_end.setValue(duration_s)

    def _on_apply_shortcuts(self) -> None:
        """Emit shortcuts_changed with the current key mapping."""
        names = self.get_shortcut_names()
        self.shortcuts_changed.emit(names)

    def get_shortcut_names(self) -> dict[int, str]:
        """Return {1: key_name, 2: key_name, ...} from the shortcut edits."""
        return {
            i + 1: edit.text().strip() or self._default_keys[i]
            for i, edit in enumerate(self._shortcut_edits)
        }

    def set_tracking(self, active: bool) -> None:
        """Update button states when tracking starts/stops."""
        self.btn_track.setEnabled(not active)
        self.btn_free_track.setEnabled(not active)
        self.btn_stop.setEnabled(active)
        self.btn_segment.setEnabled(not active)
