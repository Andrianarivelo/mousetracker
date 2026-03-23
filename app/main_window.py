"""MainWindow with docked layout for MouseTracker Pro."""

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage, QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Default Qt key codes → entity index (1-based).
# Covers both QWERTY number row and AZERTY unshifted equivalents.
_DEFAULT_DIGIT_KEYS: dict[int, int] = {
    int(Qt.Key_1): 1, int(Qt.Key_2): 2, int(Qt.Key_3): 3,
    int(Qt.Key_4): 4, int(Qt.Key_5): 5, int(Qt.Key_6): 6,
    # AZERTY: & é " ' ( -
    0x0026: 1, 0x00e9: 2, 0x0022: 3,
    0x0027: 4, 0x0028: 5, 0x002d: 6,
}

from app.config import (
    IDENTITY_COLORS,
    MASK_ALPHA,
    VIEWER_UPDATE_EVERY_N_FRAMES,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from app.core.identity_manager import IdentityManager
from app.core.keypoint_estimator import KeypointEstimator, estimate_all_frames
from app.core.mask_recovery import SizeValidator
from app.core.roi_analyzer import ROIAnalyzer
from app.core.sam3_engine import SAM3Engine
from app.core.tracker import IdentityTracker
from app.core.video_io import (
    VideoReader,
    compose_mask_overlay,
    draw_bboxes,
    draw_entity_labels,
    draw_keypoints,
)
from app.ui.examples_panel import ExamplesPanel
from app.ui.export_dialog import ExportDialog
from app.ui.action_bar import ActionBar
from app.ui.file_panel import FilePanel
from app.ui.filter_panel import FilterPanel
from app.ui.fine_tune_dialog import FineTuneDialog
from app.ui.identity_panel import IdentityPanel
from app.ui.keypoint_panel import KeypointPanel
from app.ui.preprocessing_panel import PreprocessingPanel
from app.ui.progress_widget import ProgressWidget
from app.ui.roi_drawer import ROIOverlayManager, ROIPanel, draw_rois_on_frame
from app.ui.style import DARK_THEME_QSS
from app.ui.timeline_widget import TimelineWidget
from app.ui.video_viewer import VideoViewer
from app.workers.export_worker import ExportWorker
from app.workers.fine_tune_worker import FineTuneWorker
from app.workers.preprocessing_worker import PreprocessingWorker
from app.workers.tracking_worker import TrackingWorker

logger = logging.getLogger(__name__)

_FINE_TUNE_DATASET_RE = re.compile(r"Raw dataset length\s*=\s*(\d+)")
_FINE_TUNE_PROGRESS_RE = re.compile(r"(Train|Val) Epoch: \[(\d+)\]\[(\d+)/(\d+)\]")
_FINE_TUNE_REMAINING_RE = re.compile(r"Estimated time remaining:\s*(.+)")


class MainWindow(QMainWindow):
    """
    Top-level application window for MouseTracker Pro.

    Layout:
      Central:  VideoViewer + Timeline + ActionBar + ProgressWidget
      Right bar: Identity, Filter, File, Preprocess, Dataset, Keypoints, ROI
    """

    @classmethod
    def create_as_widget(cls, parent=None) -> "QWidget":
        """Embed MouseTracker Pro inside another widget (e.g. a QTabWidget).

        Converts the QMainWindow into an embedded widget by setting
        Qt.WindowType.Widget on it, so it behaves as a child panel rather
        than a top-level window while retaining full dock-widget functionality.
        """
        win = cls()
        win.setWindowFlags(Qt.WindowType.Widget)
        return win

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MouseTracker Pro")
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(DARK_THEME_QSS)

        # ── Core objects ──────────────────────────────────────────────────────
        self._engine = SAM3Engine()
        self._tracker = IdentityTracker(n_mice=2)  # updated dynamically as entities change
        self._identity_mgr = IdentityManager()
        self._roi_analyzer = ROIAnalyzer()
        self._keypoint_estimator = KeypointEstimator()
        self._size_validator = SizeValidator(tolerance=0.20)
        self._video_reader: Optional[VideoReader] = None
        self._video_path: Optional[str] = None

        # Tracking state
        self._current_frame_idx = 0
        self._all_masks: dict[int, dict[int, np.ndarray]] = {}  # {frame_idx: {mouse_id: mask}}
        self._keypoints_by_frame: dict = {}
        self._tracking_worker: Optional[TrackingWorker] = None
        self._export_worker: Optional[ExportWorker] = None
        self._preprocess_worker: Optional[PreprocessingWorker] = None
        self._fine_tune_worker: Optional[FineTuneWorker] = None
        self._fine_tune_log_dialog: Optional[QDialog] = None
        self._fine_tune_log_view: Optional[QPlainTextEdit] = None
        self._fine_tune_log_status: Optional[QLabel] = None
        self._fine_tune_log_path: Optional[QLabel] = None
        self._fine_tune_output_dir: str = ""

        # Pending SAM3 prompt outputs (for interactive setup)
        self._pending_outputs: Optional[dict] = None   # raw SAM3 outputs on prompted frame
        self._prompt_frame_idx: int = 0
        self._rejected_masks: dict[int, np.ndarray] = {}  # SAM obj IDs rejected by filter
        self._prompt_points: list[tuple[int, int, int]] = []  # (x, y, label) refinement clicks
        self._segmentation_examples: dict[int, dict] = {}  # {frame_idx: {"prompt": str, "obj_count": int}}
        self._split_polygon_active: bool = False
        self._split_polygon_frame_idx: Optional[int] = None
        self._split_polygon_points: list[tuple[int, int]] = []
        self._split_polygon_freehand: bool = False
        self._split_polygon_temporary: bool = False
        self._show_keypoints: bool = False
        # Mutable key→entity mapping (can be customized from settings)
        self._entity_keys: dict[int, int] = dict(_DEFAULT_DIGIT_KEYS)

        # Batch tracking queue
        self._batch_queue: list[str] = []
        self._batch_index: int = 0
        self._batch_output_dir: str = ""
        self._batch_active: bool = False

        # ── UI ────────────────────────────────────────────────────────────────
        self._build_ui()
        self._connect_signals()

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.lbl_status = QLabel("Ready — load a video to begin")
        self.status_bar.addWidget(self.lbl_status)

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        import sys as _sys
        from pathlib import Path as _Path
        _sam3_root = str(_Path(__file__).parent.parent.parent)
        if _sam3_root not in _sys.path:
            _sys.path.insert(0, _sam3_root)
        from shared.sidebar_layout import SidebarLayout

        # ── Instantiate all panels ─────────────────────────────────────────────
        self.file_panel        = FilePanel()
        self.identity_panel    = IdentityPanel()
        self.preprocess_panel  = PreprocessingPanel()
        self.examples_panel    = ExamplesPanel()
        self.examples_panel.set_split_mode_active(False)
        self.keypoint_panel    = KeypointPanel()
        self.roi_panel         = ROIPanel(self._roi_analyzer)
        self.filter_panel      = FilterPanel()
        self.action_bar        = ActionBar()
        self.progress_widget   = ProgressWidget()
        self.viewer            = VideoViewer()
        self.timeline          = TimelineWidget()

        # ── Build sidebar layout (right-side vertical bar) ─────────────────────
        sidebar = SidebarLayout(bar_position="right")

        # Right-bar activity buttons
        sidebar.add_activity("identity",   "mouse-pointer", "Identity",   "Identity",            self.identity_panel)
        sidebar.add_bar_separator()
        sidebar.add_activity("filter",     "sliders",       "Filter",     "Filters",             self.filter_panel)
        sidebar.add_activity("file",       "folder-open",   "File",       "File Manager",        self.file_panel)
        sidebar.add_activity("preprocess", "upload",        "Preprocess", "Preprocessing",       self.preprocess_panel)
        sidebar.add_activity("dataset",    "layers",        "Dataset",    "Dataset",             self.examples_panel)
        sidebar.add_activity("keypoints",  "crosshair",     "Keypoints",  "Keypoints",           self.keypoint_panel)
        sidebar.add_activity("roi",        "map-pin",       "ROI",        "Regions of Interest", self.roi_panel)

        # Central: video viewer + timeline + action bar + progress (pinned below)
        center = QWidget()
        center_lay = QVBoxLayout(center)
        center_lay.setContentsMargins(0, 0, 0, 0)
        center_lay.setSpacing(0)
        center_lay.addWidget(self.viewer, 1)
        center_lay.addWidget(self.timeline)
        center_lay.addWidget(self.action_bar)
        center_lay.addWidget(self.progress_widget)
        sidebar.set_central(center)

        # Pre-open identity panel (most used)
        sidebar.show_activity("identity")

        # Store reference so _connect_signals can reach it
        self._sidebar = sidebar

        # Embed sidebar as the central widget of the QMainWindow
        self.setCentralWidget(sidebar)

        # Accept drag-and-drop on the whole window
        self.setAcceptDrops(True)

        # ROI overlay manager
        self._roi_overlay = ROIOverlayManager(self.roi_panel, self._roi_analyzer)

        # Build menu
        self._build_menu()

    def _build_menu(self) -> None:
        """Build the main menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        file_menu.addAction("Open Video…", self._menu_open_video)
        file_menu.addAction("Open Folder…", self._menu_open_folder)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        # Track menu
        track_menu = menubar.addMenu("Track")
        track_menu.addAction("Segment Current Frame", self._run_text_prompt)
        track_menu.addAction("Track Video", self._start_tracking)
        track_menu.addSeparator()
        track_menu.addAction("Estimate Keypoints", self._estimate_keypoints)
        track_menu.addAction("Analyze ROIs", self._analyze_rois)

        # Export menu
        export_menu = menubar.addMenu("Export")
        export_menu.addAction("Export…", self._show_export_dialog)

    # ── Global drag-and-drop ─────────────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        from app.ui.file_panel import VIDEO_EXTENSIONS, _collect_videos_from_dir
        paths: list[str] = []
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if Path(p).is_dir():
                paths.extend(_collect_videos_from_dir(p))
            elif Path(p).suffix.lower() in VIDEO_EXTENSIONS:
                paths.append(p)
        if not paths:
            return
        # Add to file list
        self.file_panel._add_videos(paths)
        # If no video is loaded yet, open the first one
        if self._video_reader is None:
            self._load_video(paths[0])

    # ── Signal connections ────────────────────────────────────────────────────

    def _apply_initial_dock_sizes(self) -> None:
        pass  # No docks in sidebar layout; sizing is handled by fixed widths

    def _connect_signals(self) -> None:
        # File panel
        self.file_panel.video_selected.connect(self._load_video)

        # Timeline
        self.timeline.frame_changed.connect(self._on_frame_changed)

        # Video viewer clicks
        self.viewer.click_point.connect(self._on_viewer_click)
        self.viewer.right_click_point.connect(self._on_viewer_right_click)
        self.viewer.lasso_started.connect(self._on_viewer_lasso_started)
        self.viewer.lasso_moved.connect(self._on_viewer_lasso_moved)
        self.viewer.lasso_finished.connect(self._on_viewer_lasso_finished)

        # Identity / entity panel
        self.identity_panel.mouse_selected.connect(self._on_mouse_selected)
        self.identity_panel.assignment_cleared.connect(self._on_assignment_cleared)
        self.identity_panel.entity_added.connect(self._on_entity_added)
        self.identity_panel.entity_removed.connect(self._on_entity_removed)
        self.identity_panel.swap_requested.connect(self._apply_swap)
        self.identity_panel.spin_swap_from.valueChanged.connect(self._sync_swap_flags)
        self.identity_panel.spin_swap_to.valueChanged.connect(self._sync_swap_flags)

        # Filter panel — sliders/toggles trigger live preview
        self.filter_panel.filter_changed.connect(self._on_filter_preview_changed)

        # Action bar
        self.action_bar.segment_requested.connect(self._run_text_prompt)
        self.action_bar.track_requested.connect(self._start_tracking)
        self.action_bar.free_track_requested.connect(self._start_free_track)
        self.action_bar.stop_requested.connect(self._stop_tracking)
        self.action_bar.export_csv_requested.connect(self._export_csv)
        self.action_bar.export_h5_requested.connect(self._export_h5)
        self.action_bar.export_video_requested.connect(self._export_video_overlay)

        # Keypoint panel
        self.keypoint_panel.estimate_requested.connect(self._estimate_keypoints)
        self.keypoint_panel.selection_changed.connect(self._on_keypoint_selection_changed)

        # Preprocessing panel
        self.preprocess_panel.preprocess_requested.connect(self._start_preprocessing)
        self.preprocess_panel.crop_draw_requested.connect(self._on_crop_draw_requested)
        self.preprocess_panel.crop_rect_changed.connect(self._on_crop_rect_changed)
        self.viewer.crop_drawn.connect(self._on_viewer_crop_drawn)

        # Repaint when overlay checkboxes toggle
        self.action_bar.chk_show_labels.stateChanged.connect(
            lambda: self._render_frame(self._current_frame_idx)
        )
        self.action_bar.chk_show_bbox.stateChanged.connect(
            lambda: self._render_frame(self._current_frame_idx)
        )

        # Keyboard shortcut customization
        self.action_bar.shortcuts_changed.connect(self._on_shortcuts_changed)

        # Segmentation examples panel
        self.examples_panel.frame_selected.connect(self._on_example_frame_selected)
        self.examples_panel.reprompt_requested.connect(self._on_example_reprompt_requested)
        self.examples_panel.remove_requested.connect(self._on_example_remove_requested)
        self.examples_panel.clear_requested.connect(self._on_examples_clear_requested)
        self.examples_panel.capture_frame_requested.connect(self._on_capture_frame)
        self.examples_panel.split_merged_requested.connect(self._on_split_merged_requested)

        # Batch tracking
        self.file_panel.batch_track_requested.connect(self._start_batch_track)

        # Dataset management
        self.examples_panel.build_dataset_requested.connect(self._on_build_dataset)
        self.examples_panel.add_to_dataset_requested.connect(self._on_add_to_dataset)
        self.examples_panel.load_labels_requested.connect(self._on_load_labels)
        self.examples_panel.fine_tune_requested.connect(self._on_fine_tune_requested)

    # ── Video loading ─────────────────────────────────────────────────────────

    def _load_video(self, path: str) -> None:
        try:
            self._cancel_split_polygon_mode(rerender=False)
            if self._video_reader:
                self._video_reader.close()
            self._video_reader = VideoReader(path)
            self._video_path = path
            info = self._video_reader.info

            self.timeline.load_video(info.frame_count, info.fps)
            self.file_panel.set_video_info(
                f"{info.name}\n{info.width}×{info.height}  {info.fps:.1f} fps\n"
                f"{info.frame_count} frames  {info.duration_s:.1f} s"
            )
            self.preprocess_panel.populate_from_video(info.width, info.height, info.duration_s)
            self.action_bar.set_video_duration(info.duration_s)

            # Show first frame
            frame = self._video_reader.read_frame(0)
            if frame is not None:
                self.viewer.display_first_frame(frame)

            # Reset tracking state
            self._all_masks.clear()
            self._keypoints_by_frame.clear()
            self._tracker.reset()
            self._identity_mgr.reset()
            self._size_validator.reset()
            self._pending_outputs = None
            self._prompt_frame_idx = 0
            self._rejected_masks = {}
            self._prompt_points.clear()
            self._segmentation_examples.clear()
            self.examples_panel.clear_examples()
            self.progress_widget.reset()

            self.lbl_status.setText(f"Loaded: {info.name}")
            logger.info(f"Video loaded: {path}")

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load video:\n{e}")
            logger.exception(f"Load error: {e}")

    def _menu_open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v *.webm);;All Files (*)"
        )
        if path:
            self._load_video(path)

    def _menu_open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open Folder")
        if folder:
            from app.ui.file_panel import _collect_videos_from_dir
            paths = _collect_videos_from_dir(folder)
            if paths:
                self.file_panel._add_videos(paths)
                self._load_video(paths[0])

    # ── Frame display ─────────────────────────────────────────────────────────

    def _on_frame_changed(self, frame_idx: int) -> None:
        if self._split_polygon_active and frame_idx != self._split_polygon_frame_idx:
            self._cancel_split_polygon_mode(
                "Split mode cancelled after changing frame.",
                rerender=False,
            )
        self._current_frame_idx = frame_idx
        self.identity_panel.set_current_frame(frame_idx)
        self._render_frame(frame_idx)
        self._update_assignment_cursor()

    def _render_frame(self, frame_idx: int) -> None:
        """Read and display a frame with all overlays."""
        if self._video_reader is None:
            return
        frame = self._video_reader.read_frame(frame_idx)
        if frame is None:
            return

        composite = frame.copy()

        # Mask overlay (fall back to nearest earlier tracked frame when frame_skip > 1)
        masks = self._nearest_masks(frame_idx)
        if masks:
            composite = compose_mask_overlay(composite, masks, IDENTITY_COLORS, MASK_ALPHA)

        # Rejected masks — faint red overlay, only shown on the active prompt frame
        if frame_idx == self._prompt_frame_idx and self._rejected_masks:
            rej_norm = {i + 1: m for i, m in enumerate(self._rejected_masks.values())}
            rej_colors = {k: (200, 60, 60) for k in rej_norm}
            composite = compose_mask_overlay(composite, rej_norm, rej_colors, alpha=0.25)

        # Prompt-mode overlay: instruction banner + click-point markers
        if frame_idx == self._prompt_frame_idx and self._pending_outputs is not None:
            composite = self._draw_prompt_overlay(composite)

        if self._split_polygon_active and frame_idx == self._split_polygon_frame_idx:
            composite = self._draw_split_polygon_overlay(composite)

        # ROI overlay
        if self._roi_analyzer.rois:
            composite = draw_rois_on_frame(composite, self._roi_analyzer)

        # Keypoint overlay
        if self._show_keypoints and self._keypoints_by_frame:
            from app.config import KEYPOINT_COLORS
            fkps = self._keypoints_by_frame.get(frame_idx)
            if fkps:
                composite = draw_keypoints(
                    composite, fkps, IDENTITY_COLORS,
                    keypoint_colors=KEYPOINT_COLORS, show_labels=True,
                )

        # Entity labels (ID + confidence)
        if self.action_bar.chk_show_labels.isChecked():
            state = self._tracker.get_state_at(frame_idx)
            if state and state.centroids:
                names = {
                    eid: self.identity_panel.entity_name(eid)
                    for eid in state.centroids
                }
                active_entity_id = None
                if frame_idx == self._prompt_frame_idx and self._pending_outputs is not None:
                    active_entity_id = self.identity_panel.selected_mouse()
                composite = draw_entity_labels(
                    composite, state.centroids, state.confidences,
                    IDENTITY_COLORS, names,
                    active_entity_id=active_entity_id,
                )

        # Bounding boxes
        if self.action_bar.chk_show_bbox.isChecked():
            state = self._tracker.get_state_at(frame_idx)
            if state and state.bboxes:
                composite = draw_bboxes(composite, state.bboxes, IDENTITY_COLORS)

        self.viewer.display_frame(composite)

    def _draw_prompt_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw instruction banner and point refinement markers on the prompt frame."""
        import cv2
        frame = frame.copy()
        h, w = frame.shape[:2]

        # Choose banner text/colour based on assignment state
        selected = self.identity_panel.selected_mouse()
        n_assigned = self.identity_panel.assigned_count()
        n_total = self.identity_panel.entity_count()
        if selected is not None and n_total > 0 and n_assigned >= n_total:
            # Allow manual correction after auto-assignment: once an entity is
            # selected, prefer the click-to-reassign banner over the "all done" prompt.
            n_assigned = max(0, n_total - 1)

        if n_total > 0 and n_assigned >= n_total:
            banner = f"All {n_total} entities assigned — click   Track   to start"
            banner_color = (60, 210, 100)
        elif selected is not None:
            name = self.identity_panel.entity_name(selected)
            banner = f"LEFT-CLICK on  {name}  in the video     |     RIGHT-CLICK to erase a region"
            banner_color = (255, 200, 50)
        else:
            banner = "  Select an entity from the Identity panel (right), then click on it in the video"
            banner_color = (100, 180, 255)

        # Semi-transparent banner bar at the top
        bar_h = 26
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (12, 12, 30), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.putText(frame, banner, (6, bar_h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, banner_color, 1, cv2.LINE_AA)

        # Click-point markers
        for px, py, label in self._prompt_points:
            pt_color = (50, 230, 80) if label == 1 else (230, 50, 50)
            cv2.circle(frame, (px, py), 9, (10, 10, 10), -1)
            cv2.circle(frame, (px, py), 7, pt_color, -1)
            cv2.circle(frame, (px, py), 7, (255, 255, 255), 1)
            sym = "+" if label == 1 else "-"
            cv2.putText(frame, sym, (px - 4, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Bottom-right: prompt-type badge
        n_fg = sum(1 for _, _, l in self._prompt_points if l == 1)
        n_bg = sum(1 for _, _, l in self._prompt_points if l == 0)
        if self._prompt_points:
            parts = []
            if n_fg:
                parts.append(f"+{n_fg} add")
            if n_bg:
                parts.append(f"-{n_bg} erase")
            badge = "Point-refined: " + "  ".join(parts)
            badge_color = (80, 220, 120)
        else:
            badge = "Text prompt only"
            badge_color = (150, 150, 150)

        font_scale = 0.38
        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ix, iy = w - tw - 8, h - 7
        cv2.rectangle(frame, (ix - 4, iy - th - 3), (w - 2, iy + 3), (12, 12, 30), -1)
        cv2.putText(frame, badge, (ix, iy),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, badge_color, 1, cv2.LINE_AA)

        return frame

    def _update_assignment_cursor(self) -> None:
        """Show crosshair on the viewer when click-to-assign is active."""
        active = (
            not self._split_polygon_active
            and
            self._pending_outputs is not None
            and self._current_frame_idx == self._prompt_frame_idx
            and self.identity_panel.selected_mouse() is not None
        )
        self.viewer.set_assignment_cursor(active)

    def _start_split_polygon_mode(
        self,
        *,
        freehand: bool = False,
        temporary: bool = False,
    ) -> None:
        self._split_polygon_active = True
        self._split_polygon_frame_idx = self._current_frame_idx
        self._split_polygon_points.clear()
        self._split_polygon_freehand = freehand
        self._split_polygon_temporary = temporary
        self.examples_panel.set_split_mode_active(True)
        self._update_assignment_cursor()
        self._render_frame(self._current_frame_idx)
        if freehand:
            self.lbl_status.setText(
                "Split lasso: hold Ctrl and drag around one animal, release to apply, Esc to cancel."
            )
        else:
            self.lbl_status.setText(
                "Split mode: left-click around one animal, right-click to apply, Esc to cancel."
            )

    def _cancel_split_polygon_mode(
        self,
        status: Optional[str] = None,
        rerender: bool = True,
    ) -> None:
        was_active = self._split_polygon_active or bool(self._split_polygon_points)
        self._split_polygon_active = False
        self._split_polygon_frame_idx = None
        self._split_polygon_points.clear()
        self._split_polygon_freehand = False
        self._split_polygon_temporary = False
        self.examples_panel.set_split_mode_active(False)
        self._update_assignment_cursor()
        if rerender and was_active and self._video_reader is not None:
            self._render_frame(self._current_frame_idx)
        if status:
            self.lbl_status.setText(status)

    def _append_split_polygon_point(
        self,
        x: int,
        y: int,
        *,
        min_distance: int = 0,
    ) -> bool:
        point = (int(x), int(y))
        if not self._split_polygon_points:
            self._split_polygon_points.append(point)
            return True

        last_x, last_y = self._split_polygon_points[-1]
        if min_distance > 0:
            dx = point[0] - last_x
            dy = point[1] - last_y
            if (dx * dx) + (dy * dy) < (min_distance * min_distance):
                return False
        elif point == self._split_polygon_points[-1]:
            return False

        self._split_polygon_points.append(point)
        return True

    def _can_begin_split_polygon_mode(self) -> bool:
        if self._video_reader is None:
            return False

        frame_idx = self._current_frame_idx
        info = self._video_reader.info
        frame_shape = (info.height, info.width)
        masks = self._current_split_source_masks(frame_idx, frame_shape)
        if not masks:
            self.lbl_status.setText("No masks on current frame to split.")
            return False

        n_entities = self.identity_panel.entity_count()
        if n_entities < 2:
            self.lbl_status.setText("Need at least 2 entities to split merged masks.")
            return False
        return True

    def _draw_split_polygon_overlay(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        if (
            not self._split_polygon_active
            or self._split_polygon_frame_idx != self._current_frame_idx
        ):
            return frame

        out = frame.copy()
        points = np.array(self._split_polygon_points, dtype=np.int32)
        accent = (80, 220, 255)

        if len(points) >= 3:
            overlay = out.copy()
            cv2.fillPoly(overlay, [points], accent)
            cv2.addWeighted(overlay, 0.16, out, 0.84, 0, out)

        if len(points) >= 2:
            cv2.polylines(out, [points], isClosed=False, color=accent, thickness=2)

        if self._split_polygon_freehand:
            if len(self._split_polygon_points) > 0:
                x, y = self._split_polygon_points[-1]
                cv2.circle(out, (int(x), int(y)), 4, accent, -1)
                cv2.circle(out, (int(x), int(y)), 6, (12, 12, 24), 1)
            badge = "SPLIT LASSO  hold Ctrl and drag  release to apply"
        else:
            for idx, (x, y) in enumerate(self._split_polygon_points, start=1):
                cv2.circle(out, (int(x), int(y)), 4, accent, -1)
                cv2.circle(out, (int(x), int(y)), 6, (12, 12, 24), 1)
                cv2.putText(
                    out,
                    str(idx),
                    (int(x) + 6, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    accent,
                    1,
                    cv2.LINE_AA,
                )
            badge = "SPLIT MODE  left-click vertices  right-click apply"

        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(out, (8, 8), (18 + tw, 18 + th), (12, 12, 28), -1)
        cv2.putText(
            out,
            badge,
            (14, 14 + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            accent,
            1,
            cv2.LINE_AA,
        )
        return out

    @staticmethod
    def _mask_centroid_x(mask: np.ndarray) -> float:
        ys, xs = np.where(mask)
        return float(xs.mean()) if len(xs) > 0 else 0.0

    def _current_split_source_masks(
        self,
        frame_idx: int,
        frame_shape: tuple[int, int],
    ) -> Optional[dict[int, np.ndarray]]:
        masks = None
        if self._pending_outputs is not None and frame_idx == self._prompt_frame_idx:
            masks, rejected = self._to_filtered_masks(self._pending_outputs, frame_shape)
            self._rejected_masks = rejected
        if not masks:
            masks = self._all_masks.get(frame_idx)
        if not masks:
            return None
        return {int(mask_id): mask.astype(bool) for mask_id, mask in masks.items()}

    def _commit_split_masks(
        self,
        frame_idx: int,
        mask_list: list[np.ndarray],
        source_count: int,
        n_entities: int,
    ) -> None:
        if self._video_reader is None:
            return

        info = self._video_reader.info
        frame_shape = (info.height, info.width)
        mask_list = sorted(mask_list, key=self._mask_centroid_x)
        split_masks = {i + 1: mask for i, mask in enumerate(mask_list)}

        self._prompt_frame_idx = frame_idx
        self._pending_outputs = self._outputs_from_masks(split_masks)

        self._identity_mgr.reset()
        self.identity_panel.reset()
        sam_masks, rejected = self._to_filtered_masks(self._pending_outputs, frame_shape)
        self._rejected_masks = rejected
        self._all_masks[frame_idx] = {
            i + 1: mask for i, mask in enumerate(sam_masks.values())
        }

        n_auto = self._auto_assign_entities(sam_masks, frame_idx)
        if n_auto > 0:
            mapping = self._identity_mgr.get_full_mapping()
            display_masks = {}
            for mid, sid in mapping.items():
                if sid in sam_masks:
                    display_masks[mid] = sam_masks[sid]
            self._all_masks[frame_idx] = display_masks
            self._tracker.initialize(frame_idx, sam_masks, mapping)

        prompt = str(
            self._segmentation_examples.get(frame_idx, {}).get("prompt")
            or self.action_bar.text_prompt()
        )
        self._record_segmentation_example(
            frame_idx,
            prompt,
            len(sam_masks),
            self._pending_outputs,
        )
        self._render_frame(frame_idx)
        next_unassigned = self._refresh_prompt_assignment_ui(frame_idx, sam_masks)

        n_assigned = self.identity_panel.assigned_count()
        if n_entities > 0 and n_assigned < n_entities and next_unassigned is not None:
            next_name = self.identity_panel.entity_name(next_unassigned)
            self.lbl_status.setText(
                f"Split merged mask: {source_count} -> {len(sam_masks)} objects, "
                f"assigned {n_assigned}/{n_entities} - click the mask for {next_name}"
            )
        else:
            self.lbl_status.setText(
                f"Split merged mask: {source_count} -> {len(sam_masks)} objects, "
                f"assigned {n_assigned}/{n_entities}"
            )

    def _apply_split_polygon(self) -> None:
        from app.core.mask_recovery import split_mask_by_polygon

        if self._video_reader is None:
            return
        if len(self._split_polygon_points) < 3:
            status = "Need at least 3 polygon points to split a merged mask."
            if self._split_polygon_temporary:
                self._cancel_split_polygon_mode(status)
            else:
                self.lbl_status.setText(status)
            return

        frame_idx = (
            self._split_polygon_frame_idx
            if self._split_polygon_frame_idx is not None
            else self._current_frame_idx
        )
        info = self._video_reader.info
        frame_shape = (info.height, info.width)
        masks = self._current_split_source_masks(frame_idx, frame_shape)
        if not masks:
            self.lbl_status.setText("No masks on current frame to split.")
            return

        n_entities = self.identity_panel.entity_count()
        total_area = sum(int(mask.sum()) for mask in masks.values())
        expected = max(1, total_area // max(1, n_entities or len(masks)))
        min_piece_area = max(25, int(expected * 0.15))

        best_index: Optional[int] = None
        best_overlap = 0
        best_splits: list[np.ndarray] = []
        polygon = list(self._split_polygon_points)
        mask_list = [mask.astype(bool) for mask in masks.values()]

        for idx, mask in enumerate(mask_list):
            splits = split_mask_by_polygon(mask, polygon, min_area=min_piece_area)
            if len(splits) != 2:
                continue
            overlap = int(splits[0].sum())
            if overlap > best_overlap:
                best_index = idx
                best_overlap = overlap
                best_splits = sorted(splits, key=self._mask_centroid_x)

        if best_index is None:
            status = (
                "Polygon did not split any mask into two sizeable parts. Draw a tighter outline and try again."
            )
            if self._split_polygon_temporary:
                self._cancel_split_polygon_mode(status)
            else:
                self._split_polygon_points.clear()
                self._render_frame(frame_idx)
                self.lbl_status.setText(status)
            return

        mask_list = mask_list[:best_index] + best_splits + mask_list[best_index + 1:]
        self._cancel_split_polygon_mode(rerender=False)
        self._commit_split_masks(frame_idx, mask_list, len(masks), n_entities)

    # ── Interactive segmentation (click-to-assign) ─────────────────────────────

    def _run_text_prompt(self) -> None:
        """Run SAM3 text prompt on the current frame."""
        if not self._video_path:
            QMessageBox.warning(self, "No Video", "Please load a video first.")
            return
        self._cancel_split_polygon_mode(rerender=False)

        prompt = self.action_bar.text_prompt()
        frame_idx = self._current_frame_idx
        self._prompt_frame_idx = frame_idx
        self._prompt_points.clear()
        self.progress_widget.set_indeterminate(f"Segmenting frame {frame_idx}…")
        self.lbl_status.setText(f"Running SAM3 prompt '{prompt}' on frame {frame_idx}…")

        try:
            if not self._engine.is_loaded():
                self.progress_widget.set_indeterminate("Loading SAM3 model…")
                self._engine.load_model()

            # Always open a fresh single-frame session for the prompt step.
            # This is near-instant for any video length because only one JPEG
            # is loaded into SAM3 (vs. the full video).
            self._engine.start_session_on_frame(self._video_path, frame_idx)

            outputs = self._engine.add_text_prompt(0, prompt)  # frame_index=0 in single-frame session
            self._pending_outputs = outputs
            self.progress_widget.set_determinate()
            self.progress_widget.reset("Segmentation done — click masks to assign IDs")

            # Display detected masks with SAM IDs
            if self._video_reader:
                info = self._video_reader.info
                frame_shape = (info.height, info.width)
                sam_masks, rejected = self._to_filtered_masks(outputs, frame_shape)
                self._rejected_masks = rejected
                # Remap SAM IDs to 1-based so identity colors apply (SAM may return obj_id=0)
                self._all_masks[frame_idx] = {
                    i + 1: m for i, m in enumerate(sam_masks.values())
                }
                self._render_frame(frame_idx)

            obj_ids = list(outputs.get("out_obj_ids", []))
            n_filtered = len(sam_masks) if self._video_reader else len(obj_ids)
            self._record_segmentation_example(frame_idx, prompt, n_filtered, outputs)

            # Auto-create entities if none exist — saves the user a manual step
            n_entities = self.identity_panel.entity_count()
            if n_entities == 0 and n_filtered > 0:
                for _ in range(n_filtered):
                    self.identity_panel._add_entity("mouse")
                n_entities = self.identity_panel.entity_count()
                logger.info("Auto-created %d mouse entities for %d detections", n_entities, n_filtered)

            # Auto-assign entities to masks (user can override by clicking)
            if n_entities > 0 and n_filtered > 0:
                n_auto = self._auto_assign_entities(sam_masks, frame_idx)
                if n_auto > 0:
                    # Rebuild display masks with stable mouse IDs
                    mapping = self._identity_mgr.get_full_mapping()
                    display_masks = {}
                    for mid, sid in mapping.items():
                        if sid in sam_masks:
                            display_masks[mid] = sam_masks[sid]
                    self._all_masks[frame_idx] = display_masks
                    self._tracker.initialize(frame_idx, sam_masks, mapping)
                    self._render_frame(frame_idx)

            first_unassigned = self._refresh_prompt_assignment_ui(frame_idx, sam_masks)

            n_assigned = self.identity_panel.assigned_count()
            if n_entities > 0:
                if n_assigned >= n_entities:
                    self.lbl_status.setText(
                        f"Found {n_filtered} object(s), auto-assigned all {n_entities} — "
                        "click a mask to override, or Track"
                    )
                elif n_assigned > 0:
                    name = self.identity_panel.entity_name(first_unassigned) if first_unassigned else ""
                    self.lbl_status.setText(
                        f"Found {n_filtered} object(s), auto-assigned {n_assigned}/{n_entities} — "
                        f"click the mask for {name}"
                    )
                elif first_unassigned is not None:
                    name = self.identity_panel.entity_name(first_unassigned)
                    self.lbl_status.setText(
                        f"Found {n_filtered} object(s) — click the mask for {name}"
                    )
                else:
                    self.lbl_status.setText(f"Found {n_filtered} object(s)")
            else:
                self.lbl_status.setText(
                    f"Found {n_filtered} object(s) — add entities in Identity panel, then click masks"
                )
            self._update_assignment_cursor()

        except Exception as e:
            self.progress_widget.set_determinate()
            self.progress_widget.reset()
            QMessageBox.critical(self, "Segmentation Error", str(e))
            logger.exception(f"Segmentation failed: {e}")

    def _auto_assign_entities(self, sam_masks: dict[int, np.ndarray], frame_idx: int) -> int:
        """
        Auto-assign entity IDs to SAM masks.

        If previous tracked masks exist, match by mask area similarity.
        Otherwise, sort masks left-to-right by centroid x and assign
        entities in list order (first entity → leftmost mask).

        Returns the number of assignments made.
        """
        entity_ids = list(self.identity_panel._entities.keys())
        sam_ids = sorted(sam_masks.keys())
        if not entity_ids or not sam_ids:
            return 0

        # Clear previous assignments for a fresh auto-assign
        self._identity_mgr.reset()
        self.identity_panel.reset()

        # Check if we have previous tracked masks to match against
        prev_masks: dict[int, np.ndarray] | None = None
        for fi in sorted(self._all_masks.keys(), reverse=True):
            if fi != frame_idx and self._all_masks[fi]:
                prev_masks = self._all_masks[fi]
                break

        if prev_masks and len(prev_masks) > 0:
            # Match by area similarity: build cost matrix (entity prev area vs new mask area)
            prev_areas = {mid: int(m.sum()) for mid, m in prev_masks.items()}
            new_areas = {sid: int(m.sum()) for sid, m in sam_masks.items()}
            # Map prev entity IDs to entity_ids order
            prev_eids = [eid for eid in entity_ids if eid in prev_areas]
            if prev_eids:
                from scipy.optimize import linear_sum_assignment
                n_prev = len(prev_eids)
                n_new = len(sam_ids)
                cost = np.zeros((n_prev, n_new))
                for i, eid in enumerate(prev_eids):
                    pa = prev_areas[eid]
                    for j, sid in enumerate(sam_ids):
                        na = new_areas[sid]
                        # Normalized area difference as cost
                        cost[i, j] = abs(pa - na) / max(pa, na, 1)
                row_idx, col_idx = linear_sum_assignment(cost)
                assigned = 0
                for ri, ci in zip(row_idx, col_idx):
                    eid = prev_eids[ri]
                    sid = sam_ids[ci]
                    self._identity_mgr.assign(eid, sid)
                    self.identity_panel.mark_assigned(eid, sid)
                    self._size_validator.record(eid, sam_masks[sid])
                    assigned += 1
                    logger.info("Auto-assign (area match): entity %d → SAM obj %d", eid, sid)
                return assigned

        # No previous state: sort masks by centroid x (left-to-right)
        centroids = {}
        for sid, mask in sam_masks.items():
            ys, xs = np.where(mask)
            if len(xs) > 0:
                centroids[sid] = float(xs.mean())
            else:
                centroids[sid] = 0.0
        sorted_sids = sorted(sam_ids, key=lambda s: centroids.get(s, 0.0))

        assigned = 0
        for i, eid in enumerate(entity_ids):
            if i >= len(sorted_sids):
                break
            sid = sorted_sids[i]
            self._identity_mgr.assign(eid, sid)
            self.identity_panel.mark_assigned(eid, sid)
            self._size_validator.record(eid, sam_masks[sid])
            assigned += 1
            logger.info("Auto-assign (spatial): entity %d → SAM obj %d", eid, sid)
        return assigned

    def _on_viewer_click(self, x: int, y: int) -> None:
        """Handle click on the video viewer."""
        if self._split_polygon_active:
            if self._split_polygon_freehand:
                return
            if self._split_polygon_frame_idx != self._current_frame_idx:
                self.lbl_status.setText(
                    "Split mode only applies on the frame where it was started."
                )
                return
            self._append_split_polygon_point(int(x), int(y))
            self._render_frame(self._current_frame_idx)
            self.lbl_status.setText(
                f"Split polygon: {len(self._split_polygon_points)} point(s) placed - right-click to apply."
            )
            return

        # ROI drawing mode takes priority
        if self._roi_overlay.is_drawing:
            completed_roi = self._roi_overlay.handle_click(float(x), float(y))
            if completed_roi:
                self.lbl_status.setText(f"ROI '{completed_roi}' added")
                self._render_frame(self._current_frame_idx)
            return

        # Identity assignment mode
        selected_mouse = self.identity_panel.selected_mouse()
        if selected_mouse is not None and self._pending_outputs is not None:
            if self._current_frame_idx != self._prompt_frame_idx:
                self.lbl_status.setText(
                    "Assignment is only active on the segmented example frame. "
                    "Select an example or segment this frame first."
                )
                return
            self._assign_mask_at_click(x, y, selected_mouse)

    def _on_viewer_right_click(self, x: int, y: int) -> None:
        if self._split_polygon_active:
            if self._split_polygon_freehand:
                self._cancel_split_polygon_mode("Split lasso cancelled.")
            else:
                self._apply_split_polygon()
            return

        if self._roi_overlay.is_drawing:
            completed = self._roi_overlay.handle_right_click(float(x), float(y))
            if completed:
                self.lbl_status.setText(f"ROI '{completed}' added")
                self._render_frame(self._current_frame_idx)
            return

        selected_mouse = self.identity_panel.selected_mouse()
        if (
            selected_mouse is not None
            and self._pending_outputs is not None
            and self._current_frame_idx == self._prompt_frame_idx
        ):
            # Right-click adds a background point to split/trim merged masks.
            self._refine_mask_with_point(x, y, selected_mouse, point_label=0)

    def _assign_mask_at_click(self, x: int, y: int, mouse_id: int) -> None:
        """Assign the mask under click position to mouse_id."""
        if self._pending_outputs is None or self._video_reader is None:
            return

        info = self._video_reader.info
        frame_shape = (info.height, info.width)
        sam_masks, _ = self._to_filtered_masks(self._pending_outputs, frame_shape)

        # Find which SAM mask contains this click
        clicked_sam_id = None
        for sam_id, mask in sam_masks.items():
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                clicked_sam_id = sam_id
                break

        if clicked_sam_id is None:
            # Add a foreground point to refine a merged/missed detection.
            if not self._refine_mask_with_point(x, y, mouse_id, point_label=1):
                return
            sam_masks, _ = self._to_filtered_masks(self._pending_outputs, frame_shape)
            for sam_id, mask in sam_masks.items():
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                    clicked_sam_id = sam_id
                    break
            if clicked_sam_id is None:
                self.lbl_status.setText(
                    "Refinement applied but no mask covers the clicked point yet. "
                    "Add more foreground/background clicks."
                )
                return

        # Assign identity and, when possible, swap with the previous owner so a
        # wrong auto-assignment can be corrected with a single click.
        old_sam_id = self._identity_mgr.sam_id_for_mouse(mouse_id)
        other_mouse_id = self._identity_mgr.mouse_id_for_sam(clicked_sam_id)
        swapped_mouse_id: Optional[int] = None
        self._identity_mgr.assign(mouse_id, clicked_sam_id)
        self.identity_panel.mark_assigned(mouse_id, clicked_sam_id)
        if (
            other_mouse_id is not None
            and other_mouse_id != mouse_id
            and old_sam_id is not None
            and old_sam_id != clicked_sam_id
        ):
            self._identity_mgr.assign(other_mouse_id, old_sam_id)
            self.identity_panel.mark_assigned(other_mouse_id, old_sam_id)
            swapped_mouse_id = other_mouse_id
        if clicked_sam_id in sam_masks:
            self._size_validator.record(mouse_id, sam_masks[clicked_sam_id])
        if swapped_mouse_id is not None and old_sam_id in sam_masks:
            self._size_validator.record(swapped_mouse_id, sam_masks[old_sam_id])

        # Rebuild masks with stable mouse IDs
        mapping = self._identity_mgr.get_full_mapping()
        display_masks = {}
        for mid, sid in mapping.items():
            if sid in sam_masks:
                display_masks[mid] = sam_masks[sid]

        self._all_masks[self._prompt_frame_idx] = display_masks

        # Initialize tracker with this assignment
        self._tracker.initialize(
            self._prompt_frame_idx,
            sam_masks,
            mapping,
        )

        self._render_frame(self._current_frame_idx)
        next_unassigned = self._refresh_prompt_assignment_ui(
            self._prompt_frame_idx,
            sam_masks,
        )

        n_total = self.identity_panel.entity_count()
        n_done = self.identity_panel.assigned_count()
        if swapped_mouse_id is not None and n_done >= n_total:
            other_name = self.identity_panel.entity_name(swapped_mouse_id)
            self.lbl_status.setText(
                f"Swapped assignments for {self.identity_panel.entity_name(mouse_id)} and "
                f"{other_name} - all {n_total} entities assigned"
            )
        elif n_done >= n_total:
            self.lbl_status.setText(
                f"All {n_total} entities assigned — click Track to start tracking"
            )
        else:
            if next_unassigned is not None:
                next_name = self.identity_panel.entity_name(next_unassigned)
                self.lbl_status.setText(
                    f"Assigned entity {mouse_id} ({n_done}/{n_total}) — "
                    f"now click the mask for {next_name}"
                )
            else:
                self.lbl_status.setText(
                    f"Entity {mouse_id} → SAM obj {clicked_sam_id}  "
                    f"({n_done}/{n_total} assigned)"
                )
        logger.info(f"Identity assigned: entity {mouse_id} → SAM obj {clicked_sam_id}")

    def _refine_mask_with_point(self, x: int, y: int, mouse_id: int, point_label: int) -> bool:
        """
        Refine segmentation on the prompt frame with one point.
        point_label: 1=foreground, 0=background.
        """
        if self._pending_outputs is None or self._video_reader is None:
            return False
        if self._current_frame_idx != self._prompt_frame_idx:
            return False

        frame_shape = (self._video_reader.info.height, self._video_reader.info.width)
        obj_hint = self._identity_mgr.sam_id_for_mouse(mouse_id)
        if point_label == 0 and obj_hint is None:
            self.lbl_status.setText(
                "Background refinement needs an assigned entity first. "
                "Assign this entity with a left click before right-click refinement."
            )
            return False

        outputs = self._engine.add_point_prompt(
            0,  # single-frame session always has frame index 0
            [[float(x), float(y)]],
            [int(point_label)],
            obj_id=obj_hint,
            frame_shape=frame_shape,
        )
        self._pending_outputs = outputs
        self._prompt_points.append((x, y, point_label))
        sam_masks, rejected = self._to_filtered_masks(outputs, frame_shape)
        self._rejected_masks = rejected

        mapping = self._identity_mgr.get_full_mapping()
        display_masks: dict[int, np.ndarray] = {}
        for mid, sid in mapping.items():
            if sid in sam_masks:
                display_masks[mid] = sam_masks[sid]
        if not display_masks:
            # Fallback while no assignment exists yet: show raw SAM masks.
            display_masks = sam_masks
        self._all_masks[self._prompt_frame_idx] = display_masks
        self._render_frame(self._prompt_frame_idx)

        prompt = self.action_bar.text_prompt()
        obj_count = len(outputs.get("out_obj_ids", []))
        self._record_segmentation_example(self._prompt_frame_idx, prompt, obj_count, outputs)

        action = "foreground" if point_label == 1 else "background"
        self.lbl_status.setText(f"Added {action} refinement point at ({x}, {y})")
        return True

    def _on_split_merged(self) -> None:
        """
        Run watershed / connected-component split on the current frame's masks.

        Looks for masks that are oversized (> expected area for N entities in the
        frame) and splits them, then re-assigns entity IDs.
        """
        from app.core.mask_recovery import watershed_split, connected_components_in_range

        frame_idx = self._current_frame_idx
        masks = self._all_masks.get(frame_idx)
        if not masks:
            self.lbl_status.setText("No masks on current frame to split.")
            return
        if self._video_reader is None:
            return

        n_entities = self.identity_panel.entity_count()
        if n_entities < 2:
            self.lbl_status.setText("Need at least 2 entities to split merged masks.")
            return

        # Total mask area / n_entities = expected per-entity area
        total_area = sum(int(m.sum()) for m in masks.values())
        expected = total_area // n_entities
        min_area = int(expected * 0.3)
        max_area = int(expected * 1.7)

        new_masks: dict[int, np.ndarray] = {}
        next_id = 1
        for mid, mask in masks.items():
            area = int(mask.sum())
            if area > max_area:
                # Try watershed first
                splits = watershed_split(mask, min_area, max_area)
                if not splits:
                    splits = connected_components_in_range(mask, min_area, max_area)
                if splits:
                    for sub in splits:
                        new_masks[next_id] = sub
                        next_id += 1
                    continue
            new_masks[next_id] = mask
            next_id += 1

        if len(new_masks) == len(masks):
            self.lbl_status.setText("No merged masks found to split.")
            return

        # Store split masks and re-render
        self._all_masks[frame_idx] = {i + 1: m for i, m in enumerate(new_masks.values())}
        # Rebuild pending outputs so filter preview and assignment still work
        # Update the SAM outputs to reflect the new mask count
        if self._pending_outputs is not None:
            info = self._video_reader.info
            frame_shape = (info.height, info.width)
            split_masks = self._all_masks[frame_idx]
            # Re-create outputs dict with the split masks
            import cv2
            out_ids = np.array(list(split_masks.keys()))
            out_masks = np.stack([m.astype(np.uint8) for m in split_masks.values()])
            self._pending_outputs = {
                "out_obj_ids": out_ids,
                "out_binary_masks": out_masks,
                "out_probs": np.ones((len(out_ids), 1)),
                "out_boxes_xywh": np.zeros((len(out_ids), 4)),
            }

        # Clear assignments and re-run auto-assign
        self._identity_mgr.reset()
        self.identity_panel.reset()
        info = self._video_reader.info
        frame_shape = (info.height, info.width)
        sam_masks, rejected = self._to_filtered_masks(self._pending_outputs, frame_shape)
        self._rejected_masks = rejected
        self._all_masks[frame_idx] = {i + 1: m for i, m in enumerate(sam_masks.values())}

        n_auto = self._auto_assign_entities(sam_masks, frame_idx)
        if n_auto > 0:
            mapping = self._identity_mgr.get_full_mapping()
            display_masks = {}
            for mid, sid in mapping.items():
                if sid in sam_masks:
                    display_masks[mid] = sam_masks[sid]
            self._all_masks[frame_idx] = display_masks
            self._tracker.initialize(frame_idx, sam_masks, mapping)

        self._render_frame(frame_idx)
        self.identity_panel.set_detection_hint(len(sam_masks), n_entities)
        self.lbl_status.setText(
            f"Split masks: {len(masks)} → {len(sam_masks)} objects, "
            f"{n_auto} auto-assigned"
        )

    def _on_split_merged_requested(self) -> None:
        """Toggle guided polygon split mode on the current frame."""
        if self._video_reader is None:
            return

        if self._split_polygon_active:
            self._cancel_split_polygon_mode("Split mode cancelled.")
            return

        if not self._can_begin_split_polygon_mode():
            return

        self._start_split_polygon_mode()

    def _on_viewer_lasso_started(self, x: int, y: int) -> None:
        if not self._can_begin_split_polygon_mode():
            return

        if self._split_polygon_active and not self._split_polygon_temporary:
            self._cancel_split_polygon_mode(rerender=False)

        if (
            not self._split_polygon_active
            or not self._split_polygon_freehand
            or self._split_polygon_frame_idx != self._current_frame_idx
        ):
            self._start_split_polygon_mode(freehand=True, temporary=True)

        self._split_polygon_points.clear()
        self._append_split_polygon_point(x, y)
        self._render_frame(self._current_frame_idx)
        self.lbl_status.setText(
            "Split lasso: keep dragging around one animal and release to apply."
        )

    def _on_viewer_lasso_moved(self, x: int, y: int) -> None:
        if (
            not self._split_polygon_active
            or not self._split_polygon_freehand
            or self._split_polygon_frame_idx != self._current_frame_idx
        ):
            return

        if self._append_split_polygon_point(x, y, min_distance=3):
            self._render_frame(self._current_frame_idx)

    def _on_viewer_lasso_finished(self, x: int, y: int) -> None:
        if (
            not self._split_polygon_active
            or not self._split_polygon_freehand
            or self._split_polygon_frame_idx != self._current_frame_idx
        ):
            return

        self._append_split_polygon_point(x, y, min_distance=1)
        if len(self._split_polygon_points) < 3:
            self._cancel_split_polygon_mode(
                "Split lasso cancelled. Hold Ctrl and drag around one animal to split it."
            )
            return

        self._apply_split_polygon()

    def _on_filter_preview_changed(self) -> None:
        """Re-apply mask filter and re-render when a filter slider moves."""
        if self._pending_outputs is None or self._video_reader is None:
            return
        info = self._video_reader.info
        frame_shape = (info.height, info.width)
        sam_masks, rejected = self._to_filtered_masks(self._pending_outputs, frame_shape)
        self._rejected_masks = rejected

        # Rebuild display masks: keep entity→SAM assignments where still valid
        mapping = self._identity_mgr.get_full_mapping()
        display_masks: dict = {}
        for mid, sid in mapping.items():
            if sid in sam_masks:
                display_masks[mid] = sam_masks[sid]
        if display_masks:
            self._all_masks[self._prompt_frame_idx] = display_masks
        else:
            # No assignments yet — remap SAM IDs to 1-based for display
            self._all_masks[self._prompt_frame_idx] = {
                i + 1: m for i, m in enumerate(sam_masks.values())
            }
        self._render_frame(self._prompt_frame_idx)

        n = len(sam_masks)
        n_entities = self.identity_panel.entity_count()
        self.identity_panel.set_detection_hint(n, n_entities)
        filters = []
        if self.filter_panel.area_filter_enabled():
            filters.append(f"area ≤ {self.filter_panel.slider_max_area.value()}%")
        if self.filter_panel.edge_filter_enabled():
            filters.append(f"edge ≤ {self.filter_panel.slider_max_edge.value()}%")
        filter_str = "  ".join(filters) if filters else "no filters"
        self.lbl_status.setText(f"Filter preview: {n} mask(s) pass — {filter_str}")

    def _to_filtered_masks(
        self, outputs: dict, frame_shape: tuple[int, int]
    ) -> tuple[dict, dict]:
        """Convert outputs to masks, apply filters, return (kept, rejected) dicts.

        When "Raw SAM output" is enabled ALL custom filtering is bypassed and the
        raw outputs_to_masks result is returned directly so the user can diagnose
        whether a filter is causing missed detections.

        Otherwise kept masks are limited to the N smallest surviving masks (N =
        entity count) so background/bedding regions are discarded in favour of
        compact animal blobs even when the area filter is off.
        """
        all_masks = self._engine.outputs_to_masks(outputs, frame_shape)
        if self.filter_panel.raw_sam_enabled():
            # Zero filtering — return everything SAM produced
            return all_masks, {}
        n_entities = self.identity_panel.entity_count()
        kept = self._engine.filter_masks(
            all_masks,
            frame_shape,
            max_area_frac=self.filter_panel.max_area_frac(),
            max_edge_frac=self.filter_panel.max_edge_frac(),
            use_area_filter=self.filter_panel.area_filter_enabled(),
            use_edge_filter=self.filter_panel.edge_filter_enabled(),
            max_detections=n_entities if n_entities > 0 else 0,
        )
        rejected = {k: v for k, v in all_masks.items() if k not in kept}
        return kept, rejected

    def _make_thumbnail_pixmap(self, frame_rgb: np.ndarray, max_w: int = 220, max_h: int = 124) -> Optional[QPixmap]:
        if frame_rgb is None or frame_rgb.size == 0:
            return None
        h, w = frame_rgb.shape[:2]
        bytes_per_line = int(frame_rgb.strides[0])
        image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        return pixmap.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def _outputs_from_masks(self, masks: dict[int, np.ndarray]) -> dict:
        out_ids: list[int] = []
        out_masks: list[np.ndarray] = []
        out_boxes: list[list[float]] = []
        for obj_id, mask in sorted(masks.items()):
            ys, xs = np.where(mask)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x1 = int(xs.min())
            x2 = int(xs.max())
            y1 = int(ys.min())
            y2 = int(ys.max())
            out_ids.append(int(obj_id))
            out_masks.append(mask.astype(np.uint8))
            out_boxes.append([x1, y1, x2 - x1 + 1, y2 - y1 + 1])

        if not out_ids:
            return {
                "out_obj_ids": np.array([], dtype=np.int32),
                "out_binary_masks": np.zeros((0, 0, 0), dtype=np.uint8),
                "out_probs": np.zeros((0, 1), dtype=np.float32),
                "out_boxes_xywh": np.zeros((0, 4), dtype=np.float32),
            }

        return {
            "out_obj_ids": np.asarray(out_ids, dtype=np.int32),
            "out_binary_masks": np.stack(out_masks).astype(np.uint8),
            "out_probs": np.ones((len(out_ids), 1), dtype=np.float32),
            "out_boxes_xywh": np.asarray(out_boxes, dtype=np.float32),
        }

    def _next_unassigned_entity_id(self) -> Optional[int]:
        for eid in self.identity_panel._entities.keys():
            if self._identity_mgr.sam_id_for_mouse(eid) is None:
                return eid
        return None

    def _update_example_assignment_note(
        self,
        frame_idx: int,
        obj_count: Optional[int] = None,
    ) -> None:
        entry = self._segmentation_examples.get(frame_idx)
        if entry is None:
            return

        if obj_count is None:
            obj_count = int(entry.get("obj_count", 0))
        entry["obj_count"] = int(obj_count)

        note_parts = [f"objs={int(obj_count)}"]
        n_entities = self.identity_panel.entity_count()
        if n_entities > 0:
            note_parts.append(
                f"assigned={self.identity_panel.assigned_count()}/{n_entities}"
            )

        prompt = str(entry.get("prompt", "")).strip()
        if prompt:
            note_parts.append(f"prompt='{prompt}'")

        self.examples_panel.set_example_note(frame_idx, " | ".join(note_parts))

    def _refresh_prompt_assignment_ui(
        self,
        frame_idx: int,
        sam_masks: dict[int, np.ndarray],
    ) -> Optional[int]:
        n_entities = self.identity_panel.entity_count()
        self.identity_panel.sync_assignments(self._identity_mgr.get_full_mapping())
        self.identity_panel.clear_detection_status()
        for eid in list(self.identity_panel._entities.keys()):
            sam_id = self._identity_mgr.sam_id_for_mouse(eid)
            if sam_id is not None:
                self.identity_panel.set_entity_detected(eid, sam_id in sam_masks)

        self.identity_panel.set_detection_hint(len(sam_masks), n_entities)
        self._update_example_assignment_note(frame_idx, len(sam_masks))

        next_unassigned = self._next_unassigned_entity_id()
        if next_unassigned is not None:
            self.identity_panel.select_entity(next_unassigned)

        self._update_assignment_cursor()
        return next_unassigned

    def _record_segmentation_example(
        self,
        frame_idx: int,
        prompt: str,
        obj_count: int,
        outputs: Optional[dict] = None,
    ) -> None:
        if self._video_reader is None:
            return
        frame = self._video_reader.read_frame(frame_idx)
        thumb = self._make_thumbnail_pixmap(frame) if frame is not None else None
        entry = self._segmentation_examples.get(frame_idx, {})
        entry["prompt"] = prompt
        entry["obj_count"] = int(obj_count)
        if outputs is not None:
            entry["outputs"] = outputs
        self._segmentation_examples[frame_idx] = entry
        self.examples_panel.upsert_example(frame_idx, thumb)
        self._update_example_assignment_note(frame_idx, obj_count)

    def _on_example_frame_selected(self, frame_idx: int) -> None:
        self.timeline.set_frame(frame_idx, emit=True)
        self._prompt_frame_idx = int(frame_idx)
        example = self._segmentation_examples.get(frame_idx)
        if example is not None and "outputs" in example:
            self._pending_outputs = example["outputs"]
        else:
            self._pending_outputs = None
        self.lbl_status.setText(f"Example frame {frame_idx} selected")

    def _on_example_reprompt_requested(self, frame_idx: int) -> None:
        self.timeline.set_frame(frame_idx, emit=True)
        self._run_text_prompt()

    def _on_example_remove_requested(self, frame_idx: int) -> None:
        if self._split_polygon_frame_idx == frame_idx:
            self._cancel_split_polygon_mode(rerender=False)
        self.examples_panel.remove_example(frame_idx)
        self._segmentation_examples.pop(frame_idx, None)
        self._all_masks.pop(frame_idx, None)
        if frame_idx == self._prompt_frame_idx:
            self._engine.close_session()
            self._pending_outputs = None
            self._prompt_frame_idx = self._current_frame_idx
        self._render_frame(self._current_frame_idx)
        self.lbl_status.setText(f"Removed example frame {frame_idx}")

    def _on_examples_clear_requested(self) -> None:
        self._cancel_split_polygon_mode(rerender=False)
        for frame_idx in list(self._segmentation_examples.keys()):
            self._all_masks.pop(frame_idx, None)
        self._segmentation_examples.clear()
        self.examples_panel.clear_examples()
        self._pending_outputs = None
        self._prompt_frame_idx = self._current_frame_idx
        self._engine.close_session()
        if len(self._tracker.history) <= 1:
            self._tracker.reset()
            self._identity_mgr.reset()
            self.identity_panel.reset()
        self._render_frame(self._current_frame_idx)
        self.lbl_status.setText("Segmentation examples cleared")

    def _on_capture_frame(self) -> None:
        """Capture the current tracked frame as a new segmentation example."""
        if self._video_reader is None:
            return
        frame_idx = self._current_frame_idx
        masks = self._all_masks.get(frame_idx)
        if not masks:
            self.lbl_status.setText(
                f"Frame {frame_idx} has no tracked masks — nothing to capture"
            )
            return
        frame = self._video_reader.read_frame(frame_idx)
        thumb = self._make_thumbnail_pixmap(frame) if frame is not None else None
        n_objs = len(masks)
        self._segmentation_examples[frame_idx] = {
            "prompt": "(captured)",
            "obj_count": n_objs,
        }
        self.examples_panel.upsert_example(frame_idx, thumb)
        self._update_example_assignment_note(frame_idx, n_objs)
        self.lbl_status.setText(
            f"Captured frame {frame_idx} with {n_objs} masks as example"
        )

    def _on_mouse_selected(self, mouse_id: int) -> None:
        self.lbl_status.setText(f"Entity {mouse_id} selected — click its mask in the video")
        self._update_assignment_cursor()

    def _on_assignment_cleared(self, mouse_id: int) -> None:
        self._identity_mgr.unassign_mouse(mouse_id)
        self._update_example_assignment_note(self._prompt_frame_idx)

    def _on_entity_added(self, entity_id: int, name: str, entity_type: str) -> None:
        n = self.identity_panel.entity_count()
        self._tracker.n_mice = n
        self._identity_mgr.set_n_mice(n)

    def _on_entity_removed(self, entity_id: int) -> None:
        n = self.identity_panel.entity_count()
        self._tracker.n_mice = n
        self._identity_mgr.set_n_mice(n)
        self._identity_mgr.unassign_mouse(entity_id)

    # ── Tracking ──────────────────────────────────────────────────────────────

    def _get_frame_range(self) -> tuple[int, int]:
        """Return (start_frame, end_frame) based on the time-window setting.

        end_frame is exclusive (like range()).  If the time window is disabled,
        returns (0, total_frame_count).
        """
        if self._video_reader is None:
            return 0, 0
        info = self._video_reader.info
        if not self.action_bar.time_window_enabled():
            return 0, info.frame_count
        start_s, end_s = self.action_bar.time_window()
        fps = info.fps if info.fps > 0 else 25.0
        start_f = max(0, int(start_s * fps))
        end_f = min(info.frame_count, int(end_s * fps))
        return start_f, end_f

    def _nearest_masks(self, frame_idx: int) -> dict | None:
        """Return masks for *frame_idx*, falling back to the nearest earlier
        tracked frame.  Handles frame_skip > 1 where only every Nth frame
        has actual SAM3 outputs stored in ``_all_masks``."""
        masks = self._all_masks.get(frame_idx)
        if masks is not None:
            return masks
        skip = self.action_bar.frame_skip()
        for f in range(frame_idx - 1, max(frame_idx - skip, -1), -1):
            masks = self._all_masks.get(f)
            if masks is not None:
                return masks
        return None

    def _filtered_history(self):
        """Return tracker history filtered to the active time window."""
        start_f, end_f = self._get_frame_range()
        return [s for s in self._tracker.history if start_f <= s.frame_idx < end_f]

    def _start_tracking(self) -> None:
        if self._video_path is None:
            QMessageBox.warning(self, "No Video", "Please load a video first.")
            return
        if self._tracking_worker and self._tracking_worker.isRunning():
            return

        if self._video_reader is None:
            return
        info = self._video_reader.info

        self.action_bar.set_tracking(True)
        self.viewer.set_assignment_cursor(False)
        self.progress_widget.reset("Starting tracking…")

        # If user hasn't done interactive segmentation yet, run a quick text
        # prompt on a single frame so the tracker has something to start from.
        if not self._tracker.history:
            try:
                self._run_text_prompt()
            except Exception as e:
                logger.exception(e)

        # The single-frame session from the prompt step is now closed by the
        # worker when it opens its own session (full or chunked).

        # Collect initial masks from tracker history (frame 0 assignment)
        initial_masks: dict[int, np.ndarray] = {}
        if self._tracker.history:
            state0 = self._tracker.history[0]
            # Map back to SAM obj IDs via identity manager
            for mouse_id, mask in state0.masks.items():
                sam_id = self._identity_mgr.sam_id_for_mouse(mouse_id)
                if sam_id is not None:
                    initial_masks[sam_id] = mask

        # Determine frame range from time-window setting
        tw_start, tw_end = self._get_frame_range()

        seg_size = self.action_bar.segment_size()
        track_frames = tw_end - tw_start
        if track_frames > seg_size:
            self.lbl_status.setText(
                f"Long range ({track_frames} frames) — using chunked mode "
                f"({seg_size}-frame segments via ffmpeg)"
            )
        if self.action_bar.time_window_enabled():
            start_s, end_s = self.action_bar.time_window()
            self.lbl_status.setText(
                f"Time window: {start_s:.1f}s – {end_s:.1f}s "
                f"(frames {tw_start}–{tw_end - 1})"
            )

        # Build keyframes from all segmented examples that have mask data.
        # {frame_idx: {mouse_id: mask}} sorted by frame index.
        keyframes: list[tuple[int, dict[int, np.ndarray]]] = []
        for kf_idx in sorted(self._segmentation_examples.keys()):
            if not (tw_start <= kf_idx < tw_end):
                continue
            kf_masks = self._all_masks.get(kf_idx)
            if kf_masks:
                keyframes.append((kf_idx, dict(kf_masks)))
        if len(keyframes) >= 2:
            kf_frames = [kf[0] for kf in keyframes]
            logger.info(
                "Multi-segment tracking: %d keyframes at frames %s",
                len(keyframes), kf_frames,
            )
            self.lbl_status.setText(
                f"Multi-segment mode: {len(keyframes)} keyframes at frames "
                + ", ".join(str(f) for f in kf_frames)
            )

        self._tracking_worker = TrackingWorker(
            engine=self._engine,
            tracker=self._tracker,
            video_path=self._video_path,
            frame_count=tw_end,
            frame_shape=(info.height, info.width),
            fps=info.fps,
            initial_masks=initial_masks,
            start_frame=tw_start,
            update_every=VIEWER_UPDATE_EVERY_N_FRAMES,
            frame_skip=self.action_bar.frame_skip(),
            max_area_frac=self.filter_panel.max_area_frac(),
            max_edge_frac=self.filter_panel.max_edge_frac(),
            use_area_filter=self.filter_panel.area_filter_enabled(),
            use_edge_filter=self.filter_panel.edge_filter_enabled(),
            bypass_filters=self.filter_panel.raw_sam_enabled(),
            adaptive_reprompt=self.action_bar.adaptive_reprompt_enabled(),
            keyframes=keyframes if len(keyframes) >= 2 else None,
            chunk_size=self.action_bar.segment_size(),
            size_validator=self._size_validator if self._size_validator.any_reference() else None,
        )
        self._tracking_worker.progress.connect(self._on_tracking_progress)
        self._tracking_worker.status.connect(self._on_tracking_status)
        self._tracking_worker.frame_result.connect(self._on_tracking_frame)
        self._tracking_worker.chunk_complete.connect(self._on_chunk_complete)
        self._tracking_worker.finished.connect(self._on_tracking_finished)
        self._tracking_worker.model_loading.connect(
            lambda: self.progress_widget.set_indeterminate("Loading SAM3 model…")
        )
        self._tracking_worker.start()
        logger.info("Tracking started")

    def _start_free_track(self) -> None:
        """
        Free Mode: auto-segment the current frame with the text prompt, assign
        detected objects to entities in order (no user clicks needed), then
        immediately start tracking.
        """
        if not self._video_path or self._video_reader is None:
            QMessageBox.warning(self, "No Video", "Please load a video first.")
            return
        if self._tracking_worker and self._tracking_worker.isRunning():
            return

        # Reset previous tracking state so we start clean
        self._all_masks.clear()
        self._tracker.reset()
        self._identity_mgr.reset()
        self._size_validator.reset()
        self._rejected_masks = {}

        # Run text prompt on the current frame
        self._run_text_prompt()
        if self._pending_outputs is None:
            return

        info = self._video_reader.info
        frame_shape = (info.height, info.width)
        sam_masks, rejected = self._to_filtered_masks(self._pending_outputs, frame_shape)
        self._rejected_masks = rejected

        if not sam_masks:
            QMessageBox.warning(
                self, "Free Track",
                "No objects detected after filtering.\n"
                "Try adjusting the text prompt or filter thresholds."
            )
            return

        # Ensure enough entities exist (add Mouse entities as needed)
        n_needed = len(sam_masks)
        while self.identity_panel.entity_count() < n_needed:
            self.identity_panel._add_entity("mouse")

        # Auto-assign: SAM obj IDs (sorted) → entity IDs (in order)
        entity_ids = list(self.identity_panel._entities.keys())
        for i, sam_obj_id in enumerate(sorted(sam_masks.keys())):
            if i >= len(entity_ids):
                break
            self._identity_mgr.assign(entity_ids[i], sam_obj_id)
            self._size_validator.record(entity_ids[i], sam_masks[sam_obj_id])

        # Initialize tracker from this auto-assignment
        mapping = self._identity_mgr.get_full_mapping()
        self._tracker.initialize(self._prompt_frame_idx, sam_masks, mapping)

        self.lbl_status.setText(
            f"Free Track: auto-assigned {len(sam_masks)} object(s) — starting tracking…"
        )

        # Hand off to the standard tracking path (it will see history is populated
        # and skip the auto-prompt step)
        self._start_tracking()

    def _stop_tracking(self) -> None:
        if self._tracking_worker and self._tracking_worker.isRunning():
            self._tracking_worker.abort()
            self.lbl_status.setText("Stopping tracking…")

    def _on_chunk_complete(self, last_frame: int) -> None:
        """After a chunk finishes, jump the viewer to show the latest tracked frame."""
        self.timeline.set_frame(last_frame, emit=True)

    # ── Keyboard navigation ────────────────────────────────────────────────────

    def keyPressEvent(self, event: QKeyEvent) -> None:
        # Don't intercept shortcuts while a text field has focus
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QAbstractSpinBox)):
            super().keyPressEvent(event)
            return

        key = int(event.key())
        if key == int(Qt.Key_Escape) and self._split_polygon_active:
            self._cancel_split_polygon_mode("Split mode cancelled.")
        elif key in self._entity_keys and not event.isAutoRepeat():
            n = self._entity_keys[key]
            if self.identity_panel.select_nth_entity(n):
                entity_id = self.identity_panel.nth_entity_id(n)
                if entity_id is not None:
                    name = self.identity_panel.entity_name(entity_id)
                    self.lbl_status.setText(
                        f"Entity {n} selected ({name}) — click its mask in the video"
                    )
        elif key == int(Qt.Key_Right):
            self._step_frame(+1)
        elif key == int(Qt.Key_Left):
            self._step_frame(-1)
        elif key == int(Qt.Key_S) and not event.isAutoRepeat():
            self._run_text_prompt()
        else:
            super().keyPressEvent(event)

    def _on_shortcuts_changed(self, names: dict) -> None:
        """Rebuild key→entity mapping from user-chosen shortcut keys."""
        new_keys: dict[int, int] = {}
        for idx, key_name in names.items():
            if not key_name:
                continue
            ch = key_name[0]
            qt_key = ord(ch.upper())
            new_keys[qt_key] = idx
            # Also map lowercase
            if ch.upper() != ch.lower():
                new_keys[ord(ch.lower())] = idx
        # Keep AZERTY fallbacks for digit keys that are still mapped to digits
        for qt_key, entity_idx in _DEFAULT_DIGIT_KEYS.items():
            if entity_idx in names and names[entity_idx] == str(entity_idx):
                new_keys[qt_key] = entity_idx
        self._entity_keys = new_keys
        self.identity_panel.set_shortcut_names(names)
        self.lbl_status.setText("Keyboard shortcuts updated")

    def _step_frame(self, delta: int) -> None:
        """Move the timeline by delta frames."""
        if self._video_reader is None:
            return
        max_f = self._video_reader.info.frame_count - 1
        new_idx = max(0, min(max_f, self._current_frame_idx + delta))
        self.timeline.set_frame(new_idx, emit=True)

    def _on_tracking_status(self, msg: str) -> None:
        self.progress_widget.set_indeterminate(msg)
        self.lbl_status.setText(msg)

    def _on_tracking_progress(self, percent: int, eta_s: float) -> None:
        self.progress_widget.set_determinate()
        self.progress_widget.update_progress(percent, f"Tracking… {percent}%", eta_s)

    def _on_tracking_frame(self, frame_idx: int, state) -> None:
        """Receive periodic frame results during tracking — always follow latest frame."""
        self._all_masks[frame_idx] = dict(state.masks)
        self.timeline.set_frame(frame_idx, emit=True)

    def _on_tracking_finished(self, success: bool, error: str) -> None:
        self.action_bar.set_tracking(False)
        if success:
            # Post-process: smooth trajectories then detect velocity spikes
            fps = self._video_reader.info.fps if self._video_reader else 25.0
            self._tracker.smooth_trajectories(window_length=11, polyorder=3)
            new_spikes = self._tracker.detect_velocity_swaps(fps=fps)
            if new_spikes:
                logger.info(
                    "Velocity-spike detector added %d new suspect frame(s): %s",
                    len(new_spikes), new_spikes[:10],
                )
            # Store all masks from history
            for state in self._tracker.history:
                self._all_masks[state.frame_idx] = dict(state.masks)
            # Set ID switch markers on timeline
            self.timeline.set_id_switch_markers(self._tracker.swap_log)
            # Update swap spinbox range now that we know total frame count
            if self._tracker.history:
                max_f = max(s.frame_idx for s in self._tracker.history)
                self.identity_panel.set_swap_max_frame(max_f)
            n_frames = len(self._tracker.history)
            n_swaps = len(self._tracker.swap_log)
            if n_swaps > 0:
                swap_frames_str = ", ".join(str(f) for f in self._tracker.swap_log[:5])
                if n_swaps > 5:
                    swap_frames_str += f" … (+{n_swaps - 5} more)"
                self.progress_widget.update_progress(
                    100,
                    f"Tracking complete — {n_swaps} possible ID switch(es) detected "
                    f"(red marks on timeline). Use Correct ID Swap to fix."
                )
                self.lbl_status.setText(
                    f"Done: {n_frames} frames — \u26a0 {n_swaps} ID switch(es) at frames: "
                    f"{swap_frames_str}.  Scrub timeline red marks, then use Correct ID Swap."
                )
                # Jump to first switch so user sees the problem immediately
                self.timeline.set_frame(self._tracker.swap_log[0], emit=True)
            else:
                self.progress_widget.update_progress(100, "Tracking complete")
                self.lbl_status.setText(f"Tracking complete — {n_frames} frames, no ID switches detected")
            self._render_frame(self._current_frame_idx)
        else:
            self.progress_widget.reset("Tracking failed")
            if not self._batch_active:
                QMessageBox.critical(self, "Tracking Error", error)
        logger.info(f"Tracking finished: success={success}")

        # Chain batch processing
        if self._batch_active:
            self._batch_post_track(success)

    # ── Batch tracking ─────────────────────────────────────────────────────────

    def _start_batch_track(self) -> None:
        """Prompt for output dir and start processing all listed videos."""
        videos = self.file_panel.all_video_paths()
        if not videos:
            QMessageBox.warning(self, "No Videos", "Add videos to the list first.")
            return
        if self._tracking_worker and self._tracking_worker.isRunning():
            QMessageBox.warning(self, "Busy", "Tracking is already running.")
            return

        out_dir = QFileDialog.getExistingDirectory(
            self, "Batch Output Directory", ""
        )
        if not out_dir:
            return

        self._batch_queue = list(videos)
        self._batch_index = 0
        self._batch_output_dir = out_dir
        self._batch_active = True

        self.lbl_status.setText(
            f"Batch Track: {len(videos)} video(s) → {out_dir}"
        )
        self.file_panel.set_batch_progress(0, len(videos), "starting…")
        self._batch_process_next()

    def _batch_process_next(self) -> None:
        """Load the next video in the queue and run free-track on it."""
        if self._batch_index >= len(self._batch_queue):
            self._batch_finish()
            return

        video_path = self._batch_queue[self._batch_index]
        total = len(self._batch_queue)
        name = Path(video_path).name
        self.file_panel.set_batch_progress(self._batch_index, total, name)
        self.lbl_status.setText(
            f"Batch {self._batch_index + 1}/{total}: loading {name}…"
        )

        # Load the video (reuses existing _load_video)
        self._load_video(video_path)
        if self._video_reader is None:
            logger.error("Batch: failed to load %s, skipping", video_path)
            self._batch_index += 1
            self._batch_process_next()
            return

        # Run free-track (auto-segment + auto-assign + start tracking)
        self._start_free_track()

    def _batch_post_track(self, success: bool) -> None:
        """After a batch video finishes tracking, export CSV and move to next."""
        video_path = self._batch_queue[self._batch_index]
        stem = Path(video_path).stem
        total = len(self._batch_queue)

        if success and self._tracker.history:
            # Auto-export CSV
            csv_path = str(Path(self._batch_output_dir) / f"{stem}_tracking.csv")
            try:
                result = self._do_export_csv(csv_path)
                logger.info("Batch export: %s", result)
            except Exception as e:
                logger.error("Batch CSV export failed for %s: %s", stem, e)

        self._batch_index += 1
        if self._batch_index < total:
            self.file_panel.set_batch_progress(
                self._batch_index, total, "next…"
            )
            # Use a short timer so the UI can repaint between videos
            QTimer.singleShot(100, self._batch_process_next)
        else:
            self._batch_finish()

    def _batch_finish(self) -> None:
        """Clean up after batch processing completes."""
        total = len(self._batch_queue)
        self._batch_active = False
        self._batch_queue.clear()
        self.file_panel.set_batch_progress(total, total, "done")
        self.progress_widget.update_progress(100, "Batch complete")
        self.lbl_status.setText(
            f"Batch Track complete — {total} video(s) processed, "
            f"CSV files in {self._batch_output_dir}"
        )
        logger.info("Batch tracking complete: %d videos", total)

    # ── Keypoints ─────────────────────────────────────────────────────────────

    def _on_keypoint_selection_changed(self, selected: list[str]) -> None:
        self._keypoint_estimator.selected = selected

    def _estimate_keypoints(self) -> None:
        if not self._tracker.history:
            QMessageBox.warning(self, "No Data", "Run tracking first.")
            return
        self.progress_widget.set_indeterminate("Estimating keypoints…")
        try:
            self._keypoints_by_frame = estimate_all_frames(
                self._tracker.history, self._keypoint_estimator
            )
            self._show_keypoints = True
            self._render_frame(self._current_frame_idx)
            self.progress_widget.reset(f"Keypoints estimated ({len(self._keypoints_by_frame)} frames)")
            self.lbl_status.setText("Keypoints estimated — scrub to view")
        except Exception as e:
            self.progress_widget.set_determinate()
            self.progress_widget.reset()
            QMessageBox.critical(self, "Keypoint Error", str(e))

    # ── ROI analysis ──────────────────────────────────────────────────────────

    def _analyze_rois(self) -> None:
        if not self._tracker.history:
            QMessageBox.warning(self, "No Data", "Run tracking first.")
            return
        if not self._roi_analyzer.rois:
            QMessageBox.warning(self, "No ROIs", "Draw at least one ROI first.")
            return

        fps = self._video_reader.info.fps if self._video_reader else 25.0
        trajs = self._tracker.get_trajectories()
        df = self._roi_analyzer.analyze(trajs, fps)

        from PySide6.QtWidgets import QDialog, QTextEdit
        dlg = QDialog(self)
        dlg.setWindowTitle("ROI Analysis Results")
        dlg.resize(600, 400)
        te = QTextEdit(dlg)
        te.setReadOnly(True)
        te.setPlainText(df.to_string(index=False))
        layout = QVBoxLayout(dlg)
        layout.addWidget(te)
        dlg.exec()

    # ── Export ────────────────────────────────────────────────────────────────

    def _show_export_dialog(self) -> None:
        default_dir = str(Path(self._video_path).parent) if self._video_path else ""
        dlg = ExportDialog(default_dir, self)
        if dlg.exec() != ExportDialog.Accepted:
            return
        cfg = dlg.get_config()
        out_dir = cfg["output_dir"] or default_dir
        stem = Path(self._video_path).stem if self._video_path else "output"

        if cfg["export_csv"]:
            self._run_export(
                "csv",
                self._do_export_csv,
                output_path=str(Path(out_dir) / f"{stem}_tracking.csv"),
            )
        if cfg["export_h5"]:
            self._run_export(
                "h5",
                self._do_export_h5,
                output_path=str(Path(out_dir) / f"{stem}_masks.h5"),
            )
        if cfg["export_video"]:
            self._run_export(
                "video",
                self._do_export_video,
                output_path=str(Path(out_dir) / f"{stem}_overlay.mp4"),
                draw_masks=cfg["draw_masks"],
                draw_bbox=cfg["draw_bbox"],
                draw_kps=cfg["draw_kps"],
                draw_labels=cfg["draw_labels"],
            )

    def _export_csv(self) -> None:
        if not self._tracker.history:
            QMessageBox.warning(self, "No Data", "Run tracking first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "", "CSV files (*.csv)"
        )
        if path:
            self._run_export("csv", self._do_export_csv, output_path=path)

    def _export_h5(self) -> None:
        if not self._tracker.history:
            QMessageBox.warning(self, "No Data", "Run tracking first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export HDF5", "", "HDF5 files (*.h5)"
        )
        if path:
            self._run_export("h5", self._do_export_h5, output_path=path)

    def _export_video_overlay(self) -> None:
        if not self._tracker.history:
            QMessageBox.warning(self, "No Data", "Run tracking first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Overlay Video", "", "Video files (*.mp4)"
        )
        if path:
            self._run_export("video", self._do_export_video, output_path=path)

    def _run_export(self, kind: str, fn, **kwargs) -> None:
        self.progress_widget.reset(f"Exporting {kind}…")
        self._export_worker = ExportWorker(fn, **kwargs)
        self._export_worker.progress.connect(
            lambda p: self.progress_widget.update_progress(p, f"Exporting {kind}… {p}%")
        )
        self._export_worker.finished.connect(self._on_export_finished)
        self._export_worker.start()

    def _do_export_csv(self, output_path: str, progress_callback=None) -> str:
        from app.export.csv_exporter import export_csv
        fps = self._video_reader.info.fps if self._video_reader else 25.0
        history = self._filtered_history()
        trajs = self._tracker.get_trajectories()
        roi_occ = self._roi_analyzer.get_occupancy_per_frame(trajs) if self._roi_analyzer.rois else None
        return export_csv(
            history, output_path, fps,
            keypoints_by_frame=self._keypoints_by_frame or None,
            roi_occupancy=roi_occ,
            progress_callback=progress_callback,
        )

    def _do_export_h5(self, output_path: str, progress_callback=None) -> str:
        from app.export.h5_exporter import export_h5
        if self._video_reader is None:
            raise RuntimeError("No video loaded")
        info = self._video_reader.info
        return export_h5(
            self._filtered_history(), output_path,
            (info.height, info.width), info.fps,
            self._video_path or "",
            keypoints_by_frame=self._keypoints_by_frame or None,
            progress_callback=progress_callback,
        )

    def _do_export_video(
        self,
        output_path: str,
        draw_masks: bool = True,
        draw_bbox: bool = False,
        draw_kps: bool = False,
        draw_labels: bool = True,
        progress_callback=None,
    ) -> str:
        from app.export.video_exporter import export_video
        start_f, end_f = self._get_frame_range()
        return export_video(
            self._video_path, self._filtered_history(), output_path,
            draw_masks=draw_masks, draw_bbox=draw_bbox,
            draw_kps=draw_kps, draw_labels=draw_labels,
            keypoints_by_frame=self._keypoints_by_frame or None,
            progress_callback=progress_callback,
            start_frame=start_f, end_frame=end_f,
        )

    def _on_export_finished(self, success: bool, result: str) -> None:
        if success:
            self.progress_widget.update_progress(100, "Export complete")
            self.lbl_status.setText(f"Exported: {result}")
        else:
            self.progress_widget.reset("Export failed")
            QMessageBox.critical(self, "Export Error", result)

    # ── Swap correction ───────────────────────────────────────────────────────

    def _sync_swap_flags(self) -> None:
        """Keep timeline flag markers in sync with the swap-range spinboxes."""
        from_f = self.identity_panel.spin_swap_from.value()
        to_f = self.identity_panel.spin_swap_to.value()
        self.timeline.set_swap_flags(from_f, to_f)

    def _apply_swap(self, id_a: int, id_b: int, from_frame: int, to_frame: int) -> None:
        if not self._tracker.history:
            QMessageBox.warning(self, "Swap", "No tracking data yet — run tracking first.")
            return
        max_tracked = max(s.frame_idx for s in self._tracker.history)
        frame_range = (
            max(0, from_frame),
            min(to_frame, max_tracked),
        )
        if frame_range[0] > frame_range[1]:
            QMessageBox.warning(
                self, "Swap",
                f"Invalid range: From {from_frame} > To {to_frame}."
            )
            return
        self._tracker.correct_swap(frame_range, id_a, id_b)
        # Update local mask cache for the corrected range
        for fi in range(frame_range[0], frame_range[1] + 1):
            state = self._tracker.get_state_at(fi)
            if state:
                self._all_masks[fi] = dict(state.masks)
        # Show swap range on timeline
        self.timeline.set_swap_flags(frame_range[0], frame_range[1])
        self._render_frame(self._current_frame_idx)
        name_a = self.identity_panel.entity_name(id_a)
        name_b = self.identity_panel.entity_name(id_b)
        self.lbl_status.setText(
            f"Swapped {name_a} ↔ {name_b}  frames {frame_range[0]}–{frame_range[1]}"
        )

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _on_crop_draw_requested(self, active: bool) -> None:
        """Toggle VideoViewer's crop-draw mode from the Draw Crop button."""
        if active:
            self.viewer.start_crop_draw_mode()
            self.lbl_status.setText(
                "Crop draw: click the first corner on the video, then the second"
            )
        else:
            self.viewer.stop_crop_draw_mode()

    def _on_viewer_crop_drawn(self, x: int, y: int, w: int, h: int) -> None:
        """Crop rect finalised by user drag — fill the spinboxes."""
        self.preprocess_panel.fill_crop_from_rect(x, y, w, h)
        self.lbl_status.setText(f"Crop set: x={x} y={y}  {w}×{h}")

    def _on_crop_rect_changed(self, x: int, y: int, w: int, h: int) -> None:
        """Update the live crop-rect overlay in the VideoViewer."""
        self.viewer.set_crop_rect(x, y, w, h)

    def _start_preprocessing(self, config) -> None:
        if not self._video_path:
            QMessageBox.warning(self, "No Video", "Please load a video first.")
            return

        from pathlib import Path
        p = Path(self._video_path)
        config.output_path = str(p.parent / f"{p.stem}_processed{p.suffix}")

        duration = self._video_reader.info.duration_s if self._video_reader else 0.0
        self.progress_widget.set_determinate()
        self.progress_widget.update_progress(0, "Preprocessing with ffmpeg…")
        self._preprocess_worker = PreprocessingWorker(
            self._video_path, config, total_duration_s=duration
        )
        self._preprocess_worker.progress.connect(
            lambda p: self.progress_widget.update_progress(p, f"Preprocessing… {p}%")
        )
        self._preprocess_worker.finished.connect(self._on_preprocess_finished)
        self._preprocess_worker.start()

    def _on_preprocess_finished(self, success: bool, result: str) -> None:
        if success:
            self.progress_widget.reset("Preprocessing complete")
            msg = QMessageBox(self)
            msg.setWindowTitle("Preprocessing Complete")
            msg.setText(f"Saved to:\n{result}\n\nLoad the preprocessed video?")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            if msg.exec() == QMessageBox.Yes:
                self._load_video(result)
        else:
            self.progress_widget.reset("Preprocessing failed")
            QMessageBox.critical(self, "Preprocessing Error", result)

    # ── Dataset management ─────────────────────────────────────────────────────

    def _on_build_dataset(self, dataset_dir: str) -> None:
        from app.core.dataset_manager import build_dataset
        try:
            class_names = self._get_entity_class_names()
            yaml_path = build_dataset(dataset_dir, class_names)
            self.examples_panel.set_dataset_status(f"Dataset created: {yaml_path}")
            self.lbl_status.setText(f"YOLO dataset built at {dataset_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Build Dataset", str(e))
            logger.exception("build_dataset failed: %s", e)

    def _on_add_to_dataset(self, dataset_dir: str, split: str) -> None:
        from app.core.dataset_manager import add_to_dataset, get_dataset_stats
        if not self._video_path or not self._video_reader:
            QMessageBox.warning(self, "No Video", "Please load a video first.")
            return

        # Only export frames that are in the examples list, not all tracked frames
        example_frames = set(self.examples_panel.frames())
        if not example_frames:
            QMessageBox.warning(
                self, "No Examples",
                "No example frames selected. Use Segment or Capture Current Frame to add examples first."
            )
            return
        example_masks = {
            fi: masks for fi, masks in self._all_masks.items()
            if fi in example_frames and masks
        }
        if not example_masks:
            QMessageBox.warning(
                self, "No Annotations",
                "Example frames have no masks. Segment and assign entities first."
            )
            return

        self.progress_widget.set_indeterminate(f"Saving {len(example_masks)} example(s) to {split} split…")
        try:
            class_names = self._get_entity_class_names()
            result = add_to_dataset(
                dataset_dir=dataset_dir,
                video_path=self._video_path,
                annotated_frames=example_masks,
                video_reader=self._video_reader,
                class_names=class_names,
                split=split,
            )
            stats = get_dataset_stats(dataset_dir, self._video_path)
            status = (
                f"Added {result['added']} frames ({result['skipped']} skipped) "
                f"| Dataset: {stats['total_images']} images "
                f"| This video: {stats['video_frames']} frames"
            )
            if stats["last_updated"]:
                status += f" | Last: {stats['last_updated']}"
            self.examples_panel.set_dataset_status(status)
            self.lbl_status.setText(
                f"Dataset: +{result['added']} frames to {split} "
                f"(total {stats['total_images']})"
            )
            self.progress_widget.reset(status)
        except Exception as e:
            self.progress_widget.reset("Dataset export failed")
            QMessageBox.critical(self, "Add to Dataset", str(e))
            logger.exception("add_to_dataset failed: %s", e)

    def _on_load_labels(self, dataset_dir: str) -> None:
        from app.core.dataset_manager import load_labels_from_dataset, get_dataset_stats
        if not self._video_path or not self._video_reader:
            QMessageBox.warning(self, "No Video", "Please load a video first.")
            return

        info = self._video_reader.info
        frame_shape = (info.height, info.width)

        try:
            loaded = load_labels_from_dataset(dataset_dir, self._video_path, frame_shape)
            if not loaded:
                QMessageBox.information(
                    self, "Load Labels",
                    f"No labels found for '{Path(self._video_path).name}' in this dataset."
                )
                return

            # Merge loaded masks into the app annotation state
            for frame_idx, masks in loaded.items():
                self._all_masks[frame_idx] = masks

                # Create example entries so they show in the panel
                if frame_idx not in self._segmentation_examples:
                    n_objs = len(masks)
                    frame = self._video_reader.read_frame(frame_idx)
                    thumb = self._make_thumbnail_pixmap(frame) if frame is not None else None
                    note = f"objs={n_objs} | loaded from dataset"
                    self.examples_panel.upsert_example(frame_idx, thumb, note)
                    self._segmentation_examples[frame_idx] = {
                        "prompt": "(loaded)",
                        "obj_count": n_objs,
                    }

            self._render_frame(self._current_frame_idx)

            stats = get_dataset_stats(dataset_dir, self._video_path)
            self.examples_panel.set_dataset_status(
                f"Loaded {len(loaded)} frames from dataset "
                f"| Dataset: {stats['total_images']} images"
            )
            self.lbl_status.setText(
                f"Loaded {len(loaded)} annotated frames from dataset"
            )
        except Exception as e:
            QMessageBox.critical(self, "Load Labels", str(e))
            logger.exception("load_labels_from_dataset failed: %s", e)

    def _on_fine_tune_requested(self, dataset_dir: str) -> None:
        from app.core.dataset_manager import get_dataset_stats
        from app.core.sam3_finetune import (
            default_finetune_params,
            prepare_finetune_job,
            runner_script_path,
        )

        if self._tracking_worker and self._tracking_worker.isRunning():
            QMessageBox.warning(
                self,
                "Fine-Tune SAM3",
                "Stop tracking before starting fine-tuning.",
            )
            return
        if self._fine_tune_worker and self._fine_tune_worker.isRunning():
            QMessageBox.information(
                self,
                "Fine-Tune SAM3",
                "A fine-tuning job is already running.",
            )
            return

        root = Path(dataset_dir)
        if not dataset_dir or not root.exists():
            QMessageBox.warning(
                self,
                "Fine-Tune SAM3",
                "Select an existing dataset directory first.",
            )
            return

        stats = get_dataset_stats(dataset_dir, self._video_path)
        if int(stats.get("total_images", 0)) <= 0:
            QMessageBox.warning(
                self,
                "Fine-Tune SAM3",
                "The selected dataset has no exported images yet.",
            )
            return

        defaults = default_finetune_params(dataset_dir)
        dlg = FineTuneDialog(dataset_dir, defaults, stats, self)
        if dlg.exec() != FineTuneDialog.Accepted:
            return

        params = dlg.get_params()
        self.progress_widget.set_indeterminate("Preparing SAM3 fine-tune job...")
        self.lbl_status.setText("Preparing SAM3 fine-tune job")

        try:
            class_names = self._get_entity_class_names()
            job = prepare_finetune_job(
                dataset_dir,
                params,
                sam3_class_names=class_names,
            )
            self._engine.unload_model()
        except Exception as error:
            self.progress_widget.reset("Fine-tune setup failed")
            QMessageBox.critical(self, "Fine-Tune SAM3", str(error))
            logger.exception("prepare_finetune_job failed: %s", error)
            return

        train_count = int(job["dataset_info"]["train_count"])
        val_count = int(job["dataset_info"]["val_count"])
        categories = ", ".join(job["dataset_info"].get("categories") or [])
        setup_status = (
            f"SAM3 fine-tune prepared | train={train_count} val={val_count}"
        )
        if categories:
            setup_status += f" | categories: {categories}"
        self.examples_panel.set_dataset_status(setup_status)

        self._fine_tune_worker = FineTuneWorker(
            runner_script=runner_script_path(),
            config_path=job["config_path"],
            output_dir=job["output_dir"],
            parent=self,
        )
        self._prepare_fine_tune_log_view(job["output_dir"], job["config_path"])
        self._fine_tune_worker.status.connect(self._on_fine_tune_status)
        self._fine_tune_worker.log_line.connect(self._on_fine_tune_log_line)
        self._fine_tune_worker.finished.connect(self._on_fine_tune_finished)
        self._fine_tune_worker.start()

    def _on_fine_tune_status(self, text: str) -> None:
        message = str(text).strip()
        if not message:
            return
        summary = self._summarize_fine_tune_message(message)
        self.progress_widget.set_indeterminate(summary)
        self.lbl_status.setText(summary)
        if self._fine_tune_log_status is not None:
            self._fine_tune_log_status.setText(summary)

    def _on_fine_tune_log_line(self, text: str) -> None:
        line = str(text).rstrip()
        if not line:
            return
        if self._fine_tune_log_view is None:
            self._ensure_fine_tune_log_dialog()
        if self._fine_tune_log_view is not None:
            self._fine_tune_log_view.appendPlainText(line)

    def _on_fine_tune_finished(self, success: bool, result: str) -> None:
        self.progress_widget.set_determinate()
        if success:
            status = f"SAM3 fine-tune complete: {result}"
            self.progress_widget.update_progress(100, status)
            self.examples_panel.set_dataset_status(status)
            self.lbl_status.setText(status)
            if self._fine_tune_log_status is not None:
                self._fine_tune_log_status.setText(status)
            if self._fine_tune_log_view is not None:
                self._fine_tune_log_view.appendPlainText("[MouseTracker] Training completed successfully.")
            QMessageBox.information(
                self,
                "Fine-Tune SAM3",
                f"Training finished.\n\nOutput directory:\n{result}",
            )
        else:
            self.progress_widget.reset("Fine-tune failed")
            self.lbl_status.setText("SAM3 fine-tune failed")
            if self._fine_tune_log_status is not None:
                self._fine_tune_log_status.setText("SAM3 fine-tune failed")
            if self._fine_tune_log_view is not None:
                self._fine_tune_log_view.appendPlainText("[MouseTracker] Training failed.")
            QMessageBox.critical(self, "Fine-Tune SAM3", result)
        self._fine_tune_worker = None

    def _ensure_fine_tune_log_dialog(self) -> None:
        if self._fine_tune_log_dialog is not None:
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("SAM3 Fine-Tune Progress")
        dlg.resize(880, 560)
        layout = QVBoxLayout(dlg)

        self._fine_tune_log_status = QLabel("Preparing SAM3 fine-tune...")
        self._fine_tune_log_status.setWordWrap(True)
        layout.addWidget(self._fine_tune_log_status)

        self._fine_tune_log_path = QLabel("")
        self._fine_tune_log_path.setWordWrap(True)
        layout.addWidget(self._fine_tune_log_path)

        self._fine_tune_log_view = QPlainTextEdit(dlg)
        self._fine_tune_log_view.setReadOnly(True)
        self._fine_tune_log_view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._fine_tune_log_view.setMaximumBlockCount(5000)
        layout.addWidget(self._fine_tune_log_view, 1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)

        btn_copy = QPushButton("Copy Log")
        btn_copy.setObjectName("secondary_button")
        btn_copy.clicked.connect(self._copy_fine_tune_log)
        button_row.addWidget(btn_copy)

        btn_hide = QPushButton("Hide")
        btn_hide.setObjectName("secondary_button")
        btn_hide.clicked.connect(dlg.hide)
        button_row.addWidget(btn_hide)

        layout.addLayout(button_row)
        self._fine_tune_log_dialog = dlg

    def _prepare_fine_tune_log_view(self, output_dir: str, config_path: str) -> None:
        self._fine_tune_output_dir = output_dir
        self._ensure_fine_tune_log_dialog()
        if self._fine_tune_log_view is None:
            return

        self._fine_tune_log_view.clear()
        if self._fine_tune_log_status is not None:
            self._fine_tune_log_status.setText("Launching SAM3 fine-tune...")
        if self._fine_tune_log_path is not None:
            log_path = Path(output_dir) / "logs" / "log.txt"
            self._fine_tune_log_path.setText(
                f"Output: {output_dir}\n"
                f"Config: {config_path}\n"
                f"Detailed trainer log: {log_path}"
            )

        self._fine_tune_log_view.appendPlainText("[MouseTracker] Live SAM3 fine-tune log")
        self._fine_tune_log_view.appendPlainText(
            "[MouseTracker] Progress guide: "
            "'Raw dataset length = N' means N training samples were loaded."
        )
        self._fine_tune_log_view.appendPlainText(
            "[MouseTracker] Progress guide: "
            "'Train Epoch: [e][i/n]' means epoch e+1, batch i+1 of n."
        )
        self._fine_tune_log_view.appendPlainText("")

        if self._fine_tune_log_dialog is not None:
            self._fine_tune_log_dialog.show()
            self._fine_tune_log_dialog.raise_()
            self._fine_tune_log_dialog.activateWindow()

    def _copy_fine_tune_log(self) -> None:
        if self._fine_tune_log_view is None:
            return
        QApplication.clipboard().setText(self._fine_tune_log_view.toPlainText())

    def _summarize_fine_tune_message(self, message: str) -> str:
        dataset_match = _FINE_TUNE_DATASET_RE.search(message)
        if dataset_match:
            return f"Loaded training dataset: {dataset_match.group(1)} samples"

        progress_match = _FINE_TUNE_PROGRESS_RE.search(message)
        if progress_match:
            phase, epoch, batch_idx, batch_total = progress_match.groups()
            current_batch = min(int(batch_idx) + 1, int(batch_total))
            prefix = "Training" if phase == "Train" else "Validation"
            return (
                f"{prefix} epoch {int(epoch) + 1} | "
                f"batch {current_batch}/{batch_total}"
            )

        remaining_match = _FINE_TUNE_REMAINING_RE.search(message)
        if remaining_match:
            return f"Estimated time remaining: {remaining_match.group(1)}"

        if "Setting up components:" in message:
            return "Initializing model, optimizer, and training components..."
        if "Finished setting up components:" in message:
            return "Trainer setup complete. Starting training..."
        if "Moving components to device" in message:
            return "Moving SAM3 model to the selected device..."
        if "TensorBoard SummaryWriter instantiated." in message:
            return "TensorBoard logging is ready."
        if "Experiment Log Dir:" in message:
            return "Preparing log and checkpoint directories..."
        if "Losses and meters:" in message:
            return "Completed a training epoch. Updating metrics..."
        if "Meters:" in message:
            return "Completed validation. Updating metrics..."
        return message

    def _get_entity_class_names(self) -> dict[int, str]:
        """Return {mouse_id: name} from the identity panel entities."""
        names: dict[int, str] = {}
        for eid in list(self.identity_panel._entities.keys()):
            names[eid] = self.identity_panel.entity_name(eid)
        if not names:
            names = {1: "mouse"}
        return names

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if self._tracking_worker and self._tracking_worker.isRunning():
            self._tracking_worker.abort()
            self._tracking_worker.wait(3000)
        if self._fine_tune_worker and self._fine_tune_worker.isRunning():
            self._fine_tune_worker.abort()
            self._fine_tune_worker.wait(3000)
        self._engine.close_session()
        if self._video_reader:
            self._video_reader.close()
        event.accept()
