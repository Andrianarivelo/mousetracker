"""Export format selection dialog."""

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class ExportDialog(QDialog):
    """Dialog for configuring export options."""

    def __init__(self, default_dir: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Tracking Data")
        self.setMinimumWidth(400)
        self._default_dir = default_dir
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Output directory
        dir_group = QGroupBox("Output Directory")
        dir_layout = QHBoxLayout(dir_group)
        self.edit_output_dir = QLineEdit(self._default_dir)
        dir_layout.addWidget(self.edit_output_dir)
        btn_browse = QPushButton("Browse…")
        btn_browse.setObjectName("secondary_button")
        btn_browse.clicked.connect(self._browse_dir)
        dir_layout.addWidget(btn_browse)
        layout.addWidget(dir_group)

        # Format options
        fmt_group = QGroupBox("Export Formats")
        fmt_form = QFormLayout(fmt_group)

        self.chk_csv = QCheckBox("Trajectory CSV (frame, mouse_id, x, y, keypoints, ROIs)")
        self.chk_csv.setChecked(True)
        fmt_form.addRow(self.chk_csv)

        self.chk_h5 = QCheckBox("HDF5 Masks (per-frame binary masks)")
        self.chk_h5.setChecked(False)
        fmt_form.addRow(self.chk_h5)

        self.chk_video = QCheckBox("Overlay Video (with colored mask overlays)")
        self.chk_video.setChecked(False)
        fmt_form.addRow(self.chk_video)

        layout.addWidget(fmt_group)

        # Video overlay options
        overlay_group = QGroupBox("Video Overlay Options")
        overlay_form = QFormLayout(overlay_group)

        self.chk_draw_masks = QCheckBox("Draw masks")
        self.chk_draw_masks.setChecked(True)
        overlay_form.addRow(self.chk_draw_masks)

        self.chk_draw_bbox = QCheckBox("Draw bounding boxes")
        overlay_form.addRow(self.chk_draw_bbox)

        self.chk_draw_kps = QCheckBox("Draw keypoints")
        overlay_form.addRow(self.chk_draw_kps)

        self.chk_draw_labels = QCheckBox("Draw identity labels")
        self.chk_draw_labels.setChecked(True)
        overlay_form.addRow(self.chk_draw_labels)

        layout.addWidget(overlay_group)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self._default_dir
        )
        if d:
            self.edit_output_dir.setText(d)

    def get_config(self) -> dict:
        return dict(
            output_dir=self.edit_output_dir.text(),
            export_csv=self.chk_csv.isChecked(),
            export_h5=self.chk_h5.isChecked(),
            export_video=self.chk_video.isChecked(),
            draw_masks=self.chk_draw_masks.isChecked(),
            draw_bbox=self.chk_draw_bbox.isChecked(),
            draw_kps=self.chk_draw_kps.isChecked(),
            draw_labels=self.chk_draw_labels.isChecked(),
        )
