"""Dialog for configuring a local SAM3 fine-tuning run."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.core.sam3_finetune import Sam3FineTuneParams, max_supported_local_gpus


class FineTuneDialog(QDialog):
    """Collect the training parameters for a SAM3 fine-tuning job."""

    def __init__(
        self,
        dataset_dir: str,
        defaults: Sam3FineTuneParams,
        dataset_info: Optional[dict] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._dataset_dir = dataset_dir
        self._defaults = defaults
        self._dataset_info = dataset_info or {}
        self.setWindowTitle("Fine-Tune SAM3")
        self.setMinimumWidth(560)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        summary = QGroupBox("Dataset")
        summary_layout = QVBoxLayout(summary)
        train_count = int(self._dataset_info.get("train_images", 0))
        val_count = int(self._dataset_info.get("val_images", 0))
        total_count = int(self._dataset_info.get("total_images", train_count + val_count))
        categories = self._dataset_info.get("categories") or []
        category_text = ", ".join(str(name) for name in categories) if categories else "unknown"
        self.lbl_summary = QLabel(
            f"{Path(self._dataset_dir).resolve()}\n"
            f"Images: {total_count} total ({train_count} train, {val_count} val)\n"
            f"Categories: {category_text}"
        )
        self.lbl_summary.setWordWrap(True)
        summary_layout.addWidget(self.lbl_summary)
        layout.addWidget(summary)

        paths_group = QGroupBox("Paths")
        paths_form = QFormLayout(paths_group)
        self.edit_run_name = QLineEdit(self._defaults.run_name)
        paths_form.addRow("Run name", self.edit_run_name)
        self.edit_output_dir = self._browse_row(
            self._defaults.output_dir,
            self._browse_output_dir,
        )
        paths_form.addRow("Output dir", self.edit_output_dir.parentWidget())
        self.edit_checkpoint = self._browse_row(
            self._defaults.checkpoint_path,
            self._browse_checkpoint,
        )
        paths_form.addRow("Checkpoint", self.edit_checkpoint.parentWidget())
        self.edit_bpe = self._browse_row(
            self._defaults.bpe_path,
            self._browse_bpe_path,
        )
        paths_form.addRow("BPE vocab", self.edit_bpe.parentWidget())
        layout.addWidget(paths_group)

        train_group = QGroupBox("Training")
        train_form = QFormLayout(train_group)
        self.spin_epochs = self._make_spinbox(1, 100000, self._defaults.max_epochs)
        train_form.addRow("Epochs", self.spin_epochs)
        self.spin_train_batch = self._make_spinbox(1, 256, self._defaults.train_batch_size)
        train_form.addRow("Train batch", self.spin_train_batch)
        self.spin_val_batch = self._make_spinbox(1, 256, self._defaults.val_batch_size)
        train_form.addRow("Val batch", self.spin_val_batch)
        self.spin_num_workers = self._make_spinbox(0, 64, self._defaults.num_workers)
        train_form.addRow("Workers", self.spin_num_workers)
        self.spin_num_gpus = self._make_spinbox(
            1,
            max_supported_local_gpus(),
            self._defaults.num_gpus,
        )
        if self.spin_num_gpus.maximum() == 1:
            self.spin_num_gpus.setToolTip(
                "Single-GPU mode is used for local SAM3 fine-tuning in this setup."
            )
        train_form.addRow("GPUs", self.spin_num_gpus)
        self.spin_grad_accum = self._make_spinbox(
            1,
            64,
            self._defaults.gradient_accumulation_steps,
        )
        train_form.addRow("Grad accum", self.spin_grad_accum)
        layout.addWidget(train_group)

        data_group = QGroupBox("Data & Optimization")
        data_form = QFormLayout(data_group)
        self.spin_resolution = self._make_spinbox(256, 4096, self._defaults.resolution)
        self.spin_resolution.setSingleStep(32)
        data_form.addRow("Resolution", self.spin_resolution)
        self.spin_limit_train = self._make_spinbox(0, 1000000, self._defaults.limit_train_images)
        data_form.addRow("Train image limit", self.spin_limit_train)
        self.spin_val_freq = self._make_spinbox(1, 100000, self._defaults.val_epoch_freq)
        data_form.addRow("Val every", self.spin_val_freq)
        self.spin_lr_scale = QDoubleSpinBox()
        self.spin_lr_scale.setDecimals(4)
        self.spin_lr_scale.setRange(0.0001, 100.0)
        self.spin_lr_scale.setSingleStep(0.05)
        self.spin_lr_scale.setValue(float(self._defaults.learning_rate_scale))
        data_form.addRow("LR scale", self.spin_lr_scale)
        layout.addWidget(data_group)

        hint = QLabel(
            "The app will generate SAM3 COCO annotations from the current dataset and "
            "start local training in a background process."
        )
        if self.spin_num_gpus.maximum() == 1:
            hint.setText(
                hint.text()
                + "\n\nNote: this environment runs SAM3 fine-tuning in single-GPU mode."
            )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_row(self, value: str, callback) -> QLineEdit:
        container = QWidget(self)
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        edit = QLineEdit(value)
        row.addWidget(edit)
        button = QPushButton("Browse...")
        button.setObjectName("secondary_button")
        button.clicked.connect(callback)
        row.addWidget(button)
        edit.setParent(container)
        return edit

    @staticmethod
    def _make_spinbox(minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _browse_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Training Output Directory",
            self.edit_output_dir.text().strip() or self._dataset_dir,
        )
        if folder:
            self.edit_output_dir.setText(folder)

    def _browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SAM3 Checkpoint",
            str(Path(self.edit_checkpoint.text().strip() or self._dataset_dir).parent),
            "Model files (*.pt *.pth *.safetensors);;All files (*.*)",
        )
        if path:
            self.edit_checkpoint.setText(path)

    def _browse_bpe_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select BPE Vocabulary",
            str(Path(self.edit_bpe.text().strip() or self._dataset_dir).parent),
            "Gzip files (*.gz);;All files (*.*)",
        )
        if path:
            self.edit_bpe.setText(path)

    def get_params(self) -> Sam3FineTuneParams:
        return Sam3FineTuneParams(
            run_name=self.edit_run_name.text().strip(),
            output_dir=self.edit_output_dir.text().strip() or self._defaults.output_dir,
            checkpoint_path=self.edit_checkpoint.text().strip(),
            bpe_path=self.edit_bpe.text().strip(),
            max_epochs=self.spin_epochs.value(),
            train_batch_size=self.spin_train_batch.value(),
            val_batch_size=self.spin_val_batch.value(),
            num_workers=self.spin_num_workers.value(),
            num_gpus=self.spin_num_gpus.value(),
            learning_rate_scale=self.spin_lr_scale.value(),
            resolution=self.spin_resolution.value(),
            limit_train_images=self.spin_limit_train.value(),
            val_epoch_freq=self.spin_val_freq.value(),
            gradient_accumulation_steps=self.spin_grad_accum.value(),
        ).normalized()
