"""Tests for SAM3 fine-tune config generation."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf


class MockVideoReader:
    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width

    def read_frame(self, idx: int):
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)


def _circle_mask(height: int, width: int, center: tuple[int, int]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, 35, 1, -1)
    return mask.astype(bool)


def test_prepare_finetune_job_generates_training_config():
    from app.core.dataset_manager import add_to_dataset, build_dataset
    from app.core.sam3_finetune import (
        Sam3FineTuneParams,
        default_amp_dtype,
        default_distributed_backend,
        prepare_finetune_job,
    )

    height = 180
    width = 240
    video_reader = MockVideoReader(height, width)

    with tempfile.TemporaryDirectory() as tmp:
        dataset_dir = Path(tmp)
        build_dataset(str(dataset_dir), {1: "Mouse 1"})

        train_masks = {0: {1: _circle_mask(height, width, (80, 90))}}
        val_masks = {1: {1: _circle_mask(height, width, (150, 90))}}
        add_to_dataset(
            dataset_dir=str(dataset_dir),
            video_path="session.mp4",
            annotated_frames=train_masks,
            video_reader=video_reader,
            class_names={1: "Mouse 1"},
            split="train",
        )
        add_to_dataset(
            dataset_dir=str(dataset_dir),
            video_path="session.mp4",
            annotated_frames=val_masks,
            video_reader=video_reader,
            class_names={1: "Mouse 1"},
            split="val",
        )

        params = Sam3FineTuneParams(
            run_name="trial",
            output_dir=str(dataset_dir / "runs" / "trial"),
            checkpoint_path="",
            bpe_path="",
            max_epochs=2,
            train_batch_size=1,
            val_batch_size=1,
            num_workers=0,
            num_gpus=1,
            learning_rate_scale=0.1,
            resolution=768,
            limit_train_images=0,
            val_epoch_freq=1,
            gradient_accumulation_steps=1,
        )
        job = prepare_finetune_job(
            str(dataset_dir),
            params,
            sam3_class_names={1: "Mouse 1"},
        )

        config_path = Path(job["config_path"])
        assert config_path.exists()
        assert job["dataset_info"]["train_count"] == 1
        assert job["dataset_info"]["val_count"] == 1
        assert job["dataset_info"]["categories"] == ["mouse"]

        cfg = OmegaConf.load(str(config_path))
        assert cfg.trainer.distributed.backend == default_distributed_backend()
        accelerator = "cuda" if cfg.trainer.accelerator == "cuda" else "cpu"
        assert cfg.trainer.optim.amp.amp_dtype == default_amp_dtype(accelerator)
        assert cfg.trainer.mode == "train"
        assert cfg.trainer.model.enable_segmentation is True
        assert cfg.trainer.data.train.dataset.ann_file == job["dataset_info"]["train_annotations"]
        assert cfg.trainer.data.val.dataset.ann_file == job["dataset_info"]["val_annotations"]
