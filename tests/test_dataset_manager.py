"""Tests for the YOLO dataset manager."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_build_dataset():
    from app.core.dataset_manager import build_dataset

    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = build_dataset(tmp, {0: "mouse_1", 1: "mouse_2"})
        root = Path(tmp)

        assert (root / "images" / "train").is_dir()
        assert (root / "images" / "val").is_dir()
        assert (root / "labels" / "train").is_dir()
        assert (root / "labels" / "val").is_dir()
        assert (root / "metadata" / "index.json").exists()
        assert (root / "metadata" / "sam3_category_map.json").exists()
        assert (root / "metadata" / "sam3_train_annotations.coco.json").exists()
        assert (root / "metadata" / "sam3_val_annotations.coco.json").exists()
        assert Path(yaml_path).exists()

        yaml_text = Path(yaml_path).read_text()
        assert "mouse_1" in yaml_text
        assert "mouse_2" in yaml_text
        assert "train: images/train" in yaml_text

        category_map = json.loads((root / "metadata" / "sam3_category_map.json").read_text())
        assert category_map == {"0": "mouse", "1": "mouse"}

        train_annotations = json.loads(
            (root / "metadata" / "sam3_train_annotations.coco.json").read_text()
        )
        assert train_annotations["categories"] == [{"id": 1, "name": "mouse"}]


def test_mask_to_yolo_polygons():
    from app.core.dataset_manager import _mask_to_yolo_polygons

    # Create a simple rectangular mask
    mask = np.zeros((200, 300), dtype=bool)
    mask[50:150, 80:220] = True

    polygons = _mask_to_yolo_polygons(mask, 300, 200, min_area=50)
    assert len(polygons) >= 1
    # Check normalization: all coords should be in [0, 1]
    for poly in polygons:
        for x, y in poly:
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0


def test_add_and_load_roundtrip():
    """Test that add_to_dataset → load_labels_from_dataset preserves masks."""
    from app.core.dataset_manager import build_dataset, add_to_dataset, load_labels_from_dataset

    h, w = 200, 300

    # Create a mock video reader
    class MockVideoReader:
        def read_frame(self, idx):
            return np.zeros((h, w, 3), dtype=np.uint8)

    # Create a circular mask
    mask = np.zeros((h, w), dtype=bool)
    cv2 = __import__("cv2")
    cv2.circle(mask.view(np.uint8), (150, 100), 40, 1, -1)
    mask = mask.view(np.uint8).astype(bool)

    annotated = {0: {1: mask}}

    with tempfile.TemporaryDirectory() as tmp:
        build_dataset(tmp, {0: "mouse"})

        result = add_to_dataset(
            dataset_dir=tmp,
            video_path="test_video.mp4",
            annotated_frames=annotated,
            video_reader=MockVideoReader(),
            split="train",
        )
        assert result["added"] == 1
        assert result["skipped"] == 0

        # Second call should skip
        result2 = add_to_dataset(
            dataset_dir=tmp,
            video_path="test_video.mp4",
            annotated_frames=annotated,
            video_reader=MockVideoReader(),
            split="train",
        )
        assert result2["added"] == 0
        assert result2["skipped"] == 1

        # Load back
        loaded = load_labels_from_dataset(tmp, "test_video.mp4", (h, w))
        assert 0 in loaded
        assert 1 in loaded[0]
        loaded_mask = loaded[0][1]
        assert loaded_mask.shape == (h, w)
        # The loaded mask should overlap significantly with the original
        overlap = np.logical_and(mask, loaded_mask).sum()
        original_area = mask.sum()
        assert overlap / original_area > 0.85, "Roundtrip mask overlap too low"


def test_get_dataset_stats():
    from app.core.dataset_manager import build_dataset, get_dataset_stats

    with tempfile.TemporaryDirectory() as tmp:
        build_dataset(tmp)
        stats = get_dataset_stats(tmp)
        assert stats["total_images"] == 0
        assert stats["train_images"] == 0
        assert stats["val_images"] == 0
        assert stats["video_frames"] == 0
