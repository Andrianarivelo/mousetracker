"""YOLO instance-segmentation dataset manager for MouseTracker Pro."""

import json
import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def build_dataset(
    dataset_dir: str,
    class_names: Optional[dict[int, str]] = None,
    sam3_class_names: Optional[dict[int, str]] = None,
) -> str:
    """
    Create the YOLO-ready directory structure and config files.

    Args:
        dataset_dir: Root directory for the dataset.
        class_names: {class_index: name} mapping. Defaults to {0: "mouse"}.

    Returns:
        Path to dataset.yaml.
    """
    root = Path(dataset_dir)
    for sub in (
        "images/train", "images/val",
        "labels/train", "labels/val",
        "metadata",
    ):
        (root / sub).mkdir(parents=True, exist_ok=True)

    class_names = _normalize_yolo_class_names_input(class_names)

    # dataset.yaml
    yaml_path = root / "dataset.yaml"
    names_list = [class_names.get(i, f"class_{i}") for i in range(max(class_names.keys()) + 1)]
    yaml_content = (
        f"path: {root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"names:\n"
    )
    for i, name in enumerate(names_list):
        yaml_content += f"  {i}: {name}\n"

    _atomic_write(yaml_path, yaml_content)

    # metadata/index.json
    index_path = root / "metadata" / "index.json"
    if not index_path.exists():
        _atomic_write_json(index_path, {"videos": {}})

    sam3_names = _resolve_sam3_class_names(class_names, explicit=sam3_class_names)
    _write_sam3_category_map(root, sam3_names)
    for split in ("train", "val"):
        _atomic_write_json(
            _sam3_annotations_path(root, split),
            _empty_sam3_coco(root, split, sam3_names),
        )

    logger.info("Dataset built at %s (%d classes)", dataset_dir, len(names_list))
    return str(yaml_path)


def add_to_dataset(
    dataset_dir: str,
    video_path: str,
    annotated_frames: dict[int, dict[int, np.ndarray]],
    video_reader,
    class_names: Optional[dict[int, str]] = None,
    split: str = "train",
    sam3_class_names: Optional[dict[int, str]] = None,
) -> dict:
    """
    Export annotated frames to the YOLO dataset.

    Args:
        dataset_dir:      Root dataset directory.
        video_path:       Path to the source video.
        annotated_frames: {frame_idx: {mouse_id: bool_mask}} — the annotation data.
        video_reader:     VideoReader instance for frame extraction.
        class_names:      {mouse_id: name} for dataset.yaml names field.
        split:            "train" or "val".

    Returns:
        Summary dict: {"added": int, "skipped": int, "total_dataset": int}.
    """
    root = Path(dataset_dir)
    if not (root / "dataset.yaml").exists():
        build_dataset(
            dataset_dir,
            _mouse_ids_to_class_names(class_names),
            sam3_class_names=sam3_class_names,
        )

    # Update dataset.yaml names if class_names changed
    if class_names:
        _update_yaml_names(root, _mouse_ids_to_class_names(class_names))
    sam3_names = _resolve_sam3_class_names(
        _mouse_ids_to_class_names(class_names),
        explicit=sam3_class_names,
    )
    _write_sam3_category_map(root, sam3_names)

    index_path = root / "metadata" / "index.json"
    index = _load_json(index_path) if index_path.exists() else {"videos": {}}

    video_name = Path(video_path).name
    video_entry = index["videos"].get(video_name, {"frames": [], "files": []})
    existing_frames = set(video_entry["frames"])

    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(video_path).stem
    added = 0
    skipped = 0

    for frame_idx in sorted(annotated_frames.keys()):
        if frame_idx in existing_frames:
            skipped += 1
            continue

        masks = annotated_frames[frame_idx]
        if not masks:
            continue

        # Extract frame (VideoReader.read_frame returns RGB)
        frame_rgb = video_reader.read_frame(frame_idx)
        if frame_rgb is None:
            logger.warning("Cannot read frame %d — skipping", frame_idx)
            continue
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        h, w = frame_bgr.shape[:2]

        # File names
        fname = f"{stem}_f{frame_idx:06d}"
        img_file = img_dir / f"{fname}.jpg"
        lbl_file = lbl_dir / f"{fname}.txt"

        # Save image
        cv2.imwrite(str(img_file), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Build label lines
        label_lines = []
        for mouse_id, mask in masks.items():
            # mouse_id is 1-based, class_index is 0-based
            class_idx = mouse_id - 1
            polygons = _mask_to_yolo_polygons(mask, w, h)
            for poly in polygons:
                coords_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
                label_lines.append(f"{class_idx} {coords_str}")

        _atomic_write(lbl_file, "\n".join(label_lines) + "\n" if label_lines else "")

        video_entry["frames"].append(frame_idx)
        video_entry["files"].append({
            "image": str(img_file.relative_to(root)),
            "label": str(lbl_file.relative_to(root)),
        })
        added += 1

    # Keep frames and files lists in sync — deduplicate and sort together
    seen: dict[int, dict] = {}
    for fidx, finfo in zip(video_entry["frames"], video_entry["files"]):
        seen[fidx] = finfo  # latest entry wins if duplicated
    sorted_pairs = sorted(seen.items())
    video_entry["frames"] = [f for f, _ in sorted_pairs]
    video_entry["files"] = [fi for _, fi in sorted_pairs]

    video_entry["last_updated"] = datetime.now().isoformat(timespec="seconds")
    index["videos"][video_name] = video_entry
    _atomic_write_json(index_path, index)
    _rebuild_sam3_annotations(root, split, sam3_names)

    total = _count_images(root)
    logger.info(
        "add_to_dataset: added %d, skipped %d (already present), total %d",
        added, skipped, total,
    )
    return {"added": added, "skipped": skipped, "total_dataset": total}


def load_labels_from_dataset(
    dataset_dir: str,
    video_path: str,
    frame_shape: tuple[int, int],
) -> dict[int, dict[int, np.ndarray]]:
    """
    Load YOLO polygon labels back as pixel-space masks for the given video.

    Args:
        dataset_dir: Root dataset directory.
        video_path:  Path to the source video.
        frame_shape: (H, W) of the video frames.

    Returns:
        {frame_idx: {mouse_id: bool_mask (HxW)}} ready for the app's annotation state.
    """
    root = Path(dataset_dir)
    index_path = root / "metadata" / "index.json"
    if not index_path.exists():
        logger.warning("No index.json in %s", dataset_dir)
        return {}

    index = _load_json(index_path)
    video_name = Path(video_path).name
    video_entry = index.get("videos", {}).get(video_name)
    if video_entry is None:
        logger.info("Video '%s' not found in dataset index", video_name)
        return {}

    h, w = frame_shape
    result: dict[int, dict[int, np.ndarray]] = {}

    # Build robust lookup: frame_idx → file_info by parsing filename,
    # rather than relying on positional correspondence between lists.
    import re
    file_lookup: dict[int, dict] = {}
    for file_info in video_entry.get("files", []):
        lbl_stem = Path(file_info["label"]).stem  # e.g. "video_f000100"
        m = re.search(r"_f(\d+)$", lbl_stem)
        if m:
            file_lookup[int(m.group(1))] = file_info

    for frame_idx in video_entry.get("frames", []):
        file_info = file_lookup.get(frame_idx)
        if file_info is None:
            continue

        lbl_path = root / file_info["label"]
        if not lbl_path.exists():
            continue

        masks_for_frame: dict[int, np.ndarray] = {}
        text = lbl_path.read_text().strip()
        if not text:
            continue

        for line in text.split("\n"):
            parts = line.strip().split()
            if len(parts) < 7:  # class_idx + at least 3 points (6 coords)
                continue
            class_idx = int(parts[0])
            mouse_id = class_idx + 1  # back to 1-based

            coords = list(map(float, parts[1:]))
            points = []
            for j in range(0, len(coords) - 1, 2):
                px = coords[j] * w
                py = coords[j + 1] * h
                points.append([int(round(px)), int(round(py))])

            if len(points) < 3:
                continue

            poly = np.array(points, dtype=np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 1)

            if mouse_id in masks_for_frame:
                # Multiple polygons for same class: union
                masks_for_frame[mouse_id] = masks_for_frame[mouse_id] | (mask > 0)
            else:
                masks_for_frame[mouse_id] = mask > 0

        if masks_for_frame:
            result[frame_idx] = masks_for_frame

    logger.info(
        "Loaded labels for '%s': %d frames from %s",
        video_name, len(result), dataset_dir,
    )
    return result


def get_dataset_stats(dataset_dir: str, video_path: Optional[str] = None) -> dict:
    """
    Return summary statistics for the dataset.

    Returns:
        {"total_images": int, "video_frames": int, "last_updated": str}
    """
    root = Path(dataset_dir)
    total = _count_images(root)

    video_frames = 0
    last_updated = ""
    if video_path:
        index_path = root / "metadata" / "index.json"
        if index_path.exists():
            index = _load_json(index_path)
            video_name = Path(video_path).name
            entry = index.get("videos", {}).get(video_name, {})
            video_frames = len(entry.get("frames", []))
            last_updated = entry.get("last_updated", "")

    return {
        "total_images": total,
        "train_images": _count_split_images(root, "train"),
        "val_images": _count_split_images(root, "val"),
        "video_frames": video_frames,
        "last_updated": last_updated,
    }


def ensure_sam3_training_dataset(
    dataset_dir: str,
    sam3_class_names: Optional[dict[int, str]] = None,
) -> dict:
    """
    Ensure the dataset has SAM3-compatible COCO annotation files for train/val.

    Returns:
        {
            "train_images_dir": str,
            "train_annotations": str,
            "train_count": int,
            "val_images_dir": str,
            "val_annotations": str,
            "val_count": int,
            "categories": list[str],
        }
    """
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    class_names = _load_yaml_names(root) or {0: "mouse"}
    sam3_names = _resolve_sam3_class_names(class_names, explicit=sam3_class_names)
    _write_sam3_category_map(root, sam3_names)

    split_info: dict[str, dict] = {}
    for split in ("train", "val"):
        ann_path = _rebuild_sam3_annotations(root, split, sam3_names)
        split_info[split] = {
            "images_dir": str((root / "images" / split).resolve()),
            "annotations": str(ann_path.resolve()),
            "count": _count_split_images(root, split),
        }

    categories = sorted({name for name in sam3_names.values() if name})
    return {
        "train_images_dir": split_info["train"]["images_dir"],
        "train_annotations": split_info["train"]["annotations"],
        "train_count": split_info["train"]["count"],
        "val_images_dir": split_info["val"]["images_dir"],
        "val_annotations": split_info["val"]["annotations"],
        "val_count": split_info["val"]["count"],
        "categories": categories,
    }


# ── Internal helpers ─────────────────────────────────────────────────────────


def _mask_to_yolo_polygons(
    mask: np.ndarray,
    img_w: int,
    img_h: int,
    min_area: int = 100,
) -> list[list[tuple[float, float]]]:
    """
    Convert a boolean mask to normalized YOLO polygon(s).

    Returns list of polygons, each polygon = [(x_norm, y_norm), ...].
    """
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Simplify
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.005 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 3:
            continue

        # Normalize coordinates
        points = []
        for pt in approx:
            x_norm = float(pt[0][0]) / img_w
            y_norm = float(pt[0][1]) / img_h
            # Clamp to [0, 1]
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))
            points.append((x_norm, y_norm))

        polygons.append(points)

    return polygons


def _mask_to_coco_polygons(mask: np.ndarray, min_area: int = 10) -> list[list[float]]:
    """Convert a boolean mask to COCO-style polygon coordinates in pixels."""
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons: list[list[float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        epsilon = max(1.0, 0.005 * perimeter)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue

        flat: list[float] = []
        for pt in approx[:, 0, :]:
            flat.extend([float(pt[0]), float(pt[1])])
        if len(flat) >= 6:
            polygons.append(flat)

    return polygons


def _mouse_ids_to_class_names(
    names: Optional[dict[int, str]],
) -> dict[int, str]:
    """Convert {mouse_id (1-based): name} → {class_index (0-based): name}."""
    if not names:
        return {0: "mouse"}
    return {mid - 1: name for mid, name in names.items() if mid >= 1}


def _normalize_yolo_class_names_input(
    names: Optional[dict[int, str]],
) -> dict[int, str]:
    if not names:
        return {0: "mouse"}
    if 0 in names:
        return {int(idx): str(name) for idx, name in names.items() if int(idx) >= 0}
    return _mouse_ids_to_class_names(names)


def _normalize_sam3_category_name(name: str) -> str:
    text = re.sub(r"[_\-]+", " ", str(name).strip().lower())
    if any(token in text for token in ("mouse", "mice", "rodent")):
        return "mouse"
    if "object" in text:
        return "object"
    text = re.sub(r"\b\d+\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "object"


def _resolve_sam3_class_names(
    yolo_class_names: Optional[dict[int, str]],
    explicit: Optional[dict[int, str]] = None,
) -> dict[int, str]:
    source = explicit if explicit else yolo_class_names
    if not source:
        return {0: "mouse"}
    return {
        int(class_idx): _normalize_sam3_category_name(name)
        for class_idx, name in source.items()
    }


def _update_yaml_names(root: Path, class_names: dict[int, str]) -> None:
    """Rewrite the names section in dataset.yaml."""
    yaml_path = root / "dataset.yaml"
    if not yaml_path.exists():
        return

    names_list = [class_names.get(i, f"class_{i}") for i in range(max(class_names.keys()) + 1)]

    lines = yaml_path.read_text().splitlines()
    # Remove old names section
    new_lines = []
    in_names = False
    for line in lines:
        if line.startswith("names:"):
            in_names = True
            continue
        if in_names and (line.startswith("  ") or line.strip() == ""):
            continue
        in_names = False
        new_lines.append(line)

    # Append new names
    new_lines.append("")
    new_lines.append("names:")
    for i, name in enumerate(names_list):
        new_lines.append(f"  {i}: {name}")

    _atomic_write(yaml_path, "\n".join(new_lines) + "\n")


def _load_yaml_names(root: Path) -> dict[int, str]:
    yaml_path = root / "dataset.yaml"
    if not yaml_path.exists():
        return {}

    names: dict[int, str] = {}
    in_names = False
    for line in yaml_path.read_text().splitlines():
        if line.startswith("names:"):
            in_names = True
            continue
        if in_names:
            match = re.match(r"^\s*(\d+)\s*:\s*(.+?)\s*$", line)
            if match:
                names[int(match.group(1))] = match.group(2)
            elif line.strip():
                break
    return names


def _count_images(root: Path) -> int:
    """Count total .jpg images across train and val."""
    total = 0
    for split in ("train", "val"):
        total += _count_split_images(root, split)
    return total


def _count_split_images(root: Path, split: str) -> int:
    img_dir = root / "images" / split
    if not img_dir.exists():
        return 0
    return sum(1 for f in img_dir.iterdir() if f.suffix.lower() == ".jpg")


def _sam3_category_map_path(root: Path) -> Path:
    return root / "metadata" / "sam3_category_map.json"


def _sam3_annotations_path(root: Path, split: str) -> Path:
    return root / "metadata" / f"sam3_{split}_annotations.coco.json"


def _write_sam3_category_map(root: Path, class_names: dict[int, str]) -> None:
    payload = {str(int(idx)): str(name) for idx, name in class_names.items()}
    _atomic_write_json(_sam3_category_map_path(root), payload)


def _load_sam3_category_map(root: Path) -> dict[int, str]:
    path = _sam3_category_map_path(root)
    if not path.exists():
        return {}
    raw = _load_json(path)
    return {int(idx): str(name) for idx, name in raw.items()}


def _empty_sam3_coco(root: Path, split: str, class_names: dict[int, str]) -> dict:
    categories = _build_sam3_categories(class_names)
    return {
        "info": {
            "description": f"MouseTracker SAM3 fine-tune {split} split",
            "version": "1.0",
        },
        "images": [],
        "annotations": [],
        "categories": categories,
    }


def _build_sam3_categories(class_names: dict[int, str]) -> list[dict]:
    unique_names: list[str] = []
    for _, name in sorted(class_names.items()):
        if name not in unique_names:
            unique_names.append(name)
    if not unique_names:
        unique_names = ["mouse"]
    return [
        {"id": idx + 1, "name": name}
        for idx, name in enumerate(unique_names)
    ]


def _rebuild_sam3_annotations(
    root: Path,
    split: str,
    class_names: Optional[dict[int, str]] = None,
) -> Path:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    ann_path = _sam3_annotations_path(root, split)

    resolved_names = (
        class_names
        or _load_sam3_category_map(root)
        or _resolve_sam3_class_names(_load_yaml_names(root))
    )
    categories = _build_sam3_categories(resolved_names)
    category_id_by_name = {
        item["name"]: int(item["id"]) for item in categories
    }

    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 1

    if img_dir.exists():
        image_files = sorted(
            [
                path for path in img_dir.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )
    else:
        image_files = []

    for image_id, img_path in enumerate(image_files, start=1):
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Cannot read dataset image %s while rebuilding SAM3 annotations", img_path)
            continue
        height, width = image.shape[:2]
        images.append(
            {
                "id": image_id,
                "file_name": img_path.name,
                "width": width,
                "height": height,
            }
        )

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        text = lbl_path.read_text().strip()
        if not text:
            continue

        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            class_idx = int(parts[0])
            semantic_name = resolved_names.get(
                class_idx,
                _normalize_sam3_category_name(f"class_{class_idx}"),
            )
            category_id = category_id_by_name.setdefault(
                semantic_name,
                len(category_id_by_name) + 1,
            )
            coords = list(map(float, parts[1:]))
            if len(coords) < 6 or len(coords) % 2 != 0:
                continue

            abs_points = np.array(
                [
                    [coords[idx] * width, coords[idx + 1] * height]
                    for idx in range(0, len(coords), 2)
                ],
                dtype=np.float32,
            )
            if len(abs_points) < 3:
                continue

            xs = abs_points[:, 0]
            ys = abs_points[:, 1]
            bbox_x = float(xs.min())
            bbox_y = float(ys.min())
            bbox_w = float(xs.max() - bbox_x)
            bbox_h = float(ys.max() - bbox_y)
            if bbox_w <= 0 or bbox_h <= 0:
                continue

            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [np.round(abs_points).astype(np.int32)], 1)
            segmentation = _mask_to_coco_polygons(mask > 0)
            if not segmentation:
                continue

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(category_id),
                    "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
                    "area": float(mask.sum()),
                    "segmentation": segmentation,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    categories = [
        {"id": int(cat_id), "name": name}
        for name, cat_id in sorted(category_id_by_name.items(), key=lambda item: item[1])
    ]
    payload = {
        "info": {
            "description": f"MouseTracker SAM3 fine-tune {split} split",
            "version": "1.0",
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    _atomic_write_json(ann_path, payload)
    return ann_path


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _atomic_write(path: Path, content: str) -> None:
    """Write content to a temp file then rename for atomicity."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        # On Windows, must remove target first
        if path.exists():
            path.unlink()
        os.rename(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_json(path: Path, data: dict) -> None:
    _atomic_write(path, json.dumps(data, indent=2) + "\n")
