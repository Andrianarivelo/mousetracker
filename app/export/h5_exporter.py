"""HDF5 export: per-frame binary masks + metadata."""

import logging
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


def export_h5(
    tracker_history,
    output_path: str,
    frame_shape: tuple[int, int],
    fps: float,
    video_path: str = "",
    keypoints_by_frame: Optional[dict] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> str:
    """
    Export masks to HDF5.

    Dataset "masks": shape (n_frames, H, W) dtype uint8
        Pixel value = mouse_id (0 = background)

    Dataset "keypoints": shape (n_frames, n_mice, n_keypoints, 2)

    Attributes: fps, video_path, frame_count, height, width
    """
    import h5py

    h, w = frame_shape
    n_frames = len(tracker_history)

    with h5py.File(output_path, "w") as f:
        # Masks dataset
        masks_ds = f.create_dataset(
            "masks",
            shape=(n_frames, h, w),
            dtype=np.uint8,
            compression="gzip",
            compression_opts=6,
        )

        # Frame index dataset
        frame_indices = f.create_dataset("frame_indices", shape=(n_frames,), dtype=np.int32)

        for i, state in enumerate(tracker_history):
            combined = np.zeros((h, w), dtype=np.uint8)
            for mouse_id, mask in state.masks.items():
                if mask.shape == (h, w):
                    combined[mask] = np.uint8(mouse_id)
                elif mask.shape[:2] == (h, w):
                    combined[mask[:h, :w].astype(bool)] = np.uint8(mouse_id)
            masks_ds[i] = combined
            frame_indices[i] = state.frame_idx

            if progress_callback and n_frames > 0:
                progress_callback(int((i + 1) * 70 / n_frames))

        # Centroid trajectories
        mouse_ids_all = set()
        for state in tracker_history:
            mouse_ids_all.update(state.centroids.keys())
        mouse_ids_sorted = sorted(mouse_ids_all)

        if mouse_ids_sorted:
            traj_ds = f.create_dataset(
                "centroids",
                shape=(n_frames, len(mouse_ids_sorted), 2),
                dtype=np.float32,
            )
            mid_idx = {mid: i for i, mid in enumerate(mouse_ids_sorted)}
            for i, state in enumerate(tracker_history):
                for mid, (cx, cy) in state.centroids.items():
                    traj_ds[i, mid_idx[mid]] = [cx, cy]

        # Keypoints
        if keypoints_by_frame:
            kp_names = set()
            for fkps in keypoints_by_frame.values():
                for mkps in fkps.values():
                    kp_names.update(mkps.keys())
            kp_names_sorted = sorted(kp_names)
            if kp_names_sorted and mouse_ids_sorted:
                kp_ds = f.create_dataset(
                    "keypoints",
                    shape=(n_frames, len(mouse_ids_sorted), len(kp_names_sorted), 2),
                    dtype=np.float32,
                )
                kp_ds.attrs["keypoint_names"] = kp_names_sorted
                for i, state in enumerate(tracker_history):
                    fkps = keypoints_by_frame.get(state.frame_idx, {})
                    for mid, mkps in fkps.items():
                        if mid in mid_idx:
                            for ki, kn in enumerate(kp_names_sorted):
                                if kn in mkps:
                                    kp_ds[i, mid_idx[mid], ki] = list(mkps[kn])

        # Attributes
        f.attrs["fps"] = fps
        f.attrs["video_path"] = video_path
        f.attrs["frame_count"] = n_frames
        f.attrs["height"] = h
        f.attrs["width"] = w
        f.attrs["mouse_ids"] = mouse_ids_sorted

    if progress_callback:
        progress_callback(100)

    logger.info(f"HDF5 exported: {output_path}")
    return output_path
