"""
Mask-to-keypoint estimation using skeletonization and geometric analysis.
No additional neural network required.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class KeypointEstimator:
    """
    Deduce anatomical keypoints from a binary mouse mask.

    Strategy:
      1. Skeletonize the mask (medial axis / thinning)
      2. Find skeleton endpoints
      3. Order skeleton as a spine path from nose to tail
      4. Derive keypoints from proportional positions along skeleton
      5. Ears = local curvature maxima on contour near head region
    """

    AVAILABLE_KEYPOINTS = [
        "nose_tip",
        "head_center",
        "neck",
        "body_center",
        "left_hip",
        "right_hip",
        "hip_center",
        "tail_base",
        "tail_mid",
        "tail_tip",
        "left_ear",
        "right_ear",
    ]

    def __init__(self, selected_keypoints: Optional[list[str]] = None) -> None:
        self.selected = selected_keypoints or ["nose_tip", "body_center", "tail_base"]

    def estimate(self, mask: np.ndarray) -> dict[str, tuple[float, float]]:
        """
        Return {keypoint_name: (x, y)} for selected keypoints.
        Returns empty dict if mask is too small or degenerate.
        """
        if mask is None or mask.sum() < 9:
            return {}

        try:
            skeleton = self._skeletonize(mask)
            endpoints = self._find_endpoints(skeleton)
            spine = self._order_spine(skeleton, endpoints, mask)

            if len(spine) < 3:
                # Fallback: just return centroid
                cx, cy = _centroid(mask)
                return {k: (cx, cy) for k in self.selected}

            return self._compute_keypoints(mask, skeleton, spine)
        except Exception as e:
            logger.debug(f"Keypoint estimation failed: {e}")
            cx, cy = _centroid(mask)
            return {"body_center": (cx, cy)} if "body_center" in self.selected else {}

    # ── Core computation ───────────────────────────────────────────────────────

    def _compute_keypoints(
        self,
        mask: np.ndarray,
        skeleton: np.ndarray,
        spine: np.ndarray,  # (N, 2) ordered [nose→tail] in (y, x)
    ) -> dict[str, tuple[float, float]]:
        n = len(spine)
        result: dict[str, tuple[float, float]] = {}

        def at(frac: float) -> tuple[float, float]:
            """Return (x, y) at fractional position along spine."""
            idx = min(int(frac * n), n - 1)
            y, x = spine[idx]
            return (float(x), float(y))

        def width_at_index(i: int) -> float:
            """Estimate local mask width perpendicular to spine at index i."""
            y, x = spine[i]
            # Simple: measure mask extent in orthogonal directions
            col = mask[:, x] if 0 <= x < mask.shape[1] else np.array([False])
            row = mask[y, :] if 0 <= y < mask.shape[0] else np.array([False])
            return float(min(col.sum(), row.sum()) + 1)

        # Determine nose vs tail endpoint by local width.
        # Mouse anatomy: the head end is WIDER than the tail tip.
        # Nose is at the tip of the head (wide end), tail tip is the
        # thinnest extremity.  We want spine[0] = nose (head side).
        w_start = width_at_index(0)
        w_end = width_at_index(n - 1)
        # Also check slightly inward (10-20% from each end) to avoid
        # noise at the very tip.
        w_near_start = width_at_index(min(n - 1, max(1, int(0.15 * n))))
        w_near_end = width_at_index(max(0, int(0.85 * n)))
        head_score_start = w_start + w_near_start
        head_score_end = w_end + w_near_end
        if head_score_start < head_score_end:
            # spine[0] is the thin (tail) end — reverse so nose comes first
            spine = spine[::-1]

        # --- Keypoints by fractional spine position ---
        # Spine runs nose (0.0) → tail tip (1.0).
        # Proportions tuned for typical mouse body plan:
        #   head ~20%, body ~45%, tail ~35% of skeleton length.
        if "nose_tip" in self.selected:
            result["nose_tip"] = at(0.0)
        if "head_center" in self.selected:
            result["head_center"] = at(0.10)
        if "neck" in self.selected:
            result["neck"] = at(0.20)
        if "body_center" in self.selected:
            result["body_center"] = at(0.40)
        if "hip_center" in self.selected:
            result["hip_center"] = at(0.58)
        if "tail_base" in self.selected:
            result["tail_base"] = at(0.65)
        if "tail_mid" in self.selected:
            result["tail_mid"] = at(0.82)
        if "tail_tip" in self.selected:
            result["tail_tip"] = at(1.0)

        # Hip offsets
        if "left_hip" in self.selected or "right_hip" in self.selected:
            hc = at(0.58)
            # Perpendicular offset based on local mask width
            idx_hip = min(int(0.58 * n), n - 1)
            offset = width_at_index(idx_hip) * 0.35
            # Spine direction at hip
            i0 = max(0, idx_hip - 2)
            i1 = min(n - 1, idx_hip + 2)
            dy = float(spine[i1][0] - spine[i0][0])
            dx = float(spine[i1][1] - spine[i0][1])
            length = np.sqrt(dx * dx + dy * dy) + 1e-6
            # Perpendicular: (-dy, dx) normalized
            px, py = -dy / length, dx / length
            if "left_hip" in self.selected:
                result["left_hip"] = (hc[0] + px * offset, hc[1] + py * offset)
            if "right_hip" in self.selected:
                result["right_hip"] = (hc[0] - px * offset, hc[1] - py * offset)

        # Ears — local curvature maxima on contour near head region
        if "left_ear" in self.selected or "right_ear" in self.selected:
            ears = self._find_ears(mask, at(0.10), at(0.20))
            if len(ears) >= 2:
                if "left_ear" in self.selected:
                    result["left_ear"] = ears[0]
                if "right_ear" in self.selected:
                    result["right_ear"] = ears[1]
            elif len(ears) == 1:
                if "left_ear" in self.selected:
                    result["left_ear"] = ears[0]
                if "right_ear" in self.selected:
                    result["right_ear"] = ears[0]

        return result

    # ── Skeleton utilities ─────────────────────────────────────────────────────

    def _skeletonize(self, mask: np.ndarray) -> np.ndarray:
        from skimage.morphology import skeletonize
        return skeletonize(mask > 0)

    def _find_endpoints(self, skeleton: np.ndarray) -> list[tuple[int, int]]:
        """Find skeleton endpoints (pixels with exactly 1 neighbor)."""
        if not skeleton.any():
            return []
        endpoints = []
        ys, xs = np.where(skeleton)
        for y, x in zip(ys, xs):
            neighbors = int(skeleton[
                max(0, y-1):y+2, max(0, x-1):x+2
            ].sum()) - 1  # subtract self
            if neighbors == 1:
                endpoints.append((y, x))
        return endpoints

    def _order_spine(
        self,
        skeleton: np.ndarray,
        endpoints: list[tuple[int, int]],
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Order skeleton pixels as a path from one endpoint to another.
        Returns array of (y, x) in order.
        """
        if not skeleton.any():
            ys, xs = np.where(mask > 0)
            if len(ys) == 0:
                return np.array([[0, 0]])
            cy, cx = int(np.mean(ys)), int(np.mean(xs))
            return np.array([[cy, cx]])

        skel_pts = list(zip(*np.where(skeleton)))
        if len(skel_pts) == 0:
            cy, cx = _centroid(mask)
            return np.array([[int(cy), int(cx)]])

        if len(endpoints) < 2:
            # Closed loop or single point — just return all skeleton pts
            return np.array(skel_pts)

        # Pick the two most distant endpoints for a robust spine path
        if len(endpoints) > 2:
            best_dist = 0
            best_pair = (endpoints[0], endpoints[1])
            for i in range(len(endpoints)):
                for j in range(i + 1, len(endpoints)):
                    d = abs(endpoints[i][0] - endpoints[j][0]) + abs(endpoints[i][1] - endpoints[j][1])
                    if d > best_dist:
                        best_dist = d
                        best_pair = (endpoints[i], endpoints[j])
            endpoints = [best_pair[0], best_pair[1]]

        # BFS/greedy walk from first endpoint to second
        start = endpoints[0]
        visited = set()
        path = [start]
        visited.add(start)
        skel_set = set(skel_pts)

        current = start
        while True:
            y, x = current
            neighbors = [
                (y + dy, x + dx)
                for dy in [-1, 0, 1]
                for dx in [-1, 0, 1]
                if (dy, dx) != (0, 0)
                and (y + dy, x + dx) in skel_set
                and (y + dy, x + dx) not in visited
            ]
            if not neighbors:
                break
            # Prefer neighbors closest to the other endpoint
            goal = endpoints[1]
            neighbors.sort(key=lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1]))
            nxt = neighbors[0]
            path.append(nxt)
            visited.add(nxt)
            current = nxt

        return np.array(path)

    def _find_ears(
        self,
        mask: np.ndarray,
        head_pt: tuple[float, float],
        neck_pt: tuple[float, float],
    ) -> list[tuple[float, float]]:
        """Find ear candidates as contour curvature maxima near the head."""
        import cv2
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return []
        contour = max(contours, key=cv2.contourArea)
        pts = contour[:, 0, :]  # (N, 2) in x,y

        # Filter to head region
        hx, hy = head_pt
        nx, ny = neck_pt
        head_r = max(20, np.sqrt((hx - nx) ** 2 + (hy - ny) ** 2) * 1.5)
        near_head = [
            (float(p[0]), float(p[1]))
            for p in pts
            if np.sqrt((p[0] - hx) ** 2 + (p[1] - hy) ** 2) < head_r
        ]
        if len(near_head) < 6:
            return []

        # Compute curvature (cross-product of successive vectors)
        arr = np.array(near_head)
        k = 5
        curvatures = []
        for i in range(k, len(arr) - k):
            v1 = arr[i] - arr[i - k]
            v2 = arr[i + k] - arr[i]
            cross = float(np.abs(np.cross(v1, v2)))
            curvatures.append((cross, arr[i]))

        if not curvatures:
            return []
        curvatures.sort(key=lambda t: t[0], reverse=True)

        # Pick top-2 well-separated peaks
        ears: list[tuple[float, float]] = []
        for _, pt in curvatures:
            if not ears:
                ears.append((float(pt[0]), float(pt[1])))
            elif np.sqrt((pt[0] - ears[0][0]) ** 2 + (pt[1] - ears[0][1]) ** 2) > 15:
                ears.append((float(pt[0]), float(pt[1])))
                break

        return ears


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0.0, 0.0)
    return (float(np.mean(xs)), float(np.mean(ys)))


def estimate_all_frames(
    tracker_history,
    estimator: KeypointEstimator,
) -> dict[int, dict[int, dict[str, tuple[float, float]]]]:
    """
    Run keypoint estimation on all tracked frames.

    Returns:
        {frame_idx: {mouse_id: {kp_name: (x, y)}}}
    """
    result: dict[int, dict] = {}
    for state in tracker_history:
        frame_kps: dict[int, dict] = {}
        for mouse_id, mask in state.masks.items():
            kps = estimator.estimate(mask)
            if kps:
                frame_kps[mouse_id] = kps
        result[state.frame_idx] = frame_kps
    return result
