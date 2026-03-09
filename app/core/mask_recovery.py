"""Size-aware mask validation, adaptive prompting, and watershed/CC recovery."""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _peak_marker_components(
    dist: np.ndarray,
    mask_u8: np.ndarray,
    expected_parts: Optional[int],
) -> list[np.ndarray]:
    """
    Extract connected peak regions from a distance transform.

    When ``expected_parts`` is provided, keep at most that many strongest peak
    regions. Otherwise return the strongest set that could plausibly split the
    mask.
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    best_components: list[np.ndarray] = []
    target_parts = expected_parts if expected_parts is not None and expected_parts > 1 else 2

    for peak_ratio in (0.45, 0.35, 0.25, 0.18, 0.12):
        peak_floor = float(dist.max()) * peak_ratio
        if peak_floor <= 0:
            continue

        local_max = (
            (dist >= cv2.dilate(dist, kernel))
            & (dist >= peak_floor)
            & (mask_u8 > 0)
        )
        peaks_u8 = local_max.astype(np.uint8)
        n_labels, labels = cv2.connectedComponents(peaks_u8)

        components: list[tuple[float, np.ndarray]] = []
        for label_val in range(1, n_labels):
            component = labels == label_val
            if not np.any(component):
                continue
            components.append((float(dist[component].max()), component))

        components.sort(key=lambda item: item[0], reverse=True)
        parts = [component for _, component in components]
        if len(parts) > target_parts:
            parts = parts[:target_parts]

        if len(parts) > len(best_components):
            best_components = parts

        if len(parts) >= target_parts:
            return parts

    return best_components


def _seed_points_from_mask(
    mask: np.ndarray,
    dist: np.ndarray,
    count: int,
) -> list[tuple[int, int]]:
    """
    Build fallback watershed seeds by sampling the mask along its principal axis.

    This is only used for explicit user-driven splits when the natural distance
    peaks are not distinct enough on their own.
    """
    ys, xs = np.where(mask)
    if len(xs) < count:
        return []

    coords = np.column_stack((ys, xs)).astype(np.float32)
    centered = coords - coords.mean(axis=0, keepdims=True)
    if len(coords) >= 2:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        projections = centered @ axis
    else:
        projections = coords[:, 1]

    seeds: list[tuple[int, int]] = []
    min_seed_dist = max(4.0, 0.10 * float(np.hypot(*mask.shape)))
    candidate_indices = np.arange(len(xs))
    high_dist = np.flatnonzero(dist[ys, xs] >= float(dist.max()) * 0.45)
    if high_dist.size >= count:
        candidate_indices = high_dist

    target_quantiles = np.linspace(0.25, 0.75, count)
    for quantile in target_quantiles:
        target_projection = float(np.quantile(projections, quantile))
        best_idx = min(
            candidate_indices,
            key=lambda i: (
                abs(float(projections[i]) - target_projection),
                -float(dist[ys[i], xs[i]]),
            ),
        )
        point = (int(ys[best_idx]), int(xs[best_idx]))
        if any(
            (point[0] - py) ** 2 + (point[1] - px) ** 2 < min_seed_dist ** 2
            for py, px in seeds
        ):
            continue
        seeds.append(point)

    if len(seeds) >= count:
        return seeds[:count]

    ranked = np.argsort(dist[ys, xs])[::-1]
    for index in ranked:
        point = (int(ys[index]), int(xs[index]))
        if any(
            (point[0] - py) ** 2 + (point[1] - px) ** 2 < min_seed_dist ** 2
            for py, px in seeds
        ):
            continue
        seeds.append(point)
        if len(seeds) >= count:
            break

    return seeds[:count]


def _markers_from_components(
    mask_u8: np.ndarray,
    components: list[np.ndarray],
) -> np.ndarray:
    markers = np.zeros(mask_u8.shape, dtype=np.int32)
    markers[mask_u8 == 0] = 1
    for label_val, component in enumerate(components, start=2):
        markers[component] = label_val
    return markers


def _markers_from_seed_points(
    mask_u8: np.ndarray,
    dist: np.ndarray,
    seed_points: list[tuple[int, int]],
) -> np.ndarray:
    markers = np.zeros(mask_u8.shape, dtype=np.int32)
    markers[mask_u8 == 0] = 1
    for label_val, (y, x) in enumerate(seed_points, start=2):
        radius = max(1, int(round(float(dist[y, x]) * 0.35)))
        cv2.circle(markers, (int(x), int(y)), radius, int(label_val), -1)
    markers[mask_u8 == 0] = 1
    return markers


def _run_marker_watershed(
    mask: np.ndarray,
    dist: np.ndarray,
    markers: np.ndarray,
) -> list[np.ndarray]:
    label_values = sorted(int(v) for v in np.unique(markers) if v > 1)
    if len(label_values) < 2:
        return []

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []

    seed_points: list[tuple[float, float]] = []
    for label_val in label_values:
        sy, sx = np.where(markers == label_val)
        if len(sx) == 0:
            continue
        seed_points.append((float(sy.mean()), float(sx.mean())))
    if len(seed_points) < 2:
        return []

    distances = []
    for sy, sx in seed_points:
        distances.append((ys - sy) ** 2 + (xs - sx) ** 2)
    assignments = np.argmin(np.stack(distances, axis=0), axis=0)

    result: list[np.ndarray] = []
    for label_idx in range(len(seed_points)):
        sub = np.zeros(mask.shape, dtype=bool)
        sub[ys, xs] = assignments == label_idx
        if int(sub.sum()) > 0:
            result.append(sub)
    return result


def _select_split_candidates(
    submasks: list[np.ndarray],
    mask_area: int,
    min_area: int,
    max_area: int,
    expected_parts: Optional[int],
    allow_relaxed_area: bool,
) -> list[np.ndarray]:
    valid = [sub for sub in submasks if min_area <= int(sub.sum()) <= max_area]
    if expected_parts is None:
        return valid

    if len(valid) >= expected_parts:
        valid.sort(key=lambda sub: int(sub.sum()), reverse=True)
        return valid[:expected_parts]

    if not allow_relaxed_area or len(submasks) < expected_parts:
        return []

    min_piece_area = max(1, int(mask_area / (expected_parts * 3)))
    relaxed = [sub for sub in submasks if int(sub.sum()) >= min_piece_area]
    if len(relaxed) >= expected_parts:
        relaxed.sort(key=lambda sub: int(sub.sum()), reverse=True)
        return relaxed[:expected_parts]
    return []


class SizeValidator:
    """
    Records reference mask areas per entity from the example/assignment phase
    and validates tracked masks against expected sizes.
    """

    def __init__(self, tolerance: float = 0.20) -> None:
        # {entity_id: reference_area_in_pixels}
        self._ref_areas: dict[int, int] = {}
        self._tolerance = tolerance  # +/-20% by default

    def record(self, entity_id: int, mask: np.ndarray) -> None:
        """Store reference area for an entity from the example mask."""
        area = int(mask.sum())
        if area > 0:
            self._ref_areas[entity_id] = area
            logger.debug(
                "SizeValidator: entity %d ref area = %d px", entity_id, area
            )

    def has_reference(self, entity_id: int) -> bool:
        return entity_id in self._ref_areas

    def get_range(self, entity_id: int) -> tuple[int, int]:
        """Return (min_area, max_area) for an entity."""
        ref = self._ref_areas.get(entity_id, 0)
        if ref <= 0:
            return 0, 0
        lo = int(ref * (1.0 - self._tolerance))
        hi = int(ref * (1.0 + self._tolerance))
        return lo, hi

    def validate(
        self, masks: dict[int, np.ndarray]
    ) -> tuple[dict[int, bool], dict[int, str]]:
        """
        Check each mask against its reference area.

        Returns:
            ok: {entity_id: True if within range}
            reasons: {entity_id: "too_large" | "too_small" | ""} for failed masks
        """
        ok: dict[int, bool] = {}
        reasons: dict[int, str] = {}
        for eid, mask in masks.items():
            if eid not in self._ref_areas:
                ok[eid] = True
                reasons[eid] = ""
                continue
            area = int(mask.sum())
            lo, hi = self.get_range(eid)
            if area < lo:
                ok[eid] = False
                reasons[eid] = "too_small"
            elif area > hi:
                ok[eid] = False
                reasons[eid] = "too_large"
            else:
                ok[eid] = True
                reasons[eid] = ""
        return ok, reasons

    def any_reference(self) -> bool:
        return len(self._ref_areas) > 0

    def reset(self) -> None:
        self._ref_areas.clear()

    @property
    def ref_areas(self) -> dict[int, int]:
        return dict(self._ref_areas)


def watershed_split(
    mask: np.ndarray,
    min_area: int,
    max_area: int,
    expected_parts: Optional[int] = None,
    allow_relaxed_area: bool = False,
) -> list[np.ndarray]:
    """
    Split an oversized binary mask using watershed on the distance transform.

    Returns a list of sub-masks whose areas fall within [min_area, max_area].
    If ``expected_parts`` is provided, try to return that many parts. If no
    valid split is found, returns an empty list.
    """
    mask = mask.astype(bool)
    mask_u8 = mask.astype(np.uint8) * 255
    if mask_u8.sum() == 0:
        return []

    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    if float(dist.max()) <= 0:
        return []

    mask_area = int(mask.sum())
    best_result: list[np.ndarray] = []
    peak_components = _peak_marker_components(dist, mask_u8, expected_parts)
    candidate_markers: list[tuple[str, np.ndarray, int]] = []

    if len(peak_components) >= 2:
        candidate_markers.append(
            (
                "peaks",
                _markers_from_components(mask_u8, peak_components),
                len(peak_components),
            )
        )

    if expected_parts is not None and expected_parts > 1 and len(peak_components) < expected_parts:
        seed_points = _seed_points_from_mask(mask, dist, expected_parts)
        if len(seed_points) >= expected_parts:
            candidate_markers.append(
                (
                    "forced",
                    _markers_from_seed_points(mask_u8, dist, seed_points),
                    len(seed_points),
                )
            )

    for mode, markers, marker_count in candidate_markers:
        submasks = _run_marker_watershed(mask, dist, markers)
        result = _select_split_candidates(
            submasks,
            mask_area,
            min_area,
            max_area,
            expected_parts,
            allow_relaxed_area,
        )
        if expected_parts is not None and len(result) >= expected_parts:
            logger.debug(
                "watershed_split[%s]: %d markers, %d valid sub-masks (range %d-%d)",
                mode,
                marker_count,
                len(result),
                min_area,
                max_area,
            )
            return result[:expected_parts]
        if len(result) > len(best_result):
            best_result = result

    logger.debug(
        "watershed_split: %d peak markers, %d valid sub-masks (range %d-%d)",
        len(peak_components),
        len(best_result),
        min_area,
        max_area,
    )
    return best_result


def connected_components_in_range(
    mask: np.ndarray,
    min_area: int,
    max_area: int,
) -> list[np.ndarray]:
    """
    Find connected components within the binary mask whose areas
    fall in [min_area, max_area].

    Returns a list of individual component masks.
    """
    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    result: list[np.ndarray] = []
    for comp in range(1, n_labels):  # skip background
        area = int(stats[comp, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            result.append(labels == comp)

    logger.debug(
        "CC filter: %d components, %d in range %d-%d",
        n_labels - 1,
        len(result),
        min_area,
        max_area,
    )
    return result


def split_mask_by_polygon(
    mask: np.ndarray,
    polygon: list[tuple[float, float]],
    min_area: int = 1,
) -> list[np.ndarray]:
    """
    Split a binary mask into polygon-selected and remaining regions.

    The polygon is interpreted as "this region should become one of the split
    masks". The result is therefore:
      1. mask pixels inside the polygon
      2. mask pixels outside the polygon

    Returns two cleaned boolean masks when both parts are large enough,
    otherwise an empty list.
    """
    mask = np.asarray(mask).astype(bool)
    if mask.ndim != 2 or not mask.any() or len(polygon) < 3:
        return []

    points = np.array(
        [[int(round(x)), int(round(y))] for x, y in polygon],
        dtype=np.int32,
    )
    if points.ndim != 2 or points.shape[0] < 3:
        return []

    poly_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(poly_mask, [points], 1)

    inside = mask & (poly_mask > 0)
    outside = mask & (poly_mask == 0)
    if int(inside.sum()) < min_area or int(outside.sum()) < min_area:
        return []

    noise_floor = max(1, min_area // 4)
    result: list[np.ndarray] = []
    for part in (inside, outside):
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            part.astype(np.uint8),
            connectivity=8,
        )
        cleaned = np.zeros(mask.shape, dtype=bool)
        for comp in range(1, n_labels):
            area = int(stats[comp, cv2.CC_STAT_AREA])
            if area >= noise_floor:
                cleaned |= labels == comp
        if int(cleaned.sum()) < min_area:
            return []
        result.append(cleaned)

    return result


def recover_masks(
    masks: dict[int, np.ndarray],
    validator: SizeValidator,
) -> dict[int, np.ndarray]:
    """
    For masks that fail size validation, try watershed split and CC extraction
    to find correctly-sized sub-regions. Returns a corrected mask dict.

    Masks that pass validation are returned as-is.
    """
    ok, reasons = validator.validate(masks)
    if all(ok.values()):
        return masks

    corrected = dict(masks)

    for eid, mask in masks.items():
        if ok.get(eid, True):
            continue

        lo, hi = validator.get_range(eid)
        reason = reasons[eid]
        area = int(mask.sum())
        logger.info(
            "recover_masks: entity %d %s (area=%d, range=%d-%d) - trying recovery",
            eid,
            reason,
            area,
            lo,
            hi,
        )

        if reason == "too_large":
            splits = watershed_split(mask, lo, hi)
            if splits:
                ref = validator.ref_areas.get(eid, (lo + hi) // 2)
                best = min(splits, key=lambda s: abs(int(s.sum()) - ref))
                corrected[eid] = best
                logger.info(
                    "  watershed: found %d valid split(s), picked area=%d",
                    len(splits),
                    int(best.sum()),
                )
                continue

        components = connected_components_in_range(mask, lo, hi)
        if components:
            ref = validator.ref_areas.get(eid, (lo + hi) // 2)
            best = min(components, key=lambda c: abs(int(c.sum()) - ref))
            corrected[eid] = best
            logger.info(
                "  CC: found %d valid component(s), picked area=%d",
                len(components),
                int(best.sum()),
            )
            continue

        if reason == "too_large":
            broad_lo = int(validator.ref_areas.get(eid, lo) * 0.6)
            broad_hi = int(validator.ref_areas.get(eid, hi) * 1.4)
            components = connected_components_in_range(mask, broad_lo, broad_hi)
            if components:
                ref = validator.ref_areas.get(eid, (lo + hi) // 2)
                best = min(components, key=lambda c: abs(int(c.sum()) - ref))
                corrected[eid] = best
                logger.info(
                    "  CC (broad): found %d component(s), picked area=%d",
                    len(components),
                    int(best.sum()),
                )
                continue

        logger.warning(
            "  recovery failed for entity %d - keeping original mask", eid
        )

    return corrected
