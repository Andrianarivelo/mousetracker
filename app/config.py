"""App-wide settings, paths, and defaults for MouseTracker Pro."""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent
RESOURCES_DIR = ROOT_DIR / "resources"
ICONS_DIR = RESOURCES_DIR / "icons"

# ── SAM3 model checkpoint ─────────────────────────────────────────────────────
# Resolution order:
#   1) Runtime override selected by the user in-app
#   2) Env vars SAM3_CHECKPOINT_PATH / SAM3_SAFETENSORS_PATH
#   3) Original git defaults (C:\sam3_pt\...)
#   4) Local D: fallback (D:\Analysis\sam3-weights\...)
#
# If no local file is found, the model builder may attempt HuggingFace.
SAM3_CHECKPOINT_PATH: str = os.environ.get("SAM3_CHECKPOINT_PATH", "").strip()
SAM3_SAFETENSORS_PATH: str = os.environ.get("SAM3_SAFETENSORS_PATH", "").strip()

_RUNTIME_SAM3_CHECKPOINT_PATH: str = ""

GIT_DEFAULT_SAM3_PT = Path(r"C:\sam3_pt\sam3.pt")
GIT_DEFAULT_SAM3_SAFETENSORS = Path(r"C:\sam3_pt\sam3.safetensors")

D_FALLBACK_SAM3_PT = Path(r"D:\Analysis\sam3-weights\sam3.pt")
D_FALLBACK_SAM3_SAFETENSORS = Path(r"D:\Analysis\sam3-weights\sam3.safetensors")


def _sam3_checkpoint_candidates() -> list[Path]:
    """Ordered list of possible local checkpoint files."""
    candidates: list[Path] = []

    # Runtime selection in GUI wins.
    if _RUNTIME_SAM3_CHECKPOINT_PATH:
        candidates.append(Path(_RUNTIME_SAM3_CHECKPOINT_PATH).expanduser())

    # Explicit env-var paths win.
    if SAM3_CHECKPOINT_PATH:
        candidates.append(Path(SAM3_CHECKPOINT_PATH).expanduser())
    if SAM3_SAFETENSORS_PATH:
        candidates.append(Path(SAM3_SAFETENSORS_PATH).expanduser())

    # Original git defaults.
    candidates.extend(
        [
            GIT_DEFAULT_SAM3_PT,
            GIT_DEFAULT_SAM3_SAFETENSORS,
        ]
    )

    # D: fallback.
    candidates.extend(
        [
            D_FALLBACK_SAM3_PT,
            D_FALLBACK_SAM3_SAFETENSORS,
        ]
    )
    return candidates


def set_runtime_sam3_checkpoint(path: str | None) -> None:
    """Set or clear the runtime checkpoint override selected by the user."""
    global _RUNTIME_SAM3_CHECKPOINT_PATH
    _RUNTIME_SAM3_CHECKPOINT_PATH = str(path or "").strip()


def get_sam3_checkpoint() -> str | None:
    """Return the first existing local SAM3 checkpoint path, if any."""
    for candidate in _sam3_checkpoint_candidates():
        if candidate.exists():
            return str(candidate.resolve())
    return None

# ── Identity colors (R, G, B) ─────────────────────────────────────────────────
IDENTITY_COLORS: dict[int, tuple[int, int, int]] = {
    1: (0, 200, 255),    # Cyan
    2: (255, 140, 0),    # Orange
    3: (0, 255, 100),    # Green
    4: (255, 50, 150),   # Pink
    5: (180, 100, 255),  # Purple
    6: (255, 255, 0),    # Yellow
}

IDENTITY_COLORS_HEX: dict[int, str] = {
    1: "#00c8ff",
    2: "#ff8c00",
    3: "#00ff64",
    4: "#ff3296",
    5: "#b464ff",
    6: "#ffff00",
}

IDENTITY_NAMES: dict[int, str] = {
    1: "Mouse 1",
    2: "Mouse 2",
    3: "Mouse 3",
    4: "Mouse 4",
    5: "Mouse 5",
    6: "Mouse 6",
}

MAX_MICE = 6

# ── SAM3 defaults ─────────────────────────────────────────────────────────────
DEFAULT_TEXT_PROMPT = "separate dark animal on clear bedding"
AUTO_PROMPTS = [
    "separate dark animal on clear bedding",
    "black mouse",
    "dark mouse on white bedding",
    "mouse",
    "small dark animal",
    "rodent",
]
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_SAMPLE_FRAMES = 5

# ── Tracking defaults ─────────────────────────────────────────────────────────
TRACKING_COST_WEIGHTS = {
    "centroid_distance": 0.4,
    "mask_iou": 0.35,
    "appearance": 0.25,
}
ID_SWITCH_COST_THRESHOLD = 0.7
MIN_MOUSE_AREA_PIXELS = 500   # reject masks smaller than this
MAX_MOUSE_AREA_FRACTION = 0.3  # reject masks larger than this fraction of frame

# ── Mask overlay ──────────────────────────────────────────────────────────────
MASK_ALPHA = 0.4  # opacity of mask overlay (0=transparent, 1=opaque)

# ── Viewer update rate during tracking ───────────────────────────────────────
VIEWER_UPDATE_EVERY_N_FRAMES = 30

# ── Keypoints ─────────────────────────────────────────────────────────────────
ALL_KEYPOINTS = [
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
DEFAULT_KEYPOINTS = ["nose_tip", "body_center", "tail_base"]

# Per-keypoint colors (R, G, B) — mapped to ALL_KEYPOINTS order
KEYPOINT_COLORS: dict[str, tuple[int, int, int]] = {
    "nose_tip":    (255, 50, 50),     # Red
    "head_center": (255, 140, 50),    # Orange
    "neck":        (255, 220, 50),    # Yellow
    "body_center": (50, 220, 50),     # Green
    "left_hip":    (50, 200, 200),    # Teal
    "right_hip":   (50, 150, 255),    # Light blue
    "hip_center":  (50, 100, 255),    # Blue
    "tail_base":   (140, 80, 255),    # Purple
    "tail_mid":    (200, 50, 255),    # Magenta
    "tail_tip":    (255, 50, 200),    # Pink
    "left_ear":    (180, 255, 50),    # Lime
    "right_ear":   (50, 255, 180),    # Mint
}

# ── UI dimensions ─────────────────────────────────────────────────────────────
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
FILE_PANEL_WIDTH = 220
IDENTITY_PANEL_WIDTH = 200
TIMELINE_HEIGHT = 80
SETTINGS_PANEL_HEIGHT = 130

# ── Playback ──────────────────────────────────────────────────────────────────
DEFAULT_FPS = 25
PLAYBACK_TIMER_INTERVAL_MS = 40  # ~25 fps display
