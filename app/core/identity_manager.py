"""ID consistency management — stores user-assigned identities across frames."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class IdentityManager:
    """
    Manages the mapping between user-assigned mouse IDs and SAM3 object IDs.

    During the interactive setup phase, the user clicks on masks and assigns
    them to Mouse 1, 2, 3, etc.  This class stores and updates that mapping.
    """

    def __init__(self) -> None:
        # {mouse_id: sam_obj_id}  — current canonical mapping
        self._mouse_to_sam: dict[int, int] = {}
        # {sam_obj_id: mouse_id}
        self._sam_to_mouse: dict[int, int] = {}
        # Names the user gave to each mouse
        self._mouse_names: dict[int, str] = {}
        self._n_mice: int = 0

    # ── Setup ─────────────────────────────────────────────────────────────────

    def set_n_mice(self, n: int) -> None:
        """Set the expected number of mice."""
        self._n_mice = n

    def assign(self, mouse_id: int, sam_obj_id: int) -> None:
        """Assign a SAM3 object ID to a mouse identity."""
        # Remove any existing cross-references
        if mouse_id in self._mouse_to_sam:
            old_sam = self._mouse_to_sam[mouse_id]
            self._sam_to_mouse.pop(old_sam, None)
        if sam_obj_id in self._sam_to_mouse:
            old_mouse = self._sam_to_mouse[sam_obj_id]
            self._mouse_to_sam.pop(old_mouse, None)

        self._mouse_to_sam[mouse_id] = sam_obj_id
        self._sam_to_mouse[sam_obj_id] = mouse_id
        logger.info(f"Assigned Mouse {mouse_id} → SAM obj {sam_obj_id}")

    def unassign_mouse(self, mouse_id: int) -> None:
        """Remove the assignment for a mouse."""
        sam_id = self._mouse_to_sam.pop(mouse_id, None)
        if sam_id is not None:
            self._sam_to_mouse.pop(sam_id, None)

    def set_name(self, mouse_id: int, name: str) -> None:
        self._mouse_names[mouse_id] = name

    def get_name(self, mouse_id: int) -> str:
        return self._mouse_names.get(mouse_id, f"Mouse {mouse_id}")

    # ── Queries ───────────────────────────────────────────────────────────────

    def mouse_id_for_sam(self, sam_obj_id: int) -> Optional[int]:
        return self._sam_to_mouse.get(sam_obj_id)

    def sam_id_for_mouse(self, mouse_id: int) -> Optional[int]:
        return self._mouse_to_sam.get(mouse_id)

    def get_full_mapping(self) -> dict[int, int]:
        """Return {mouse_id: sam_obj_id} mapping."""
        return dict(self._mouse_to_sam)

    def assigned_mice(self) -> list[int]:
        return sorted(self._mouse_to_sam.keys())

    def is_complete(self) -> bool:
        """True if all expected mice have been assigned."""
        return (
            self._n_mice > 0
            and len(self._mouse_to_sam) >= self._n_mice
        )

    def reset(self) -> None:
        self._mouse_to_sam.clear()
        self._sam_to_mouse.clear()
        self._mouse_names.clear()
