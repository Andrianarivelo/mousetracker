"""Tests for the identity assignment panel."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.ui.identity_panel import IdentityPanel


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_reassigning_to_existing_sam_obj_clears_previous_owner() -> None:
    _app()
    panel = IdentityPanel()

    panel.mark_assigned(1, 10)
    panel.mark_assigned(2, 20)
    panel.mark_assigned(1, 20)

    assert panel.assigned_count() == 1
    assert panel._assignments == {1: 20}
    assert panel._entities[1]["lbl_assign"].text() == "Assigned"
    assert panel._entities[2]["lbl_assign"].text() == "Open"


def test_select_entity_highlights_active_row() -> None:
    _app()
    panel = IdentityPanel()

    assert panel.select_entity(2) is True

    assert panel.selected_mouse() == 2
    assert "border: 2px solid" in panel._entities[2]["row_widget"].styleSheet()
    assert panel._entities[1]["row_widget"].styleSheet() == ""
