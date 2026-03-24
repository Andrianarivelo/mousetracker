"""Dynamic entity (mouse / object) identity assignment panel."""

import logging
from typing import Optional

from PySide6.QtCore import QEvent, QObject, QRect, QSize, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.config import MAX_MICE, identity_color_hex, identity_color_rgb

logger = logging.getLogger(__name__)


def _color_hex_for(entity_id: int) -> str:
    return identity_color_hex(entity_id)


def _make_icon(painter_fn, size: int = 18) -> QIcon:
    """Create a QIcon by painting onto a transparent pixmap."""
    pix = QPixmap(size, size)
    pix.fill(QColor(0, 0, 0, 0))
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter_fn(p, size)
    p.end()
    return QIcon(pix)


def _icon_rename(size: int = 18) -> QIcon:
    """Pencil icon."""
    def _draw(p: QPainter, s: int) -> None:
        pen = QPen(QColor("#cdd6f4"), 1.6)
        pen.setCapStyle(pen.capStyle().RoundCap)
        p.setPen(pen)
        # pencil body (diagonal line)
        p.drawLine(4, s - 5, s - 5, 4)
        # pencil tip
        p.drawLine(3, s - 4, 4, s - 5)
        # small nib mark
        p.drawLine(s - 5, 4, s - 3, 2)
    return _make_icon(_draw, size)


def _icon_clear(size: int = 18) -> QIcon:
    """Eraser / undo-arrow icon."""
    def _draw(p: QPainter, s: int) -> None:
        pen = QPen(QColor("#f9e2af"), 1.6)
        pen.setCapStyle(pen.capStyle().RoundCap)
        pen.setJoinStyle(pen.joinStyle().RoundJoin)
        p.setPen(pen)
        # circular arrow (undo)
        from PySide6.QtCore import QPointF
        cx, cy, r = s / 2, s / 2, s * 0.34
        # arc approximation with lines
        import math
        pts = []
        for deg in range(-30, 260, 15):
            rad = math.radians(deg)
            pts.append(QPointF(cx + r * math.cos(rad), cy - r * math.sin(rad)))
        for i in range(len(pts) - 1):
            p.drawLine(pts[i], pts[i + 1])
        # arrowhead at the end
        end = pts[-1]
        p.drawLine(end, QPointF(end.x() - 3, end.y() - 2))
        p.drawLine(end, QPointF(end.x() + 1, end.y() - 3))
    return _make_icon(_draw, size)


def _icon_remove(size: int = 18) -> QIcon:
    """Trash-can icon."""
    def _draw(p: QPainter, s: int) -> None:
        pen = QPen(QColor("#f38ba8"), 1.4)
        pen.setCapStyle(pen.capStyle().RoundCap)
        pen.setJoinStyle(pen.joinStyle().RoundJoin)
        p.setPen(pen)
        # lid
        p.drawLine(3, 5, s - 3, 5)
        # handle on lid
        p.drawLine(7, 5, 7, 3)
        p.drawLine(7, 3, s - 7, 3)
        p.drawLine(s - 7, 3, s - 7, 5)
        # body
        p.drawLine(4, 5, 5, s - 3)
        p.drawLine(5, s - 3, s - 5, s - 3)
        p.drawLine(s - 5, s - 3, s - 4, 5)
        # vertical lines inside
        mid = s // 2
        p.drawLine(mid, 7, mid, s - 5)
        p.drawLine(mid - 2, 7, mid - 2, s - 5)
        p.drawLine(mid + 2, 7, mid + 2, s - 5)
    return _make_icon(_draw, size)


def _icon_flag(size: int = 18) -> QIcon:
    """Pin/target icon for frame marking."""
    def _draw(p: QPainter, s: int) -> None:
        pen = QPen(QColor("#89dceb"), 1.5)
        pen.setCapStyle(pen.capStyle().RoundCap)
        p.setPen(pen)
        # crosshair circle
        cx, cy, r = s / 2, s / 2, s * 0.3
        p.drawEllipse(QRect(int(cx - r), int(cy - r), int(2 * r), int(2 * r)))
        # crosshair lines
        p.drawLine(int(cx), 2, int(cx), int(cy - r - 1))
        p.drawLine(int(cx), int(cy + r + 1), int(cx), s - 2)
        p.drawLine(2, int(cy), int(cx - r - 1), int(cy))
        p.drawLine(int(cx + r + 1), int(cy), s - 2, int(cy))
    return _make_icon(_draw, size)


class IdentityPanel(QWidget):
    """
    Panel for dynamically managing tracked entities (mice, objects).

    Interaction model:
      - Single click on dot or name -> select entity for mask assignment
      - Double click on name       -> rename inline via dialog
      - Pencil button              -> rename via dialog
      - X button                   -> clear mask assignment
      - Minus button               -> remove entity from session

    Signals:
        entity_added(int, str, str): (entity_id, name, type)
        entity_removed(int): entity_id
        mouse_selected(int): selected entity_id
        assignment_cleared(int): entity_id
    """

    entity_added = Signal(int, str, str)
    entity_removed = Signal(int)
    mouse_selected = Signal(int)
    assignment_cleared = Signal(int)
    swap_requested = Signal(int, int, int, int)
    shortcuts_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._entities: dict[int, dict] = {}
        self._next_id = 1
        self._selected_id: Optional[int] = None
        self._assignments: dict[int, Optional[int]] = {}
        self._current_frame: int = 0
        self._shortcut_names: dict[int, str] = {
            i: str(i) for i in range(1, MAX_MICE + 1)
        }
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        lbl_title = QLabel("Entities")
        lbl_title.setStyleSheet("font-weight: bold; color: #cba6f7;")
        layout.addWidget(lbl_title)

        lbl_hint = QLabel(
            "Click dot or name to select. Use icons to rename, clear, or remove."
        )
        lbl_hint.setWordWrap(True)
        lbl_hint.setStyleSheet("color: #6c7086; font-size: 11px;")
        layout.addWidget(lbl_hint)

        add_row = QHBoxLayout()
        add_row.setSpacing(4)
        self.btn_add_mouse = QPushButton("+ Mouse")
        self.btn_add_mouse.setObjectName("secondary_button")
        self.btn_add_mouse.setToolTip("Add a mouse entity")
        self.btn_add_mouse.clicked.connect(lambda: self._add_entity("mouse"))
        add_row.addWidget(self.btn_add_mouse)

        self.btn_add_object = QPushButton("+ Object")
        self.btn_add_object.setObjectName("secondary_button")
        self.btn_add_object.setToolTip("Add a generic tracked object")
        self.btn_add_object.clicked.connect(lambda: self._add_entity("object"))
        add_row.addWidget(self.btn_add_object)
        add_row.addStretch()
        layout.addLayout(add_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        self._entity_container = QWidget()
        self._entity_layout = QVBoxLayout(self._entity_container)
        self._entity_layout.setSpacing(4)
        self._entity_layout.setContentsMargins(0, 0, 0, 0)
        self._entity_layout.addStretch(1)
        scroll.setWidget(self._entity_container)
        layout.addWidget(scroll)

        self.lbl_detection_hint = QLabel("")
        self.lbl_detection_hint.setWordWrap(True)
        self.lbl_detection_hint.setStyleSheet("color: #a6e3a1; font-size: 11px;")
        self.lbl_detection_hint.hide()
        layout.addWidget(self.lbl_detection_hint)

        swap_group = QGroupBox("Correct ID Swap")
        swap_layout = QVBoxLayout(swap_group)
        swap_layout.setSpacing(4)

        lbl_swap = QLabel(
            "Swap two entity IDs within a frame range.\n"
            "Use flag buttons to set From/To from the current timeline position."
        )
        lbl_swap.setWordWrap(True)
        lbl_swap.setStyleSheet("color: #6c7086; font-size: 11px;")
        swap_layout.addWidget(lbl_swap)

        combo_row = QHBoxLayout()
        combo_row.setSpacing(4)
        self.combo_swap_a = QComboBox()
        self.combo_swap_b = QComboBox()
        combo_row.addWidget(self.combo_swap_a)
        combo_row.addWidget(QLabel("↔"))
        combo_row.addWidget(self.combo_swap_b)
        swap_layout.addLayout(combo_row)

        from_row = QHBoxLayout()
        from_row.setSpacing(4)
        from_row.addWidget(QLabel("From:"))
        self.spin_swap_from = QSpinBox()
        self.spin_swap_from.setRange(0, 999999)
        self.spin_swap_from.setValue(0)
        self.spin_swap_from.setMaximumWidth(80)
        from_row.addWidget(self.spin_swap_from)
        btn_flag_from = QPushButton()
        btn_flag_from.setFixedSize(28, 24)
        btn_flag_from.setIcon(_icon_flag())
        btn_flag_from.setIconSize(QSize(16, 16))
        btn_flag_from.setObjectName("secondary_button")
        btn_flag_from.setToolTip("Set From frame to current timeline position")
        btn_flag_from.clicked.connect(
            lambda: self.spin_swap_from.setValue(self._current_frame)
        )
        from_row.addWidget(btn_flag_from)
        from_row.addStretch()
        swap_layout.addLayout(from_row)

        to_row = QHBoxLayout()
        to_row.setSpacing(4)
        to_row.addWidget(QLabel("To:"))
        self.spin_swap_to = QSpinBox()
        self.spin_swap_to.setRange(0, 999999)
        self.spin_swap_to.setValue(999999)
        self.spin_swap_to.setMaximumWidth(80)
        to_row.addWidget(self.spin_swap_to)
        btn_flag_to = QPushButton()
        btn_flag_to.setFixedSize(28, 24)
        btn_flag_to.setIcon(_icon_flag())
        btn_flag_to.setIconSize(QSize(16, 16))
        btn_flag_to.setObjectName("secondary_button")
        btn_flag_to.setToolTip("Set To frame to current timeline position")
        btn_flag_to.clicked.connect(
            lambda: self.spin_swap_to.setValue(self._current_frame)
        )
        to_row.addWidget(btn_flag_to)
        to_row.addStretch()
        swap_layout.addLayout(to_row)

        self.btn_apply_swap = QPushButton("Apply Swap in Range")
        self.btn_apply_swap.setObjectName("secondary_button")
        self.btn_apply_swap.setToolTip(
            "Swap the two entity IDs within [From, To] frames only"
        )
        self.btn_apply_swap.clicked.connect(self._emit_swap_requested)
        swap_layout.addWidget(self.btn_apply_swap)
        layout.addWidget(swap_group)

        self._add_entity("mouse")
        self._add_entity("mouse")

    def _add_entity(self, entity_type: str) -> int:
        eid = self._next_id
        self._next_id += 1

        existing = sum(1 for e in self._entities.values() if e["type"] == entity_type)
        default_name = (
            f"Mouse {existing + 1}"
            if entity_type == "mouse"
            else f"Object {existing + 1}"
        )
        color = _color_hex_for(eid)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        btn_select = QPushButton("●")
        btn_select.setFixedWidth(28)
        btn_select.setCheckable(True)
        btn_select.setStyleSheet(
            f"QPushButton {{ color: {color}; font-size: 16px; border: none; background: transparent; }}"
            f"QPushButton:checked {{ background: {color}33; border-radius: 4px; }}"
        )
        btn_select.setToolTip("Select this entity for mask assignment")
        btn_select.clicked.connect(lambda _checked, e=eid: self._on_select(e))
        row_layout.addWidget(btn_select)

        entity_index = len(self._entities) + 1
        shortcut_text = self._shortcut_names.get(entity_index, "")
        lbl_shortcut = QLabel(f"[{shortcut_text}]" if shortcut_text else "")
        lbl_shortcut.setFixedWidth(24)
        lbl_shortcut.setStyleSheet(
            "color: #6c7086; font-size: 10px; font-weight: bold;"
        )
        lbl_shortcut.setToolTip(f"Press {shortcut_text} to select this entity")
        row_layout.addWidget(lbl_shortcut)

        btn_name = QPushButton(default_name)
        btn_name.setFlat(True)
        btn_name.setStyleSheet(
            f"QPushButton {{ color: {color}; text-align: left; border: none; "
            f"background: transparent; padding: 0 2px; font-weight: bold; }}"
        )
        btn_name.setToolTip("Click to select. Double-click or ✎ to rename")
        btn_name.clicked.connect(lambda _checked, e=eid: self._on_select(e))
        btn_name.installEventFilter(self)
        row_layout.addWidget(btn_name, stretch=1)

        btn_rename = QPushButton()
        btn_rename.setFixedSize(24, 24)
        btn_rename.setIcon(_icon_rename())
        btn_rename.setIconSize(QSize(16, 16))
        btn_rename.setObjectName("secondary_button")
        btn_rename.setToolTip("Rename this entity")
        btn_rename.clicked.connect(lambda _checked, e=eid: self._rename_entity(e))
        row_layout.addWidget(btn_rename)

        lbl_assign = QLabel("Open")
        lbl_assign.setFixedWidth(62)
        row_layout.addWidget(lbl_assign)

        lbl_det = QLabel("")
        lbl_det.setFixedWidth(22)
        lbl_det.setStyleSheet("font-size: 13px;")
        lbl_det.setToolTip(
            "Detection status: check = detected by SAM / x = not found / blank = not yet segmented"
        )
        row_layout.addWidget(lbl_det)

        btn_clear = QPushButton()
        btn_clear.setFixedSize(24, 24)
        btn_clear.setIcon(_icon_clear())
        btn_clear.setIconSize(QSize(16, 16))
        btn_clear.setObjectName("secondary_button")
        btn_clear.setToolTip("Clear mask assignment (keep entity)")
        btn_clear.clicked.connect(lambda _checked, e=eid: self._clear_assignment(e))
        row_layout.addWidget(btn_clear)

        btn_remove = QPushButton()
        btn_remove.setFixedSize(24, 24)
        btn_remove.setIcon(_icon_remove())
        btn_remove.setIconSize(QSize(16, 16))
        btn_remove.setObjectName("secondary_button")
        btn_remove.setToolTip("Remove this entity")
        btn_remove.clicked.connect(lambda _checked, e=eid: self._remove_entity(e))
        row_layout.addWidget(btn_remove)

        insert_pos = max(0, self._entity_layout.count() - 1)
        self._entity_layout.insertWidget(insert_pos, row)

        self._entities[eid] = {
            "name": default_name,
            "type": entity_type,
            "sam_id": None,
            "row_widget": row,
            "btn_select": btn_select,
            "btn_name": btn_name,
            "lbl_assign": lbl_assign,
            "lbl_det": lbl_det,
            "lbl_shortcut": lbl_shortcut,
            "color": color,
        }
        self._set_assignment_label(eid, assigned=False)

        self._refresh_swap_combos()
        self._refresh_shortcut_labels()
        self.entity_added.emit(eid, default_name, entity_type)
        logger.debug("Entity added: id=%d name=%r type=%s", eid, default_name, entity_type)
        return eid

    def _remove_entity(self, entity_id: int) -> None:
        entity = self._entities.pop(entity_id, None)
        if entity is None:
            return
        row = entity["row_widget"]
        self._entity_layout.removeWidget(row)
        row.deleteLater()
        if self._selected_id == entity_id:
            self._selected_id = None
        self._assignments.pop(entity_id, None)
        self._refresh_swap_combos()
        self._refresh_shortcut_labels()
        self.entity_removed.emit(entity_id)

    def _apply_entity_visual_state(self, entity_id: int) -> None:
        entity = self._entities.get(entity_id)
        if entity is None:
            return

        color = entity["color"]
        is_selected = self._selected_id == entity_id
        entity["btn_select"].setChecked(is_selected)
        if is_selected:
            entity["row_widget"].setStyleSheet(
                f"QWidget {{ background: {color}26; border: 2px solid {color}; border-radius: 6px; }}"
            )
            entity["btn_name"].setStyleSheet(
                f"QPushButton {{ color: {color}; text-align: left; border: 1px solid {color}99; "
                f"background: {color}22; padding: 2px 6px; border-radius: 5px; font-weight: bold; }}"
            )
        else:
            entity["row_widget"].setStyleSheet("")
            entity["btn_name"].setStyleSheet(
                f"QPushButton {{ color: {color}; text-align: left; border: none; "
                f"background: transparent; padding: 0 2px; font-weight: bold; }}"
            )

    def _on_select(self, entity_id: int) -> None:
        self._selected_id = entity_id
        for eid, entity in self._entities.items():
            self._apply_entity_visual_state(eid)
        self.mouse_selected.emit(entity_id)

    def _rename_entity(self, entity_id: int) -> None:
        entity = self._entities.get(entity_id)
        if entity is None:
            return
        new_name, ok = QInputDialog.getText(
            self, "Rename Entity", "New name:", text=entity["name"]
        )
        if ok and new_name.strip():
            new_name = new_name.strip()
            entity["name"] = new_name
            entity["btn_name"].setText(new_name)
            self._refresh_swap_combos()

    def _clear_assignment(self, entity_id: int) -> None:
        self._assignments.pop(entity_id, None)
        self._set_assignment_label(entity_id, assigned=False)
        self.assignment_cleared.emit(entity_id)

    def _set_assignment_label(
        self,
        entity_id: int,
        assigned: bool,
        sam_obj_id: Optional[int] = None,
    ) -> None:
        entity = self._entities.get(entity_id)
        if entity is None:
            return

        lbl_assign = entity["lbl_assign"]
        color = entity["color"]
        if assigned:
            lbl_assign.setText("Assigned")
            if self._selected_id == entity_id:
                lbl_assign.setStyleSheet(
                    f"color: {color}; background: {color}22; border: 1px solid {color}99; "
                    "border-radius: 9px; padding: 1px 6px; font-size: 11px; font-weight: bold;"
                )
            else:
                lbl_assign.setStyleSheet(
                    f"color: {color}; font-size: 11px; font-weight: bold;"
                )
            lbl_assign.setToolTip(f"Assigned to SAM object {sam_obj_id}")
        else:
            lbl_assign.setText("Open")
            if self._selected_id == entity_id:
                lbl_assign.setStyleSheet(
                    "color: #cdd6f4; background: #45475a; border: 1px solid #6c7086; "
                    "border-radius: 9px; padding: 1px 6px; font-size: 11px; font-weight: bold;"
                )
            else:
                lbl_assign.setStyleSheet("color: #9399b2; font-size: 11px;")
            lbl_assign.setToolTip("No mask assigned yet")

        self._apply_entity_visual_state(entity_id)

    def _refresh_swap_combos(self) -> None:
        for combo in (self.combo_swap_a, self.combo_swap_b):
            combo.clear()
            for eid, entity in self._entities.items():
                combo.addItem(entity["name"], eid)

    def _emit_swap_requested(self) -> None:
        ids = self.get_swap_selection()
        if len(ids) != 2:
            return
        from_f = self.spin_swap_from.value()
        to_f = self.spin_swap_to.value()
        if from_f > to_f:
            from_f, to_f = to_f, from_f
        self.swap_requested.emit(ids[0], ids[1], from_f, to_f)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.MouseButtonDblClick:
            for eid, entity in self._entities.items():
                if obj is entity.get("btn_name"):
                    self._rename_entity(eid)
                    return True
        return super().eventFilter(obj, event)

    def mark_assigned(self, entity_id: int, sam_obj_id: int) -> None:
        for eid, asgn in list(self._assignments.items()):
            if eid != entity_id and asgn == sam_obj_id:
                self._assignments.pop(eid, None)
                self._set_assignment_label(eid, assigned=False)
                break

        self._assignments[entity_id] = sam_obj_id
        self._set_assignment_label(entity_id, assigned=True, sam_obj_id=sam_obj_id)

    def sync_assignments(self, assignments: dict[int, int]) -> None:
        self._assignments = {}
        for eid in self._entities.keys():
            sam_obj_id = assignments.get(eid)
            if sam_obj_id is None:
                self._set_assignment_label(eid, assigned=False)
                continue
            self._assignments[eid] = sam_obj_id
            self._set_assignment_label(eid, assigned=True, sam_obj_id=sam_obj_id)

    def selected_mouse(self) -> Optional[int]:
        return self._selected_id

    def get_swap_selection(self) -> list[int]:
        a = self.combo_swap_a.currentData()
        b = self.combo_swap_b.currentData()
        if a is not None and b is not None and a != b:
            return [a, b]
        return []

    def select_entity(self, entity_id: int) -> bool:
        if entity_id in self._entities:
            self._on_select(entity_id)
            return True
        return False

    def select_nth_entity(self, n: int) -> bool:
        keys = list(self._entities.keys())
        if 1 <= n <= len(keys):
            self._on_select(keys[n - 1])
            return True
        return False

    def assigned_count(self) -> int:
        return sum(1 for v in self._assignments.values() if v is not None)

    def entity_count(self) -> int:
        return len(self._entities)

    def get_identity_colors(self) -> dict[int, tuple[int, int, int]]:
        """Return {entity_id: (R, G, B)} for every current entity."""
        return {eid: identity_color_rgb(eid) for eid in self._entities}

    def entity_name(self, entity_id: int) -> str:
        entity = self._entities.get(entity_id)
        if entity is None:
            return f"Entity {entity_id}"
        return str(entity.get("name", f"Entity {entity_id}"))

    def nth_entity_id(self, n: int) -> Optional[int]:
        keys = list(self._entities.keys())
        if 1 <= n <= len(keys):
            return int(keys[n - 1])
        return None

    def reset(self) -> None:
        self._assignments.clear()
        self._selected_id = None
        for eid, entity in self._entities.items():
            self._set_assignment_label(eid, assigned=False)
            self._apply_entity_visual_state(eid)

    def set_n_mice(self, n: int) -> None:
        pass

    def set_current_frame(self, frame: int) -> None:
        self._current_frame = frame

    def set_entity_detected(self, entity_id: int, detected: bool) -> None:
        entity = self._entities.get(entity_id)
        if entity is None:
            return
        lbl = entity.get("lbl_det")
        if lbl is None:
            return
        if detected:
            lbl.setText("✓")
            lbl.setStyleSheet("font-size: 13px; color: #a6e3a1;")
            lbl.setToolTip("Detected by SAM in the last segmentation")
        else:
            lbl.setText("✗")
            lbl.setStyleSheet("font-size: 13px; color: #f38ba8;")
            lbl.setToolTip("Not detected - entity may not be present in this frame")

    def clear_detection_status(self) -> None:
        for entity in self._entities.values():
            lbl = entity.get("lbl_det")
            if lbl:
                lbl.setText("")
                lbl.setStyleSheet("font-size: 13px;")

    def set_detection_hint(self, n_detected: int, n_entities: int) -> None:
        if n_detected == n_entities:
            self.lbl_detection_hint.setText(
                f"✓ All {n_entities} entities matched. Click a mask to reassign if needed."
            )
            self.lbl_detection_hint.setStyleSheet("color: #a6e3a1; font-size: 11px;")
        elif n_detected < n_entities:
            self.lbl_detection_hint.setText(
                f"⚠ {n_detected} object(s) detected, {n_entities} entities defined. "
                "Some entities may be absent - assign the visible ones and leave the rest open."
            )
            self.lbl_detection_hint.setStyleSheet("color: #f9e2af; font-size: 11px;")
        else:
            self.lbl_detection_hint.setText(
                f"{n_detected} objects found for {n_entities} entities. Click masks to assign."
            )
            self.lbl_detection_hint.setStyleSheet("color: #89dceb; font-size: 11px;")
        self.lbl_detection_hint.show()

    def _refresh_shortcut_labels(self) -> None:
        for i, (_eid, entity) in enumerate(self._entities.items()):
            idx = i + 1
            lbl = entity.get("lbl_shortcut")
            if lbl is None:
                continue
            key_name = self._shortcut_names.get(idx, "")
            lbl.setText(f"[{key_name}]" if key_name else "")
            lbl.setToolTip(f"Press {key_name} to select this entity" if key_name else "")

    def set_shortcut_names(self, names: dict[int, str]) -> None:
        self._shortcut_names = dict(names)
        self._refresh_shortcut_labels()

    def set_swap_max_frame(self, max_frame: int) -> None:
        self.spin_swap_from.setMaximum(max_frame)
        self.spin_swap_to.setMaximum(max_frame)
        self.spin_swap_to.setValue(max_frame)
