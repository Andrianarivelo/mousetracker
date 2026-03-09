"""QSS dark theme for MouseTracker Pro."""

from app.config import IDENTITY_COLORS_HEX

DARK_THEME_QSS = """
QMainWindow {
    background: #1e1e2e;
}
QWidget {
    background: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", "Arial", sans-serif;
    font-size: 13px;
}
QDockWidget {
    background: #2a2a3e;
    color: #cdd6f4;
    border: 1px solid #45475a;
    titlebar-close-icon: none;
}
QDockWidget::title {
    background: #313244;
    padding: 6px 10px;
    font-weight: bold;
    color: #cdd6f4;
    border-bottom: 1px solid #45475a;
}
QFrame, QGroupBox {
    background: #2a2a3e;
    border: 1px solid #45475a;
    border-radius: 4px;
}
QGroupBox {
    font-weight: bold;
    color: #cdd6f4;
    padding-top: 12px;
    margin-top: 4px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}
QLabel {
    color: #cdd6f4;
    background: transparent;
    border: none;
}
QLabel#title_label {
    font-size: 15px;
    font-weight: bold;
    color: #cba6f7;
}
QLabel#subtitle_label {
    font-size: 11px;
    color: #6c7086;
}
QPushButton {
    background: #7c3aed;
    color: white;
    border: none;
    padding: 4px 12px;
    border-radius: 5px;
    font-weight: bold;
    font-size: 12px;
}
QPushButton:hover {
    background: #6d28d9;
}
QPushButton:pressed {
    background: #5b21b6;
}
QPushButton:disabled {
    background: #45475a;
    color: #6c7086;
}
QPushButton#secondary_button {
    background: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
}
QPushButton#secondary_button:hover {
    background: #45475a;
}
QPushButton#danger_button {
    background: #f38ba8;
    color: #1e1e2e;
}
QPushButton#danger_button:hover {
    background: #e06c84;
}
QSlider::groove:horizontal {
    background: #45475a;
    height: 8px;
    border-radius: 4px;
}
QSlider::handle:horizontal {
    background: #cba6f7;
    width: 16px;
    height: 16px;
    margin: -4px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background: #b4befe;
}
QSlider::sub-page:horizontal {
    background: #7c3aed;
    border-radius: 4px;
}
QCheckBox {
    color: #cdd6f4;
    spacing: 8px;
    background: transparent;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 2px solid #45475a;
    border-radius: 3px;
    background: #313244;
}
QCheckBox::indicator:checked {
    background: #7c3aed;
    border-color: #7c3aed;
}
QCheckBox::indicator:hover {
    border-color: #7c3aed;
}
QProgressBar {
    background: #45475a;
    border-radius: 4px;
    text-align: center;
    color: white;
    height: 20px;
    border: none;
}
QProgressBar::chunk {
    background: #7c3aed;
    border-radius: 4px;
}
QComboBox {
    background: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    padding: 5px 10px;
    border-radius: 5px;
}
QComboBox:hover {
    border-color: #7c3aed;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background: #313244;
    color: #cdd6f4;
    selection-background-color: #7c3aed;
    border: 1px solid #45475a;
}
QLineEdit {
    background: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    padding: 5px 8px;
    border-radius: 5px;
}
QLineEdit:focus {
    border-color: #7c3aed;
}
QSpinBox, QDoubleSpinBox {
    background: #313244;
    color: #cdd6f4;
    border: 1px solid #585b70;
    padding: 3px 4px;
    border-radius: 5px;
    min-height: 24px;
}
QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #cba6f7;
}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background: #45475a;
    border: none;
    width: 16px;
    border-radius: 3px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: #585b70;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid #cdd6f4;
    width: 0; height: 0;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #cdd6f4;
    width: 0; height: 0;
}
QScrollBar:vertical {
    background: #1e1e2e;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #45475a;
    border-radius: 5px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #7c3aed;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: #1e1e2e;
    height: 10px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background: #45475a;
    border-radius: 5px;
    min-width: 20px;
}
QScrollBar::handle:horizontal:hover {
    background: #7c3aed;
}
QListWidget {
    background: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    color: #cdd6f4;
}
QListWidget::item:selected {
    background: #7c3aed;
    color: white;
}
QListWidget::item:hover {
    background: #45475a;
}
QTabWidget::pane {
    background: #2a2a3e;
    border: 1px solid #45475a;
}
QTabBar::tab {
    background: #313244;
    color: #cdd6f4;
    padding: 6px 14px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    border: 1px solid #45475a;
    border-bottom: none;
}
QTabBar::tab:selected {
    background: #7c3aed;
    color: white;
}
QTabBar::tab:hover {
    background: #45475a;
}
QSplitter::handle {
    background: #45475a;
    width: 2px;
    height: 2px;
}
QToolTip {
    background: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    padding: 4px 8px;
    border-radius: 4px;
}
QMessageBox {
    background: #2a2a3e;
    color: #cdd6f4;
}
QDialog {
    background: #2a2a3e;
}
"""


def get_identity_label_style(mouse_id: int) -> str:
    """Return a QSS style string for an identity label."""
    color = IDENTITY_COLORS_HEX.get(mouse_id, "#ffffff")
    return f"color: {color}; font-weight: bold;"


def get_mouse_button_style(mouse_id: int, selected: bool = False) -> str:
    """Return style for a mouse identity selection button."""
    color = IDENTITY_COLORS_HEX.get(mouse_id, "#ffffff")
    if selected:
        return (
            f"background: {color}; color: #1e1e2e; border: 2px solid white; "
            f"padding: 6px 12px; border-radius: 6px; font-weight: bold;"
        )
    else:
        return (
            f"background: #313244; color: {color}; border: 2px solid {color}; "
            f"padding: 6px 12px; border-radius: 6px; font-weight: bold;"
        )
