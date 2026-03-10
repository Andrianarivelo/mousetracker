"""MouseTracker Pro — entry point."""

import logging
import os
import sys
from pathlib import Path

# Reduce CUDA allocator fragmentation before torch is imported anywhere.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# Add the mousetracker package root to path so that `app.*` imports work
sys.path.insert(0, str(Path(__file__).parent))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from app.main_window import MainWindow


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy third-party loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting MouseTracker Pro")

    # High-DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("MouseTracker Pro")
    app.setOrganizationName("NeuroLab")

    window = MainWindow()
    window.show()

    logger.info("Application window shown")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
