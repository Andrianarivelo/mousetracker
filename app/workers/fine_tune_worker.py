"""QThread worker for launching SAM3 fine-tuning in a subprocess."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

from app.config import ROOT_DIR

logger = logging.getLogger(__name__)


class FineTuneWorker(QThread):
    """Run the generated SAM3 training config in a separate Python process."""

    status = Signal(str)
    log_line = Signal(str)
    finished = Signal(bool, str)

    def __init__(
        self,
        runner_script: str,
        config_path: str,
        output_dir: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.runner_script = str(Path(runner_script).resolve())
        self.config_path = str(Path(config_path).resolve())
        self.output_dir = str(Path(output_dir).resolve())
        self._process: Optional[subprocess.Popen] = None
        self._abort = False

    def abort(self) -> None:
        self._abort = True
        process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
        except Exception:
            logger.exception("Failed to terminate fine-tune subprocess")

    def run(self) -> None:
        command = [
            sys.executable,
            "-u",
            self.runner_script,
            "--config",
            self.config_path,
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        output_lines: list[str] = []

        try:
            self.status.emit(f"Launching SAM3 fine-tune: {Path(self.output_dir).name}")
            self._process = subprocess.Popen(
                command,
                cwd=str(ROOT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=creationflags,
            )

            assert self._process.stdout is not None
            for raw_line in self._process.stdout:
                if self._abort:
                    break
                line = raw_line.strip()
                if not line:
                    continue
                output_lines.append(line)
                output_lines = output_lines[-200:]
                self.log_line.emit(line)
                self.status.emit(line)

            exit_code = self._process.wait()
            if self._abort:
                self.finished.emit(False, "Fine-tuning cancelled.")
                return
            if exit_code == 0:
                self.finished.emit(True, self.output_dir)
                return

            detail = "\n".join(output_lines[-20:]).strip()
            if not detail:
                detail = f"Fine-tuning failed with exit code {exit_code}."
            self.finished.emit(False, detail)
        except Exception as error:
            logger.exception("Fine-tuning failed: %s", error)
            self.finished.emit(False, str(error))
        finally:
            self._process = None
