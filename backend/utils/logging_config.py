from __future__ import annotations

from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import os


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def _resolve_log_level(explicit_level: str | None) -> int:
    level_name = (explicit_level or os.getenv("FR_LOG_LEVEL", "INFO")).upper()
    return getattr(logging, level_name, logging.INFO)


def configure_logging(
    *,
    log_level: str | None = None,
    log_file: str | Path = "logs/system.log",
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> None:
    """Configure root logging once for API, scripts, and dashboard runtime."""
    root_logger = logging.getLogger()
    if getattr(root_logger, "_face_attendance_logging_configured", False):
        return

    resolved_level = _resolve_log_level(log_level)
    root_logger.setLevel(resolved_level)

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(resolved_level)
    root_logger.addHandler(console_handler)

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(resolved_level)
    root_logger.addHandler(file_handler)

    root_logger._face_attendance_logging_configured = True  # type: ignore[attr-defined]


def get_attendance_logger() -> logging.Logger:
    return logging.getLogger("attendance")
