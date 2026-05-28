# -*- coding: utf-8 -*-
"""Logger configuration."""

import logging
from logging.handlers import RotatingFileHandler
from backend.config import Config


class RotatingLogger:
    """Rotating logger with file and console handlers."""

    def __init__(self, name: str, cfg: Config):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, cfg.log_level))

        if not self.logger.handlers:
            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%H:%M:%S"
            ))
            self.logger.addHandler(console)

            try:
                file_handler = RotatingFileHandler(
                    cfg.log_file,
                    maxBytes=cfg.log_max_size_mb * 1024 * 1024,
                    backupCount=cfg.log_backup_count
                )
                file_handler.setFormatter(logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
                ))
                self.logger.addHandler(file_handler)
            except Exception as e:
                self.logger.warning(f"Could not setup file logging: {e}")

    def __getattr__(self, name):
        return getattr(self.logger, name)


logger = RotatingLogger("smallthing", Config())
