"""Configuration dictionary for the logger."""

import logging.config
import os
from datetime import datetime


def setup_logging(log_path, verbose):
    """

    Args:
        log_path (str): Directory holding the log file.
        verbose (bool): Send DEBUG level to std.out?

    Returns:
        Initialized and configured logging facility.
    """
    current_date = datetime.now().strftime('%Y%m%d_%H%M%S')

    console_verbose_lvl = "INFO" if verbose else "WARNING"  # Set console verbose level
    log_main_path = os.path.join(log_path, current_date + ".log")

    logger = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "verbose": {
                "format": "%(asctime)s - %(levelname)s - %(module)s : %(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M"
            },
            "simple": {
                "format": "%(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": console_verbose_lvl,
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            },
            "main": {
                "class": "logging.FileHandler",
                "level": "NOTSET",
                "formatter": "verbose",
                "filename": log_main_path,
                "mode": "w"
            }
        },
        "root": {
            "handlers": ["console", "main"],
            "level": "NOTSET"
        }
    }

    logging.config.dictConfig(logger)


if __name__ == '__main__':
    pass
