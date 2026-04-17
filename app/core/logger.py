import logging
import sys


def setup_logging(level: int = logging.DEBUG) -> None:
    """Configure the root logger once at application startup.

    Writes to stdout so Railway (and any cloud platform) captures every line
    immediately — especially important when PYTHONUNBUFFERED=1 is set.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    # Remove any handlers already attached (e.g. uvicorn's default ones)
    root.handlers.clear()
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger.  Call once at the top of each file:

        from app.core.logger import get_logger
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)
