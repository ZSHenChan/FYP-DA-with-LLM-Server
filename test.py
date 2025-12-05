from app.services.agent.utils import SessionWorkspace
import logging
from pathlib import Path
from typing import Optional
from core.config import config

def get_file_logger(
    name: str = "app",
    filename: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Return a simple file-backed logger. Creates log directory if missing.
    - name: logger name
    - filename: override file name (defaults to config.CENTRAL_LOG_FILENAME or 'app.log')
    - level: logging level (logging.INFO by default)
    """
    log_dir = getattr(config, "CENTRAL_LOG_DIR", "logs")
    default_file = getattr(config, "CENTRAL_LOG_FILENAME", "app.log")
    filepath = Path(log_dir) / (filename or default_file)

    filepath.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # avoid duplicate logs if root logger configured

    # if handlers already set up for this logger, return it (idempotent)
    if any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == filepath for h in logger.handlers if getattr(h, "baseFilename", None)):
        return logger

    fh = logging.FileHandler(filepath, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh.setFormatter(fmt)
    fh.setLevel(level)

    logger.addHandler(fh)
    return logger

# convenience short-hand
def simple_log(message: str, level: str = "info", logger_name: str = "app"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logger = get_file_logger(logger_name, level=lvl)
    logger.log(lvl, message)

logger = get_file_logger()
sess_wp = SessionWorkspace('sess_86085887','run_93fc1974')
logger.info(sess_wp.list_figures())