import logging
from pathlib import Path

from src.app.colors import Colors
from src.app.config import get_config

_LOGGER_INITIALIZED = False


class ColorFormatter(logging.Formatter):

    LEVEL_COLORS = {
        logging.DEBUG: Colors.CYAN,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.MAGENTA,
    }

    def format(self, record: logging.LogRecord) -> str:

        msg = record.getMessage()

        # Keyword highlight
        msg = msg.replace("Epoch", f"{Colors.BLUE}Epoch{Colors.RESET}")
        msg = msg.replace("Step", f"{Colors.MAGENTA}Step{Colors.RESET}")

        # Loss/metric highlight
        msg = msg.replace("train_loss", f"{Colors.CYAN}train_loss{Colors.RESET}")
        msg = msg.replace("val_loss", f"{Colors.YELLOW}val_loss{Colors.RESET}")
        msg = msg.replace("val_pairwise", f"{Colors.YELLOW}val_pairwise{Colors.RESET}")
        msg = msg.replace("test_pairwise", f"{Colors.GREEN}test_pairwise{Colors.RESET}")

        # Time and Level 
        asctime = self.formatTime(record, self.datefmt)
        level_color = self.LEVEL_COLORS.get(record.levelno, Colors.WHITE)
        levelname = f"{level_color}{record.levelname}{Colors.RESET}"

        return f"{asctime} | {levelname} | {msg}"


def _setup_root_logger(level: int) -> None:
    global _LOGGER_INITIALIZED
    if _LOGGER_INITIALIZED:
        return

    logger = logging.getLogger()
    logger.setLevel(level)

    # logs directory
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    # File handler 
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = ColorFormatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    _LOGGER_INITIALIZED = True


def get_logger(name: str = __name__) -> logging.Logger:

    global _LOGGER_INITIALIZED
    if not _LOGGER_INITIALIZED:

        try:
            cfg = get_config()
            level_name = getattr(cfg.logging, "level", "INFO")
            level = getattr(logging, level_name.upper(), logging.INFO)
        except Exception:
            level = logging.INFO

        _setup_root_logger(level)

    return logging.getLogger(name)