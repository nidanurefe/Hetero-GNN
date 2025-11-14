import logging
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"


# Setup logging configuration
def setup(level: int = logging.INFO) -> None:

    logger = logging.getLogger()

    if logger.handlers:
        return

    logger.setLevel(level)

    # File Handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(level)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Adding Handlers to Logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


