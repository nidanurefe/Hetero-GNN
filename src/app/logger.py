import logging
from .logging_config import setup

setup()

# Logger for modules 
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)