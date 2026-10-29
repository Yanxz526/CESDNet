"""Console and UTF-8 file logging used by the three training entry points."""
import logging
from pathlib import Path


def get_logger(logdir):
    path = Path(logdir)
    path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(path.resolve()))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s | %(message)s')
        for handler in (logging.StreamHandler(), logging.FileHandler(path / 'train.log', encoding='utf-8')):
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    return logger
