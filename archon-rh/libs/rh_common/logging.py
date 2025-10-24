import logging

_DEF_FMT = "[%(levelname)s] %(asctime)s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEF_FMT))
        logger.addHandler(handler)
    return logger
