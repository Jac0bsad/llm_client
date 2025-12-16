import logging as logger

logger.basicConfig(
    level=logger.INFO,
    format="%(asctime)s %(filename)s %(lineno)dL %(levelname)-8s %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
)


def get_logger():
    return logger
