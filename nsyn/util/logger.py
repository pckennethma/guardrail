import logging

# set to INFO to suppress debug messages
logging.basicConfig(level=logging.INFO)


def get_logger(name: str = "nsyn") -> logging.Logger:
    """
    Returns a logger with the given name.

    :param name: The name of the logger.

    :return: The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
