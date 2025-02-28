import os
import logging
from logging.handlers import WatchedFileHandler

def make_logger(pid, name, logdir):
    os.makedirs(logdir, exist_ok=True)
    logger = logging.getLogger(f"my_custom_logger_{pid}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    file_handler = WatchedFileHandler(os.path.join(logdir, f"{name}_{pid}.log"))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)
    
    return logger