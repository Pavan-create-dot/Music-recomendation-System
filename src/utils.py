import logging
import sys
from config import config

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a logger with a standard file and stream handler configuration.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(config.LOG_LEVEL)
    
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    # File Handler
    try:
        file_handler = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # Fallback to sys.stderr if log file is not writable
        pass
        
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
