"""Logging utilities"""

import logging
import sys
from typing import Optional

def setup_logger(
    name: str = "hello_agents",
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger
    
    Args:
        name: Logger name
        level: Log level
        format_string: Log format
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            format_string or 
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def get_logger(name: str = "hello_agents") -> logging.Logger:
    """Get logger"""
    return logging.getLogger(name)