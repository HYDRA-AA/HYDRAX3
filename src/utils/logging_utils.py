#!/usr/bin/env python3
"""
Logging utilities for the HYDRA encryption algorithm.

This module provides a consistent logging interface for the HYDRA project.
"""

import os
import sys
import logging
import platform
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union

# Default log levels
DEFAULT_CONSOLE_LEVEL = logging.INFO
DEFAULT_FILE_LEVEL = logging.DEBUG

# Default log format
DEFAULT_LOG_FORMAT = '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class HydraLogger:
    """HYDRA logger class for unified logging across the project."""
    
    def __init__(self, name: str, log_dir: Optional[str] = None, 
                 console_level: int = DEFAULT_CONSOLE_LEVEL,
                 file_level: int = DEFAULT_FILE_LEVEL):
        """
        Initialize a HYDRA logger.
        
        Args:
            name: Logger name
            log_dir: Directory to store log files
            console_level: Logging level for console output
            file_level: Logging level for file output
        """
        self.name = name
        self.logger = logging.getLogger(name)
        
        # Set the root logger level to DEBUG to allow all handlers to
        # filter according to their own levels
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self.logger.handlers = []
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Add file handler if log_dir is provided
        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_file = log_path / f"{name}_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(file_level)
            file_formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            # Log system info
            self._log_system_info()
    
    def _log_system_info(self):
        """Log system information for debugging purposes."""
        self.logger.debug("System Information:")
        self.logger.debug(f"  Platform: {platform.platform()}")
        self.logger.debug(f"  Python version: {sys.version}")
        self.logger.debug(f"  CPU count: {os.cpu_count()}")
        self.logger.debug(f"  Working directory: {os.getcwd()}")
    
    def debug(self, msg: str, *args, **kwargs):
        """Log a debug message."""
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Log an info message."""
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log a warning message."""
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Log an error message."""
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """Log a critical message."""
        self.logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, exc_info=True, **kwargs):
        """Log an exception."""
        self.logger.exception(msg, *args, exc_info=exc_info, **kwargs)
    
    def log_dict(self, level: int, msg: str, data: Dict[str, Any]) -> None:
        """
        Log a dictionary with a message at the specified level.
        
        Args:
            level: Logging level
            msg: Message to log
            data: Dictionary to log
        """
        if self.logger.isEnabledFor(level):
            data_str = "\n".join(f"  {k}: {v}" for k, v in data.items())
            self.logger.log(level, f"{msg}\n{data_str}")

# Module-level function to get a logger
def get_logger(name: str, log_dir: Optional[str] = None,
               console_level: int = DEFAULT_CONSOLE_LEVEL,
               file_level: int = DEFAULT_FILE_LEVEL) -> HydraLogger:
    """
    Get a HYDRA logger.
    
    Args:
        name: Logger name
        log_dir: Directory to store log files
        console_level: Logging level for console output
        file_level: Logging level for file output
    
    Returns:
        HYDRA logger
    """
    return HydraLogger(name, log_dir, console_level, file_level)

# Helper function to set up all loggers in a consistent way
def configure_logging(log_dir: Optional[str] = None,
                      console_level: int = DEFAULT_CONSOLE_LEVEL,
                      file_level: int = DEFAULT_FILE_LEVEL) -> None:
    """
    Configure logging for the HYDRA project.
    
    Args:
        log_dir: Directory to store log files
        console_level: Logging level for console output
        file_level: Logging level for file output
    """
    # Create log directory if provided
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if log_dir is provided
    if log_dir:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = Path(log_dir) / f"hydra_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

# Example usage
if __name__ == "__main__":
    # Example 1: Get a logger for a specific module
    logger = get_logger("hydra.example", log_dir="logs")
    logger.info("This is an info message")
    logger.debug("This is a debug message")
    
    # Example 2: Configure all logging
    configure_logging(log_dir="logs")
    root_logger = logging.getLogger()
    root_logger.info("This is a root logger info message")
    
    # Example 3: Log data
    data = {
        "operation": "encryption",
        "file_size": 1024,
        "elapsed_time": 0.05,
        "success": True
    }
    logger.log_dict(logging.INFO, "Encryption stats:", data)
