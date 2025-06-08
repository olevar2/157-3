#!/usr/bin/env python3
"""
Platform3 Logger System

Comprehensive logging system for Platform3 trading platform
with performance monitoring and error tracking capabilities.

Author: Platform3 AI Team
Date: June 2, 2025
"""

import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

@dataclass
class LogMetadata:
    """Metadata for logging operations"""
    component: str = "platform3"
    operation: str = "unknown"
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    trade_id: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None

class Platform3Logger:
    """Advanced logging system for Platform3 trading operations"""
    
    def __init__(self, name: str = "Platform3", level: int = logging.INFO):
        """
        Initialize Platform3 Logger
        
        Args:
            name: Logger name identifier
            level: Logging level (default: INFO)
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup console and file handlers"""
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        
        # File handler (optional)
        try:
            log_file = Path("logs") / "platform3.log"
            log_file.parent.mkdir(exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
        except:
            # If file logging fails, continue with console only
            pass
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message"""
        self.logger.info(message, extra=extra or {})
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        self.logger.warning(message, extra=extra or {})
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log error message"""
        self.logger.error(message, extra=extra or {})
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message"""
        self.logger.debug(message, extra=extra or {})
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log critical message"""
        self.logger.critical(message, extra=extra or {})

# Export for backwards compatibility
__all__ = ['Platform3Logger', 'LogMetadata', 'log_performance']

# Create default instance
default_logger = Platform3Logger("Platform3")

def log_performance(operation: str, duration: float, **kwargs):
    """Log performance metrics"""
    metadata = LogMetadata(
        component="Platform3",
        operation=operation,
        additional_data={"duration": duration, **kwargs}
    )
    default_logger.info(f"Performance: {operation} completed in {duration:.4f}s", extra=metadata.__dict__)