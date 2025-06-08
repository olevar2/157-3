"""
Platform3 Logging Module

Centralized logging infrastructure for Platform3 trading system.
Provides consistent logging across all platform components.
"""

from .platform3_logger import Platform3Logger, LogMetadata, log_performance

# Create default logger instance
default_logger = Platform3Logger("Platform3")

# Export main interface
__all__ = [
    'Platform3Logger',
    'LogMetadata', 
    'default_logger',
    'log_performance'
]

# Convenience functions using default logger
def info(message: str, **kwargs):
    """Log info message using default logger"""
    return default_logger.info(message, extra=kwargs)

def warning(message: str, **kwargs):
    """Log warning message using default logger"""
    return default_logger.warning(message, extra=kwargs)

def error(message: str, **kwargs):
    """Log error message using default logger"""
    return default_logger.error(message, extra=kwargs)

def debug(message: str, **kwargs):
    """Log debug message using default logger"""
    return default_logger.debug(message, extra=kwargs)

def critical(message: str, **kwargs):
    """Log critical message using default logger"""
    return default_logger.critical(message, extra=kwargs)
