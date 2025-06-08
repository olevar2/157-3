"""
Platform3 Logging Framework
Shared logging utilities for consistent Winston-style logging across Platform3
"""

from .platform3_logger import (
    Platform3Logger,
    LogMetadata,
    StructuredFormatter,
    global_logger,
    create_logger,
    get_logger
)

__all__ = [
    'Platform3Logger',
    'LogMetadata', 
    'StructuredFormatter',
    'global_logger',
    'create_logger',
    'get_logger'
]

# Version info
__version__ = "1.0.0"
__author__ = "Platform3 Development Team"
