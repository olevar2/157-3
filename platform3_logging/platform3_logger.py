"""
Platform3 Logging Module - Root Level

This module provides the logging.platform3_logger import path that
indicators are expecting.
"""

import sys

from shared.logging.platform3_logger import Platform3Logger, LogMetadata

# Create default logger instance for platform3
platform3_logger = Platform3Logger("Platform3")

# Export for compatibility
__all__ = ['platform3_logger', 'Platform3Logger', 'LogMetadata']
