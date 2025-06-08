"""
Platform3 Standardized Logging Framework
Winston-style logging for Python with structured JSON format and request correlation
"""

import logging
import json
import os
import sys
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Union
from contextlib import contextmanager
from pathlib import Path

# Robust import of logging handlers
try:
    from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
except ImportError:
    # Fallback for environments where logging.handlers is not available
    RotatingFileHandler = logging.FileHandler
    TimedRotatingFileHandler = logging.FileHandler


class LogMetadata:
    """Structured logging metadata following Platform3 standards"""
    
    def __init__(
        self,
        timestamp: str = None,
        level: str = None,
        service: str = None,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs
    ):
        self.timestamp = timestamp or datetime.utcnow().isoformat() + 'Z'
        self.level = level
        self.service = service
        self.request_id = request_id
        self.user_id = user_id
        self.extra = kwargs


class StructuredFormatter(logging.Formatter):
    """Winston-style JSON formatter for structured logging"""
    
    def __init__(self, service_name: str = "Platform3"):
        super().__init__()
        self.service_name = service_name
    
    def format(self, record: logging.LogRecord) -> str:
        # Create structured log object following Winston pattern
        log_object = {
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "level": record.levelname,
            "service": self.service_name,
            "context": getattr(record, 'context', record.name),
            "message": record.getMessage(),
        }
        
        # Add request correlation if available
        if hasattr(record, 'request_id'):
            log_object["requestId"] = record.request_id
        
        if hasattr(record, 'user_id'):
            log_object["userId"] = record.user_id
        
        # Add stack trace for errors
        if record.exc_info:
            log_object["stack"] = self.formatException(record.exc_info)
        
        # Add extra metadata
        if hasattr(record, 'meta') and record.meta:
            log_object.update(record.meta)
        
        return json.dumps(log_object, ensure_ascii=False)


class Platform3Logger:
    """
    Platform3 standardized logger following Winston patterns
    
    Features:
    - Structured JSON logging
    - Multiple transports (console, file, error)
    - Request correlation tracking
    - Log rotation and retention
    - Trading-specific logging methods
    """
    
    def __init__(
        self,
        context: str = "Application",
        service_name: str = "Platform3",
        level: str = None,
        log_to_file: bool = True,
        log_directory: str = "./logs",
        max_file_size: str = "20MB",
        max_files: int = 14,
        colorize: bool = None
    ):
        self.context = context
        self.service_name = service_name
        self.log_directory = Path(log_directory)
        self.request_id = None
        self.user_id = None
        
        # Set defaults
        if level is None:
            level = os.getenv('LOG_LEVEL', 'INFO')
        if colorize is None:
            colorize = os.getenv('NODE_ENV') != 'production'
        
        # Create logger
        self.logger = logging.getLogger(f"{service_name}.{context}")
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup transports
        self._setup_transports(log_to_file, max_file_size, max_files, colorize)
    
    def _setup_transports(self, log_to_file: bool, max_file_size: str, max_files: int, colorize: bool):
        """Setup logging transports following Winston patterns"""
        
        # Console transport
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        if colorize and sys.stdout.isatty():
            # Simple colored formatter for console
            console_formatter = logging.Formatter(
                '\033[90m%(asctime)s\033[0m [\033[%(levelcolor)s%(levelname)s\033[0m] [\033[36m%(context)s\033[0m] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            # Add level colors
            def format_with_color(record):
                colors = {
                    'DEBUG': '37m',    # White
                    'INFO': '32m',     # Green
                    'WARNING': '33m',  # Yellow
                    'ERROR': '31m',    # Red
                    'CRITICAL': '35m'  # Magenta
                }
                record.levelcolor = colors.get(record.levelname, '0m')
                record.context = getattr(record, 'context', self.context)
                return console_formatter.format(record)
            
            console_handler.format = format_with_color
        else:
            # JSON formatter for console in production
            console_handler.setFormatter(StructuredFormatter(self.service_name))
        
        self.logger.addHandler(console_handler)
        
        # File transports (if enabled)
        if log_to_file:
            self.log_directory.mkdir(parents=True, exist_ok=True)
            
            # Parse file size
            size_bytes = self._parse_file_size(max_file_size)
            
            # Combined logs
            combined_handler = RotatingFileHandler(
                self.log_directory / 'combined.log',
                maxBytes=size_bytes,
                backupCount=max_files
            )
            combined_handler.setLevel(logging.INFO)
            combined_handler.setFormatter(StructuredFormatter(self.service_name))
            self.logger.addHandler(combined_handler)
            
            # Error logs
            error_handler = RotatingFileHandler(
                self.log_directory / 'error.log',
                maxBytes=size_bytes,
                backupCount=max_files
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(StructuredFormatter(self.service_name))
            self.logger.addHandler(error_handler)
            
            # Debug logs (only in development)
            if os.getenv('NODE_ENV') != 'production':
                debug_handler = RotatingFileHandler(
                    self.log_directory / 'debug.log',
                    maxBytes=size_bytes,
                    backupCount=7
                )
                debug_handler.setLevel(logging.DEBUG)
                debug_handler.setFormatter(StructuredFormatter(self.service_name))
                self.logger.addHandler(debug_handler)
    
    def _parse_file_size(self, size_str: str) -> int:
        """Parse file size string (e.g., '20MB', '1GB') to bytes"""
        units = {
            'B': 1,
            'KB': 1024,
            'MB': 1024 * 1024,
            'GB': 1024 * 1024 * 1024
        }
        
        size_str = size_str.upper().strip()
        for unit, multiplier in units.items():
            if size_str.endswith(unit):
                try:
                    return int(size_str[:-len(unit)]) * multiplier
                except ValueError:
                    break
          # Default to 20MB
        return 20 * 1024 * 1024
    
    def _log(self, level: str, message: str, meta: Dict[str, Any] = None, **kwargs):
        """Internal logging method with metadata"""
        extra = {
            'context': self.context,
            'meta': meta or {}
        }
        
        if self.request_id:
            extra['request_id'] = self.request_id
        if self.user_id:
            extra['user_id'] = self.user_id
        
        getattr(self.logger, level.lower())(message, extra=extra, **kwargs)
    
    # Core logging methods (Winston-style)    def error(self, message: str, meta: Dict[str, Any] = None, **kwargs):
        """Log error message"""
        self._log('ERROR', message, meta, **kwargs)
    
    def warn(self, message: str, meta: Dict[str, Any] = None):
        """Log warning message"""
        self._log('WARNING', message, meta)
    
    def info(self, message: str, meta: Dict[str, Any] = None):
        """Log info message"""
        self._log('INFO', message, meta)
    
    def debug(self, message: str, meta: Dict[str, Any] = None):
        """Log debug message"""
        self._log('DEBUG', message, meta)
    
    def verbose(self, message: str, meta: Dict[str, Any] = None):
        """Log verbose message (maps to debug)"""
        self._log('DEBUG', message, meta)
    
    # Trading-specific logging methods
    def log_trade(self, message: str, trade_data: Dict[str, Any]):
        """Log trading operations"""
        self.info(message, {
            'type': 'TRADE',
            'trade': trade_data,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
    
    def log_market_data(self, message: str, market_data: Dict[str, Any]):
        """Log market data updates"""
        self.debug(message, {
            'type': 'MARKET_DATA',
            'market': market_data,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
    
    def log_event(self, message: str, event_data: Dict[str, Any]):
        """Log system events"""
        self.info(message, {
            'type': 'EVENT',
            'event': event_data,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
    
    def log_performance(self, message: str, performance_data: Dict[str, Any]):
        """Log performance metrics"""
        self.debug(message, {
            'type': 'PERFORMANCE',
            'performance': performance_data,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
    
    def log_security(self, message: str, security_data: Dict[str, Any]):
        """Log security events"""
        self.warn(message, {
            'type': 'SECURITY',
            'security': security_data,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
    
    def log_audit(self, message: str, audit_data: Dict[str, Any]):
        """Log audit events"""
        self.info(message, {
            'type': 'AUDIT',
            'audit': audit_data,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
    
    def log_ai_model(self, message: str, model_data: Dict[str, Any]):
        """Log AI model operations"""
        self.info(message, {
            'type': 'AI_MODEL',
            'model': model_data,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
    
    def log_prediction(self, message: str, prediction_data: Dict[str, Any]):
        """Log AI predictions"""
        self.debug(message, {
            'type': 'PREDICTION',
            'prediction': prediction_data,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
    
    # Request correlation methods
    def set_request_id(self, request_id: str):
        """Set request ID for correlation tracking"""
        self.request_id = request_id
    
    def set_user_id(self, user_id: str):
        """Set user ID for tracking"""
        self.user_id = user_id
    
    def clear_context(self):
        """Clear request context"""
        self.request_id = None
        self.user_id = None
    
    @contextmanager
    def request_context(self, request_id: str = None, user_id: str = None):
        """Context manager for request correlation"""
        old_request_id = self.request_id
        old_user_id = self.user_id
        
        try:
            self.request_id = request_id or str(uuid.uuid4())
            self.user_id = user_id
            yield self
        finally:
            self.request_id = old_request_id
            self.user_id = old_user_id
    
    # Child logger creation
    def create_child(self, child_context: str) -> 'Platform3Logger':
        """Create child logger with additional context"""
        full_context = f"{self.context}:{child_context}"
        return Platform3Logger(
            context=full_context,
            service_name=self.service_name
        )
    
    # Performance timing decorator
    def time_operation(self, operation_name: str):
        """Decorator for timing operations"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    self.log_performance(
                        f"Operation {operation_name} completed",
                        {
                            'operation': operation_name,
                            'duration_ms': round(duration * 1000, 2),
                            'success': True
                        }
                    )
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    self.log_performance(
                        f"Operation {operation_name} failed",
                        {
                            'operation': operation_name,
                            'duration_ms': round(duration * 1000, 2),
                            'success': False,
                            'error': str(e)
                        }
                    )
                    raise
            return wrapper
        return decorator


# Global logger instance (singleton pattern)
global_logger = Platform3Logger(
    context="Global",
    service_name="Platform3",
    level=os.getenv('LOG_LEVEL', 'INFO'),
    log_to_file=os.getenv('LOG_TO_FILE', 'true').lower() != 'false',
    log_directory=os.getenv('LOG_DIRECTORY', './logs')
)


def create_logger(context: str, service_name: str = "Platform3", **kwargs) -> Platform3Logger:
    """Helper function to create logger instances"""
    return Platform3Logger(context=context, service_name=service_name, **kwargs)


def get_logger(context: str = None) -> Platform3Logger:
    """Get logger instance (creates new or returns global)"""
    if context:
        return create_logger(context)
    return global_logger
