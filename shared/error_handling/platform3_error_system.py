#!/usr/bin/env python3
"""
Platform3 Error Handling System

Comprehensive error handling and reporting for Platform3 trading platform.

Author: Platform3 AI Team
Date: June 2, 2025
"""

import traceback
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DatabaseError(Exception):
    """Database related errors for Platform3"""
    
    def __init__(self, message: str, query: Optional[str] = None, details: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.query = query
        self.details = details or {}
        self.timestamp = datetime.now()

class ServiceError(Exception):
    """Custom service error for Platform3 operations"""
    
    def __init__(self, message: str, error_code: str = "GENERAL_ERROR", details: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format"""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }

class Platform3ErrorSystem:
    """Error handling and reporting system for Platform3"""
    
    def __init__(self):
        self.error_history = []
    
    def handle_error(self, error: Exception):
        """Handle and log error"""
        error_info = {
            'type': type(error).__name__,
            'message': str(error),
            'timestamp': datetime.now().isoformat(),
            'traceback': traceback.format_exc()
        }
        
        self.error_history.append(error_info)
        
        # In production, this would send to monitoring system
        print(f"ERROR HANDLED: {error_info['type']} - {error_info['message']}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors"""
        return {
            'total_errors': len(self.error_history),
            'recent_errors': self.error_history[-10:] if self.error_history else []
        }

class BaseService:
    """
    Base Service Class
    
    Provides common functionality for all Platform3 services including
    error handling, logging, and service lifecycle management.
    """
    
    def __init__(self, service_name: str = "Platform3Service"):
        """
        Initialize Base Service
        
        Args:
            service_name: Name of the service for logging
        """
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)
        self.is_initialized = False
        self.is_running = False
    
    def initialize(self) -> bool:
        """Initialize the service"""
        try:
            self.logger.info(f"Initializing {self.service_name}")
            self.is_initialized = True
            return True
        except Exception as e:
            self.handle_error(f"Failed to initialize {self.service_name}", e)
            return False
    
    def start(self) -> bool:
        """Start the service"""
        try:
            if not self.is_initialized:
                if not self.initialize():
                    return False
            
            self.logger.info(f"Starting {self.service_name}")
            self.is_running = True
            return True
        except Exception as e:
            self.handle_error(f"Failed to start {self.service_name}", e)
            return False
    
    def stop(self) -> bool:
        """Stop the service"""
        try:
            self.logger.info(f"Stopping {self.service_name}")
            self.is_running = False
            return True
        except Exception as e:
            self.handle_error(f"Failed to stop {self.service_name}", e)
            return False
    
    def handle_error(self, message: str, exception: Exception = None, 
                     context: Optional[Dict[str, Any]] = None):
        """Handle service errors with proper logging"""
        error_msg = f"{message}: {str(exception) if exception else 'Unknown error'}"
        self.logger.error(error_msg)
        
# Export classes
__all__ = ['ServiceError', 'Platform3ErrorSystem', 'BaseService', 'ErrorSeverity', 'DatabaseError']
