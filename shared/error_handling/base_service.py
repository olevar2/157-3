"""
Platform3 Base Service with EventEmitter Error Handling
Provides base class for all Platform3 services with comprehensive error handling,
graceful degradation, and EventEmitter patterns.
"""

import sys
import os
import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
from contextlib import asynccontextmanager

# Add Platform3 frameworks
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from logging.platform3_logger import Platform3Logger, log_performance
from error_handling.platform3_error_system import (
    EventEmitter, ServiceError, ErrorSeverity, ErrorCategory, ErrorMetadata,
    CircuitBreaker, retry_with_backoff,
    ValidationError, AuthenticationError, TradingError, AIModelError,
    DatabaseError, ExternalServiceError
)

class ServiceHealth:
    """Service health monitoring and reporting"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.start_time = datetime.utcnow()
        self.status = "healthy"
        self.error_count = 0
        self.last_error_time: Optional[datetime] = None
        self.uptime_percentage = 100.0
        self.dependencies: Dict[str, bool] = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert health status to dictionary"""
        uptime = datetime.utcnow() - self.start_time
        return {
            "service_name": self.service_name,
            "status": self.status,
            "uptime_seconds": uptime.total_seconds(),
            "uptime_percentage": self.uptime_percentage,
            "error_count": self.error_count,
            "last_error_time": self.last_error_time.isoformat() + 'Z' if self.last_error_time else None,
            "dependencies": self.dependencies,
            "timestamp": datetime.utcnow().isoformat() + 'Z'
        }

class BaseService(EventEmitter, ABC):
    """
    Base service class with comprehensive error handling and EventEmitter patterns
    """
    
    def __init__(
        self,
        service_name: str,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.service_name = service_name
        self.request_id = request_id or str(uuid.uuid4())
        self.user_id = user_id
        self.config = config or {}
        
        # Initialize logging
        self.logger = Platform3Logger.get_logger(service_name)
        
        # Service health monitoring
        self.health = ServiceHealth(service_name)
        
        # Circuit breakers for external dependencies
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Error handling configuration
        self.error_handlers: Dict[str, Callable] = {}
        self.graceful_degradation_enabled = True
        self.max_error_threshold = 10
        
        # Set up default error event handlers
        self._setup_error_handlers()
        
        with self.logger.correlation_context(self.request_id, self.user_id):
            self.logger.info(
                f"Initializing {service_name} service",
                extra={
                    "service_name": service_name,
                    "request_id": self.request_id,
                    "config": self.config
                }
            )
    
    def _setup_error_handlers(self):
        """Set up default error event handlers"""
        self.on('error', self._handle_error_event)
        self.on('critical_error', self._handle_critical_error)
        self.on('degraded_service', self._handle_service_degradation)
        self.on('service_recovery', self._handle_service_recovery)
        self.on('dependency_failure', self._handle_dependency_failure)
    
    def handle_error(self, error: Exception, operation: str = "unknown") -> ServiceError:
        """
        Central error handling method with EventEmitter support
        """
        # Convert to ServiceError if needed
        if not isinstance(error, ServiceError):
            service_error = ServiceError(
                message=str(error),
                code="UNKNOWN_ERROR",
                metadata=ErrorMetadata(
                    timestamp=datetime.utcnow().isoformat() + 'Z',
                    request_id=self.request_id,
                    user_id=self.user_id,
                    service_name=self.service_name,
                    operation=operation,
                    error_id=f"ERR_{int(datetime.utcnow().timestamp() * 1000)}"
                ),
                cause=error
            )
        else:
            service_error = error
            
        # Update health status
        self.health.error_count += 1
        self.health.last_error_time = datetime.utcnow()
          # Log the error
        with self.logger.correlation_context(self.request_id, self.user_id):
            self.logger.error(
                f"Service error in {operation}",
                exc_info=service_error.cause or service_error,
                meta={
                    "error_code": service_error.code,
                    "error_severity": service_error.severity.value,
                    "error_category": service_error.category.value,
                    "operation": operation,
                    "error_metadata": service_error.metadata.to_dict()
                }
            )
        
        # Emit error event
        self.emit_error_event(service_error)
        
        # Check if service needs degradation
        if self.health.error_count >= self.max_error_threshold:
            self.emit('degraded_service', {
                'service_name': self.service_name,
                'error_count': self.health.error_count,
                'threshold': self.max_error_threshold
            })
        
        return service_error
    
    def emit_error_event(self, error: ServiceError):
        """Emit appropriate error event based on severity"""
        if error.severity == ErrorSeverity.CRITICAL:
            self.emit('critical_error', error)
        else:
            self.emit('error', error)
    
    def _handle_error_event(self, error: ServiceError):
        """Handle general error events"""
        with self.logger.correlation_context(self.request_id, self.user_id):
            self.logger.warning(
                f"Error event emitted: {error.code}",
                extra={"error_details": error.to_json()}
            )
    
    def _handle_critical_error(self, error: ServiceError):
        """Handle critical error events"""
        with self.logger.correlation_context(self.request_id, self.user_id):
            self.logger.critical(
                f"Critical error event: {error.code}",
                extra={"error_details": error.to_json()}
            )
        
        # Trigger immediate health check
        self.health.status = "critical"
        
        # Emit health status change
        self.emit('health_status_changed', self.health.to_dict())
    
    def _handle_service_degradation(self, degradation_info: Dict[str, Any]):
        """Handle service degradation events"""
        self.health.status = "degraded"
        
        with self.logger.correlation_context(self.request_id, self.user_id):
            self.logger.warning(
                "Service entering degraded mode",
                extra=degradation_info
            )
        
        if self.graceful_degradation_enabled:
            self.implement_graceful_degradation()
    
    def _handle_service_recovery(self, recovery_info: Dict[str, Any]):
        """Handle service recovery events"""
        self.health.status = "healthy"
        self.health.error_count = 0
        
        with self.logger.correlation_context(self.request_id, self.user_id):
            self.logger.info(
                "Service recovered to healthy state",
                extra=recovery_info
            )
    
    def _handle_dependency_failure(self, dependency_info: Dict[str, Any]):
        """Handle dependency failure events"""
        dependency_name = dependency_info.get('dependency_name', 'unknown')
        self.health.dependencies[dependency_name] = False
        with self.logger.correlation_context(self.request_id, self.user_id):
            self.logger.error(
                f"Dependency failure: {dependency_name}",
                meta=dependency_info
            )
    
    @abstractmethod
    def implement_graceful_degradation(self):
        """
        Implement service-specific graceful degradation logic
        Must be implemented by each service
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Implement service-specific health check
        Must be implemented by each service
        """
        pass
    
    def add_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ) -> CircuitBreaker:
        """Add circuit breaker for external dependency"""
        circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            name=f"{self.service_name}_{name}"
        )
        
        self.circuit_breakers[name] = circuit_breaker
        
        with self.logger.correlation_context(self.request_id, self.user_id):
            self.logger.info(
                f"Added circuit breaker for {name}",
                extra={
                    "circuit_breaker_name": name,
                    "failure_threshold": failure_threshold,
                    "recovery_timeout": recovery_timeout
                }
            )
        
        return circuit_breaker
    
    @retry_with_backoff(max_retries=3, exceptions=(ExternalServiceError,))
    def call_external_service(
        self,
        service_name: str,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Call external service with circuit breaker and retry logic
        """
        if service_name not in self.circuit_breakers:
            self.add_circuit_breaker(service_name)
        
        circuit_breaker = self.circuit_breakers[service_name]
        
        try:
            result = circuit_breaker.call(operation, *args, **kwargs)
            
            # Mark dependency as healthy
            self.health.dependencies[service_name] = True
            
            return result
            
        except Exception as e:
            # Mark dependency as failed
            self.health.dependencies[service_name] = False
            
            # Emit dependency failure event
            self.emit('dependency_failure', {
                'dependency_name': service_name,
                'error': str(e),
                'operation': operation.__name__ if hasattr(operation, '__name__') else 'unknown'
            })
            
            raise ExternalServiceError(
                f"Failed to call {service_name}: {str(e)}",
                service_name=service_name
            )
    
    @log_performance
    async def execute_with_error_handling(
        self,
        operation: Callable,
        operation_name: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute operation with comprehensive error handling
        """
        try:
            with self.logger.correlation_context(self.request_id, self.user_id):
                self.logger.debug(
                    f"Executing operation: {operation_name}",
                    extra={"operation": operation_name}
                )
            
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)
            
            with self.logger.correlation_context(self.request_id, self.user_id):
                self.logger.debug(
                    f"Operation completed successfully: {operation_name}",
                    extra={"operation": operation_name}
                )
            
            return result
            
        except Exception as e:
            service_error = self.handle_error(e, operation_name)
            
            # Re-raise for caller to handle
            raise service_error
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current service health status"""
        return self.health.to_dict()
    
    def reset_error_count(self):
        """Reset error count (for recovery scenarios)"""
        old_count = self.health.error_count
        self.health.error_count = 0
        
        if old_count > 0:
            self.emit('service_recovery', {
                'service_name': self.service_name,
                'previous_error_count': old_count
            })
        
        with self.logger.correlation_context(self.request_id, self.user_id):
            self.logger.info(
                "Error count reset",
                extra={"previous_count": old_count}
            )

# Example service implementation
class TradingService(BaseService):
    """Example trading service with error handling"""
    
    def __init__(self, request_id: Optional[str] = None, user_id: Optional[str] = None):
        super().__init__(
            service_name="trading_service",
            request_id=request_id,
            user_id=user_id
        )
        
        # Add circuit breakers for external dependencies
        self.add_circuit_breaker("broker_api", failure_threshold=3)
        self.add_circuit_breaker("market_data_feed", failure_threshold=5)
    
    def implement_graceful_degradation(self):
        """Implement trading service graceful degradation"""
        with self.logger.correlation_context(self.request_id, self.user_id):
            self.logger.warning("Trading service entering degraded mode - limiting order sizes")
        
        # Example: Reduce order sizes, disable high-risk strategies
        self.config['max_order_size'] = self.config.get('max_order_size', 1000) * 0.5
        self.config['high_risk_strategies_enabled'] = False
    
    async def health_check(self) -> Dict[str, Any]:
        """Trading service health check"""
        health_status = self.get_health_status()
        
        # Check broker connection
        try:
            # Mock broker ping
            await self.ping_broker()
            health_status['broker_connected'] = True
        except Exception:
            health_status['broker_connected'] = False
            health_status['status'] = 'degraded'
        
        return health_status
    
    async def ping_broker(self):
        """Mock broker ping for health check"""
        # This would be actual broker API call
        pass
