#!/usr/bin/env python3
"""
Platform3 Microservices Communication Enhancement Framework
Implements service discovery, health checks, message queuing, and request correlation
"""

import os
import json
import time
import uuid
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

class ServiceType(Enum):
    """Enumeration of Platform3 service types"""
    API_GATEWAY = "api-gateway"
    TRADING_SERVICE = "trading-service"
    USER_SERVICE = "user-service"
    NOTIFICATION_SERVICE = "notification-service"
    MARKET_DATA_SERVICE = "market-data-service"
    EVENT_SYSTEM = "event-system"
    AI_MODEL_SERVICE = "ai-model-service"
    PAYMENT_SERVICE = "payment-service"
    ANALYTICS_SERVICE = "analytics-service"

class ServiceStatus(Enum):
    """Service health status enumeration"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"

class MessagePriority(Enum):
    """Message priority levels for queuing"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

@dataclass
class ServiceInstance:
    """Represents a service instance in the service registry"""
    service_id: str
    service_type: ServiceType
    host: str
    port: int
    version: str
    status: ServiceStatus
    health_check_url: str
    metadata: Dict[str, Any]
    last_heartbeat: float
    registration_time: float
    tags: List[str]
    load_balancer_weight: int = 100
    circuit_breaker_state: str = "closed"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert service instance to dictionary"""
        result = asdict(self)
        result['service_type'] = self.service_type.value
        result['status'] = self.status.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ServiceInstance':
        """Create service instance from dictionary"""
        data['service_type'] = ServiceType(data['service_type'])
        data['status'] = ServiceStatus(data['status'])
        return cls(**data)

@dataclass
class HealthCheckResult:
    """Health check result structure"""
    service_id: str
    status: ServiceStatus
    response_time: float
    timestamp: float
    details: Dict[str, Any]
    errors: List[str]

@dataclass
class Message:
    """Message structure for inter-service communication"""
    message_id: str
    correlation_id: str
    source_service: str
    target_service: str
    message_type: str
    payload: Dict[str, Any]
    priority: MessagePriority
    timestamp: float
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 30.0
    headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}

class ServiceRegistry:
    """Central service registry for Platform3 microservices"""
    
    def __init__(self, consul_host: str = "localhost", consul_port: int = 8500):
        self.consul_host = consul_host
        self.consul_port = consul_port
        self.services: Dict[str, ServiceInstance] = {}
        self.health_check_interval = 30  # seconds
        self.service_timeout = 120  # seconds
        self.logger = logging.getLogger(__name__)
        
    def register_service(self, service: ServiceInstance) -> bool:
        """Register a service instance"""
        try:
            service.registration_time = time.time()
            service.last_heartbeat = time.time()
            
            # Generate unique service ID if not provided
            if not service.service_id:
                service.service_id = f"{service.service_type.value}-{uuid.uuid4().hex[:8]}"
            
            self.services[service.service_id] = service
            
            self.logger.info(f"Service registered: {service.service_id} at {service.host}:{service.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register service: {str(e)}")
            return False
    
    def deregister_service(self, service_id: str) -> bool:
        """Deregister a service instance"""
        try:
            if service_id in self.services:
                del self.services[service_id]
                self.logger.info(f"Service deregistered: {service_id}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to deregister service {service_id}: {str(e)}")
            return False
    
    def update_heartbeat(self, service_id: str) -> bool:
        """Update service heartbeat timestamp"""
        if service_id in self.services:
            self.services[service_id].last_heartbeat = time.time()
            return True
        return False
    
    def get_healthy_services(self, service_type: ServiceType) -> List[ServiceInstance]:
        """Get all healthy instances of a service type"""
        return [
            service for service in self.services.values()
            if service.service_type == service_type and service.status == ServiceStatus.HEALTHY
        ]
    
    def get_service_by_id(self, service_id: str) -> Optional[ServiceInstance]:
        """Get service instance by ID"""
        return self.services.get(service_id)
    
    def cleanup_stale_services(self):
        """Remove services that haven't sent heartbeat"""
        current_time = time.time()
        stale_services = []
        
        for service_id, service in self.services.items():
            if current_time - service.last_heartbeat > self.service_timeout:
                stale_services.append(service_id)
        
        for service_id in stale_services:
            self.deregister_service(service_id)
            self.logger.warning(f"Removed stale service: {service_id}")

class LoadBalancer:
    """Load balancer for service instance selection"""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.round_robin_counters: Dict[str, int] = {}
    
    def select_instance(self, instances: List[ServiceInstance]) -> Optional[ServiceInstance]:
        """Select a service instance based on load balancing strategy"""
        if not instances:
            return None
        
        if self.strategy == "round_robin":
            return self._round_robin_select(instances)
        elif self.strategy == "weighted":
            return self._weighted_select(instances)
        elif self.strategy == "least_connections":
            return self._least_connections_select(instances)
        else:
            return instances[0]  # Default to first available
    
    def _round_robin_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round-robin load balancing"""
        service_type = instances[0].service_type.value
        counter = self.round_robin_counters.get(service_type, 0)
        selected = instances[counter % len(instances)]
        self.round_robin_counters[service_type] = counter + 1
        return selected
    
    def _weighted_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted load balancing based on instance weights"""
        total_weight = sum(instance.load_balancer_weight for instance in instances)
        if total_weight == 0:
            return instances[0]
        
        import random
        weight = random.randint(1, total_weight)
        current_weight = 0
        
        for instance in instances:
            current_weight += instance.load_balancer_weight
            if weight <= current_weight:
                return instance
        
        return instances[-1]
    
    def _least_connections_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance with least connections (simplified)"""
        # In a real implementation, this would track active connections
        return min(instances, key=lambda x: x.metadata.get('active_connections', 0))

class HealthChecker:
    """Health checker for service instances"""
    
    def __init__(self, service_registry: ServiceRegistry):
        self.service_registry = service_registry
        self.logger = logging.getLogger(__name__)
        self.health_check_timeout = 10  # seconds
    
    async def check_service_health(self, service: ServiceInstance) -> HealthCheckResult:
        """Perform health check on a service instance"""
        start_time = time.time()
        
        try:
            # Simulate health check (in real implementation, make HTTP request)
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # Mock health check logic
            is_healthy = True  # Replace with actual HTTP health check
            
            response_time = time.time() - start_time
            status = ServiceStatus.HEALTHY if is_healthy else ServiceStatus.UNHEALTHY
            
            # Update service status
            service.status = status
            
            return HealthCheckResult(
                service_id=service.service_id,
                status=status,
                response_time=response_time,
                timestamp=time.time(),
                details={'endpoint': service.health_check_url},
                errors=[]
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            service.status = ServiceStatus.UNHEALTHY
            
            return HealthCheckResult(
                service_id=service.service_id,
                status=ServiceStatus.UNHEALTHY,
                response_time=response_time,
                timestamp=time.time(),
                details={'endpoint': service.health_check_url},
                errors=[str(e)]
            )
    
    async def run_health_checks(self):
        """Run health checks for all registered services"""
        while True:
            try:
                tasks = []
                for service in self.service_registry.services.values():
                    task = asyncio.create_task(self.check_service_health(service))
                    tasks.append(task)
                
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for result in results:
                        if isinstance(result, HealthCheckResult):
                            if result.status == ServiceStatus.UNHEALTHY:
                                self.logger.warning(
                                    f"Service {result.service_id} is unhealthy: {result.errors}"
                                )
                
                # Cleanup stale services
                self.service_registry.cleanup_stale_services()
                
                await asyncio.sleep(self.service_registry.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Health check error: {str(e)}")
                await asyncio.sleep(5)

class MessageQueue:
    """Message queue for inter-service communication"""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.queues: Dict[str, List[Message]] = {}
        self.logger = logging.getLogger(__name__)
        self.dlq_name = "dead_letter_queue"
        
    def publish_message(self, message: Message) -> bool:
        """Publish a message to the queue"""
        try:
            queue_name = f"{message.target_service}_queue"
            
            if queue_name not in self.queues:
                self.queues[queue_name] = []
            
            # Sort by priority
            self.queues[queue_name].append(message)
            self.queues[queue_name].sort(key=lambda x: x.priority.value)
            
            self.logger.debug(f"Message published to {queue_name}: {message.message_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish message: {str(e)}")
            return False
    
    def consume_message(self, service_name: str) -> Optional[Message]:
        """Consume a message from the queue"""
        queue_name = f"{service_name}_queue"
        
        if queue_name in self.queues and self.queues[queue_name]:
            return self.queues[queue_name].pop(0)
        
        return None
    
    def move_to_dlq(self, message: Message):
        """Move message to dead letter queue"""
        if self.dlq_name not in self.queues:
            self.queues[self.dlq_name] = []
        
        message.retry_count = message.max_retries
        self.queues[self.dlq_name].append(message)
        
        self.logger.warning(f"Message moved to DLQ: {message.message_id}")

class RequestCorrelation:
    """Request correlation and tracing system"""
    
    def __init__(self):
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    def generate_correlation_id(self) -> str:
        """Generate a new correlation ID"""
        return str(uuid.uuid4())
    
    def start_request(self, correlation_id: str, service_name: str, 
                     operation: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Start tracking a request"""
        request_info = {
            'correlation_id': correlation_id,
            'service_name': service_name,
            'operation': operation,
            'start_time': time.time(),
            'metadata': metadata or {},
            'spans': []
        }
        
        self.active_requests[correlation_id] = request_info
        return request_info
    
    def add_span(self, correlation_id: str, service_name: str, 
                operation: str, start_time: float, end_time: float, 
                metadata: Dict[str, Any] = None):
        """Add a span to the request trace"""
        if correlation_id in self.active_requests:
            span = {
                'service_name': service_name,
                'operation': operation,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'metadata': metadata or {}
            }
            
            self.active_requests[correlation_id]['spans'].append(span)
    
    def complete_request(self, correlation_id: str) -> Optional[Dict[str, Any]]:
        """Complete request tracking and return trace information"""
        if correlation_id in self.active_requests:
            request_info = self.active_requests.pop(correlation_id)
            request_info['end_time'] = time.time()
            request_info['total_duration'] = request_info['end_time'] - request_info['start_time']
            
            self.logger.info(f"Request completed: {correlation_id}, "
                           f"Duration: {request_info['total_duration']:.3f}s, "
                           f"Spans: {len(request_info['spans'])}")
            
            return request_info
        
        return None

class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
        self.logger = logging.getLogger(__name__)
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                self.logger.info("Circuit breaker transitioning to half-open")
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
                self.logger.info("Circuit breaker closed")
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise e

class Platform3CommunicationFramework:
    """Main Platform3 microservices communication framework"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.service_registry = ServiceRegistry(
            consul_host=self.config.get('consul_host', 'localhost'),
            consul_port=self.config.get('consul_port', 8500)
        )
        self.load_balancer = LoadBalancer(
            strategy=self.config.get('load_balancer_strategy', 'round_robin')
        )
        self.health_checker = HealthChecker(self.service_registry)
        self.message_queue = MessageQueue(
            redis_host=self.config.get('redis_host', 'localhost'),
            redis_port=self.config.get('redis_port', 6379)
        )
        self.request_correlation = RequestCorrelation()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.logger = logging.getLogger(__name__)
        self.initialized = False
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    async def initialize(self):
        """Initialize the communication framework and all its components"""
        try:
            if self.initialized:
                self.logger.info("Platform3CommunicationFramework already initialized")
                return True

            self.logger.info("Initializing Platform3CommunicationFramework...")
            
            # Initialize message queue
            await self._initialize_message_queue()
            
            # Initialize service registry
            await self._initialize_service_registry()
            
            # Start health monitoring
            await self._initialize_health_monitoring()
            
            self.initialized = True
            self.logger.info("✅ Platform3CommunicationFramework initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Platform3CommunicationFramework: {e}")
            return False
    
    async def _initialize_message_queue(self):
        """Initialize the message queue system"""
        try:
            # For now, this is a placeholder for Redis connection setup
            # In production, this would establish Redis connections
            self.logger.info("Message queue system initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize message queue: {e}")
            raise
    
    async def _initialize_service_registry(self):
        """Initialize the service registry"""
        try:
            # For now, this is a placeholder for Consul connection setup
            # In production, this would establish Consul connections
            self.logger.info("Service registry initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize service registry: {e}")
            raise
    
    async def _initialize_health_monitoring(self):
        """Initialize health monitoring system"""
        try:
            # Start background health checking if needed
            self.logger.info("Health monitoring system initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize health monitoring: {e}")
            raise

    def register_service(self, service_type: ServiceType, host: str, port: int, 
                        version: str, health_check_url: str = None, 
                        metadata: Dict[str, Any] = None, tags: List[str] = None) -> str:
        """Register a service with the framework"""
        try:
            service_id = f"{service_type.value}-{uuid.uuid4().hex[:8]}"
            
            if health_check_url is None:
                health_check_url = f"http://{host}:{port}/health"
            
            service = ServiceInstance(
                service_id=service_id,
                service_type=service_type,
                host=host,
                port=port,
                version=version,
                status=ServiceStatus.STARTING,
                health_check_url=health_check_url,
                metadata=metadata or {},
                last_heartbeat=time.time(),
                registration_time=time.time(),
                tags=tags or []
            )
            
            if self.service_registry.register_service(service):
                self.logger.info(f"Service registered successfully: {service_id}")
                return service_id
            else:
                raise Exception("Failed to register service")
                
        except Exception as e:
            self.logger.error(f"Failed to register service: {e}")
            raise

    def discover_service(self, service_type: ServiceType) -> Optional[ServiceInstance]:
        """Discover a healthy service instance"""
        try:
            healthy_services = self.service_registry.get_healthy_services(service_type)
            
            if not healthy_services:
                self.logger.warning(f"No healthy services found for type: {service_type.value}")
                return None
            
            # Use load balancer to select an instance
            selected_service = self.load_balancer.select_instance(healthy_services)
            
            self.logger.debug(f"Selected service: {selected_service.service_id}")
            return selected_service
            
        except Exception as e:
            self.logger.error(f"Failed to discover service: {e}")
            return None

    def send_message(self, target_service: str, message_type: str, payload: Dict[str, Any],
                    source_service: str, priority: MessagePriority = MessagePriority.NORMAL,
                    correlation_id: str = None, timeout: float = 30.0) -> str:
        """Send a message to a target service"""
        try:
            message_id = str(uuid.uuid4())
            if correlation_id is None:
                correlation_id = self.request_correlation.generate_correlation_id()
            
            message = Message(
                message_id=message_id,
                correlation_id=correlation_id,
                source_service=source_service,
                target_service=target_service,
                message_type=message_type,
                payload=payload,
                priority=priority,
                timestamp=time.time(),
                timeout=timeout
            )
            
            if self.message_queue.publish_message(message):
                self.logger.debug(f"Message sent successfully: {message_id}")
                return message_id
            else:
                raise Exception("Failed to publish message")
                
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            raise

    def consume_messages(self, service_name: str) -> List[Message]:
        """Consume messages for a service"""
        try:
            messages = []
            while True:
                message = self.message_queue.consume_message(service_name)
                if message is None:
                    break
                messages.append(message)
            
            return messages
            
        except Exception as e:
            self.logger.error(f"Failed to consume messages: {e}")
            return []

    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            services_by_type = {}
            for service in self.service_registry.services.values():
                service_type = service.service_type.value
                if service_type not in services_by_type:
                    services_by_type[service_type] = []
                services_by_type[service_type].append(service.to_dict())
            
            return {
                'timestamp': time.time(),
                'total_services': len(self.service_registry.services),
                'services_by_type': services_by_type,
                'active_requests': len(self.request_correlation.active_requests),
                'framework_initialized': self.initialized
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get service status: {e}")
            return {}

    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a service"""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()
        return self.circuit_breakers[service_name]

    async def shutdown(self):
        """Gracefully shutdown the communication framework"""
        try:
            self.logger.info("Shutting down Platform3CommunicationFramework...")
            
            # Stop health monitoring
            # In a full implementation, this would stop background tasks
            
            # Clear circuit breakers
            self.circuit_breakers.clear()
            
            # Mark as not initialized
            self.initialized = False
            
            self.logger.info("✅ Platform3CommunicationFramework shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise

# Example usage and testing
async def main():
    """Example usage of the Platform3 communication framework"""
    framework = Platform3CommunicationFramework()
    
    # Register some services
    api_gateway_id = framework.register_service(
        ServiceType.API_GATEWAY, "localhost", 3000, "1.0.0",
        metadata={"role": "gateway"}, tags=["public", "gateway"]
    )
    
    trading_service_id = framework.register_service(
        ServiceType.TRADING_SERVICE, "localhost", 3001, "1.0.0",
        metadata={"role": "trading"}, tags=["core", "trading"]
    )
    
    # Discover a service
    trading_service = framework.discover_service(ServiceType.TRADING_SERVICE)
    if trading_service:
        print(f"Discovered trading service: {trading_service.service_id}")
    
    # Send a message
    message_id = framework.send_message(
        target_service="trading-service",
        message_type="execute_trade",
        payload={"symbol": "BTCUSD", "amount": 100},
        source_service="api-gateway"
    )
    
    print(f"Sent message: {message_id}")
    
    # Get system status
    status = framework.get_service_status()
    print(f"System status: {json.dumps(status, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())
