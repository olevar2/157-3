"""
API Router

Intelligent routing and management system for multiple broker APIs with 
failover, load balancing, and unified interface for trading operations.

Features:
- Multi-broker API management
- Intelligent request routing
- Failover and redundancy
- Load balancing and rate limiting
- Unified trading interface
- Connection health monitoring
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor, Future
from collections import defaultdict, deque
import threading
import hashlib
from abc import ABC, abstractmethod


class BrokerStatus(Enum):
    """Broker connection status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    RATE_LIMITED = "rate_limited"


class RequestPriority(Enum):
    """Request priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class RoutingStrategy(Enum):
    """Routing strategies for request distribution"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    BEST_LATENCY = "best_latency"
    WEIGHTED = "weighted"
    PRIMARY_BACKUP = "primary_backup"


@dataclass
class BrokerConfig:
    """Broker configuration"""
    broker_id: str
    broker_name: str
    api_class: str
    priority: int = 1
    weight: float = 1.0
    max_requests_per_second: int = 10
    max_concurrent_requests: int = 5
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    health_check_interval: float = 60.0
    connection_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestMetrics:
    """Request performance metrics"""
    broker_id: str
    request_type: str
    timestamp: datetime
    latency: float
    success: bool
    error_message: Optional[str] = None
    retry_count: int = 0


class BrokerInterface(ABC):
    """Abstract base class for broker implementations"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker API"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from broker API"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check broker connection health"""
        pass
    
    @abstractmethod
    async def place_order(self, order_data: Dict) -> Dict:
        """Place trading order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> Dict:
        """Cancel existing order"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> Dict:
        """Get current positions"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict:
        """Get account information"""
        pass
    
    @abstractmethod
    async def get_market_data(self, symbol: str) -> Dict:
        """Get market data for symbol"""
        pass


class APIRouter:
    """
    Advanced API routing and management system
    
    Provides intelligent routing for trading operations across multiple brokers with:
    - Automatic failover and redundancy
    - Load balancing and rate limiting
    - Performance monitoring and optimization
    - Unified trading interface
    - Connection health management
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize APIRouter
        
        Args:
            config: Router configuration dictionary
        """
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Broker management
        self.brokers: Dict[str, BrokerInterface] = {}
        self.broker_configs: Dict[str, BrokerConfig] = {}
        self.broker_status: Dict[str, BrokerStatus] = {}
        self.broker_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Routing configuration
        self.routing_strategy = RoutingStrategy(self.config.get('routing_strategy', 'round_robin'))
        self.default_timeout = self.config.get('default_timeout', 30.0)
        self.max_retries = self.config.get('max_retries', 3)
        
        # Load balancing
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.last_used_broker: Dict[str, str] = {}  # Per request type
        self.broker_weights: Dict[str, float] = {}
        
        # Rate limiting
        self.rate_limiters: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Request queuing and processing
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.processing_requests: Dict[str, Future] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Health monitoring
        self.health_check_interval = self.config.get('health_check_interval', 60.0)
        self.health_monitor_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.performance_metrics: deque = deque(maxlen=10000)
        
        # Circuit breaker
        self.circuit_breaker_threshold = self.config.get('circuit_breaker_threshold', 5)
        self.circuit_breaker_timeout = self.config.get('circuit_breaker_timeout', 300)  # 5 minutes
        self.circuit_breaker_status: Dict[str, Dict] = defaultdict(dict)
        
        self.logger.info("APIRouter initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('APIRouter')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def register_broker(self, broker_config: BrokerConfig, broker_instance: BrokerInterface) -> bool:
        """
        Register a broker with the router
        
        Args:
            broker_config: Broker configuration
            broker_instance: Broker implementation instance
            
        Returns:
            bool: True if successfully registered
        """
        try:
            broker_id = broker_config.broker_id
            
            if broker_id in self.brokers:
                self.logger.warning(f"Broker {broker_id} already registered")
                return False
            
            self.brokers[broker_id] = broker_instance
            self.broker_configs[broker_id] = broker_config
            self.broker_status[broker_id] = BrokerStatus.DISCONNECTED
            self.broker_weights[broker_id] = broker_config.weight
            
            # Initialize circuit breaker
            self.circuit_breaker_status[broker_id] = {
                'failures': 0,
                'last_failure': None,
                'is_open': False
            }
            
            self.logger.info(f"Registered broker: {broker_id} ({broker_config.broker_name})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering broker {broker_config.broker_id}: {e}")
            return False
    
    def unregister_broker(self, broker_id: str) -> bool:
        """
        Unregister a broker from the router
        
        Args:
            broker_id: Broker identifier
            
        Returns:
            bool: True if successfully unregistered
        """
        try:
            if broker_id not in self.brokers:
                self.logger.warning(f"Broker {broker_id} not registered")
                return False
            
            # Disconnect if connected
            if self.broker_status[broker_id] == BrokerStatus.CONNECTED:
                asyncio.create_task(self.brokers[broker_id].disconnect())
            
            # Remove from all dictionaries
            del self.brokers[broker_id]
            del self.broker_configs[broker_id]
            del self.broker_status[broker_id]
            del self.broker_weights[broker_id]
            
            self.logger.info(f"Unregistered broker: {broker_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error unregistering broker {broker_id}: {e}")
            return False
    
    async def start_router(self) -> bool:
        """
        Start the API router
        
        Returns:
            bool: True if started successfully
        """
        try:
            # Connect to all brokers
            connection_tasks = []
            for broker_id, broker in self.brokers.items():
                task = asyncio.create_task(self._connect_broker(broker_id))
                connection_tasks.append(task)
            
            # Wait for all connections
            connection_results = await asyncio.gather(*connection_tasks, return_exceptions=True)
            
            # Check results
            connected_brokers = 0
            for i, result in enumerate(connection_results):
                if isinstance(result, Exception):
                    broker_id = list(self.brokers.keys())[i]
                    self.logger.error(f"Failed to connect to {broker_id}: {result}")
                elif result:
                    connected_brokers += 1
            
            if connected_brokers == 0:
                self.logger.error("Failed to connect to any brokers")
                return False
            
            # Start health monitoring
            self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
            
            # Start request processing
            asyncio.create_task(self._process_request_queue())
            
            self.logger.info(f"Router started with {connected_brokers}/{len(self.brokers)} brokers connected")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting router: {e}")
            return False
    
    async def stop_router(self) -> bool:
        """
        Stop the API router
        
        Returns:
            bool: True if stopped successfully
        """
        try:
            # Cancel health monitoring
            if self.health_monitor_task:
                self.health_monitor_task.cancel()
            
            # Disconnect all brokers
            disconnect_tasks = []
            for broker_id, broker in self.brokers.items():
                if self.broker_status[broker_id] == BrokerStatus.CONNECTED:
                    task = asyncio.create_task(broker.disconnect())
                    disconnect_tasks.append(task)
            
            if disconnect_tasks:
                await asyncio.gather(*disconnect_tasks, return_exceptions=True)
            
            self.logger.info("Router stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping router: {e}")
            return False
    
    async def _connect_broker(self, broker_id: str) -> bool:
        """Connect to individual broker"""
        try:
            broker = self.brokers[broker_id]
            
            if await broker.connect():
                self.broker_status[broker_id] = BrokerStatus.CONNECTED
                self.logger.info(f"Connected to broker: {broker_id}")
                return True
            else:
                self.broker_status[broker_id] = BrokerStatus.ERROR
                self.logger.error(f"Failed to connect to broker: {broker_id}")
                return False
                
        except Exception as e:
            self.broker_status[broker_id] = BrokerStatus.ERROR
            self.logger.error(f"Error connecting to broker {broker_id}: {e}")
            return False
    
    async def route_request(self, request_type: str, request_data: Dict,
                           priority: RequestPriority = RequestPriority.MEDIUM,
                           target_broker: Optional[str] = None) -> Dict:
        """
        Route API request to appropriate broker
        
        Args:
            request_type: Type of request (place_order, get_positions, etc.)
            request_data: Request parameters
            priority: Request priority level
            target_broker: Specific broker to target (optional)
            
        Returns:
            Dict: Response from broker
        """
        start_time = time.time()
        
        try:
            # Select broker
            if target_broker and target_broker in self.brokers:
                selected_broker = target_broker
            else:
                selected_broker = self._select_broker(request_type)
            
            if not selected_broker:
                raise Exception("No available brokers for request")
            
            # Check circuit breaker
            if self._is_circuit_breaker_open(selected_broker):
                # Try alternative broker
                alternative_broker = self._select_alternative_broker(selected_broker, request_type)
                if alternative_broker:
                    selected_broker = alternative_broker
                else:
                    raise Exception(f"Circuit breaker open for {selected_broker} and no alternatives available")
            
            # Check rate limits
            if not self._check_rate_limit(selected_broker):
                # Try alternative broker or queue request
                alternative_broker = self._select_alternative_broker(selected_broker, request_type)
                if alternative_broker:
                    selected_broker = alternative_broker
                else:
                    # Queue request for later processing
                    await self.request_queue.put({
                        'request_type': request_type,
                        'request_data': request_data,
                        'priority': priority,
                        'timestamp': datetime.now()
                    })
                    raise Exception("Request queued due to rate limits")
            
            # Execute request
            response = await self._execute_request(selected_broker, request_type, request_data)
            
            # Record success metrics
            latency = time.time() - start_time
            self._record_metrics(selected_broker, request_type, latency, True)
            
            # Reset circuit breaker on success
            self._reset_circuit_breaker(selected_broker)
            
            return {
                'success': True,
                'data': response,
                'broker': selected_broker,
                'latency': latency
            }
            
        except Exception as e:
            latency = time.time() - start_time
            
            # Record failure metrics
            if 'selected_broker' in locals():
                self._record_metrics(selected_broker, request_type, latency, False, str(e))
                self._update_circuit_breaker(selected_broker)
            
            self.logger.error(f"Request routing failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'broker': selected_broker if 'selected_broker' in locals() else None,
                'latency': latency
            }
    
    def _select_broker(self, request_type: str) -> Optional[str]:
        """Select best broker for request based on routing strategy"""
        available_brokers = [
            broker_id for broker_id, status in self.broker_status.items()
            if status == BrokerStatus.CONNECTED and not self._is_circuit_breaker_open(broker_id)
        ]
        
        if not available_brokers:
            return None
        
        if self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(available_brokers, request_type)
        elif self.routing_strategy == RoutingStrategy.LEAST_LOADED:
            return self._least_loaded_selection(available_brokers)
        elif self.routing_strategy == RoutingStrategy.BEST_LATENCY:
            return self._best_latency_selection(available_brokers, request_type)
        elif self.routing_strategy == RoutingStrategy.WEIGHTED:
            return self._weighted_selection(available_brokers)
        elif self.routing_strategy == RoutingStrategy.PRIMARY_BACKUP:
            return self._primary_backup_selection(available_brokers)
        else:
            return available_brokers[0]  # Default to first available
    
    def _round_robin_selection(self, brokers: List[str], request_type: str) -> str:
        """Round robin broker selection"""
        last_broker = self.last_used_broker.get(request_type)
        
        if last_broker and last_broker in brokers:
            current_index = brokers.index(last_broker)
            next_index = (current_index + 1) % len(brokers)
            selected = brokers[next_index]
        else:
            selected = brokers[0]
        
        self.last_used_broker[request_type] = selected
        return selected
    
    def _least_loaded_selection(self, brokers: List[str]) -> str:
        """Select broker with least load"""
        loads = {broker: self.request_counts[broker] for broker in brokers}
        return min(loads, key=loads.get)
    
    def _best_latency_selection(self, brokers: List[str], request_type: str) -> str:
        """Select broker with best latency for request type"""
        latencies = {}
        
        for broker in brokers:
            recent_metrics = [
                m for m in self.broker_metrics[broker]
                if m.request_type == request_type and m.success
                and m.timestamp > datetime.now() - timedelta(minutes=10)
            ]
            
            if recent_metrics:
                avg_latency = sum(m.latency for m in recent_metrics) / len(recent_metrics)
                latencies[broker] = avg_latency
            else:
                latencies[broker] = float('inf')  # No recent data, lowest priority
        
        return min(latencies, key=latencies.get)
    
    def _weighted_selection(self, brokers: List[str]) -> str:
        """Weighted random selection based on broker weights"""
        import random
        
        weights = [self.broker_weights.get(broker, 1.0) for broker in brokers]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.choice(brokers)
        
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return brokers[i]
        
        return brokers[-1]  # Fallback
    
    def _primary_backup_selection(self, brokers: List[str]) -> str:
        """Primary-backup selection based on priority"""
        # Sort by priority (higher number = higher priority)
        sorted_brokers = sorted(
            brokers,
            key=lambda b: self.broker_configs[b].priority,
            reverse=True
        )
        return sorted_brokers[0]
    
    def _select_alternative_broker(self, exclude_broker: str, request_type: str) -> Optional[str]:
        """Select alternative broker excluding specified one"""
        available_brokers = [
            broker_id for broker_id, status in self.broker_status.items()
            if (status == BrokerStatus.CONNECTED and 
                broker_id != exclude_broker and 
                not self._is_circuit_breaker_open(broker_id))
        ]
        
        return self._select_broker(request_type) if available_brokers else None
    
    def _check_rate_limit(self, broker_id: str) -> bool:
        """Check if broker is within rate limits"""
        config = self.broker_configs[broker_id]
        rate_limiter = self.rate_limiters[broker_id]
        current_time = time.time()
        
        # Remove old requests outside the time window
        while rate_limiter and current_time - rate_limiter[0] > 1.0:
            rate_limiter.popleft()
        
        # Check if under limit
        return len(rate_limiter) < config.max_requests_per_second
    
    def _record_rate_limit_usage(self, broker_id: str) -> None:
        """Record rate limit usage"""
        self.rate_limiters[broker_id].append(time.time())
    
    async def _execute_request(self, broker_id: str, request_type: str, request_data: Dict) -> Any:
        """Execute request on specific broker"""
        broker = self.brokers[broker_id]
        config = self.broker_configs[broker_id]
        
        # Record rate limit usage
        self._record_rate_limit_usage(broker_id)
        
        # Increment request count
        self.request_counts[broker_id] += 1
        
        try:
            # Route to appropriate broker method
            if request_type == 'place_order':
                response = await asyncio.wait_for(
                    broker.place_order(request_data),
                    timeout=config.timeout
                )
            elif request_type == 'cancel_order':
                response = await asyncio.wait_for(
                    broker.cancel_order(request_data.get('order_id')),
                    timeout=config.timeout
                )
            elif request_type == 'get_positions':
                response = await asyncio.wait_for(
                    broker.get_positions(),
                    timeout=config.timeout
                )
            elif request_type == 'get_account_info':
                response = await asyncio.wait_for(
                    broker.get_account_info(),
                    timeout=config.timeout
                )
            elif request_type == 'get_market_data':
                response = await asyncio.wait_for(
                    broker.get_market_data(request_data.get('symbol')),
                    timeout=config.timeout
                )
            else:
                raise ValueError(f"Unknown request type: {request_type}")
            
            return response
            
        except asyncio.TimeoutError:
            raise Exception(f"Request timeout on broker {broker_id}")
        except Exception as e:
            raise Exception(f"Request failed on broker {broker_id}: {e}")
        finally:
            # Decrement request count
            self.request_counts[broker_id] -= 1
    
    def _record_metrics(self, broker_id: str, request_type: str, latency: float, 
                       success: bool, error_message: Optional[str] = None) -> None:
        """Record request metrics"""
        metrics = RequestMetrics(
            broker_id=broker_id,
            request_type=request_type,
            timestamp=datetime.now(),
            latency=latency,
            success=success,
            error_message=error_message
        )
        
        self.broker_metrics[broker_id].append(metrics)
        self.performance_metrics.append(metrics)
    
    def _is_circuit_breaker_open(self, broker_id: str) -> bool:
        """Check if circuit breaker is open for broker"""
        breaker = self.circuit_breaker_status[broker_id]
        
        if not breaker['is_open']:
            return False
        
        # Check if timeout has passed
        if (breaker['last_failure'] and 
            datetime.now() - breaker['last_failure'] > timedelta(seconds=self.circuit_breaker_timeout)):
            # Reset circuit breaker
            breaker['is_open'] = False
            breaker['failures'] = 0
            self.logger.info(f"Circuit breaker reset for broker {broker_id}")
            return False
        
        return True
    
    def _update_circuit_breaker(self, broker_id: str) -> None:
        """Update circuit breaker on failure"""
        breaker = self.circuit_breaker_status[broker_id]
        breaker['failures'] += 1
        breaker['last_failure'] = datetime.now()
        
        if breaker['failures'] >= self.circuit_breaker_threshold:
            breaker['is_open'] = True
            self.logger.warning(f"Circuit breaker opened for broker {broker_id}")
    
    def _reset_circuit_breaker(self, broker_id: str) -> None:
        """Reset circuit breaker on success"""
        breaker = self.circuit_breaker_status[broker_id]
        if breaker['failures'] > 0:
            breaker['failures'] = max(0, breaker['failures'] - 1)
    
    async def _health_monitor_loop(self) -> None:
        """Health monitoring loop"""
        while True:
            try:
                for broker_id, broker in self.brokers.items():
                    if self.broker_status[broker_id] == BrokerStatus.CONNECTED:
                        try:
                            healthy = await asyncio.wait_for(
                                broker.health_check(),
                                timeout=10.0
                            )
                            if not healthy:
                                self.broker_status[broker_id] = BrokerStatus.ERROR
                                self.logger.warning(f"Health check failed for broker {broker_id}")
                        except Exception as e:
                            self.broker_status[broker_id] = BrokerStatus.ERROR
                            self.logger.error(f"Health check error for broker {broker_id}: {e}")
                
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(5)
    
    async def _process_request_queue(self) -> None:
        """Process queued requests"""
        while True:
            try:
                # Get queued request
                queued_request = await self.request_queue.get()
                
                # Route queued request
                response = await self.route_request(
                    queued_request['request_type'],
                    queued_request['request_data'],
                    queued_request['priority']
                )
                
                # Store response for later retrieval if needed
                request_id = hashlib.md5(
                    f"{queued_request['timestamp']}{queued_request['request_type']}".encode()
                ).hexdigest()
                
                self.processing_requests[request_id] = response
                
            except Exception as e:
                self.logger.error(f"Error processing queued request: {e}")
    
    def get_broker_status(self) -> Dict[str, Dict]:
        """Get status of all brokers"""
        status_info = {}
        
        for broker_id in self.brokers:
            recent_metrics = [
                m for m in self.broker_metrics[broker_id]
                if m.timestamp > datetime.now() - timedelta(minutes=5)
            ]
            
            success_rate = (
                sum(1 for m in recent_metrics if m.success) / len(recent_metrics)
                if recent_metrics else 0
            )
            
            avg_latency = (
                sum(m.latency for m in recent_metrics if m.success) / 
                sum(1 for m in recent_metrics if m.success)
                if any(m.success for m in recent_metrics) else 0
            )
            
            status_info[broker_id] = {
                'status': self.broker_status[broker_id].value,
                'recent_requests': len(recent_metrics),
                'success_rate': success_rate,
                'avg_latency': avg_latency,
                'circuit_breaker_open': self._is_circuit_breaker_open(broker_id),
                'current_load': self.request_counts[broker_id]
            }
        
        return status_info
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        if not self.performance_metrics:
            return {'error': 'No performance data available'}
        
        recent_metrics = [
            m for m in self.performance_metrics
            if m.timestamp > datetime.now() - timedelta(hours=1)
        ]
        
        if not recent_metrics:
            return {'error': 'No recent performance data'}
        
        success_rate = sum(1 for m in recent_metrics if m.success) / len(recent_metrics)
        avg_latency = sum(m.latency for m in recent_metrics if m.success) / sum(1 for m in recent_metrics if m.success)
        
        return {
            'total_requests': len(recent_metrics),
            'success_rate': success_rate,
            'average_latency': avg_latency,
            'active_brokers': sum(1 for s in self.broker_status.values() if s == BrokerStatus.CONNECTED),
            'total_brokers': len(self.brokers),
            'routing_strategy': self.routing_strategy.value
        }


# Example usage and testing
if __name__ == "__main__":
    # This would normally be run in an async context
    async def test_router():
        # Initialize router
        router = APIRouter({
            'routing_strategy': 'round_robin',
            'health_check_interval': 30.0,
            'circuit_breaker_threshold': 3
        })
        
        # Register mock brokers (in real implementation, these would be actual broker classes)
        class MockBroker(BrokerInterface):
            def __init__(self, broker_id: str):
                self.broker_id = broker_id
            
            async def connect(self) -> bool:
                return True
            
            async def disconnect(self) -> bool:
                return True
            
            async def health_check(self) -> bool:
                return True
            
            async def place_order(self, order_data: Dict) -> Dict:
                return {'order_id': '12345', 'status': 'filled'}
            
            async def cancel_order(self, order_id: str) -> Dict:
                return {'order_id': order_id, 'status': 'cancelled'}
            
            async def get_positions(self) -> Dict:
                return {'positions': []}
            
            async def get_account_info(self) -> Dict:
                return {'balance': 100000}
            
            async def get_market_data(self, symbol: str) -> Dict:
                return {'symbol': symbol, 'bid': 1.1000, 'ask': 1.1005}
        
        # Register brokers
        broker1_config = BrokerConfig(
            broker_id='broker1',
            broker_name='Test Broker 1',
            api_class='MockBroker',
            priority=1,
            weight=1.0
        )
        
        broker2_config = BrokerConfig(
            broker_id='broker2',
            broker_name='Test Broker 2',
            api_class='MockBroker',
            priority=2,
            weight=1.5
        )
        
        router.register_broker(broker1_config, MockBroker('broker1'))
        router.register_broker(broker2_config, MockBroker('broker2'))
        
        # Start router
        if await router.start_router():
            print("Router started successfully")
            
            # Test requests
            response = await router.route_request('place_order', {
                'symbol': 'EUR/USD',
                'side': 'buy',
                'quantity': 100000
            })
            
            print(f"Order response: {response}")
            
            # Get status
            status = router.get_broker_status()
            print(f"Broker status: {status}")
            
            # Stop router
            await router.stop_router()
        else:
            print("Failed to start router")
    
    # Run test
    asyncio.run(test_router())
