"""
Phase 6.2: Scaling Orchestrator
Dynamic scaling and load management system for production environments.
"""

import os
import sys
import json
import yaml
import logging
import asyncio
import subprocess
import psutil
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone, timedelta
from enum import Enum
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ScalingDirection(Enum):
    """Scaling direction enumeration."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"

class ScalingTrigger(Enum):
    """Scaling trigger enumeration."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    QUEUE_LENGTH = "queue_length"
    CUSTOM_METRIC = "custom_metric"

class ScalingStatus(Enum):
    """Scaling status enumeration."""
    IDLE = "idle"
    MONITORING = "monitoring"
    SCALING_UP = "scaling_up"
    SCALING_DOWN = "scaling_down"
    COOLDOWN = "cooldown"
    ERROR = "error"

@dataclass
class ScalingMetric:
    """Scaling metric definition."""
    name: str
    trigger: ScalingTrigger
    scale_up_threshold: float
    scale_down_threshold: float
    evaluation_period: int = 300  # seconds
    data_points_to_alarm: int = 3
    weight: float = 1.0
    enabled: bool = True

@dataclass
class ScalingPolicy:
    """Scaling policy configuration."""
    name: str
    target_service: str
    min_instances: int
    max_instances: int
    desired_instances: int
    metrics: List[ScalingMetric] = field(default_factory=list)
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds
    scale_up_increment: int = 1
    scale_down_increment: int = 1
    enabled: bool = True

@dataclass
class ScalingEvent:
    """Scaling event record."""
    timestamp: datetime
    service: str
    direction: ScalingDirection
    trigger_metric: str
    current_value: float
    threshold: float
    from_instances: int
    to_instances: int
    reason: str
    success: bool = True
    error: Optional[str] = None

@dataclass
class ServiceInstance:
    """Service instance representation."""
    instance_id: str
    service_name: str
    status: str
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    request_count: int = 0
    response_time: float = 0.0
    last_health_check: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class ScalingOrchestrator:
    """
    Scaling Orchestrator
    Dynamic scaling and load management system for production environments.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/scaling_config.yaml"
        self.scaling_policies: Dict[str, ScalingPolicy] = {}
        self.service_instances: Dict[str, List[ServiceInstance]] = {}
        self.scaling_events: List[ScalingEvent] = []
        self.metric_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.scaling_status = ScalingStatus.IDLE
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.last_scaling_actions: Dict[str, datetime] = {}
        self.scaling_handlers: Dict[str, Callable] = {}
        
    async def initialize(self) -> bool:
        """Initialize the scaling orchestrator."""
        try:
            logger.info("Initializing Scaling Orchestrator...")
            
            # Create necessary directories
            await self._create_directories()
            
            # Load configuration
            await self._load_configuration()
            
            # Initialize scaling policies
            await self._initialize_scaling_policies()
            
            # Initialize metric collection
            await self._initialize_metric_collection()
            
            # Register default scaling handlers
            await self._register_default_handlers()
            
            logger.info("Scaling Orchestrator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Scaling Orchestrator: {e}")
            return False
    
    async def _create_directories(self):
        """Create necessary directories."""
        directories = [
            "logs/scaling", "config", "metrics", "scaling_events", "reports"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    async def _load_configuration(self):
        """Load scaling configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    # Process configuration
            else:
                # Create default configuration
                await self._create_default_config()
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    async def _create_default_config(self):
        """Create default scaling configuration."""
        default_config = {
            "global_settings": {
                "monitoring_interval": 30,
                "metric_retention_hours": 24,
                "max_scaling_events_per_hour": 10,
                "emergency_scaling_enabled": True,
                "notifications_enabled": True
            },
            "default_thresholds": {
                "cpu_scale_up": 70.0,
                "cpu_scale_down": 30.0,
                "memory_scale_up": 80.0,
                "memory_scale_down": 40.0,
                "response_time_scale_up": 2000.0,  # ms
                "response_time_scale_down": 500.0
            },
            "services": {
                "web_server": {
                    "min_instances": 2,
                    "max_instances": 10,
                    "desired_instances": 3
                },
                "api_server": {
                    "min_instances": 1,
                    "max_instances": 5,
                    "desired_instances": 2
                },
                "worker_service": {
                    "min_instances": 1,
                    "max_instances": 8,
                    "desired_instances": 2
                }
            }
        }
        
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
    
    async def _initialize_scaling_policies(self):
        """Initialize default scaling policies."""
        # Web Server Policy
        web_policy = ScalingPolicy(
            name="web_server_policy",
            target_service="web_server",
            min_instances=2,
            max_instances=10,
            desired_instances=3,
            metrics=[
                ScalingMetric(
                    name="cpu_usage",
                    trigger=ScalingTrigger.CPU_USAGE,
                    scale_up_threshold=70.0,
                    scale_down_threshold=30.0,
                    weight=1.0
                ),
                ScalingMetric(
                    name="memory_usage",
                    trigger=ScalingTrigger.MEMORY_USAGE,
                    scale_up_threshold=80.0,
                    scale_down_threshold=40.0,
                    weight=0.8
                ),
                ScalingMetric(
                    name="response_time",
                    trigger=ScalingTrigger.RESPONSE_TIME,
                    scale_up_threshold=2000.0,  # ms
                    scale_down_threshold=500.0,
                    weight=1.2
                )
            ]
        )
        
        # API Server Policy
        api_policy = ScalingPolicy(
            name="api_server_policy",
            target_service="api_server",
            min_instances=1,
            max_instances=5,
            desired_instances=2,
            metrics=[
                ScalingMetric(
                    name="cpu_usage",
                    trigger=ScalingTrigger.CPU_USAGE,
                    scale_up_threshold=75.0,
                    scale_down_threshold=25.0,
                    weight=1.0
                ),
                ScalingMetric(
                    name="request_rate",
                    trigger=ScalingTrigger.REQUEST_RATE,
                    scale_up_threshold=100.0,  # requests/minute
                    scale_down_threshold=20.0,
                    weight=1.5
                )
            ]
        )
        
        # Worker Service Policy
        worker_policy = ScalingPolicy(
            name="worker_service_policy",
            target_service="worker_service",
            min_instances=1,
            max_instances=8,
            desired_instances=2,
            metrics=[
                ScalingMetric(
                    name="queue_length",
                    trigger=ScalingTrigger.QUEUE_LENGTH,
                    scale_up_threshold=50.0,
                    scale_down_threshold=10.0,
                    weight=2.0
                ),
                ScalingMetric(
                    name="cpu_usage",
                    trigger=ScalingTrigger.CPU_USAGE,
                    scale_up_threshold=80.0,
                    scale_down_threshold=20.0,
                    weight=1.0
                )
            ]
        )
        
        self.scaling_policies = {
            "web_server": web_policy,
            "api_server": api_policy,
            "worker_service": worker_policy
        }
    
    async def _initialize_metric_collection(self):
        """Initialize metric collection systems."""
        # Initialize metric history for each service and metric
        for policy in self.scaling_policies.values():
            for metric in policy.metrics:
                metric_key = f"{policy.target_service}:{metric.name}"
                self.metric_history[metric_key] = []
    
    async def _register_default_handlers(self):
        """Register default scaling handlers."""
        self.scaling_handlers = {
            "web_server": self._scale_web_server,
            "api_server": self._scale_api_server,
            "worker_service": self._scale_worker_service
        }
    
    async def start_monitoring(self, interval_seconds: int = 30):
        """Start scaling monitoring."""
        if self.monitoring_active:
            logger.warning("Scaling monitoring is already active")
            return
        
        self.monitoring_active = True
        self.scaling_status = ScalingStatus.MONITORING
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(interval_seconds)
        )
        logger.info(f"Started scaling monitoring (interval: {interval_seconds}s)")
    
    async def stop_monitoring(self):
        """Stop scaling monitoring."""
        if not self.monitoring_active:
            logger.warning("Scaling monitoring is not active")
            return
        
        self.monitoring_active = False
        self.scaling_status = ScalingStatus.IDLE
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped scaling monitoring")
    
    async def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        try:
            while self.monitoring_active:
                logger.debug("Performing scaling evaluation...")
                
                # Collect current metrics
                await self._collect_metrics()
                
                # Evaluate scaling decisions
                await self._evaluate_scaling_decisions()
                
                # Clean up old metrics
                await self._cleanup_old_metrics()
                
                # Wait for next evaluation
                await asyncio.sleep(interval_seconds)
                
        except asyncio.CancelledError:
            logger.info("Scaling monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Scaling monitoring loop error: {e}")
            self.scaling_status = ScalingStatus.ERROR
    
    async def _collect_metrics(self):
        """Collect current metrics for all services."""
        current_time = datetime.now(timezone.utc)
        
        for service_name, policy in self.scaling_policies.items():
            if not policy.enabled:
                continue
                
            try:
                # Get current service instances
                instances = self.service_instances.get(service_name, [])
                
                # Collect metrics for each metric type
                for metric in policy.metrics:
                    if not metric.enabled:
                        continue
                    
                    try:
                        metric_value = await self._collect_metric_value(service_name, metric, instances)
                        metric_key = f"{service_name}:{metric.name}"
                        
                        # Store metric value
                        self.metric_history[metric_key].append((current_time, metric_value))
                        
                        logger.debug(f"Collected metric {metric_key}: {metric_value}")
                        
                    except Exception as e:
                        logger.error(f"Failed to collect metric {metric.name} for {service_name}: {e}")
                        
            except Exception as e:
                logger.error(f"Failed to collect metrics for service {service_name}: {e}")
    
    async def _collect_metric_value(self, service_name: str, metric: ScalingMetric, instances: List[ServiceInstance]) -> float:
        """Collect specific metric value for a service."""
        if metric.trigger == ScalingTrigger.CPU_USAGE:
            return await self._get_cpu_usage(service_name, instances)
        elif metric.trigger == ScalingTrigger.MEMORY_USAGE:
            return await self._get_memory_usage(service_name, instances)
        elif metric.trigger == ScalingTrigger.REQUEST_RATE:
            return await self._get_request_rate(service_name, instances)
        elif metric.trigger == ScalingTrigger.RESPONSE_TIME:
            return await self._get_response_time(service_name, instances)
        elif metric.trigger == ScalingTrigger.QUEUE_LENGTH:
            return await self._get_queue_length(service_name, instances)
        elif metric.trigger == ScalingTrigger.CUSTOM_METRIC:
            return await self._get_custom_metric(service_name, metric.name, instances)
        else:
            logger.warning(f"Unknown metric trigger: {metric.trigger}")
            return 0.0
    
    async def _get_cpu_usage(self, service_name: str, instances: List[ServiceInstance]) -> float:
        """Get average CPU usage for service instances."""
        if not instances:
            # Simulate system CPU usage for demo
            return psutil.cpu_percent(interval=0.1)
        
        # Average CPU usage across instances
        total_cpu = sum(instance.cpu_usage for instance in instances)
        return total_cpu / len(instances)
    
    async def _get_memory_usage(self, service_name: str, instances: List[ServiceInstance]) -> float:
        """Get average memory usage for service instances."""
        if not instances:
            # Simulate system memory usage for demo
            return psutil.virtual_memory().percent
        
        # Average memory usage across instances
        total_memory = sum(instance.memory_usage for instance in instances)
        return total_memory / len(instances)
    
    async def _get_request_rate(self, service_name: str, instances: List[ServiceInstance]) -> float:
        """Get request rate for service instances."""
        if not instances:
            # Simulate request rate
            import random
            return random.uniform(10, 150)
        
        # Sum request counts across instances
        total_requests = sum(instance.request_count for instance in instances)
        return total_requests  # requests per minute
    
    async def _get_response_time(self, service_name: str, instances: List[ServiceInstance]) -> float:
        """Get average response time for service instances."""
        if not instances:
            # Simulate response time
            import random
            return random.uniform(200, 3000)
        
        # Average response time across instances
        response_times = [instance.response_time for instance in instances if instance.response_time > 0]
        return statistics.mean(response_times) if response_times else 0.0
    
    async def _get_queue_length(self, service_name: str, instances: List[ServiceInstance]) -> float:
        """Get queue length for worker services."""
        # Simulate queue length
        import random
        base_queue = random.uniform(5, 80)
        
        # Adjust based on number of instances
        instance_count = len(instances) if instances else 1
        adjusted_queue = base_queue / max(instance_count * 0.5, 1)
        
        return max(adjusted_queue, 0)
    
    async def _get_custom_metric(self, service_name: str, metric_name: str, instances: List[ServiceInstance]) -> float:
        """Get custom metric value."""
        # Placeholder for custom metrics
        return 0.0
    
    async def _evaluate_scaling_decisions(self):
        """Evaluate scaling decisions for all services."""
        for service_name, policy in self.scaling_policies.items():
            if not policy.enabled:
                continue
                
            try:
                # Check cooldown periods
                if not await self._is_scaling_allowed(service_name, policy):
                    continue
                
                # Evaluate each metric
                scaling_signals = []
                for metric in policy.metrics:
                    if not metric.enabled:
                        continue
                    
                    signal = await self._evaluate_metric_signal(service_name, metric, policy)
                    if signal != ScalingDirection.STABLE:
                        scaling_signals.append((metric, signal))
                
                # Make scaling decision based on signals
                if scaling_signals:
                    await self._make_scaling_decision(service_name, policy, scaling_signals)
                    
            except Exception as e:
                logger.error(f"Failed to evaluate scaling for service {service_name}: {e}")
    
    async def _is_scaling_allowed(self, service_name: str, policy: ScalingPolicy) -> bool:
        """Check if scaling is allowed based on cooldown periods."""
        last_action_time = self.last_scaling_actions.get(service_name)
        if not last_action_time:
            return True
        
        current_time = datetime.now(timezone.utc)
        time_since_last_action = (current_time - last_action_time).total_seconds()
        
        # Use the longer cooldown period as minimum
        min_cooldown = max(policy.scale_up_cooldown, policy.scale_down_cooldown)
        
        return time_since_last_action >= min_cooldown
    
    async def _evaluate_metric_signal(self, service_name: str, metric: ScalingMetric, policy: ScalingPolicy) -> ScalingDirection:
        """Evaluate scaling signal for a specific metric."""
        metric_key = f"{service_name}:{metric.name}"
        metric_history = self.metric_history.get(metric_key, [])
        
        if len(metric_history) < metric.data_points_to_alarm:
            return ScalingDirection.STABLE
        
        # Get recent values within evaluation period
        current_time = datetime.now(timezone.utc)
        evaluation_start = current_time - timedelta(seconds=metric.evaluation_period)
        
        recent_values = [
            value for timestamp, value in metric_history
            if timestamp >= evaluation_start
        ]
        
        if len(recent_values) < metric.data_points_to_alarm:
            return ScalingDirection.STABLE
        
        # Calculate average of recent values
        avg_value = statistics.mean(recent_values[-metric.data_points_to_alarm:])
        
        # Determine scaling direction
        if avg_value >= metric.scale_up_threshold:
            logger.info(f"Scale UP signal for {service_name}:{metric.name}: {avg_value:.2f} >= {metric.scale_up_threshold}")
            return ScalingDirection.UP
        elif avg_value <= metric.scale_down_threshold:
            logger.info(f"Scale DOWN signal for {service_name}:{metric.name}: {avg_value:.2f} <= {metric.scale_down_threshold}")
            return ScalingDirection.DOWN
        else:
            return ScalingDirection.STABLE
    
    async def _make_scaling_decision(self, service_name: str, policy: ScalingPolicy, scaling_signals: List[Tuple[ScalingMetric, ScalingDirection]]):
        """Make final scaling decision based on all signals."""
        # Calculate weighted score
        scale_up_score = 0.0
        scale_down_score = 0.0
        
        for metric, direction in scaling_signals:
            if direction == ScalingDirection.UP:
                scale_up_score += metric.weight
            elif direction == ScalingDirection.DOWN:
                scale_down_score += metric.weight
        
        # Get current instance count
        current_instances = len(self.service_instances.get(service_name, []))
        if current_instances == 0:
            current_instances = policy.desired_instances
        
        # Make decision
        if scale_up_score > scale_down_score and current_instances < policy.max_instances:
            # Scale up
            target_instances = min(current_instances + policy.scale_up_increment, policy.max_instances)
            await self._execute_scaling_action(service_name, policy, ScalingDirection.UP, current_instances, target_instances, scaling_signals)
            
        elif scale_down_score > scale_up_score and current_instances > policy.min_instances:
            # Scale down
            target_instances = max(current_instances - policy.scale_down_increment, policy.min_instances)
            await self._execute_scaling_action(service_name, policy, ScalingDirection.DOWN, current_instances, target_instances, scaling_signals)
    
    async def _execute_scaling_action(self, service_name: str, policy: ScalingPolicy, direction: ScalingDirection, 
                                     current_instances: int, target_instances: int, signals: List[Tuple[ScalingMetric, ScalingDirection]]):
        """Execute scaling action."""
        try:
            logger.info(f"Executing scaling action: {service_name} {direction.value} from {current_instances} to {target_instances} instances")
            
            # Update scaling status
            if direction == ScalingDirection.UP:
                self.scaling_status = ScalingStatus.SCALING_UP
            else:
                self.scaling_status = ScalingStatus.SCALING_DOWN
            
            # Get scaling handler
            handler = self.scaling_handlers.get(service_name)
            if not handler:
                logger.error(f"No scaling handler registered for service: {service_name}")
                return
            
            # Execute scaling
            success = await handler(direction, current_instances, target_instances)
            
            # Record scaling event
            primary_signal = max(signals, key=lambda x: x[0].weight)
            metric_key = f"{service_name}:{primary_signal[0].name}"
            recent_values = self.metric_history.get(metric_key, [])
            current_value = recent_values[-1][1] if recent_values else 0.0
            
            event = ScalingEvent(
                timestamp=datetime.now(timezone.utc),
                service=service_name,
                direction=direction,
                trigger_metric=primary_signal[0].name,
                current_value=current_value,
                threshold=primary_signal[0].scale_up_threshold if direction == ScalingDirection.UP else primary_signal[0].scale_down_threshold,
                from_instances=current_instances,
                to_instances=target_instances,
                reason=f"Triggered by {primary_signal[0].name}: {current_value:.2f}",
                success=success
            )
            
            self.scaling_events.append(event)
            
            # Update last scaling time
            self.last_scaling_actions[service_name] = datetime.now(timezone.utc)
            
            # Save scaling event
            await self._save_scaling_event(event)
            
            if success:
                logger.info(f"Scaling action completed successfully for {service_name}")
            else:
                logger.error(f"Scaling action failed for {service_name}")
            
            # Return to monitoring status
            self.scaling_status = ScalingStatus.MONITORING
            
        except Exception as e:
            logger.error(f"Failed to execute scaling action: {e}")
            self.scaling_status = ScalingStatus.ERROR
    
    async def _scale_web_server(self, direction: ScalingDirection, current_instances: int, target_instances: int) -> bool:
        """Scale web server instances."""
        try:
            logger.info(f"Scaling web server: {current_instances} -> {target_instances}")
            
            if direction == ScalingDirection.UP:
                # Add instances
                for i in range(target_instances - current_instances):
                    instance = ServiceInstance(
                        instance_id=f"web_{int(time.time())}_{i}",
                        service_name="web_server",
                        status="running"
                    )
                    
                    if "web_server" not in self.service_instances:
                        self.service_instances["web_server"] = []
                    self.service_instances["web_server"].append(instance)
                    
                    logger.info(f"Added web server instance: {instance.instance_id}")
            
            elif direction == ScalingDirection.DOWN:
                # Remove instances
                instances = self.service_instances.get("web_server", [])
                instances_to_remove = current_instances - target_instances
                
                for i in range(instances_to_remove):
                    if instances:
                        removed_instance = instances.pop()
                        logger.info(f"Removed web server instance: {removed_instance.instance_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale web server: {e}")
            return False
    
    async def _scale_api_server(self, direction: ScalingDirection, current_instances: int, target_instances: int) -> bool:
        """Scale API server instances."""
        try:
            logger.info(f"Scaling API server: {current_instances} -> {target_instances}")
            
            if direction == ScalingDirection.UP:
                # Add instances
                for i in range(target_instances - current_instances):
                    instance = ServiceInstance(
                        instance_id=f"api_{int(time.time())}_{i}",
                        service_name="api_server",
                        status="running"
                    )
                    
                    if "api_server" not in self.service_instances:
                        self.service_instances["api_server"] = []
                    self.service_instances["api_server"].append(instance)
                    
                    logger.info(f"Added API server instance: {instance.instance_id}")
            
            elif direction == ScalingDirection.DOWN:
                # Remove instances
                instances = self.service_instances.get("api_server", [])
                instances_to_remove = current_instances - target_instances
                
                for i in range(instances_to_remove):
                    if instances:
                        removed_instance = instances.pop()
                        logger.info(f"Removed API server instance: {removed_instance.instance_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale API server: {e}")
            return False
    
    async def _scale_worker_service(self, direction: ScalingDirection, current_instances: int, target_instances: int) -> bool:
        """Scale worker service instances."""
        try:
            logger.info(f"Scaling worker service: {current_instances} -> {target_instances}")
            
            if direction == ScalingDirection.UP:
                # Add instances
                for i in range(target_instances - current_instances):
                    instance = ServiceInstance(
                        instance_id=f"worker_{int(time.time())}_{i}",
                        service_name="worker_service",
                        status="running"
                    )
                    
                    if "worker_service" not in self.service_instances:
                        self.service_instances["worker_service"] = []
                    self.service_instances["worker_service"].append(instance)
                    
                    logger.info(f"Added worker service instance: {instance.instance_id}")
            
            elif direction == ScalingDirection.DOWN:
                # Remove instances
                instances = self.service_instances.get("worker_service", [])
                instances_to_remove = current_instances - target_instances
                
                for i in range(instances_to_remove):
                    if instances:
                        removed_instance = instances.pop()
                        logger.info(f"Removed worker service instance: {removed_instance.instance_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale worker service: {e}")
            return False
    
    async def _save_scaling_event(self, event: ScalingEvent):
        """Save scaling event to file."""
        try:
            event_data = {
                "timestamp": event.timestamp.isoformat(),
                "service": event.service,
                "direction": event.direction.value,
                "trigger_metric": event.trigger_metric,
                "current_value": event.current_value,
                "threshold": event.threshold,
                "from_instances": event.from_instances,
                "to_instances": event.to_instances,
                "reason": event.reason,
                "success": event.success,
                "error": event.error
            }
            
            event_file = f"scaling_events/scaling_event_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(event_file, 'w') as f:
                json.dump(event_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save scaling event: {e}")
    
    async def _cleanup_old_metrics(self):
        """Clean up old metric data."""
        try:
            current_time = datetime.now(timezone.utc)
            retention_period = timedelta(hours=24)
            cutoff_time = current_time - retention_period
            
            for metric_key, history in self.metric_history.items():
                # Remove old entries
                self.metric_history[metric_key] = [
                    (timestamp, value) for timestamp, value in history
                    if timestamp >= cutoff_time
                ]
                
        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {e}")
    
    def register_scaling_handler(self, service_name: str, handler: Callable):
        """Register custom scaling handler for a service."""
        self.scaling_handlers[service_name] = handler
        logger.info(f"Registered scaling handler for service: {service_name}")
    
    def add_scaling_policy(self, policy: ScalingPolicy):
        """Add new scaling policy."""
        self.scaling_policies[policy.target_service] = policy
        
        # Initialize metric history for new policy
        for metric in policy.metrics:
            metric_key = f"{policy.target_service}:{metric.name}"
            if metric_key not in self.metric_history:
                self.metric_history[metric_key] = []
        
        logger.info(f"Added scaling policy for service: {policy.target_service}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        service_status = {}
        for service_name, instances in self.service_instances.items():
            policy = self.scaling_policies.get(service_name)
            service_status[service_name] = {
                "current_instances": len(instances),
                "min_instances": policy.min_instances if policy else 0,
                "max_instances": policy.max_instances if policy else 0,
                "desired_instances": policy.desired_instances if policy else 0,
                "policy_enabled": policy.enabled if policy else False
            }
        
        return {
            "overall_status": self.scaling_status.value,
            "monitoring_active": self.monitoring_active,
            "total_scaling_events": len(self.scaling_events),
            "recent_events": len([e for e in self.scaling_events if (datetime.now(timezone.utc) - e.timestamp).total_seconds() < 3600]),
            "services": service_status
        }
    
    def get_scaling_metrics(self, service_name: str, metric_name: str, hours: int = 1) -> List[Tuple[datetime, float]]:
        """Get scaling metrics for a service."""
        metric_key = f"{service_name}:{metric_name}"
        history = self.metric_history.get(metric_key, [])
        
        # Filter by time range
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [(timestamp, value) for timestamp, value in history if timestamp >= cutoff_time]
    
    def get_scaling_events(self, service_name: Optional[str] = None, hours: int = 24) -> List[ScalingEvent]:
        """Get scaling events."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        events = [e for e in self.scaling_events if e.timestamp >= cutoff_time]
        
        if service_name:
            events = [e for e in events if e.service == service_name]
        
        return sorted(events, key=lambda x: x.timestamp, reverse=True)

async def main():
    """Main execution function for testing."""
    orchestrator = ScalingOrchestrator()
    
    # Initialize
    if not await orchestrator.initialize():
        print("Failed to initialize scaling orchestrator")
        return
    
    # Start monitoring
    await orchestrator.start_monitoring(interval_seconds=10)
    
    try:
        # Let it run for a bit to collect metrics and potentially trigger scaling
        print("Scaling orchestrator is running. Monitoring for 60 seconds...")
        await asyncio.sleep(60)
        
        # Print status
        status = orchestrator.get_scaling_status()
        print(f"\nScaling Status:")
        print(f"Overall Status: {status['overall_status']}")
        print(f"Monitoring Active: {status['monitoring_active']}")
        print(f"Total Scaling Events: {status['total_scaling_events']}")
        print(f"Recent Events (1h): {status['recent_events']}")
        
        print(f"\nService Status:")
        for service, info in status['services'].items():
            print(f"  {service}: {info['current_instances']}/{info['max_instances']} instances")
        
        # Show recent events
        recent_events = orchestrator.get_scaling_events(hours=1)
        if recent_events:
            print(f"\nRecent Scaling Events:")
            for event in recent_events[:5]:  # Show last 5 events
                print(f"  {event.timestamp.strftime('%H:%M:%S')} - {event.service}: "
                      f"{event.direction.value} ({event.from_instances} -> {event.to_instances}) "
                      f"- {event.reason}")
    
    finally:
        # Stop monitoring
        await orchestrator.stop_monitoring()
        print("Scaling orchestrator stopped")

if __name__ == "__main__":
    asyncio.run(main())
