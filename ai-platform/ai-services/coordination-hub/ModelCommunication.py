"""
🔗 MODEL COMMUNICATION PROTOCOL FOR HUMANITARIAN TRADING MISSION
===============================================================

Inter-model communication system for coordinated AI decisions
Ensures all 25+ AI models work together for maximum humanitarian impact

Mission: Generate $300,000+ monthly profits for:
- Emergency medical aid for the poor and sick  
- Surgical operations for children
- Global poverty alleviation

This protocol enables seamless communication between all AI models
to optimize coordination and maximize charitable impact.
"""


import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from enum import Enum
import uuid
import time
import base64
from collections import defaultdict, deque
import threading
import weakref

# Import the new real-time infrastructure modules
from .realtime import *  # Real-time communication infrastructure
try:
    from .realtime.agent_health_monitor import (
        AgentHealthMonitor, AgentHealthStatus, AgentHealthMetrics, PerformanceMetric
    )
    from .realtime.websocket_agent_server import (
        WebSocketAgentServer, AgentConnection as WSAgentConnection, AgentConnectionState
    )
    from .realtime.message_queue_manager import (
        MessageQueueManager, MessageQueueType, MessagePriority as QueuePriority
    )
    REALTIME_INFRASTRUCTURE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Real-time infrastructure not available: {e}")
    REALTIME_INFRASTRUCTURE_AVAILABLE = False

# Import security infrastructure
try:
    from ....shared.security.agent_authentication_manager import (
        AgentSecurityManager, AgentIdentity, MessageSecurity, AgentPermission,
        SecurityAuditEntry, create_agent_security_manager
    )
    SECURITY_INFRASTRUCTURE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Security infrastructure not available: {e}")
    SECURITY_INFRASTRUCTURE_AVAILABLE = False

# Import persistence infrastructure
try:
    from ....shared.persistence.agent_state_manager import (
        AgentStateManager, StateType, StateSnapshot, create_agent_state_manager
    )
    from .AgentPersistenceCoordinator import (
        AgentPersistenceCoordinator, TransactionType, create_agent_persistence_coordinator
    )
    PERSISTENCE_INFRASTRUCTURE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Persistence infrastructure not available: {e}")
    PERSISTENCE_INFRASTRUCTURE_AVAILABLE = False

# Import configuration coordination infrastructure
try:
    from ...config.AgentConfigCoordinator import (
        AgentConfigCoordinator, ConfigChangeType, ConfigSyncMode, AgentConfigProfile
    )
    CONFIG_COORDINATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Configuration coordination infrastructure not available: {e}")
    CONFIG_COORDINATION_AVAILABLE = False

# Import collaboration monitoring infrastructure
try:
    from ...monitoring.AgentCollaborationMonitor import (
        AgentCollaborationMonitor, get_collaboration_monitor,
        MetricType, AlertLevel, CollaborationMetric
    )
    COLLABORATION_MONITORING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Collaboration monitoring infrastructure not available: {e}")
    COLLABORATION_MONITORING_AVAILABLE = False

# Import disaster recovery infrastructure
try:
    from ....shared.disaster_recovery.agent_cluster_recovery import (
        AgentClusterRecovery, RecoveryType, RecoveryStatus, AgentClusterState,
        create_agent_cluster_recovery
    )
    DISASTER_RECOVERY_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Disaster recovery infrastructure not available: {e}")
    DISASTER_RECOVERY_AVAILABLE = False

# Import Platform3 Communication Framework for enhanced service coordination
try:
    from ....shared.communication.platform3_communication_framework import (
        Platform3CommunicationFramework, ServiceType, ServiceStatus, MessagePriority as FrameworkPriority
    )
    PLATFORM3_FRAMEWORK_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Platform3 Communication Framework not available: {e}")
    PLATFORM3_FRAMEWORK_AVAILABLE = False

# Optional imports - will gracefully degrade if not available
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    
try:
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    
try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

# Import data structures
try:
    from AICoordinator import MarketContext, TradingSignals, RiskAssessment, UnifiedPrediction
except ImportError:
    # Define minimal data structures if AICoordinator is not available
    MarketContext = dict
    TradingSignals = dict
    RiskAssessment = dict
    UnifiedPrediction = dict

class MessageType(Enum):
    """Types of messages in the communication protocol"""
    MARKET_UPDATE = "market_update"
    PREDICTION_REQUEST = "prediction_request"
    PREDICTION_RESPONSE = "prediction_response"
    PERFORMANCE_FEEDBACK = "performance_feedback"
    ADAPTATION_SIGNAL = "adaptation_signal"
    ENSEMBLE_WEIGHT_UPDATE = "ensemble_weight_update"
    RISK_ALERT = "risk_alert"
    HUMANITARIAN_PRIORITY = "humanitarian_priority"
    COORDINATION_SYNC = "coordination_sync"

class MessagePriority(Enum):
    """Message priority levels for humanitarian mission"""
    CRITICAL = 1    # Emergency risk alerts, system failures
    HIGH = 2        # Market updates, urgent trading signals
    MEDIUM = 3      # Performance feedback, adaptation signals
    LOW = 4         # Routine updates, analytics

@dataclass
class CommunicationMessage:
    """Standard communication message between AI models"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.MARKET_UPDATE
    priority: MessagePriority = MessagePriority.MEDIUM
    sender: str = "unknown"
    recipient: str = "broadcast"
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    humanitarian_context: bool = True  # Always true for our mission
    requires_response: bool = False
    correlation_id: Optional[str] = None

@dataclass
class AgentConnection:
    """WebSocket connection information for an agent"""
    agent_id: str
    websocket: Optional[Any] = None
    last_heartbeat: datetime = field(default_factory=datetime.now)
    status: str = "connected"
    message_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue())
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class HeartbeatConfig:
    """Heartbeat monitoring configuration"""
    interval: int = 30  # seconds
    timeout: int = 10   # seconds
    max_missed: int = 3 # missed heartbeats before declaring unhealthy

@dataclass
class MessageDeliveryStatus:
    """Track message delivery status"""
    message_id: str
    status: str  # "pending", "delivered", "failed", "timeout"
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    error_message: Optional[str] = None

@dataclass 
class PerformanceFeedback:
    """Performance feedback for model adaptation"""
    model_name: str
    trade_results: Dict[str, Any]
    charitable_impact: float  # Actual charitable impact achieved
    accuracy: float
    profitability: float
    risk_metrics: Dict[str, float]
    adaptation_recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AdaptationPlan:
    """Adaptation plan for model improvements"""
    priorities: List[Dict[str, Any]]
    humanitarian_optimization: Dict[str, Any]
    resource_allocation: Dict[str, float]
    timeline: Dict[str, datetime]
    expected_impact: float

# ============================================================================
# AGENT DEPENDENCY RESOLUTION SYSTEM
# ============================================================================

@dataclass
class DependencyRequest:
    """Request for data from dependent agents"""
    request_id: str
    requesting_agent: str
    target_agent: str
    data_type: str
    request_data: Dict[str, Any]
    timeout_ms: int
    timestamp: datetime = field(default_factory=datetime.now)
    priority: str = "MEDIUM"

@dataclass
class DependencyResponse:
    """Response from agent with requested data"""
    request_id: str
    responding_agent: str
    success: bool
    data: Optional[Dict[str, Any]]
    error_message: Optional[str] = None
    response_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class AgentDependencyGraph:
    """
    🕸️ AGENT DEPENDENCY GRAPH MANAGER
    
    Manages the dependency relationships between all 9 genius agents
    and provides real-time dependency traversal capabilities.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.dependencies: Dict[str, List[str]] = {}
        self.dependents: Dict[str, List[str]] = {}
        self.agent_capabilities: Dict[str, Set[str]] = {}
        
        # Initialize agent dependencies based on genius agent registry
        self._initialize_agent_dependencies()
    
    def _initialize_agent_dependencies(self):
        """Initialize the dependency graph with known agent relationships"""
        # Agent dependencies from genius_agent_registry.py
        agent_deps = {
            "risk_genius": [],
            "session_expert": [],
            "pattern_master": [],
            "execution_expert": ["risk_genius", "pattern_master"],
            "pair_specialist": ["session_expert"],
            "decision_master": ["risk_genius", "pattern_master", "execution_expert"],
            "ai_model_coordinator": ["risk_genius", "session_expert", "pattern_master", 
                                   "execution_expert", "pair_specialist", "decision_master",
                                   "market_microstructure_genius", "sentiment_integration_genius"],
            "market_microstructure_genius": ["execution_expert"],
            "sentiment_integration_genius": ["decision_master"]
        }
        
        # Build dependency and dependent maps
        for agent, deps in agent_deps.items():
            self.dependencies[agent] = deps
            for dep in deps:
                if dep not in self.dependents:
                    self.dependents[dep] = []
                self.dependents[dep].append(agent)
        
        # Define agent capabilities (what data they can provide)
        self.agent_capabilities = {
            "risk_genius": {"risk_assessment", "volatility_metrics", "risk_alerts"},
            "session_expert": {"session_analysis", "market_sessions", "timezone_optimization"},
            "pattern_master": {"pattern_recognition", "technical_patterns", "trend_analysis"},
            "execution_expert": {"execution_data", "order_management", "slippage_metrics"},
            "pair_specialist": {"pair_analysis", "currency_correlations", "spread_data"},
            "decision_master": {"trading_decisions", "confidence_scores", "decision_validation"},
            "ai_model_coordinator": {"model_coordination", "ensemble_weights", "system_health"},
            "market_microstructure_genius": {"microstructure_data", "order_flow", "market_depth"},
            "sentiment_integration_genius": {"sentiment_analysis", "news_impact", "sentiment_scores"}
        }
        
        self.logger.info(f"🕸️ Agent dependency graph initialized with {len(agent_deps)} agents")
    
    def get_dependencies(self, agent_id: str) -> List[str]:
        """Get direct dependencies for an agent"""
        return self.dependencies.get(agent_id, [])
    
    def get_dependents(self, agent_id: str) -> List[str]:
        """Get agents that depend on this agent"""
        return self.dependents.get(agent_id, [])
    
    def can_provide_data(self, agent_id: str, data_type: str) -> bool:
        """Check if an agent can provide specific data type"""
        capabilities = self.agent_capabilities.get(agent_id, set())
        return data_type in capabilities
    
    def find_data_provider(self, data_type: str) -> Optional[str]:
        """Find which agent can provide specific data type"""
        for agent_id, capabilities in self.agent_capabilities.items():
            if data_type in capabilities:
                return agent_id
        return None
    
    def get_dependency_chain(self, agent_id: str) -> List[str]:
        """Get the full dependency chain for an agent (BFS traversal)"""
        visited = set()
        queue = deque([agent_id])
        chain = []
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            
            visited.add(current)
            chain.append(current)
            
            # Add dependencies to queue
            deps = self.get_dependencies(current)
            for dep in deps:
                if dep not in visited:
                    queue.append(dep)
        
        return chain[1:]  # Exclude the agent itself

class DependencyResolver:
    """
    🔗 RUNTIME AGENT DEPENDENCY RESOLVER
    
    Implements active runtime dependency resolution between agents.
    When an agent needs data from its dependencies, this resolver
    automatically coordinates the request/response flow.
    """
    
    def __init__(self, websocket_server: Optional['WebSocketAgentServer'] = None):
        self.logger = logging.getLogger(__name__)
        self.dependency_graph = AgentDependencyGraph()
        self.websocket_server = websocket_server
        self.pending_requests: Dict[str, DependencyRequest] = {}
        self.request_timeout_tasks: Dict[str, asyncio.Task] = {}
        self.response_callbacks: Dict[str, Callable] = {}
        
        # Metrics
        self.resolution_metrics = {
            "total_requests": 0,
            "successful_resolutions": 0,
            "failed_resolutions": 0,
            "average_response_time_ms": 0.0,
            "timeout_count": 0
        }
        
        self.logger.info("🔗 DependencyResolver initialized")
    
    async def resolve_agent_dependencies(self, agent_id: str, request_data: Dict[str, Any], 
                                       timeout_ms: int = 5000) -> Dict[str, Any]:
        """
        🎯 RESOLVE ALL DEPENDENCIES FOR AN AGENT
        
        Main entry point for dependency resolution. When an agent needs data
        from its dependencies, this method coordinates the entire resolution process.
        """
        start_time = time.time()
        self.logger.info(f"🔗 Resolving dependencies for {agent_id}")
        
        try:
            dependencies = self.dependency_graph.get_dependencies(agent_id)
            if not dependencies:
                self.logger.info(f"🔗 No dependencies for {agent_id}")
                return {}
            
            # Create requests for all dependencies
            requests = []
            for dep_agent in dependencies:
                request_id = str(uuid.uuid4())
                dep_request = DependencyRequest(
                    request_id=request_id,
                    requesting_agent=agent_id,
                    target_agent=dep_agent,
                    data_type=self._determine_data_type(agent_id, dep_agent),
                    request_data=request_data,
                    timeout_ms=timeout_ms
                )
                requests.append(dep_request)
                self.pending_requests[request_id] = dep_request
            
            # Send requests concurrently
            response_tasks = []
            for req in requests:
                task = asyncio.create_task(self._send_dependency_request(req))
                response_tasks.append(task)
            
            # Wait for all responses or timeout
            responses = await asyncio.gather(*response_tasks, return_exceptions=True)
            
            # Process responses
            resolved_data = {}
            successful_count = 0
            
            for i, response in enumerate(responses):
                req = requests[i]
                if isinstance(response, DependencyResponse) and response.success:
                    resolved_data[req.target_agent] = response.data
                    successful_count += 1
                else:
                    self.logger.warning(f"🔗 Failed to resolve dependency: {req.target_agent}")
                    resolved_data[req.target_agent] = None
            
            # Update metrics
            total_time_ms = (time.time() - start_time) * 1000
            self._update_resolution_metrics(successful_count, len(requests), total_time_ms)
            
            self.logger.info(f"🔗 Resolved {successful_count}/{len(requests)} dependencies for {agent_id} in {total_time_ms:.1f}ms")
            return resolved_data
            
        except Exception as e:
            self.logger.error(f"🔗 Error resolving dependencies for {agent_id}: {e}")
            self.metrics["failed_resolutions"] += 1
            return {}
    
    async def _send_dependency_request(self, request: DependencyRequest) -> DependencyResponse:
        """Send a dependency request to target agent via WebSocket"""
        try:
            if not self.websocket_server:
                raise Exception("WebSocket server not available")
            
            # Prepare request message
            message = {
                "type": "dependency_request",
                "request_id": request.request_id,
                "requesting_agent": request.requesting_agent,
                "data_type": request.data_type,
                "request_data": request.request_data,
                "timestamp": request.timestamp.isoformat()
            }
            
            # Set up response waiting
            response_future = asyncio.Future()
            self.response_callbacks[request.request_id] = response_future.set_result
            
            # Set up timeout
            timeout_task = asyncio.create_task(self._handle_request_timeout(request.request_id, request.timeout_ms))
            self.request_timeout_tasks[request.request_id] = timeout_task
            
            # Send request via WebSocket
            success = await self.websocket_server.send_to_agent(request.target_agent, message)
            if not success:
                raise Exception(f"Failed to send request to {request.target_agent}")
            
            # Wait for response or timeout
            try:
                response = await response_future
                timeout_task.cancel()
                return response
            except asyncio.TimeoutError:
                self.logger.warning(f"🔗 Timeout waiting for response from {request.target_agent}")
                self.metrics["timeout_count"] += 1
                return DependencyResponse(
                    request_id=request.request_id,
                    responding_agent=request.target_agent,
                    success=False,
                    data=None,
                    error_message="Request timeout"
                )
            
        except Exception as e:
            self.logger.error(f"🔗 Error sending dependency request: {e}")
            return DependencyResponse(
                request_id=request.request_id,
                responding_agent=request.target_agent,
                success=False,
                data=None,
                error_message=str(e)
            )
        finally:
            # Cleanup
            self.pending_requests.pop(request.request_id, None)
            self.response_callbacks.pop(request.request_id, None)
            if request.request_id in self.request_timeout_tasks:
                self.request_timeout_tasks[request.request_id].cancel()
                del self.request_timeout_tasks[request.request_id]
    
    async def handle_dependency_response(self, response_data: Dict[str, Any]):
        """Handle incoming dependency response from an agent"""
        try:
            request_id = response_data.get("request_id")
            if not request_id or request_id not in self.response_callbacks:
                self.logger.warning(f"🔗 Received response for unknown request: {request_id}")
                return
            
            response = DependencyResponse(
                request_id=request_id,
                responding_agent=response_data.get("responding_agent"),
                success=response_data.get("success", False),
                data=response_data.get("data"),
                error_message=response_data.get("error_message"),
                response_time_ms=response_data.get("response_time_ms", 0.0)
            )
            
            # Trigger the waiting future
            callback = self.response_callbacks.get(request_id)
            if callback:
                callback(response)
            
        except Exception as e:
            self.logger.error(f"🔗 Error handling dependency response: {e}")
    
    async def _handle_request_timeout(self, request_id: str, timeout_ms: int):
        """Handle request timeout"""
        await asyncio.sleep(timeout_ms / 1000.0)
        
        callback = self.response_callbacks.get(request_id)
        if callback:
            timeout_response = DependencyResponse(
                request_id=request_id,
                responding_agent="unknown",
                success=False,
                data=None,
                error_message="Request timeout"
            )
            callback(timeout_response)
    
    def _determine_data_type(self, requesting_agent: str, target_agent: str) -> str:
        """Determine what data type the requesting agent needs from target agent"""
        # Map based on agent capabilities and typical data flows
        data_type_map = {
            ("decision_master", "risk_genius"): "risk_assessment",
            ("decision_master", "pattern_master"): "pattern_recognition", 
            ("decision_master", "execution_expert"): "execution_data",
            ("execution_expert", "risk_genius"): "risk_metrics",
            ("execution_expert", "pattern_master"): "technical_patterns",
            ("ai_model_coordinator", "decision_master"): "trading_decisions",
            ("market_microstructure_genius", "execution_expert"): "order_management",
            ("sentiment_integration_genius", "decision_master"): "decision_validation",
            ("pair_specialist", "session_expert"): "session_analysis"
        }
        
        return data_type_map.get((requesting_agent, target_agent), "general_data")
    
    def _update_resolution_metrics(self, successful: int, total: int, response_time_ms: float):
        """Update dependency resolution metrics"""
        self.resolution_metrics["total_requests"] += total
        self.resolution_metrics["successful_resolutions"] += successful
        self.resolution_metrics["failed_resolutions"] += (total - successful)
        
        # Update average response time
        current_avg = self.resolution_metrics["average_response_time_ms"]
        current_total = self.resolution_metrics["total_requests"]
        new_avg = ((current_avg * (current_total - total)) + response_time_ms) / current_total
        self.resolution_metrics["average_response_time_ms"] = new_avg
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get dependency resolution metrics"""
        success_rate = 0.0
        if self.resolution_metrics["total_requests"] > 0:
            success_rate = (self.resolution_metrics["successful_resolutions"] / 
                          self.resolution_metrics["total_requests"]) * 100
        
        return {
            **self.resolution_metrics,
            "success_rate_percent": success_rate
        }

# ============================================================================

class ModelCommunicationProtocol:
    """
    🔗 INTER-MODEL COMMUNICATION FOR COORDINATED HUMANITARIAN AI
    
    Enhanced with real-time WebSocket communication, message queuing,
    heartbeat monitoring, and connection recovery for robust agent coordination.
    
    Manages communication between all AI models in the platform:
    - Real-time bidirectional WebSocket channels
    - Redis/Kafka message queuing for reliability
    - Heartbeat monitoring and health tracking
    - Automatic connection recovery
    - Distributed event bus for coordination
    - Performance metrics and monitoring
      
    Ensures all models work together optimally for humanitarian mission success.
    """
    
    def __init__(self, start_background_tasks=True, redis_url="redis://localhost:6379", 
                 kafka_bootstrap_servers="localhost:9092", websocket_port=8765):
        """Initialize the enhanced communication protocol"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("🔗 Initializing Enhanced Model Communication Protocol")
        
        # Initialize the new real-time infrastructure components
        if REALTIME_INFRASTRUCTURE_AVAILABLE:
            self.logger.info("🚀 Initializing real-time infrastructure components")
            
            # Agent Health Monitor for comprehensive health tracking with auto-recovery
            recovery_config = {
                'auto_recovery_enabled': True,
                'max_recovery_attempts': 3,
                'recovery_timeout_seconds': 60,
                'escalation_threshold': 5,
                'cooldown_period_minutes': 10
            }
            health_monitor_config = {
                'heartbeat_interval': 30,
                'health_check_interval': 60,
                'recovery_config': recovery_config
            }
            self.health_monitor = AgentHealthMonitor(health_monitor_config)
            
            # WebSocket Agent Server for real-time communication
            self.websocket_server_component = WebSocketAgentServer(
                host="localhost",
                port=websocket_port,
                message_handler=self._handle_websocket_message,
                connection_handler=self._handle_websocket_connection
            )
            
            # Message Queue Manager for reliable messaging
            self.message_queue_manager = MessageQueueManager(
                redis_url=redis_url,
                kafka_bootstrap_servers=kafka_bootstrap_servers,
                enable_redis=True,
                enable_kafka=True
            )
            
            # Dependency Resolver for runtime agent coordination
            self.dependency_resolver = DependencyResolver(
                websocket_server=self.websocket_server_component
            )
            
            self.logger.info("✅ Real-time infrastructure components initialized")
        else:
            self.logger.warning("⚠️ Real-time infrastructure not available - using legacy mode")
            self.health_monitor = None
            self.websocket_server_component = None
            self.message_queue_manager = None
            self.dependency_resolver = None
        
        # Initialize Security Manager for agent authentication and authorization
        if SECURITY_INFRASTRUCTURE_AVAILABLE:
            self.logger.info("🔐 Initializing Agent Security Manager")
            try:
                self.security_manager = create_agent_security_manager()
                self.security_enabled = True
                self.logger.info("✅ Agent Security Manager initialized with mTLS integration")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Agent Security Manager: {e}")
                self.security_manager = None
                self.security_enabled = False
        else:
            self.logger.warning("⚠️ Security infrastructure not available - running in insecure mode")
            self.security_manager = None
            self.security_enabled = False
        
        # Initialize Agent State Persistence for agent state management and recovery
        if PERSISTENCE_INFRASTRUCTURE_AVAILABLE:
            self.logger.info("💾 Initializing Agent State Persistence System")
            try:
                self.state_manager = None  # Will be initialized async
                self.persistence_coordinator = None  # Will be initialized async
                self.persistence_enabled = True
                self.auto_snapshot_interval_hours = 6  # Create snapshots every 6 hours
                self.state_backup_enabled = True
                self.logger.info("✅ Agent State Persistence System ready for initialization")
            except Exception as e:
                self.logger.error(f"❌ Failed to prepare Agent State Persistence System: {e}")
                self.state_manager = None
                self.persistence_coordinator = None
                self.persistence_enabled = False
        else:
            self.logger.warning("⚠️ Persistence infrastructure not available - states will not be saved")
            self.state_manager = None
            self.persistence_coordinator = None
            self.persistence_enabled = False
        
        # Initialize Disaster Recovery System for agent cluster recovery
        if DISASTER_RECOVERY_AVAILABLE:
            self.logger.info("🚑 Initializing Disaster Recovery System")
            try:
                self.disaster_recovery = None  # Will be initialized async
                self.disaster_recovery_enabled = True
                self.recovery_time_target_minutes = 15  # 15-minute recovery target
                self.auto_backup_interval_hours = 4  # Create backups every 4 hours
                self.recovery_validation_timeout_minutes = 5
                self.logger.info("✅ Disaster Recovery System ready for initialization")
            except Exception as e:
                self.logger.error(f"❌ Failed to prepare Disaster Recovery System: {e}")
                self.disaster_recovery = None
                self.disaster_recovery_enabled = False
        else:
            self.logger.warning("⚠️ Disaster Recovery infrastructure not available - no recovery capabilities")
            self.disaster_recovery = None
            self.disaster_recovery_enabled = False
        
        # Initialize Platform3 Communication Framework integration
        if PLATFORM3_FRAMEWORK_AVAILABLE:
            self.logger.info("🌐 Initializing Platform3 Communication Framework integration")
            try:
                self.platform3_framework = Platform3CommunicationFramework({
                    'redis_host': redis_url.split('://')[1].split(':')[0] if '://' in redis_url else 'localhost',
                    'redis_port': int(redis_url.split(':')[-1]) if ':' in redis_url else 6379,
                    'load_balancer_strategy': 'weighted'
                })
                self.framework_service_id = None  # Will be set during service registration
                self.logger.info("✅ Platform3 Communication Framework integration initialized")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Platform3 Communication Framework: {e}")
                self.platform3_framework = None
        else:
            self.logger.warning("⚠️ Platform3 Communication Framework not available")
            self.platform3_framework = None
        
        # Initialize Agent Configuration Coordinator for dynamic config management
        if CONFIG_COORDINATION_AVAILABLE:
            self.logger.info("⚙️ Initializing Agent Configuration Coordinator")
            try:
                self.config_coordinator = None  # Will be initialized async
                self.config_coordination_enabled = True
                self.config_change_timeout_seconds = 10
                self.auto_config_validation = True
                self.logger.info("✅ Agent Configuration Coordinator ready for initialization")
            except Exception as e:
                self.logger.error(f"❌ Failed to prepare Agent Configuration Coordinator: {e}")
                self.config_coordinator = None
                self.config_coordination_enabled = False
        else:
            self.logger.warning("⚠️ Configuration coordination infrastructure not available")
            self.config_coordinator = None
            self.config_coordination_enabled = False
        
        # WebSocket server configuration (legacy)
        self.websocket_port = websocket_port
        self.websocket_server = None
        self.agent_connections = {}  # agent_id -> AgentConnection
        self.connection_lock = asyncio.Lock()
        
        # Redis configuration for message queuing (legacy)
        self.redis_url = redis_url
        self.redis_client = None
        self.redis_pubsub = None
        
        # Kafka configuration for distributed messaging (legacy)
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.kafka_producer = None
        self.kafka_consumer = None
        self.kafka_topics = {
            'agent_communication': 'platform3_agent_comm',
            'market_updates': 'platform3_market_updates',
            'health_monitoring': 'platform3_health'
        }
        
        # Heartbeat monitoring
        self.heartbeat_config = HeartbeatConfig()
        self.heartbeat_status = {}  # agent_id -> last_heartbeat_time
        self.health_monitor_active = True
        
        # Message delivery tracking
        self.message_delivery_status = {}  # message_id -> MessageDeliveryStatus
        self.delivery_timeout = 30  # seconds
        
        # Event bus for distributed coordination
        self.event_subscribers = defaultdict(list)  # event_type -> [callback_functions]
        self.event_history = []
        self.max_event_history = 1000
        
        # Message queues for different priorities
        self.message_queues = {
            MessagePriority.CRITICAL: asyncio.Queue(maxsize=100),
            MessagePriority.HIGH: asyncio.Queue(maxsize=500),  
            MessagePriority.MEDIUM: asyncio.Queue(maxsize=1000),
            MessagePriority.LOW: asyncio.Queue(maxsize=2000)
        }
        
        # Registered models and their communication handlers
        self.registered_models = {}
        self.model_capabilities = {}
        self.message_handlers = {}
        
        # Performance tracking
        self.communication_stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'failed_deliveries': 0,
            'average_latency': 0.0
        }
        
        # Humanitarian mission settings
        self.humanitarian_mode = True
        self.charitable_impact_tracking = True
        self.risk_protection_enabled = True
        
        # Communication history for analysis
        self.message_history = []
        self.max_history_size = 10000
        
        # Start background processing
        self._processing_active = True
          # Only start background tasks if requested and if we have an event loop
        if start_background_tasks:
            try:
                self._start_background_tasks()
            except RuntimeError as e:
                if "no running event loop" in str(e):
                    self.logger.info("⏳ Background tasks will start when event loop is available")
                else:
                    raise
        
        self.logger.info("✅ Enhanced Model Communication Protocol initialized for humanitarian mission")
    
    def _start_background_tasks(self):
        """Start background communication processing tasks"""
        # Initialize the new infrastructure components if available
        if REALTIME_INFRASTRUCTURE_AVAILABLE and self.health_monitor:
            asyncio.create_task(self.health_monitor.start_monitoring())
            asyncio.create_task(self.websocket_server_component.start_server())
            asyncio.create_task(self.message_queue_manager.initialize())
        
        # Initialize persistence infrastructure if available
        if PERSISTENCE_INFRASTRUCTURE_AVAILABLE and self.persistence_enabled:
            asyncio.create_task(self._initialize_persistence_infrastructure())
        
        # Initialize Platform3 Communication Framework if available
        if PLATFORM3_FRAMEWORK_AVAILABLE and self.platform3_framework:
            asyncio.create_task(self._initialize_platform3_framework())
        
        # Initialize Agent Configuration Coordinator if available
        if CONFIG_COORDINATION_AVAILABLE and self.config_coordination_enabled:
            asyncio.create_task(self._initialize_config_coordination())
        
        # Initialize Disaster Recovery System if available
        if DISASTER_RECOVERY_AVAILABLE and self.disaster_recovery_enabled:
            asyncio.create_task(self._initialize_disaster_recovery_system())
        
        # Start legacy background tasks
        asyncio.create_task(self.initialize_infrastructure())
        asyncio.create_task(self._process_message_queues())
        asyncio.create_task(self._monitor_communication_health())
        asyncio.create_task(self._optimize_humanitarian_coordination())
        asyncio.create_task(self.start_heartbeat_monitoring())
        asyncio.create_task(self._cleanup_message_delivery_status())
        asyncio.create_task(self._process_redis_messages())
        
    async def _initialize_platform3_framework(self):
        """Initialize the Platform3 Communication Framework integration"""
        try:
            self.logger.info("🌐 Initializing Platform3 Communication Framework...")
            
            # Initialize the framework
            await self.platform3_framework.initialize()
            
            # Register this service with the framework
            self.framework_service_id = self.platform3_framework.register_service(
                service_type=ServiceType.AI_MODEL_SERVICE,
                host="localhost",
                port=self.websocket_port,
                version="1.0.0",
                health_check_url=f"http://localhost:{self.websocket_port}/health",
                metadata={
                    "service_name": "agent_communication_protocol",
                    "description": "Enhanced Runtime Agent Communication System",
                    "humanitarian_mission": True,
                    "agents_supported": True
                },
                tags=["coordination", "humanitarian", "agents", "realtime"]
            )
            
            self.logger.info(f"✅ Service registered with Platform3 framework: {self.framework_service_id}")
            
            # Setup event subscribers for framework integration
            asyncio.create_task(self._setup_framework_event_subscribers())
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Platform3 framework: {e}")
            # Don't raise - allow system to work in degraded mode
    
    async def _setup_framework_event_subscribers(self):
        """Setup event subscribers for inter-service communication"""
        try:
            # Subscribe to relevant events for agent coordination
            await self.subscribe_to_event("service_registry_update", self._handle_service_registry_update)
            await self.subscribe_to_event("agent_health_update", self._handle_agent_health_update_event)
            await self.subscribe_to_event("market_data_update", self._handle_market_data_update_event)
            
            # Publish initial service status
            await self.publish_event("agent_coordinator_status", {
                "status": "online",
                "websocket_port": self.websocket_port,
                "agents_connected": len(self.agent_connections),
                "timestamp": datetime.now().isoformat()
            })
            
            self.logger.info("✅ Framework event subscribers configured")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup framework event subscribers: {e}")
            # Don't raise - allow system to work in degraded mode
            
    async def _initialize_config_coordination(self):
        """Initialize the Agent Configuration Coordinator"""
        try:
            self.logger.info("⚙️ Initializing Agent Configuration Coordinator...")
            
            # Initialize the configuration coordinator
            from ...config.production_config_manager import Environment
            self.config_coordinator = AgentConfigCoordinator(Environment.PRODUCTION)
            
            # Subscribe to configuration change events
            await self.subscribe_to_event("agent_config_change", self._handle_agent_config_change)
            await self.subscribe_to_event("config_rollback_request", self._handle_config_rollback_request)
            
            # Register configuration change notification handler
            asyncio.create_task(self._monitor_config_changes())
            
            # Get initial configuration states for all agents
            registered_agents = self.get_registered_agents()
            for agent_id in registered_agents:
                try:
                    config_profile = await self.config_coordinator.get_agent_config(agent_id)
                    if config_profile:
                        self.logger.info(f"📝 Loaded configuration for agent: {agent_id}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not load config for agent {agent_id}: {e}")
            
            self.logger.info(f"✅ Agent Configuration Coordinator initialized with {len(registered_agents)} agents")
            
            # Publish configuration coordination availability to event bus
            await self.publish_event("config_coordination_status", {
                "status": "online",
                "agents_registered": len(registered_agents),
                "sync_mode": self.config_coordinator.sync_mode.value,
                "validation_enabled": self.config_coordinator.validation_enabled,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize configuration coordination: {e}")
            self.config_coordination_enabled = False
            # Don't raise - allow system to work in degraded mode
    
    async def _initialize_disaster_recovery_system(self):
        """Initialize the Disaster Recovery System for agent cluster recovery"""
        try:
            self.logger.info("🚑 Initializing Disaster Recovery System...")
            
            # Initialize the disaster recovery system
            self.disaster_recovery = await create_agent_cluster_recovery(
                backup_base_path="d:/Platform3/backups/agent_clusters",
                postgres_config=self.postgres_config if hasattr(self, 'postgres_config') else None,
                redis_config=self.redis_config if hasattr(self, 'redis_config') else None
            )
            
            # Subscribe to system failure events
            await self.subscribe_to_event("agent_cluster_failure", self._handle_cluster_failure)
            await self.subscribe_to_event("recovery_request", self._handle_recovery_request)
            await self.subscribe_to_event("backup_request", self._handle_backup_request)
            
            # Start background disaster recovery monitoring
            asyncio.create_task(self._monitor_disaster_recovery())
            asyncio.create_task(self._run_periodic_cluster_backups())
            
            # Create initial cluster backup
            try:
                initial_backup = await self.disaster_recovery.create_cluster_backup("main_cluster")
                self.logger.info(f"📦 Created initial cluster backup: {initial_backup.cluster_id}")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not create initial cluster backup: {e}")
            
            # Get registered agents for disaster recovery coverage
            registered_agents = self.get_registered_agents()
            
            self.logger.info(f"✅ Disaster Recovery System initialized for {len(registered_agents)} agents")
            
            # Publish disaster recovery availability to event bus
            await self.publish_event("disaster_recovery_status", {
                "status": "online",
                "agents_covered": len(registered_agents),
                "recovery_target_minutes": self.recovery_time_target_minutes,
                "backup_interval_hours": self.auto_backup_interval_hours,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize disaster recovery system: {e}")
            self.disaster_recovery_enabled = False
            # Don't raise - allow system to work in degraded mode
            
    async def _initialize_persistence_infrastructure(self):
        """Initialize the Agent State Persistence and Recovery System"""
        try:
            self.logger.info("💾 Initializing Agent State Persistence infrastructure...")
            
            # Initialize the state manager
            self.state_manager = await create_agent_state_manager()
            
            # Initialize the persistence coordinator with our state manager
            self.persistence_coordinator = await create_agent_persistence_coordinator(self.state_manager)
            
            # Configure auto-snapshot intervals
            self.auto_snapshot_enabled = True
            self.auto_snapshot_timer = time.time()
            
            # Register agents for state persistence based on genius_agent_registry
            registered_agents = self.get_registered_agents()
            for agent_id in registered_agents:
                # Create initial state snapshots for each agent
                await self._create_agent_initial_state(agent_id)
            
            # Start background state persistence tasks
            asyncio.create_task(self._run_periodic_state_snapshots())
            asyncio.create_task(self._monitor_state_consistency())
            
            self.logger.info(f"✅ Agent State Persistence infrastructure initialized with {len(registered_agents)} agents")
            
            # Publish state persistence availability to event bus
            await self.publish_event("persistence_system_status", {
                "status": "online",
                "agents_registered": len(registered_agents),
                "auto_snapshot_interval_hours": self.auto_snapshot_interval_hours,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize persistence infrastructure: {e}")
            self.persistence_enabled = False
            # Don't raise - allow system to work in degraded mode
            
    async def _create_agent_initial_state(self, agent_id: str):
        """Create initial state records for an agent"""
        try:
            # Create configuration state
            config_state = self._get_agent_configuration(agent_id)
            if config_state:
                await self.state_manager.save_agent_state(
                    agent_name=agent_id,
                    state_type=StateType.CONFIGURATION,
                    state_data=config_state,
                    version="1.0",
                    tags=["initial", "configuration"]
                )
            
            # Create communication state
            comm_state = self._get_agent_communication_state(agent_id)
            if comm_state:
                await self.state_manager.save_agent_state(
                    agent_name=agent_id,
                    state_type=StateType.COMMUNICATION_STATE,
                    state_data=comm_state,
                    version="1.0",
                    tags=["initial", "communication"]
                )
                
            # Create dependency state
            dep_state = self._get_agent_dependency_state(agent_id)
            if dep_state:
                await self.state_manager.save_agent_state(
                    agent_name=agent_id,
                    state_type=StateType.DEPENDENCY_STATE,
                    state_data=dep_state,
                    version="1.0",
                    tags=["initial", "dependency"]
                )
                
            self.logger.debug(f"Created initial states for agent: {agent_id}")
            return True
        
        except Exception as e:
            self.logger.warning(f"Failed to create initial states for agent {agent_id}: {e}")
            return False
    
    async def _run_periodic_state_snapshots(self):
        """Background task to create periodic state snapshots of all agents"""
        while self.persistence_enabled and self._processing_active:
            try:
                # Only create snapshots if enough time has passed
                current_time = time.time()
                if current_time - self.auto_snapshot_timer >= self.auto_snapshot_interval_hours * 3600:
                    self.logger.info(f"📸 Creating periodic agent state snapshots")
                    
                    # Get all registered agents
                    agents = self.get_registered_agents()
                    
                    # Create distributed snapshot through coordinator
                    if agents and self.persistence_coordinator:
                        await self.persistence_coordinator.create_distributed_snapshot(agents)
                        
                    # Reset timer
                    self.auto_snapshot_timer = current_time
                    
                    self.logger.info(f"✅ Created periodic state snapshots for {len(agents)} agents")
                
                # Sleep until next check
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in periodic state snapshot task: {e}")
                await asyncio.sleep(600)  # Retry after 10 minutes on error
    
    async def _monitor_state_consistency(self):
        """Background task to monitor and maintain state consistency across agents"""
        while self.persistence_enabled and self._processing_active:
            try:
                # Run state consistency check every hour
                await asyncio.sleep(3600)
                
                if not self.persistence_coordinator:
                    continue
                
                # Check for inconsistencies between dependent agents
                self.logger.debug("Running agent state consistency check")
                
                # Get agent dependency map
                dependency_map = self.dependency_resolver.get_agent_dependency_graph()
                
                # Check each agent's dependency states
                for agent_id, dependencies in dependency_map.items():
                    if not dependencies:
                        continue
                        
                    # If agent has dependencies, check if states are in sync
                    try:
                        # Only sync communication and dependency states periodically
                        await self.persistence_coordinator.synchronize_agent_states(
                            primary_agent=agent_id,
                            dependent_agents=dependencies,
                            state_types=[StateType.COMMUNICATION_STATE, StateType.DEPENDENCY_STATE],
                            sync_direction="bidirectional"
                        )
                    except Exception as e:
                        self.logger.warning(f"State synchronization failed for {agent_id}: {e}")
                
            except Exception as e:
                self.logger.error(f"Error in state consistency monitoring: {e}")
                await asyncio.sleep(1800)  # Retry after 30 minutes on error
                
    def _get_agent_configuration(self, agent_id: str) -> Dict[str, Any]:
        """Get configuration state for an agent"""
        # Base configuration that all agents have
        config = {
            "agent_id": agent_id,
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
            "humanitarian_mode": self.humanitarian_mode,
            "communication_protocol_version": "3.0",
            "heartbeat_interval_seconds": self.heartbeat_config.interval,
            "system_defaults": {
                "message_retry_count": 3,
                "timeout_seconds": 30,
                "priority_boost_humanitarian": True
            }
        }
        
        # Agent-specific configuration
        if agent_id == "decision_master":
            config.update({
                "risk_threshold": 0.75,
                "decision_timeout_ms": 2000,
                "mission_priority": "humanitarian_impact"
            })
        elif agent_id == "risk_genius":
            config.update({
                "risk_models": ["market_volatility", "catastrophic_loss", "gradient_protection"],
                "max_acceptable_var": 0.15,
                "update_frequency_ms": 500
            })
        elif agent_id == "pattern_master":
            config.update({
                "pattern_detection_models": ["trend", "reversal", "volatility", "momentum"],
                "realtime_analysis": True,
                "minimum_pattern_strength": 0.65
            })
        elif agent_id == "execution_expert":
            config.update({
                "slippage_tolerance": 0.005,
                "execution_modes": ["instant", "twap", "vwap", "iceberg"],
                "default_execution_mode": "instant"
            })
        
        return config
    
    def _get_agent_communication_state(self, agent_id: str) -> Dict[str, Any]:
        """Get communication state for an agent"""
        return {
            "agent_id": agent_id,
            "last_communication": datetime.now().isoformat(),
            "active_connections": [],
            "message_queues": {
                "inbound": 0,
                "outbound": 0,
                "priority_high": 0,
                "priority_medium": 0,
                "priority_low": 0
            },
            "last_heartbeat": datetime.now().isoformat(),
            "status": "online"
        }
    
    def _get_agent_dependency_state(self, agent_id: str) -> Dict[str, Any]:
        """Get dependency state for an agent"""
        dependency_graph = self.dependency_resolver.get_agent_dependency_graph() if self.dependency_resolver else {}
        dependencies = dependency_graph.get(agent_id, [])
        
        return {
            "agent_id": agent_id,
            "dependencies": dependencies,
            "dependency_states": {dep: "unknown" for dep in dependencies},
            "last_sync": datetime.now().isoformat(),
            "dependency_data": {}
        }
    
    # Public API for persistence functions
    
    async def save_agent_state(self, agent_id: str, state_type: str, state_data: Dict[str, Any]) -> Optional[str]:
        """Save agent state (public API method)"""
        if not self.persistence_enabled or not self.state_manager:
            self.logger.warning(f"Cannot save state: Persistence not enabled")
            return None
        
        try:
            # Convert string state_type to enum
            enum_state_type = getattr(StateType, state_type.upper(), None)
            if not enum_state_type:
                raise ValueError(f"Invalid state type: {state_type}")
                
            state_id = await self.state_manager.save_agent_state(
                agent_name=agent_id,
                state_type=enum_state_type,
                state_data=state_data,
                version=str(time.time()),
                tags=["api_save"]
            )
            
            return state_id
            
        except Exception as e:
            self.logger.error(f"Error saving agent state: {e}")
            return None
    
    async def load_agent_state(self, agent_id: str, state_type: str) -> Optional[Dict[str, Any]]:
        """Load agent state (public API method)"""
        if not self.persistence_enabled or not self.state_manager:
            self.logger.warning(f"Cannot load state: Persistence not enabled")
            return None
            
        try:
            # Convert string state_type to enum
            enum_state_type = getattr(StateType, state_type.upper(), None)
            if not enum_state_type:
                raise ValueError(f"Invalid state type: {state_type}")
                
            result = await self.state_manager.load_agent_state(
                agent_name=agent_id,
                state_type=enum_state_type
            )
            
            if result:
                state_data, _ = result
                return state_data
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading agent state: {e}")
            return None
            
    async def create_agent_snapshot(self, agent_id: str) -> Optional[str]:
        """Create a snapshot of all agent states (public API method)"""
        if not self.persistence_enabled or not self.state_manager:
            self.logger.warning(f"Cannot create snapshot: Persistence not enabled")
            return None
            
        try:
            snapshot_id = await self.state_manager.create_agent_snapshot(agent_id)
            self.logger.info(f"Created snapshot for agent {agent_id}: {snapshot_id}")
            return snapshot_id
            
        except Exception as e:
            self.logger.error(f"Error creating agent snapshot: {e}")
            return None
    
    async def restore_agent_from_snapshot(self, agent_id: str, snapshot_id: str) -> bool:
        """Restore agent state from a snapshot (public API method)"""
        if not self.persistence_enabled or not self.state_manager:
            self.logger.warning(f"Cannot restore snapshot: Persistence not enabled")
            return False
            
        try:
            result = await self.state_manager.restore_agent_from_snapshot(agent_id, snapshot_id)
            if result:
                self.logger.info(f"Successfully restored agent {agent_id} from snapshot {snapshot_id}")
            else:
                self.logger.warning(f"Failed to restore agent {agent_id} from snapshot {snapshot_id}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error restoring agent from snapshot: {e}")
            return False
    
    async def synchronize_agent_states(self, primary_agent: str, dependent_agents: List[str], 
                                      state_types: List[str], sync_direction: str) -> bool:
        """Synchronize states between agents (public API method)"""
        if not self.persistence_enabled or not self.persistence_coordinator:
            self.logger.warning(f"Cannot synchronize states: Persistence not enabled")
            return False
            
        try:
            # Convert string state_types to enum
            enum_state_types = []
            for state_type in state_types:
                enum_type = getattr(StateType, state_type.upper(), None)
                if enum_type:
                    enum_state_types.append(enum_type)
                    
            if not enum_state_types:
                raise ValueError(f"No valid state types in: {state_types}")
                
            sync_id = await self.persistence_coordinator.synchronize_agent_states(
                primary_agent=primary_agent,
                dependent_agents=dependent_agents,
                state_types=enum_state_types,
                sync_direction=sync_direction
            )
            
            self.logger.info(f"Successfully synchronized states, sync_id: {sync_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error synchronizing agent states: {e}")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup framework event subscribers: {e}")
    
    async def _handle_service_registry_update(self, event_data: Dict[str, Any]):
        """Handle service registry update events"""
        self.logger.debug(f"Service registry update received: {event_data}")
        # Implementation to handle service changes
    
    async def _handle_agent_health_update_event(self, event_data: Dict[str, Any]):
        """Handle agent health update events from the framework"""
        agent_id = event_data.get('agent_id')
        health_status = event_data.get('health_status')
        
        if agent_id and health_status:
            self.logger.info(f"Health update from framework for agent {agent_id}: {health_status}")
            
            # Update health monitor with this information
            if self.health_monitor and agent_id in self.health_monitor.agents:
                await self.health_monitor.update_agent_status(
                    agent_id, 
                    AgentHealthStatus(health_status) if isinstance(health_status, str) else health_status
                )
    
    async def _handle_market_data_update_event(self, event_data: Dict[str, Any]):
        """Handle market data update events for agent coordination"""
        # Process market updates and coordinate agents as needed
        if 'market_data' in event_data:
            await self.broadcast_market_update(event_data['market_data'])
    
    # Callback handlers for the new infrastructure components
    async def _handle_agent_health_alert(self, agent_id: str, health_status: AgentHealthStatus, metrics: AgentHealthMetrics):
        """Handle health alerts from the agent health monitor"""
        self.logger.warning(f"🚨 Health alert for agent {agent_id}: {health_status.value}")
        
        # Broadcast health alert to other agents
        alert_data = {
            'agent_id': agent_id,
            'health_status': health_status.value,
            'metrics': {
                'response_time_ms': metrics.response_time_ms,
                'error_rate_percent': metrics.error_rate_percent,
                'cpu_usage_percent': metrics.cpu_usage_percent,
                'memory_usage_mb': metrics.memory_usage_mb
            },
            'timestamp': datetime.now().isoformat()
        }
        
        await self.broadcast_message(
            MessageType.RISK_ALERT,
            alert_data,
            priority=MessagePriority.HIGH,
            sender="health_monitor"
        )
        
        # Take corrective action based on health status
        if health_status == AgentHealthStatus.CRITICAL:
            await self._handle_critical_agent_failure(agent_id)
        elif health_status == AgentHealthStatus.WARNING:
            await self._handle_agent_warning(agent_id, metrics)
    
    async def _handle_websocket_message(self, agent_id: str, message: Dict[str, Any]):
        """Handle incoming WebSocket messages from agents with security validation"""
        try:
            # Security validation if security manager is available
            if self.security_enabled and self.security_manager:
                # Check if agent is authenticated
                agent_status = self.security_manager.get_agent_security_status(agent_id)
                if not agent_status:
                    self.logger.warning(f"🔒 Rejecting message from unauthenticated agent: {agent_id}")
                    return
                
                # Decrypt message if it's encrypted
                if isinstance(message, dict) and 'encrypted_message' in message:
                    try:
                        # Convert to MessageSecurity object for decryption
                        secure_msg = MessageSecurity(**message['encrypted_message'])
                        decrypted_message = self.security_manager.decrypt_message(agent_id, secure_msg)
                        if not decrypted_message:
                            self.logger.error(f"🔒 Failed to decrypt message from agent: {agent_id}")
                            return
                        message = decrypted_message
                    except Exception as e:
                        self.logger.error(f"🔒 Message decryption error for agent {agent_id}: {e}")
                        return
            
            message_type = message.get('type', 'unknown')
            payload = message.get('payload', {})
            
            # Additional authorization checks based on message type
            if self.security_enabled and self.security_manager:
                required_permission = self._get_required_permission_for_message_type(message_type)
                if required_permission and not self.security_manager.authorize_action(agent_id, required_permission):
                    self.logger.warning(f"🔒 Agent {agent_id} unauthorized for message type: {message_type}")
                    return
            
            # Route to appropriate handler
            if message_type == 'heartbeat':
                await self.handle_heartbeat(agent_id, payload)
            elif message_type == 'agent_communication':
                await self.handle_agent_communication(agent_id, payload)
            elif message_type == 'status_update':
                await self.handle_status_update(agent_id, payload)
            elif message_type == 'performance_metrics':
                await self._handle_performance_metrics(agent_id, payload)
            elif message_type == 'dependency_request':
                await self._handle_dependency_request(agent_id, message)
            elif message_type == 'dependency_response':
                await self._handle_dependency_response(agent_id, message)
            else:
                self.logger.warning(f"Unknown WebSocket message type from {agent_id}: {message_type}")
                
        except Exception as e:
            self.logger.error(f"Error handling WebSocket message from {agent_id}: {e}")
    
    async def _handle_websocket_connection(self, agent_id: str, connection_state: AgentConnectionState):
        """Handle WebSocket connection state changes"""
        self.logger.info(f"🔗 Agent {agent_id} connection state: {connection_state.value}")
        
        if connection_state == AgentConnectionState.CONNECTED:
            # Register the agent in the health monitor
            if self.health_monitor:
                await self.health_monitor.register_agent(agent_id)
            
            # Notify other agents about the new connection
            await self.broadcast_message(
                MessageType.COORDINATION_SYNC,
                {
                    'action': 'agent_connected',
                    'agent_id': agent_id,
                    'timestamp': datetime.now().isoformat()
                },
                priority=MessagePriority.MEDIUM,
                sender="connection_manager"
            )
        
        elif connection_state == AgentConnectionState.DISCONNECTED:
            # Unregister from health monitor
            if self.health_monitor:
                await self.health_monitor.unregister_agent(agent_id)
            
            # Notify about disconnection
            await self.broadcast_message(
                MessageType.COORDINATION_SYNC,
                {
                    'action': 'agent_disconnected',
                    'agent_id': agent_id,
                    'timestamp': datetime.now().isoformat()
                },
                priority=MessagePriority.MEDIUM,
                sender="connection_manager"
            )
    
    async def _handle_performance_metrics(self, agent_id: str, metrics: Dict[str, Any]):
        """Handle performance metrics from agents"""
        if self.health_monitor:
            # Update health monitor with new metrics
            health_metrics = AgentHealthMetrics(
                agent_id=agent_id,
                status=AgentHealthStatus.HEALTHY,  # Will be determined by monitor
                last_heartbeat=datetime.now(),
                response_time_ms=metrics.get('response_time_ms', 0.0),
                throughput_per_minute=metrics.get('throughput_per_minute', 0.0),
                error_rate_percent=metrics.get('error_rate_percent', 0.0),
                cpu_usage_percent=metrics.get('cpu_usage_percent', 0.0),
                memory_usage_mb=metrics.get('memory_usage_mb', 0.0),
                accuracy_percent=metrics.get('accuracy_percent', 0.0),
                profit_contribution=metrics.get('profit_contribution', 0.0),
                updated_at=datetime.now()
            )
            await self.health_monitor.update_agent_metrics(agent_id, health_metrics)
    
    async def _handle_critical_agent_failure(self, agent_id: str):
        """Handle critical agent failure"""
        self.logger.error(f"🚨 CRITICAL: Agent {agent_id} has failed")
        
        # Attempt to failover to backup agents
        await self.broadcast_message(
            MessageType.RISK_ALERT,
            {
                'alert_level': 'CRITICAL',
                'failed_agent': agent_id,
                'action_required': 'immediate_failover',
                'humanitarian_protection': True
            },
            priority=MessagePriority.CRITICAL,
            sender="failover_manager"
        )
    
    async def _handle_agent_warning(self, agent_id: str, metrics: AgentHealthMetrics):
        """Handle agent warning conditions"""
        self.logger.warning(f"⚠️ Agent {agent_id} performance warning")
        
        # Send optimization signal to the agent
        await self.send_message_to_model(
            agent_id,
            MessageType.ADAPTATION_SIGNAL,
            {
                'optimization_needed': True,
                'current_metrics': {
                    'response_time': metrics.response_time_ms,
                    'error_rate': metrics.error_rate_percent
                },
                'recommended_actions': ['reduce_load', 'optimize_performance']
            },
            priority=MessagePriority.HIGH
        )
    
    async def _handle_dependency_request(self, agent_id: str, message: Dict[str, Any]):
        """
        🔗 Handle incoming dependency request from an agent
        
        When an agent receives a dependency request, it should respond with
        the requested data from its capabilities.
        """
        try:
            request_id = message.get('request_id')
            requesting_agent = message.get('requesting_agent')
            data_type = message.get('data_type')
            request_data = message.get('request_data', {})
            
            self.logger.info(f"🔗 Received dependency request: {requesting_agent} -> {agent_id} ({data_type})")
            
            # For now, simulate data response based on agent capabilities
            # In a real implementation, this would call the actual agent's data provider
            response_data = await self._generate_agent_response_data(agent_id, data_type, request_data)
            
            # Send response back to requesting agent
            response_message = {
                "type": "dependency_response",
                "request_id": request_id,
                "responding_agent": agent_id,
                "success": response_data is not None,
                "data": response_data,
                "error_message": None if response_data else f"Unable to provide {data_type}",
                "response_time_ms": 10.0,  # Simulated response time
                "timestamp": datetime.now().isoformat()
            }
            
            if self.websocket_server_component:
                await self.websocket_server_component.send_to_agent(requesting_agent, response_message)
            
        except Exception as e:
            self.logger.error(f"🔗 Error handling dependency request: {e}")
            
            # Send error response
            error_response = {
                "type": "dependency_response",
                "request_id": message.get('request_id'),
                "responding_agent": agent_id,
                "success": False,
                "data": None,
                "error_message": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            if self.websocket_server_component:
                requesting_agent = message.get('requesting_agent')
                await self.websocket_server_component.send_to_agent(requesting_agent, error_response)
    
    async def _handle_dependency_response(self, agent_id: str, message: Dict[str, Any]):
        """
        🔗 Handle incoming dependency response from an agent
        
        Forward the response to the dependency resolver for processing.
        """
        try:
            if self.dependency_resolver:
                await self.dependency_resolver.handle_dependency_response(message)
            else:
                self.logger.warning("🔗 Dependency resolver not available")
                
        except Exception as e:
            self.logger.error(f"🔗 Error handling dependency response: {e}")
    
    async def _generate_agent_response_data(self, agent_id: str, data_type: str, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        🎯 Generate response data based on agent capabilities
        
        This is a simulation - in a real implementation, this would interface
        with the actual agent's data providers and models.
        """
        try:
            # Simulate agent-specific data based on their capabilities
            if agent_id == "risk_genius":
                if data_type == "risk_assessment":
                    return {
                        "risk_level": "MEDIUM",
                        "volatility_score": 0.65,
                        "risk_factors": ["market_volatility", "liquidity_risk"],
                        "risk_limits": {"max_position": 10000, "max_drawdown": 0.05},
                        "timestamp": datetime.now().isoformat()
                    }
                elif data_type == "risk_metrics":
                    return {
                        "var_95": 2500.0,
                        "expected_shortfall": 3200.0,
                        "sharpe_ratio": 1.85,
                        "max_drawdown": 0.12,
                        "timestamp": datetime.now().isoformat()
                    }
            
            elif agent_id == "pattern_master":
                if data_type == "pattern_recognition":
                    return {
                        "detected_patterns": ["ascending_triangle", "support_level"],
                        "pattern_confidence": 0.78,
                        "trend_direction": "bullish",
                        "support_levels": [1.2150, 1.2100],
                        "resistance_levels": [1.2250, 1.2300],
                        "timestamp": datetime.now().isoformat()
                    }
                elif data_type == "technical_patterns":
                    return {
                        "patterns": {
                            "head_and_shoulders": {"probability": 0.15, "target": 1.2050},
                            "double_bottom": {"probability": 0.72, "target": 1.2280}
                        },
                        "trend_strength": 0.68,
                        "timestamp": datetime.now().isoformat()
                    }
            
            elif agent_id == "execution_expert":
                if data_type == "execution_data":
                    return {
                        "optimal_execution_time": "2025-06-04T10:30:00Z",
                        "expected_slippage": 0.0015,
                        "liquidity_score": 0.85,
                        "market_impact": 0.002,
                        "execution_venues": ["venue_a", "venue_b"],
                        "timestamp": datetime.now().isoformat()
                    }
                elif data_type == "order_management":
                    return {
                        "order_status": "READY",
                        "execution_strategy": "TWAP",
                        "time_horizon": 300,  # seconds
                        "slice_size": 1000,
                        "timestamp": datetime.now().isoformat()
                    }
            
            elif agent_id == "session_expert":
                if data_type == "session_analysis":
                    return {
                        "current_session": "London",
                        "session_volatility": 0.45,
                        "optimal_trading_hours": ["09:00-11:00", "14:00-16:00"],
                        "session_overlap": "London-NY",
                        "timestamp": datetime.now().isoformat()
                    }
            
            elif agent_id == "decision_master":
                if data_type == "trading_decisions":
                    return {
                        "decision": "BUY",
                        "confidence": 0.82,
                        "position_size": 5000,
                        "entry_price": 1.2185,
                        "stop_loss": 1.2150,
                        "take_profit": 1.2250,
                        "timestamp": datetime.now().isoformat()
                    }
                elif data_type == "decision_validation":
                    return {
                        "validation_status": "APPROVED",
                        "validation_score": 0.91,
                        "risk_approval": True,
                        "compliance_check": True,
                        "timestamp": datetime.now().isoformat()
                    }
            
            else:
                # Generic data for other agents
                return {
                    "agent_id": agent_id,
                    "data_type": data_type,
                    "status": "available",
                    "data": request_data,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"🔗 Error generating response data for {agent_id}: {e}")
            return None
    
    async def register_model(self, model_name: str, model_instance: Any, capabilities: List[str]):
        """
        Register an AI model for communication
        
        Args:
            model_name: Unique name for the model
            model_instance: The actual model instance
            capabilities: List of capabilities (e.g., ['scalping', 'pattern_recognition'])
        """
        self.logger.info(f"📝 Registering model: {model_name}")
        
        self.registered_models[model_name] = model_instance
        self.model_capabilities[model_name] = capabilities
        
        # Set up message handler for this model
        if hasattr(model_instance, 'handle_message'):
            self.message_handlers[model_name] = model_instance.handle_message
        else:
            self.message_handlers[model_name] = self._default_message_handler
        
        # Send registration confirmation
        await self.broadcast_message(
            MessageType.COORDINATION_SYNC,
            {'action': 'model_registered', 'model': model_name, 'capabilities': capabilities},
            priority=MessagePriority.MEDIUM,
            sender="communication_protocol"
        )
        
        self.logger.info(f"✅ Model {model_name} registered successfully")
    
    async def broadcast_market_update(self, market_data: Dict[str, Any]):
        """
        🌍 BROADCAST MARKET UPDATES TO ALL AI MODELS
        
        Distributes critical market data to all registered models
        for coordinated humanitarian trading decisions
        """
        self.logger.info("📡 Broadcasting market update to all AI models")
        
        message_data = {
            'market_data': market_data,
            'humanitarian_context': {
                'mission_priority': 'HIGH',
                'charitable_focus': True,
                'fund_protection': True
            },
            'timestamp': datetime.now().isoformat(),
            'update_type': 'real_time_market_data'
        }
        
        # Broadcast to all models with high priority
        await self.broadcast_message(
            MessageType.MARKET_UPDATE,
            message_data,
            priority=MessagePriority.HIGH,
            sender="market_data_service"
        )
        
        # Update specific model categories
        await asyncio.gather(
            self._update_trading_models(market_data),
            self._update_analysis_models(market_data),
            self._update_adaptive_models(market_data)
        )
        
        self.logger.info("📡 Market update broadcast completed")
    
    async def coordinate_model_adaptation(self, performance_feedback: PerformanceFeedback):
        """
        🧠 COORDINATE MODEL ADAPTATION FOR HUMANITARIAN OPTIMIZATION
        
        Coordinates adaptation across all models based on performance feedback
        Prioritizes models with highest humanitarian impact potential
        """
        self.logger.info(f"🧠 Coordinating adaptation for {performance_feedback.model_name}")
        
        # Create adaptation plan
        adaptation_plan = await self._create_adaptation_plan(performance_feedback)
        
        # Prioritize models with highest humanitarian impact
        humanitarian_priorities = self._calculate_humanitarian_priorities(adaptation_plan)
        
        # Execute adaptation in priority order
        for priority_item in humanitarian_priorities:
            model_name = priority_item['model']
            adaptation_strategy = priority_item['strategy']
            
            await self._adapt_model(model_name, adaptation_strategy)
            
            self.logger.info(f"🎯 Adapted {model_name} for humanitarian optimization")
        
        # Broadcast adaptation completion
        await self.broadcast_message(
            MessageType.ADAPTATION_SIGNAL,
            {
                'adaptation_plan': adaptation_plan.__dict__,
                'humanitarian_impact': performance_feedback.charitable_impact,
                'models_adapted': [p['model'] for p in humanitarian_priorities]
            },
            priority=MessagePriority.MEDIUM,
            sender="adaptation_coordinator"
        )
        
        self.logger.info("🧠 Model adaptation coordination completed")
    
    async def synchronize_ensemble_weights(self, performance_data: Dict[str, Dict[str, float]]):
        """
        ⚖️ SYNCHRONIZE ENSEMBLE WEIGHTS FOR OPTIMAL HUMANITARIAN IMPACT
        
        Updates ensemble weights across all trading models based on
        humanitarian performance metrics and charitable impact
        """
        self.logger.info("⚖️ Synchronizing ensemble weights for humanitarian optimization")
        
        # Calculate optimal weights with humanitarian focus
        optimal_weights = await self._calculate_optimal_weights(performance_data)
        
        # Update trading model ensembles
        ensemble_updates = []
        
        # Scalping ensemble update
        if 'scalping' in optimal_weights:
            update_msg = await self.send_message_to_model(
                'scalping_ensemble',
                MessageType.ENSEMBLE_WEIGHT_UPDATE,
                {'weights': optimal_weights['scalping'], 'humanitarian_focus': True},
                priority=MessagePriority.HIGH
            )
            ensemble_updates.append(update_msg)
        
        # Day trading ensemble update  
        if 'daytrading' in optimal_weights:
            update_msg = await self.send_message_to_model(
                'daytrading_ensemble',
                MessageType.ENSEMBLE_WEIGHT_UPDATE,
                {'weights': optimal_weights['daytrading'], 'humanitarian_focus': True},
                priority=MessagePriority.HIGH
            )
            ensemble_updates.append(update_msg)
        
        # Swing trading ensemble update
        if 'swing' in optimal_weights:
            update_msg = await self.send_message_to_model(
                'swing_ensemble',
                MessageType.ENSEMBLE_WEIGHT_UPDATE,
                {'weights': optimal_weights['swing'], 'humanitarian_focus': True},
                priority=MessagePriority.HIGH
            )
            ensemble_updates.append(update_msg)
        
        # Wait for all updates to complete
        await asyncio.gather(*ensemble_updates)
        
        # Log humanitarian impact
        total_expected_impact = sum(
            weights.get('humanitarian_multiplier', 1.0) 
            for weights in optimal_weights.values()
        )
        
        self.logger.info(f"⚖️ Ensemble weights synchronized - Expected humanitarian boost: {total_expected_impact:.2f}x")
    
    async def broadcast_risk_alert(self, risk_data: Dict[str, Any], alert_level: str = "HIGH"):
        """
        🚨 BROADCAST RISK ALERT FOR CHARITABLE FUND PROTECTION
        
        Immediately alerts all models about risk conditions
        to protect humanitarian trading capital
        """
        self.logger.warning(f"🚨 Broadcasting {alert_level} risk alert")
        
        alert_data = {
            'alert_level': alert_level,
            'risk_data': risk_data,
            'humanitarian_protection': {
                'immediate_action_required': alert_level == "CRITICAL",
                'fund_protection_mode': True,
                'reduce_exposure': True if alert_level in ["CRITICAL", "HIGH"] else False
            },
            'timestamp': datetime.now().isoformat(),
            'protective_measures': self._get_protective_measures(alert_level)
        }
        
        # Critical alerts get highest priority
        priority = MessagePriority.CRITICAL if alert_level == "CRITICAL" else MessagePriority.HIGH
        
        await self.broadcast_message(
            MessageType.RISK_ALERT,
            alert_data,
            priority=priority,
            sender="risk_management_system"
        )
        
        self.logger.warning(f"🚨 Risk alert broadcast completed - Charitable funds protected")
    
    # Configuration Coordination Event Handlers
    
    async def _handle_agent_config_change(self, event_data: Dict[str, Any]):
        """Handle agent configuration change events"""
        try:
            agent_id = event_data.get('agent_id')
            change_type = event_data.get('change_type')
            config_section = event_data.get('config_section')
            new_value = event_data.get('new_value')
            
            self.logger.info(f"⚙️ Processing config change for {agent_id}: {config_section} -> {change_type}")
            
            # Validate the configuration change if validation is enabled
            if self.auto_config_validation and self.config_coordinator:
                validation_result = await self.config_coordinator.validate_config_change(
                    agent_id, config_section, new_value
                )
                if not validation_result.get('valid', False):
                    self.logger.error(f"❌ Config validation failed for {agent_id}: {validation_result.get('error')}")
                    return
            
            # Notify dependent agents about the configuration change
            if self.config_coordinator:
                dependent_agents = await self.config_coordinator.get_dependent_agents(agent_id)
                for dependent_agent in dependent_agents:
                    await self.send_message_to_model(
                        dependent_agent,
                        MessageType.COORDINATION_SYNC,
                        {
                            'type': 'config_dependency_update',
                            'source_agent': agent_id,
                            'config_section': config_section,
                            'change_type': change_type,
                            'propagation_timeout_seconds': self.config_change_timeout_seconds,
                            'timestamp': datetime.now().isoformat()
                        },
                        priority=MessagePriority.HIGH
                    )
                    self.logger.info(f"📢 Notified {dependent_agent} of config change in {agent_id}")
            
            # Update local tracking of agent configurations
            await self.publish_event("config_change_processed", {
                "agent_id": agent_id,
                "change_type": change_type,
                "config_section": config_section,
                "dependent_agents_notified": len(dependent_agents) if self.config_coordinator else 0,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"❌ Error handling config change event: {e}")
    
    async def _handle_config_rollback_request(self, event_data: Dict[str, Any]):
        """Handle configuration rollback requests"""
        try:
            agent_id = event_data.get('agent_id')
            rollback_version = event_data.get('rollback_version')
            rollback_reason = event_data.get('reason', 'Manual rollback')
            
            self.logger.warning(f"🔄 Processing config rollback for {agent_id} to version {rollback_version}")
            
            if self.config_coordinator:
                # Perform the rollback
                rollback_result = await self.config_coordinator.rollback_agent_config(
                    agent_id, rollback_version
                )
                
                if rollback_result.get('success', False):
                    # Notify all agents about the rollback
                    await self.broadcast_message(
                        MessageType.COORDINATION_SYNC,
                        {
                            'type': 'config_rollback_completed',
                            'agent_id': agent_id,
                            'rollback_version': rollback_version,
                            'reason': rollback_reason,
                            'timestamp': datetime.now().isoformat()
                        },
                        priority=MessagePriority.HIGH,
                        sender="config_coordinator"
                    )
                    
                    self.logger.info(f"✅ Config rollback completed for {agent_id}")
                else:
                    self.logger.error(f"❌ Config rollback failed for {agent_id}: {rollback_result.get('error')}")
            
        except Exception as e:
            self.logger.error(f"❌ Error handling config rollback request: {e}")
    
    async def _monitor_config_changes(self):
        """Monitor and log configuration changes across all agents"""
        try:
            while self.config_coordination_enabled:
                if self.config_coordinator:
                    # Check for any pending configuration changes
                    pending_changes = await self.config_coordinator.get_pending_changes()
                    
                    for change in pending_changes:
                        # Process each pending change
                        await self._handle_agent_config_change(change)
                    
                    # Check for configuration inconsistencies
                    consistency_report = await self.config_coordinator.run_consistency_check()
                    if not consistency_report.get('consistent', True):
                        self.logger.warning(f"⚠️ Configuration inconsistencies detected: {consistency_report}")
                        
                        # Publish inconsistency alert
                        await self.publish_event("config_inconsistency_detected", consistency_report)
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
        except asyncio.CancelledError:
            self.logger.info("🔄 Configuration monitoring task cancelled")
        except Exception as e:
            self.logger.error(f"❌ Error in configuration monitoring: {e}")
    
    # Configuration API Methods
    
    async def update_agent_configuration(self, agent_id: str, config_section: str, 
                                      new_value: Any, change_type: str = "update") -> bool:
        """
        ⚙️ Update agent configuration with dependency propagation
        
        Args:
            agent_id: ID of the agent to update
            config_section: Configuration section to update
            new_value: New configuration value
            change_type: Type of change (update, add, remove)
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            if not self.config_coordination_enabled or not self.config_coordinator:
                self.logger.warning("⚠️ Configuration coordination not available")
                return False
            
            # Update the configuration
            update_result = await self.config_coordinator.update_agent_config(
                agent_id, config_section, new_value, change_type
            )
            
            if update_result.get('success', False):
                # Trigger configuration change event
                await self.publish_event("agent_config_change", {
                    "agent_id": agent_id,
                    "config_section": config_section,
                    "new_value": new_value,
                    "change_type": change_type,
                    "change_id": update_result.get('change_id'),
                    "timestamp": datetime.now().isoformat()
                })
                
                self.logger.info(f"✅ Configuration updated for {agent_id}: {config_section}")
                return True
            else:
                self.logger.error(f"❌ Configuration update failed for {agent_id}: {update_result.get('error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error updating agent configuration: {e}")
            return False
    
    async def get_agent_configuration(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get current configuration for an agent"""
        try:
            if not self.config_coordination_enabled or not self.config_coordinator:
                return None
                
            config_profile = await self.config_coordinator.get_agent_config(agent_id)
            return config_profile.base_config if config_profile else None
            
        except Exception as e:
            self.logger.error(f"❌ Error getting agent configuration: {e}")
            return None
    
    # Security Helper Methods
    
    def _get_required_permission_for_message_type(self, message_type: str) -> Optional[AgentPermission]:
        """Map message types to required permissions"""
        permission_map = {
            'heartbeat': None,  # No special permission required
            'status_update': AgentPermission.ACCESS_HEALTH_DATA,
            'performance_metrics': AgentPermission.ACCESS_HEALTH_DATA,
            'agent_communication': None,  # Will be checked per content
            'dependency_request': AgentPermission.MANAGE_DEPENDENCIES,
            'dependency_response': AgentPermission.MANAGE_DEPENDENCIES,
            'market_data': AgentPermission.READ_MARKET_DATA,
            'risk_analysis': AgentPermission.READ_RISK_ANALYSIS,
            'pattern_analysis': AgentPermission.READ_PATTERN_ANALYSIS,
            'execution_signals': AgentPermission.READ_EXECUTION_SIGNALS,
            'coordination': AgentPermission.COORDINATE_AGENTS
        }
        return permission_map.get(message_type)
    
    # Security API Methods
    
    async def authenticate_agent(self, agent_id: str, certificate_data: bytes) -> bool:
        """
        Authenticate an agent using mTLS certificate
        
        Args:
            agent_id: Agent identifier
            certificate_data: Agent's certificate in PEM format
            
        Returns:
            True if authentication successful
        """
        if not self.security_enabled or not self.security_manager:
            self.logger.warning("🔒 Security not enabled - allowing unauthenticated access")
            return True
        
        identity = self.security_manager.authenticate_agent(agent_id, certificate_data)
        if identity:
            self.logger.info(f"🔒 Agent {agent_id} authenticated successfully as {identity.agent_name}")
            return True
        
        self.logger.warning(f"🔒 Authentication failed for agent {agent_id}")
        return False
    
    def authorize_agent_action(self, agent_id: str, permission: AgentPermission, target_agent_id: Optional[str] = None) -> bool:
        """
        Authorize agent action based on permissions
        
        Args:
            agent_id: Agent requesting permission
            permission: Required permission
            target_agent_id: Target agent (for inter-agent actions)
            
        Returns:
            True if authorized
        """
        if not self.security_enabled or not self.security_manager:
            return True
        
        return self.security_manager.authorize_action(agent_id, permission, target_agent_id)
    
    async def send_secure_message(self, sender_id: str, recipient_id: str, message_data: Dict[str, Any]) -> bool:
        """
        Send encrypted message between agents
        
        Args:
            sender_id: Sending agent ID
            recipient_id: Receiving agent ID
            message_data: Message payload to encrypt
            
        Returns:
            True if message sent successfully
        """
        if not self.security_enabled or not self.security_manager:
            # Fall back to regular message sending
            return await self.send_message_to_model(recipient_id, MessageType.AGENT_COMMUNICATION, message_data, sender=sender_id)
        
        # Encrypt message
        secure_message = self.security_manager.encrypt_message(sender_id, recipient_id, message_data)
        if not secure_message:
            self.logger.error(f"🔒 Failed to encrypt message from {sender_id} to {recipient_id}")
            return False
        
        # Send encrypted message
        encrypted_payload = {
            'type': 'encrypted_message',
            'encrypted_message': {
                'message_id': secure_message.message_id,
                'sender_id': secure_message.sender_id,
                'recipient_id': secure_message.recipient_id,
                'timestamp': secure_message.timestamp.isoformat(),
                'encrypted_payload': base64.b64encode(secure_message.encrypted_payload).decode('utf-8'),
                'signature': base64.b64encode(secure_message.signature).decode('utf-8'),
                'nonce': base64.b64encode(secure_message.nonce).decode('utf-8'),
                'auth_tag': base64.b64encode(secure_message.auth_tag).decode('utf-8')
            }
        }
        
        return await self.send_message_to_model(recipient_id, MessageType.AGENT_COMMUNICATION, encrypted_payload, sender=sender_id)
    
    def get_agent_security_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get security status for an agent"""
        if not self.security_enabled or not self.security_manager:
            return None
        
        return self.security_manager.get_agent_security_status(agent_id)
    
    def get_security_audit_trail(self, agent_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get security audit trail"""
        if not self.security_enabled or not self.security_manager:
            return []
        
        entries = self.security_manager.get_security_audit_trail(agent_id=agent_id, limit=limit)
        return [
            {
                'event_id': entry.event_id,
                'event_type': entry.event_type,
                'agent_id': entry.agent_id,
                'target_agent_id': entry.target_agent_id,
                'timestamp': entry.timestamp.isoformat(),
                'success': entry.success,
                'details': entry.details,
                'risk_level': entry.risk_level
            }
            for entry in entries
        ]
    
    def revoke_agent_access(self, agent_id: str, reason: str) -> bool:
        """Revoke agent access and terminate sessions"""
        if not self.security_enabled or not self.security_manager:
            return False
        
        success = self.security_manager.revoke_agent_certificate(agent_id, reason)
        if success:
            self.logger.warning(f"🔒 Agent {agent_id} access revoked: {reason}")
            # Also disconnect from WebSocket if connected
            if self.websocket_server_component:
                asyncio.create_task(self.websocket_server_component.disconnect_agent(agent_id))
        
        return success
    
    async def send_message_to_model(self, recipient: str, msg_type: MessageType, data: Dict[str, Any], 
                                   priority: MessagePriority = MessagePriority.MEDIUM,
                                   sender: str = "coordination_hub") -> bool:
        """
        Send a message to a specific AI model using enhanced infrastructure
        
        Returns:
            bool: True if message was successfully sent, False otherwise
        """
        if recipient not in self.registered_models:
            self.logger.warning(f"⚠️ Model {recipient} not registered - message dropped")
            return False
        
        message = CommunicationMessage(
            type=msg_type,
            priority=priority,
            sender=sender,
            recipient=recipient,
            data=data,
            humanitarian_context=True
        )
        
        try:
            # Use the new message queue manager if available
            if REALTIME_INFRASTRUCTURE_AVAILABLE and self.message_queue_manager:
                # Convert priority to queue priority
                queue_priority = QueuePriority.CRITICAL if priority == MessagePriority.CRITICAL else \
                                QueuePriority.HIGH if priority == MessagePriority.HIGH else \
                                QueuePriority.NORMAL if priority == MessagePriority.MEDIUM else \
                                QueuePriority.LOW
                
                # Send via the enhanced message queue
                success = await self.message_queue_manager.send_message(
                    recipient=recipient,
                    message=message.__dict__,
                    priority=queue_priority,
                    delivery_mode='reliable'
                )
                
                # Also try WebSocket if agent is connected
                if self.websocket_server_component and self.websocket_server_component.is_agent_connected(recipient):
                    await self.websocket_server_component.send_to_agent(recipient, {
                        'type': 'communication_message',
                        'message': message.__dict__,
                        'timestamp': datetime.now().isoformat()
                    })
                
                if success:
                    self.communication_stats['messages_sent'] += 1
                    self._add_to_history(message)
                    return True
                else:
                    self.communication_stats['failed_deliveries'] += 1
                    return False
            
            # Fallback to legacy method
            else:
                # Add to appropriate priority queue
                await self.message_queues[priority].put(message)
                self.communication_stats['messages_sent'] += 1
                
                # Add to history
                self._add_to_history(message)
                
                return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to send message to {recipient}: {str(e)}")
            self.communication_stats['failed_deliveries'] += 1
            return False
    
    async def broadcast_message(self, msg_type: MessageType, data: Dict[str, Any], 
                               priority: MessagePriority = MessagePriority.MEDIUM,
                               sender: str = "coordination_hub"):
        """Broadcast a message to all registered models"""
        message = CommunicationMessage(
            type=msg_type,
            priority=priority,
            sender=sender,
            recipient="broadcast",
            data=data,
            humanitarian_context=True
        )
        
        # Add to all priority queues for processing
        await self.message_queues[priority].put(message)
        self.communication_stats['messages_sent'] += 1
        
        # Add to history
        self._add_to_history(message)
    
    async def _process_message_queues(self):
        """Background task to process message queues by priority"""
        while self._processing_active:
            try:
                # Process critical messages first
                for priority in [MessagePriority.CRITICAL, MessagePriority.HIGH, 
                               MessagePriority.MEDIUM, MessagePriority.LOW]:
                    
                    queue = self.message_queues[priority]
                    
                    # Process up to 10 messages per priority level per cycle
                    for _ in range(10):
                        try:
                            message = queue.get_nowait()
                            await self._deliver_message(message)
                        except asyncio.QueueEmpty:
                            break
                
                # Short delay before next processing cycle
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"❌ Error in message processing: {str(e)}")
                await asyncio.sleep(1.0)
    
    async def _deliver_message(self, message: CommunicationMessage):
        """Deliver a message to its intended recipient(s)"""
        start_time = datetime.now()
        
        try:
            if message.recipient == "broadcast":
                # Deliver to all registered models
                delivery_tasks = []
                for model_name in self.registered_models:
                    task = self._deliver_to_model(model_name, message)
                    delivery_tasks.append(task)
                
                if delivery_tasks:
                    await asyncio.gather(*delivery_tasks, return_exceptions=True)
            else:
                # Deliver to specific model
                await self._deliver_to_model(message.recipient, message)
            
            # Update latency stats
            latency = (datetime.now() - start_time).total_seconds()
            self._update_latency_stats(latency)
            
        except Exception as e:
            self.logger.error(f"❌ Message delivery failed: {str(e)}")
            self.communication_stats['failed_deliveries'] += 1
    
    async def _deliver_to_model(self, model_name: str, message: CommunicationMessage):
        """Deliver message to a specific model"""
        if model_name not in self.message_handlers:
            return
        
        try:
            handler = self.message_handlers[model_name]
            await handler(message)
            self.communication_stats['messages_received'] += 1
            
        except Exception as e:
            self.logger.error(f"❌ Failed to deliver message to {model_name}: {str(e)}")
    
    async def _default_message_handler(self, message: CommunicationMessage):
        """Default message handler for models without custom handlers"""
        self.logger.info(f"📩 Default handler processed message: {message.type.value}")
    
    async def _update_trading_models(self, market_data: Dict[str, Any]):
        """Update all trading models with market data"""
        trading_models = ['scalping_ensemble', 'daytrading_ensemble', 'swing_ensemble']
        
        for model_name in trading_models:
            if model_name in self.registered_models:
                await self.send_message_to_model(
                    model_name,
                    MessageType.MARKET_UPDATE,
                    {'market_data': market_data, 'focus': 'trading_signals'},
                    priority=MessagePriority.HIGH
                )
    
    async def _update_analysis_models(self, market_data: Dict[str, Any]):
        """Update all analysis models with market data"""
        analysis_models = ['pattern_recognition_ai', 'sentiment_analysis_ai', 'risk_assessment_ai']
        
        for model_name in analysis_models:
            if model_name in self.registered_models:
                await self.send_message_to_model(
                    model_name,
                    MessageType.MARKET_UPDATE,
                    {'market_data': market_data, 'focus': 'market_analysis'},
                    priority=MessagePriority.HIGH
                )
    
    async def _update_adaptive_models(self, market_data: Dict[str, Any]):
        """Update adaptive learning models with market data"""
        adaptive_models = ['adaptive_learner', 'rapid_pipeline', 'performance_optimizer']
        
        for model_name in adaptive_models:
            if model_name in self.registered_models:
                await self.send_message_to_model(
                    model_name,
                    MessageType.MARKET_UPDATE,
                    {'market_data': market_data, 'focus': 'adaptive_learning'},
                    priority=MessagePriority.MEDIUM
                )
    
    async def _create_adaptation_plan(self, feedback: PerformanceFeedback) -> AdaptationPlan:
        """Create adaptation plan based on performance feedback"""
        priorities = []
        
        # Prioritize based on humanitarian impact potential
        if feedback.charitable_impact < 1000:  # Low impact
            priorities.append({
                'model': feedback.model_name,
                'strategy': 'aggressive_optimization',
                'priority': 1,
                'humanitarian_multiplier': 2.0
            })
        elif feedback.accuracy < 0.6:  # Low accuracy
            priorities.append({
                'model': feedback.model_name, 
                'strategy': 'accuracy_improvement',
                'priority': 2,
                'humanitarian_multiplier': 1.5
            })
        else:  # Fine-tuning
            priorities.append({
                'model': feedback.model_name,
                'strategy': 'fine_tuning',
                'priority': 3,
                'humanitarian_multiplier': 1.2
            })
        
        return AdaptationPlan(
            priorities=priorities,
            humanitarian_optimization={'charitable_impact_weight': 0.7},
            resource_allocation={'compute': 0.8, 'memory': 0.6},
            timeline={'start': datetime.now(), 'completion': datetime.now() + timedelta(hours=2)},
            expected_impact=feedback.charitable_impact * 1.3
        )
    
    def _calculate_humanitarian_priorities(self, plan: AdaptationPlan) -> List[Dict[str, Any]]:
        """Calculate humanitarian-focused priorities"""
        # Sort by humanitarian impact potential
        priorities = sorted(
            plan.priorities,
            key=lambda x: x.get('humanitarian_multiplier', 1.0),
            reverse=True
        )
        return priorities
    
    async def _adapt_model(self, model_name: str, strategy: str):
        """Adapt a specific model with given strategy"""
        if model_name in self.registered_models:
            await self.send_message_to_model(
                model_name,
                MessageType.ADAPTATION_SIGNAL,
                {
                    'strategy': strategy,
                    'humanitarian_focus': True,
                    'charitable_optimization': True
                },
                priority=MessagePriority.HIGH
            )
    
    async def _calculate_optimal_weights(self, performance_data: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Calculate optimal ensemble weights with humanitarian focus"""
        optimal_weights = {}
        
        for ensemble_name, model_performance in performance_data.items():
            weights = {}
            total_humanitarian_score = 0.0
            
            # Calculate humanitarian-weighted scores
            for model_name, metrics in model_performance.items():
                accuracy = metrics.get('accuracy', 0.5)
                profitability = metrics.get('profitability', 0.5)
                charitable_impact = metrics.get('charitable_impact', 0.0)
                
                # Humanitarian score = weighted combination favoring charitable impact
                humanitarian_score = (
                    accuracy * 0.3 + 
                    profitability * 0.3 + 
                    (charitable_impact / 1000) * 0.4  # Normalize impact
                )
                
                weights[model_name] = humanitarian_score
                total_humanitarian_score += humanitarian_score
            
            # Normalize weights
            if total_humanitarian_score > 0:
                for model_name in weights:
                    weights[model_name] /= total_humanitarian_score
                    
                # Add humanitarian multiplier
                weights['humanitarian_multiplier'] = min(total_humanitarian_score / len(weights), 2.0)
            
            optimal_weights[ensemble_name] = weights
        
        return optimal_weights
    
    def _get_protective_measures(self, alert_level: str) -> List[str]:
        """Get protective measures based on alert level"""
        if alert_level == "CRITICAL":
            return [
                "immediate_position_closure",
                "halt_new_positions", 
                "activate_emergency_stops",
                "protect_charitable_capital"
            ]
        elif alert_level == "HIGH":
            return [
                "reduce_position_sizes",
                "increase_stop_losses",
                "conservative_mode",
                "enhanced_monitoring"
            ]
        else:
            return [
                "cautious_trading",
                "monitor_closely"
            ]
    
    async def _monitor_communication_health(self):
        """Monitor communication system health"""
        while self._processing_active:
            try:
                # Check queue sizes
                total_queued = sum(q.qsize() for q in self.message_queues.values())
                
                if total_queued > 5000:  # High queue warning
                    self.logger.warning(f"⚠️ High message queue size: {total_queued}")
                
                # Log stats periodically
                if self.communication_stats['messages_sent'] % 1000 == 0:
                    self.logger.info(f"📊 Communication stats: {self.communication_stats}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Health monitoring error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _optimize_humanitarian_coordination(self):
        """Background optimization for humanitarian coordination"""
        while self._processing_active:
            try:
                # Analyze message patterns for optimization opportunities
                if len(self.message_history) > 100:
                    await self._analyze_coordination_patterns()
                
                # Optimize for charitable impact
                await self._optimize_charitable_impact()
                
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
            except Exception as e:
                self.logger.error(f"❌ Coordination optimization error: {str(e)}")
                await asyncio.sleep(600)
    
    async def _analyze_coordination_patterns(self):
        """Analyze communication patterns for optimization"""
        # Placeholder for pattern analysis
        self.logger.info("🔍 Analyzing coordination patterns for humanitarian optimization")
    
    async def _optimize_charitable_impact(self):
        """Optimize coordination for maximum charitable impact"""
        # Placeholder for impact optimization
        self.logger.info("💰 Optimizing coordination for maximum charitable impact")
    
    def _add_to_history(self, message: CommunicationMessage):
        """Add message to communication history"""
        self.message_history.append({
            'id': message.id,
            'type': message.type.value,
            'priority': message.priority.value,
            'sender': message.sender,
            'recipient': message.recipient,
            'timestamp': message.timestamp,
            'humanitarian_context': message.humanitarian_context
        })
        
        # Maintain history size limit
        if len(self.message_history) > self.max_history_size:
            self.message_history = self.message_history[-self.max_history_size:]
    
    def _update_latency_stats(self, latency: float):
        """Update latency statistics"""
        if self.communication_stats['average_latency'] == 0:
            self.communication_stats['average_latency'] = latency
        else:
            # Moving average
            self.communication_stats['average_latency'] = (
                self.communication_stats['average_latency'] * 0.9 + latency * 0.1
            )
    
    async def _cleanup_message_delivery_status(self):
        """Clean up old message delivery status records"""
        while self.health_monitor_active:
            try:
                current_time = datetime.now()
                expired_messages = []
                
                for message_id, status in self.message_delivery_status.items():
                    age = (current_time - status.timestamp).total_seconds()
                    if age > 300:  # Remove status older than 5 minutes
                        expired_messages.append(message_id)
                
                for message_id in expired_messages:
                    del self.message_delivery_status[message_id]
                
                await asyncio.sleep(60)  # Clean up every minute
                
            except Exception as e:
                self.logger.error(f"Error in message cleanup: {e}")
                await asyncio.sleep(10)

    async def _process_redis_messages(self):
        """Process incoming Redis messages"""
        if not self.redis_pubsub:
            return
            
        try:
            while self.health_monitor_active:
                message = await self.redis_pubsub.get_message(timeout=1)
                if message and message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        await self._handle_distributed_event(data)
                    except json.JSONDecodeError:
                        self.logger.warning("Invalid JSON in Redis message")
                    except Exception as e:
                        self.logger.error(f"Error processing Redis message: {e}")
                        
                await asyncio.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Error in Redis message processing: {e}")

    async def _handle_distributed_event(self, event_data: Dict[str, Any]):
        """Handle distributed events from Redis/Kafka"""
        event_type = event_data.get('type')
        if event_type in self.event_subscribers:
            for callback in self.event_subscribers[event_type]:
                try:
                    await callback(event_data)
                except Exception as e:
                    self.logger.error(f"Error in distributed event callback: {e}")

    async def handle_agent_communication(self, agent_id: str, data: Dict[str, Any]):
        """Handle communication messages between agents"""
        try:
            message_type = data.get('message_type', 'unknown')
            payload = data.get('payload', {})
            target_agent = data.get('target_agent')
            
            # Create communication message
            comm_message = CommunicationMessage(
                type=MessageType(message_type) if message_type in [mt.value for mt in MessageType] else MessageType.COORDINATION_SYNC,
                sender=agent_id,
                recipient=target_agent or "broadcast",
                data=payload,
                timestamp=datetime.now()
            )
            
            # Route message appropriately
            if target_agent and target_agent != "broadcast":
                # Direct message to specific agent
                await self.send_to_agent(target_agent, {
                    'type': 'agent_message',
                    'from': agent_id,
                    'message': comm_message.__dict__,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                # Broadcast to all agents
                await self.broadcast_to_all_agents({
                    'type': 'agent_broadcast',
                    'from': agent_id,
                    'message': comm_message.__dict__,
                    'timestamp': datetime.now().isoformat()
                })
            
        except Exception as e:
            self.logger.error(f"Error handling agent communication from {agent_id}: {e}")

    async def handle_status_update(self, agent_id: str, data: Dict[str, Any]):
        """Handle status updates from agents"""
        try:
            status = data.get('status', 'unknown')
            details = data.get('details', {})
            
            # Update agent registry if available
            if hasattr(self, 'registered_models') and agent_id in self.registered_models:
                # Update performance metrics or status in the registry
                pass
            
            # Broadcast status update to interested parties
            await self.publish_event('agent_status_update', {
                'agent_id': agent_id,
                'status': status,
                'details': details,
                'timestamp': datetime.now().isoformat()
            })            
        except Exception as e:
            self.logger.error(f"Error handling status update from {agent_id}: {e}")
            
    async def shutdown(self):
        """Gracefully shutdown the communication protocol"""
        self.logger.info("🔌 Shutting down communication protocol...")
        
        self.health_monitor_active = False
        self._processing_active = False
        
        # Shutdown new infrastructure components
        if REALTIME_INFRASTRUCTURE_AVAILABLE:
            if self.health_monitor:
                await self.health_monitor.stop_monitoring()
                self.logger.info("✅ Health monitor stopped")
            
            if self.websocket_server_component:
                await self.websocket_server_component.stop_server()
                self.logger.info("✅ WebSocket server stopped")
            
            if self.message_queue_manager:
                await self.message_queue_manager.shutdown()
                self.logger.info("✅ Message queue manager stopped")
        
        # Shutdown legacy components
        
        # Close WebSocket server
        if self.websocket_server:
            self.websocket_server.close()
            if WEBSOCKETS_AVAILABLE:
                await self.websocket_server.wait_closed()
        
        # Close all agent connections
        for agent_id, connection in self.agent_connections.items():
            if connection.websocket and WEBSOCKETS_AVAILABLE:
                await connection.websocket.close()
        
        # Close Redis connection
        if self.redis_client and REDIS_AVAILABLE:
            await self.redis_client.close()
          # Close Kafka producer
        if self.kafka_producer and KAFKA_AVAILABLE:
            self.kafka_producer.close()
        
        self.logger.info("✅ Communication protocol shutdown complete")
        
    async def initialize_infrastructure(self):
        """Initialize Redis, Kafka, WebSocket infrastructure, and health monitoring with auto-recovery"""
        try:
            # Initialize Redis connection if available
            if REDIS_AVAILABLE:
                try:
                    self.redis_client = aioredis.from_url(self.redis_url)
                    # Test connection
                    await self.redis_client.ping()
                    self.logger.info("✅ Redis connection established")
                except Exception as e:
                    self.logger.warning(f"⚠️ Redis connection failed: {e}")
                    self.redis_client = None
            else:
                self.logger.warning("⚠️ Redis not available - message queuing disabled")
            
            # Initialize Kafka producer if available
            if KAFKA_AVAILABLE:
                try:
                    self.kafka_producer = KafkaProducer(
                        bootstrap_servers=self.kafka_bootstrap_servers,
                        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                        retry_backoff_ms=100,
                        retries=3
                    )
                    self.logger.info("✅ Kafka producer initialized")
                except Exception as e:
                    self.logger.warning(f"⚠️ Kafka initialization failed: {e}")
                    self.kafka_producer = None
            else:
                self.logger.warning("⚠️ Kafka not available - distributed messaging disabled")
            
            # Initialize and start enhanced health monitoring with auto-recovery
            if REALTIME_INFRASTRUCTURE_AVAILABLE and self.health_monitor:
                try:
                    # Register health alert callback
                    self.health_monitor.add_alert_callback(self._handle_agent_health_alert)
                    
                    # Start health monitoring
                    await self.health_monitor.start_monitoring()
                    self.logger.info("✅ Agent health monitoring with auto-recovery started")
                except Exception as e:
                    self.logger.error(f"❌ Failed to start health monitoring: {e}")
            
            # Start WebSocket server if available
            if WEBSOCKETS_AVAILABLE:
                await self.start_websocket_server()
            else:
                self.logger.warning("⚠️ WebSockets not available - real-time communication disabled")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize infrastructure: {e}")
            # Don't raise - allow system to work in degraded mode
            # Don't raise - allow system to work in degraded mode    async def start_websocket_server(self):
        """Start WebSocket server for real-time agent communication"""
        if not WEBSOCKETS_AVAILABLE:
            self.logger.warning("⚠️ WebSockets not available")
            return
            
        try:
            self.websocket_server = await websockets.serve(
                self.handle_websocket_connection,
                "localhost",
                self.websocket_port
            )
            self.logger.info(f"🌐 WebSocket server started on port {self.websocket_port}")
        except Exception as e:
            self.logger.error(f"❌ Failed to start WebSocket server: {e}")
            # Don't raise - allow system to work without WebSockets

    async def handle_websocket_connection(self, websocket, path):
        """Handle incoming WebSocket connections from agents"""
        agent_id = None
        try:
            # Wait for agent identification
            init_message = await asyncio.wait_for(websocket.recv(), timeout=10)
            init_data = json.loads(init_message)
            
            if init_data.get('type') != 'agent_identification':
                await websocket.send(json.dumps({'error': 'Expected agent identification'}))
                return
            
            agent_id = init_data.get('agent_id')
            if not agent_id:
                await websocket.send(json.dumps({'error': 'Agent ID required'}))
                return
            
            # Register agent connection
            async with self.connection_lock:
                connection = AgentConnection(
                    agent_id=agent_id,
                    websocket=websocket,
                    last_heartbeat=datetime.now(),
                    status="connected"
                )
                self.agent_connections[agent_id] = connection
                self.heartbeat_status[agent_id] = time.time()
            
            # Send confirmation
            await websocket.send(json.dumps({
                'type': 'connection_confirmed',
                'agent_id': agent_id,
                'timestamp': datetime.now().isoformat()
            }))
            
            self.logger.info(f"🔗 Agent {agent_id} connected via WebSocket")
            
            # Handle messages from this agent
            await self.handle_agent_messages(agent_id, websocket)
            
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"🔌 Agent {agent_id} disconnected")
        except Exception as e:
            self.logger.error(f"❌ Error handling WebSocket connection: {e}")
        finally:
            # Clean up connection
            if agent_id and agent_id in self.agent_connections:
                async with self.connection_lock:
                    self.agent_connections[agent_id].status = "disconnected"
                    del self.agent_connections[agent_id]
                    if agent_id in self.heartbeat_status:
                        del self.heartbeat_status[agent_id]

    async def handle_agent_messages(self, agent_id: str, websocket):
        """Handle incoming messages from a connected agent"""
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get('type')
                    
                    if message_type == 'heartbeat':
                        await self.handle_heartbeat(agent_id, data)
                    elif message_type == 'agent_message':
                        await self.handle_agent_communication(agent_id, data)
                    elif message_type == 'status_update':
                        await self.handle_status_update(agent_id, data)
                    else:
                        self.logger.warning(f"Unknown message type from {agent_id}: {message_type}")
                        
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid JSON from agent {agent_id}")
                except Exception as e:
                    self.logger.error(f"Error processing message from {agent_id}: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            pass

    async def handle_heartbeat(self, agent_id: str, data: Dict[str, Any]):
        """Handle heartbeat messages from agents"""
        current_time = time.time()
        self.heartbeat_status[agent_id] = current_time
        
        if agent_id in self.agent_connections:
            self.agent_connections[agent_id].last_heartbeat = datetime.now()
            
        # Send heartbeat response
        response = {
            'type': 'heartbeat_response',
            'timestamp': datetime.now().isoformat(),
            'status': 'ok'
        }
        
        await self.send_to_agent(agent_id, response)

    async def send_to_agent(self, agent_id: str, message: Dict[str, Any]) -> bool:
        """Send message to a specific agent via WebSocket"""
        if agent_id not in self.agent_connections:
            self.logger.warning(f"Agent {agent_id} not connected")
            return False
            
        connection = self.agent_connections[agent_id]
        if connection.status != "connected" or not connection.websocket:
            return False
            
        try:
            await connection.websocket.send(json.dumps(message))
            return True
        except websockets.exceptions.ConnectionClosed:
            connection.status = "disconnected"
            return False
        except Exception as e:
            self.logger.error(f"Error sending message to {agent_id}: {e}")
            return False

    async def broadcast_to_all_agents(self, message: Dict[str, Any]) -> Dict[str, bool]:
        """Broadcast message to all connected agents"""
        results = {}
        
        for agent_id in list(self.agent_connections.keys()):
            success = await self.send_to_agent(agent_id, message)
            results[agent_id] = success
            
        return results

    async def send_message_with_retry(self, agent_id: str, message: Dict[str, Any], 
                                    max_retries: int = 3) -> bool:
        """Send message with automatic retry on failure"""
        message_id = str(uuid.uuid4())
        message['message_id'] = message_id
        
        # Track delivery status
        self.message_delivery_status[message_id] = MessageDeliveryStatus(
            message_id=message_id,
            status="pending"
        )
        
        for attempt in range(max_retries + 1):
            success = await self.send_to_agent(agent_id, message)
            
            if success:
                self.message_delivery_status[message_id].status = "delivered"
                return True
            else:
                self.message_delivery_status[message_id].retry_count = attempt + 1
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
        self.message_delivery_status[message_id].status = "failed"
        return False

    # ============================================================================
    # AGENT DEPENDENCY RESOLUTION API
    # ============================================================================

    async def resolve_agent_dependencies(self, agent_id: str, request_data: Dict[str, Any] = None, 
                                       timeout_ms: int = 5000) -> Dict[str, Any]:
        """
        🔗 PUBLIC API: Resolve dependencies for an agent
        
        Main entry point for agents to request data from their dependencies.
        This method coordinates the entire dependency resolution process.
        
        Args:
            agent_id: The agent requesting dependency resolution
            request_data: Context data for the dependency request
            timeout_ms: Maximum time to wait for responses
            
        Returns:
            Dictionary containing resolved data from all dependencies
        """
        if not self.dependency_resolver:
            self.logger.warning(f"🔗 Dependency resolver not available for {agent_id}")
            return {}
        
        try:
            return await self.dependency_resolver.resolve_agent_dependencies(
                agent_id, request_data or {}, timeout_ms
            )
        except Exception as e:
            self.logger.error(f"🔗 Error resolving dependencies for {agent_id}: {e}")
            return {}
    
    async def request_agent_data(self, requesting_agent: str, target_agent: str, 
                               data_type: str, request_data: Dict[str, Any] = None,
                               timeout_ms: int = 3000) -> Optional[Dict[str, Any]]:
        """
        🔗 PUBLIC API: Request specific data from a target agent
        
        Direct request for specific data from a single agent.
        
        Args:
            requesting_agent: Agent making the request
            target_agent: Agent to request data from
            data_type: Type of data being requested
            request_data: Context data for the request
            timeout_ms: Maximum time to wait for response
            
        Returns:
            Response data from target agent or None if failed
        """
        if not self.dependency_resolver:
            self.logger.warning(f"🔗 Dependency resolver not available")
            return None
        
        try:
            request_id = str(uuid.uuid4())
            request = DependencyRequest(
                request_id=request_id,
                requesting_agent=requesting_agent,
                target_agent=target_agent,
                data_type=data_type,
                request_data=request_data or {},
                timeout_ms=timeout_ms
            )
            
            response = await self.dependency_resolver._send_dependency_request(request)
            return response.data if response.success else None
            
        except Exception as e:
            self.logger.error(f"🔗 Error requesting data from {target_agent}: {e}")
            return None
    
    def get_agent_dependencies(self, agent_id: str) -> List[str]:
        """
        🔗 PUBLIC API: Get list of dependencies for an agent
        
        Returns the list of agents that the specified agent depends on.
        """
        if not self.dependency_resolver:
            return []
        
        return self.dependency_resolver.dependency_graph.get_dependencies(agent_id)
    
    def get_dependency_resolution_metrics(self) -> Dict[str, Any]:
        """
        🔗 PUBLIC API: Get dependency resolution performance metrics
        
        Returns metrics about dependency resolution performance and success rates.
        """
        if not self.dependency_resolver:
            return {}
        
        return self.dependency_resolver.get_metrics()

    # ============================================================================

    async def start_heartbeat_monitoring(self):
        """Start continuous heartbeat monitoring for all agents"""
        while self.health_monitor_active:
            try:
                current_time = time.time()
                unhealthy_agents = []
                
                for agent_id, last_heartbeat in self.heartbeat_status.items():
                    time_since_heartbeat = current_time - last_heartbeat
                    
                    if time_since_heartbeat > (self.heartbeat_config.interval * self.heartbeat_config.max_missed):
                        unhealthy_agents.append(agent_id)
                        self.logger.warning(f"💔 Agent {agent_id} missed heartbeat for {time_since_heartbeat:.1f}s")
                
                # Handle unhealthy agents
                for agent_id in unhealthy_agents:
                    await self.handle_unhealthy_agent(agent_id)
                
                # Request heartbeat from all connected agents
                heartbeat_request = {
                    'type': 'heartbeat_request',
                    'timestamp': datetime.now().isoformat()
                }
                await self.broadcast_to_all_agents(heartbeat_request)
                
                await asyncio.sleep(self.heartbeat_config.interval)
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat monitoring: {e}")
                await asyncio.sleep(5)

    async def handle_unhealthy_agent(self, agent_id: str):
        """Handle agents that have become unhealthy"""
        if agent_id in self.agent_connections:
            connection = self.agent_connections[agent_id]
            connection.status = "unhealthy"
            
            # Attempt to reconnect
            await self.attempt_agent_reconnection(agent_id)
            
            # Notify other agents about the unhealthy agent
            alert_message = {
                'type': 'agent_health_alert',
                'unhealthy_agent': agent_id,
                'timestamp': datetime.now().isoformat(),
                'action': 'attempting_reconnection'
            }
            await self.broadcast_to_all_agents(alert_message)

    async def attempt_agent_reconnection(self, agent_id: str):
        """Attempt to reconnect to an unhealthy agent"""
        if agent_id not in self.agent_connections:
            return
            
        connection = self.agent_connections[agent_id]
        connection.retry_count += 1
        
        if connection.retry_count > connection.max_retries:
            self.logger.error(f"❌ Max reconnection attempts reached for {agent_id}")
            connection.status = "failed"
            return
        
        try:
            # Close existing connection if it exists
            if connection.websocket:
                await connection.websocket.close()
              # In a real implementation, you would trigger the agent to reconnect
            # For now, we'll mark it as attempting reconnection
            connection.status = "reconnecting"
            self.logger.info(f"🔄 Attempting reconnection for agent {agent_id} (attempt {connection.retry_count})")
            
        except Exception as e:
            self.logger.error(f"Error during reconnection attempt for {agent_id}: {e}")
            
    async def publish_to_redis(self, channel: str, message: Dict[str, Any]):
        """Publish message to Redis channel for distributed communication"""
        if not self.redis_client:
            self.logger.debug("Redis client not available - skipping publish")
            return
            
        try:
            serialized_message = json.dumps(message)
            await self.redis_client.publish(channel, serialized_message)
        except Exception as e:
            self.logger.error(f"Error publishing to Redis channel {channel}: {e}")

    async def publish_to_kafka(self, topic: str, message: Dict[str, Any]):
        """Publish message to Kafka topic for reliable messaging"""
        if not self.kafka_producer:
            self.logger.debug("Kafka producer not available - skipping publish")
            return
            
        try:
            self.kafka_producer.send(topic, message)
            self.kafka_producer.flush()
        except Exception as e:
            self.logger.error(f"Error publishing to Kafka topic {topic}: {e}")

    async def subscribe_to_event(self, event_type: str, callback: Callable):
        """Subscribe to distributed events"""
        self.event_subscribers[event_type].append(callback)
        self.logger.info(f"📡 Subscribed to event type: {event_type}")

    async def publish_event(self, event_type: str, event_data: Dict[str, Any]):
        """Publish event to all subscribers"""
        event = {
            'id': str(uuid.uuid4()),
            'type': event_type,
            'data': event_data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in event history
        self.event_history.append(event)
        if len(self.event_history) > self.max_event_history:
            self.event_history.pop(0)
        
        # Publish to Redis and Kafka
        await self.publish_to_redis('platform3_agent_events', event)
        await self.publish_to_kafka(self.kafka_topics['agent_communication'], event)
        
        # Notify local subscribers
        if event_type in self.event_subscribers:
            for callback in self.event_subscribers[event_type]:
                try:
                    await callback(event)
                except Exception as e:
                    self.logger.error(f"Error in event callback for {event_type}: {e}")

    def get_agent_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current health status of all agents"""
        current_time = time.time()
        health_status = {}
        
        for agent_id in self.agent_connections:
            connection = self.agent_connections[agent_id]
            last_heartbeat = self.heartbeat_status.get(agent_id, 0)
            time_since_heartbeat = current_time - last_heartbeat
            
            health_status[agent_id] = {
                'status': connection.status,
                'last_heartbeat': datetime.fromtimestamp(last_heartbeat).isoformat(),
                'time_since_heartbeat': time_since_heartbeat,
                'retry_count': connection.retry_count,
                'is_healthy': time_since_heartbeat < (self.heartbeat_config.interval * 2)
            }
            
        return health_status

    def get_communication_metrics(self) -> Dict[str, Any]:
        """Get enhanced communication performance metrics"""
        # Get legacy metrics
        total_connections = len(self.agent_connections)
        healthy_connections = sum(1 for conn in self.agent_connections.values() if conn.status == "connected")
        
        delivered_messages = sum(1 for status in self.message_delivery_status.values() if status.status == "delivered")
        failed_messages = sum(1 for status in self.message_delivery_status.values() if status.status == "failed")
        total_messages = len(self.message_delivery_status)
        
        delivery_rate = (delivered_messages / total_messages * 100) if total_messages > 0 else 0
        
        base_metrics = {
            'total_agent_connections': total_connections,
            'healthy_connections': healthy_connections,
            'connection_health_rate': (healthy_connections / total_connections * 100) if total_connections > 0 else 0,
            'total_messages_sent': total_messages,
            'message_delivery_rate': delivery_rate,
            'failed_messages': failed_messages,
            'event_history_size': len(self.event_history),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add enhanced metrics if new infrastructure is available
        if REALTIME_INFRASTRUCTURE_AVAILABLE:
            enhanced_metrics = {}
            
            if self.health_monitor:
                health_summary = self.health_monitor.get_overall_health_summary()
                enhanced_metrics.update({
                    'agent_health_summary': health_summary,
                    'average_response_time_ms': health_summary.get('average_response_time_ms', 0),
                    'overall_health_score': health_summary.get('overall_health_score', 0)
                })
            
            if self.websocket_server_component:
                ws_metrics = self.websocket_server_component.get_connection_metrics()
                enhanced_metrics.update({
                    'websocket_connections': ws_metrics.get('active_connections', 0),
                    'websocket_messages_sent': ws_metrics.get('messages_sent', 0),
                    'websocket_messages_received': ws_metrics.get('messages_received', 0)
                })
            
            if self.message_queue_manager:
                queue_metrics = self.message_queue_manager.get_queue_metrics()
                enhanced_metrics.update({
                    'queue_depth': queue_metrics.get('total_queue_depth', 0),
                    'queue_processing_rate': queue_metrics.get('processing_rate', 0),
                    'redis_available': queue_metrics.get('redis_available', False),
                    'kafka_available': queue_metrics.get('kafka_available', False)
                })
            
            base_metrics['enhanced_metrics'] = enhanced_metrics
            base_metrics['infrastructure_status'] = 'enhanced'
        else:
            base_metrics['infrastructure_status'] = 'legacy'
        
        return base_metrics
    
    async def get_real_time_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get real-time status of all agents"""
        if not REALTIME_INFRASTRUCTURE_AVAILABLE or not self.health_monitor:
            # Return legacy status
            return self.get_agent_health_status()
        
        # Get enhanced status from health monitor
        all_agents_health = await self.health_monitor.get_all_agents_health()
        
        enhanced_status = {}
        for agent_id, health_metrics in all_agents_health.items():
            enhanced_status[agent_id] = {
                'health_status': health_metrics.status.value,
                'last_heartbeat': health_metrics.last_heartbeat.isoformat(),
                'response_time_ms': health_metrics.response_time_ms,
                'throughput_per_minute': health_metrics.throughput_per_minute,
                'error_rate_percent': health_metrics.error_rate_percent,
                'cpu_usage_percent': health_metrics.cpu_usage_percent,
                'memory_usage_mb': health_metrics.memory_usage_mb,
                'accuracy_percent': health_metrics.accuracy_percent,
                'profit_contribution': health_metrics.profit_contribution,
                'is_connected': self.websocket_server_component.is_agent_connected(agent_id) if self.websocket_server_component else False
            }
        
        return enhanced_status
    
    async def send_priority_broadcast(self, message_type: MessageType, data: Dict[str, Any], 
                                    priority: MessagePriority = MessagePriority.HIGH) -> Dict[str, bool]:
        """Send high-priority broadcast using both legacy and enhanced infrastructure"""
        message = CommunicationMessage(
            type=message_type,
            priority=priority,
            sender="coordination_hub",
            recipient="broadcast",
            data=data,
            humanitarian_context=True
        )
        
        results = {}
        
        # Send via enhanced infrastructure if available
        if REALTIME_INFRASTRUCTURE_AVAILABLE:
            if self.websocket_server_component:
                # Send via WebSocket to all connected agents
                ws_results = await self.websocket_server_component.broadcast_to_all_agents({
                    'type': 'priority_broadcast',
                    'message': message.__dict__,
                    'timestamp': datetime.now().isoformat()
                })
                results.update(ws_results)
            
            if self.message_queue_manager:
                # Send via message queues for reliability
                queue_priority = QueuePriority.CRITICAL if priority == MessagePriority.CRITICAL else \
                               QueuePriority.HIGH if priority == MessagePriority.HIGH else \
                               QueuePriority.NORMAL
                
                await self.message_queue_manager.broadcast_message(
                    message=message.__dict__,
                    priority=queue_priority
                )
        
        # Also send via legacy system
        await self.broadcast_message(message_type, data, priority, "coordination_hub")
        
        self.communication_stats['messages_sent'] += 1
        self._add_to_history(message)
        
        return results
    
    async def establish_direct_agent_channel(self, agent_id: str) -> bool:
        """Establish a direct communication channel with an agent"""
        if not REALTIME_INFRASTRUCTURE_AVAILABLE or not self.websocket_server_component:
            self.logger.warning("Direct channels require enhanced infrastructure")
            return False
        
        try:
            # Ensure agent is connected
            if not self.websocket_server_component.is_agent_connected(agent_id):
                self.logger.warning(f"Agent {agent_id} not connected via WebSocket")
                return False
            
            # Send channel establishment request
            success = await self.websocket_server_component.send_to_agent(agent_id, {
                'type': 'establish_direct_channel',
                'timestamp': datetime.now().isoformat(),
                'channel_id': str(uuid.uuid4())
            })
            
            if success:
                self.logger.info(f"✅ Direct channel established with agent {agent_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to establish direct channel with {agent_id}: {e}")
            return False
    
    # ============================================================================
    # AUTO-RECOVERY SYSTEM API 🔄
    # ============================================================================
    
    def get_recovery_metrics(self) -> Dict[str, Any]:
        """Get auto-recovery system metrics"""
        if not REALTIME_INFRASTRUCTURE_AVAILABLE or not self.health_monitor:
            return {"auto_recovery_available": False}
        
        return self.health_monitor.get_recovery_metrics()
    
    # ============================================================================
    # DISASTER RECOVERY SYSTEM API 🚑
    # ============================================================================
    
    async def create_cluster_backup(self, cluster_id: str = "main_cluster") -> bool:
        """Create immediate cluster backup"""
        if not DISASTER_RECOVERY_AVAILABLE or not self.disaster_recovery:
            self.logger.warning("Disaster recovery not available")
            return False
        
        try:
            cluster_state = await self.disaster_recovery.create_cluster_backup(cluster_id)
            self.logger.info(f"✅ Cluster backup created: {cluster_state.cluster_id}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to create cluster backup: {e}")
            return False
    
    async def initiate_cluster_recovery(self, backup_path: str = None) -> bool:
        """Initiate cluster recovery from backup"""
        if not DISASTER_RECOVERY_AVAILABLE or not self.disaster_recovery:
            self.logger.warning("Disaster recovery not available")
            return False
        
        try:
            success = await self.disaster_recovery.restore_cluster_from_backup(backup_path)
            if success:
                self.logger.info("✅ Cluster recovery completed successfully")
                # Reinitialize communication after recovery
                await self.initialize_infrastructure()
            else:
                self.logger.error("❌ Cluster recovery failed")
            return success
        except Exception as e:
            self.logger.error(f"❌ Cluster recovery error: {e}")
            return False
    
    def get_disaster_recovery_status(self) -> Dict[str, Any]:
        """Get disaster recovery system status"""
        if not DISASTER_RECOVERY_AVAILABLE or not self.disaster_recovery:
            return {"disaster_recovery_available": False}
        
        return {
            "disaster_recovery_available": True,
            "recovery_enabled": self.disaster_recovery_enabled,
            "recovery_target_minutes": self.recovery_time_target_minutes,
            "backup_interval_hours": self.auto_backup_interval_hours,
            "current_recovery": self.disaster_recovery.current_recovery is not None,
            "recovery_history_count": len(self.disaster_recovery.recovery_history)
        }
    
    async def _handle_cluster_failure(self, event_data: Dict[str, Any]):
        """Handle cluster failure events"""
        self.logger.warning(f"🚨 Cluster failure detected: {event_data}")
        
        if self.disaster_recovery_enabled and self.disaster_recovery:
            # Automatically initiate recovery for critical failures
            failure_type = event_data.get('failure_type', 'unknown')
            if failure_type in ['complete_failure', 'agent_cascade_failure']:
                self.logger.info("🚑 Initiating automatic cluster recovery")
                await self.initiate_cluster_recovery()
    
    async def _handle_recovery_request(self, event_data: Dict[str, Any]):
        """Handle manual recovery requests"""
        backup_path = event_data.get('backup_path')
        self.logger.info(f"🚑 Manual recovery requested: {backup_path}")
        await self.initiate_cluster_recovery(backup_path)
    
    async def _handle_backup_request(self, event_data: Dict[str, Any]):
        """Handle manual backup requests"""
        cluster_id = event_data.get('cluster_id', 'main_cluster')
        self.logger.info(f"📦 Manual backup requested: {cluster_id}")
        await self.create_cluster_backup(cluster_id)
    
    async def _monitor_disaster_recovery(self):
        """Background monitoring for disaster recovery system"""
        while self.disaster_recovery_enabled and self._processing_active:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                if not self.disaster_recovery:
                    continue
                
                # Check if disaster recovery system is healthy
                test_results = await self.disaster_recovery.test_recovery_system()
                overall_status = test_results.get('overall_status', False)
                
                if not overall_status:
                    self.logger.warning("⚠️ Disaster recovery system health check failed")
                    # Publish alert
                    await self.publish_event("disaster_recovery_alert", {
                        "alert_type": "system_health_degraded",
                        "test_results": test_results,
                        "timestamp": datetime.now().isoformat()
                    })
                
            except Exception as e:
                self.logger.error(f"Error in disaster recovery monitoring: {e}")
                await asyncio.sleep(1800)  # Retry after 30 minutes on error
    
    async def _run_periodic_cluster_backups(self):
        """Background task to create periodic cluster backups"""
        while self.disaster_recovery_enabled and self._processing_active:
            try:
                # Wait for backup interval
                await asyncio.sleep(self.auto_backup_interval_hours * 3600)
                
                if self.disaster_recovery:
                    self.logger.info("📦 Creating scheduled cluster backup")
                    await self.create_cluster_backup("scheduled_backup")
                
            except Exception as e:
                self.logger.error(f"Error in periodic cluster backup: {e}")
                await asyncio.sleep(3600)  # Retry after 1 hour on error
    
    def get_recovery_history(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recovery history for all agents or specific agent"""
        if not REALTIME_INFRASTRUCTURE_AVAILABLE or not self.health_monitor:
            return []
        
        attempts = self.health_monitor.get_recovery_history(agent_id)
        return [attempt.to_dict() for attempt in attempts]
    
    def simulate_agent_failure(self, agent_id: str, failure_type: str = "high_response_time"):
        """Simulate agent failure for testing auto-recovery (TESTING ONLY)"""
        if not REALTIME_INFRASTRUCTURE_AVAILABLE or not self.health_monitor:
            self.logger.warning("Auto-recovery simulation requires enhanced infrastructure")
            return
        
        self.health_monitor.simulate_agent_failure(agent_id, failure_type)
        self.logger.info(f"🧪 Simulated failure for {agent_id}: {failure_type}")
    
    def enable_auto_recovery(self):
        """Enable auto-recovery system"""
        if not REALTIME_INFRASTRUCTURE_AVAILABLE or not self.health_monitor:
            self.logger.warning("Auto-recovery not available - requires enhanced infrastructure")
            return
        
        self.health_monitor.enable_auto_recovery()
        self.logger.info("🔄 Auto-recovery system enabled")
    
    def disable_auto_recovery(self):
        """Disable auto-recovery system"""
        if not REALTIME_INFRASTRUCTURE_AVAILABLE or not self.health_monitor:
            self.logger.warning("Auto-recovery not available - requires enhanced infrastructure")
            return
        
        self.health_monitor.disable_auto_recovery()
        self.logger.info("🛑 Auto-recovery system disabled")
    
    def get_agent_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive agent health summary including recovery status"""
        if not REALTIME_INFRASTRUCTURE_AVAILABLE or not self.health_monitor:
            return {"health_monitoring_available": False}
        
        return self.health_monitor.get_system_health_summary()

    # ============================================================================
    # DISASTER RECOVERY SYSTEM API 🚑
    # ============================================================================
    
    async def create_cluster_backup(self, cluster_id: str = "main_cluster") -> Dict[str, Any]:
        """Create complete backup of agent cluster state for disaster recovery"""
        if not DISASTER_RECOVERY_AVAILABLE or not self.disaster_recovery:
            self.logger.warning("Disaster recovery not available - requires infrastructure")
            return {"error": "disaster_recovery_not_available"}
        
        try:
            cluster_state = await self.disaster_recovery.create_cluster_backup(cluster_id)
            
            # Log backup creation
            self.logger.info(f"🚑 Created disaster recovery backup for cluster {cluster_id}")
            
            # Publish backup creation event
            await self.publish_event("disaster_recovery_backup_created", {
                "cluster_id": cluster_id,
                "backup_timestamp": cluster_state.timestamp.isoformat(),
                "agents_backed_up": len(cluster_state.agents),
                "checksum": cluster_state.checksum
            })
            
            return {
                "success": True,
                "cluster_id": cluster_id,
                "backup_timestamp": cluster_state.timestamp.isoformat(),
                "agents_count": len(cluster_state.agents),
                "checksum": cluster_state.checksum
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create cluster backup: {e}")
            return {"error": str(e)}
    
    async def restore_cluster_from_backup(self, backup_path: str = None, recovery_type: str = "full_cluster") -> Dict[str, Any]:
        """Restore agent cluster from disaster recovery backup"""
        if not DISASTER_RECOVERY_AVAILABLE or not self.disaster_recovery:
            self.logger.warning("Disaster recovery not available - requires infrastructure")
            return {"error": "disaster_recovery_not_available"}
        
        try:
            # Convert recovery type string to enum
            from ....shared.disaster_recovery.agent_cluster_recovery import RecoveryType
            recovery_type_enum = RecoveryType.FULL_CLUSTER
            if recovery_type == "partial_cluster":
                recovery_type_enum = RecoveryType.PARTIAL_CLUSTER
            elif recovery_type == "single_agent":
                recovery_type_enum = RecoveryType.SINGLE_AGENT
            elif recovery_type == "coordination_only":
                recovery_type_enum = RecoveryType.COORDINATION_ONLY
            
            # Start recovery process
            success = await self.disaster_recovery.restore_cluster_from_backup(backup_path, recovery_type_enum)
            
            if success:
                self.logger.info(f"🚑 Successfully restored cluster from backup")
                
                # Publish recovery success event
                await self.publish_event("disaster_recovery_completed", {
                    "recovery_type": recovery_type,
                    "backup_path": backup_path or "latest",
                    "recovery_time": datetime.now().isoformat(),
                    "success": True
                })
                
                return {
                    "success": True,
                    "recovery_type": recovery_type,
                    "backup_path": backup_path or "latest",
                    "recovery_time": datetime.now().isoformat()
                }
            else:
                self.logger.error(f"❌ Cluster recovery failed")
                return {"error": "cluster_recovery_failed"}
            
        except Exception as e:
            self.logger.error(f"❌ Failed to restore cluster: {e}")
            return {"error": str(e)}
    
    def get_recovery_status(self) -> Dict[str, Any]:
        """Get current disaster recovery system status"""
        if not DISASTER_RECOVERY_AVAILABLE or not self.disaster_recovery:
            return {"disaster_recovery_available": False}
        
        try:
            current_recovery = self.disaster_recovery.current_recovery
            recovery_history = self.disaster_recovery.recovery_history
            
            return {
                "disaster_recovery_available": True,
                "disaster_recovery_enabled": self.disaster_recovery_enabled,
                "current_recovery": current_recovery.__dict__ if current_recovery else None,
                "recovery_history_count": len(recovery_history),
                "last_recovery": recovery_history[-1].__dict__ if recovery_history else None,
                "recovery_time_target_minutes": self.recovery_time_target_minutes,
                "auto_backup_interval_hours": self.auto_backup_interval_hours
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get recovery status: {e}")
            return {"error": str(e)}
    
    async def test_disaster_recovery_system(self) -> Dict[str, Any]:
        """Test disaster recovery system functionality"""
        if not DISASTER_RECOVERY_AVAILABLE or not self.disaster_recovery:
            return {"disaster_recovery_available": False}
        
        try:
            # Run disaster recovery system test
            test_results = await self.disaster_recovery.test_recovery_system()
            
            self.logger.info(f"🧪 Disaster recovery system test completed")
            
            return {
                "test_completed": True,
                "test_results": test_results,
                "overall_status": test_results.get("overall_status", False),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to test disaster recovery system: {e}")
            return {"error": str(e)}

# Export main classes
__all__ = [
    'ModelCommunicationProtocol', 
    'CommunicationMessage', 
    'MessageType', 
    'MessagePriority',
    'PerformanceFeedback',
    'AdaptationPlan'
]
