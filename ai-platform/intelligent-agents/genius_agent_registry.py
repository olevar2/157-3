"""
Central Registry for all Nine Platform3 Genius Agents
Enhanced with real-time communication integration
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
from datetime import datetime, timedelta

# Import enhanced communication protocol
try:
    from ..ai-services.coordination-hub.ModelCommunication import (
        ModelCommunicationProtocol, MessageType, MessagePriority
    )
    COMMUNICATION_PROTOCOL_AVAILABLE = True
except ImportError:
    COMMUNICATION_PROTOCOL_AVAILABLE = False

@dataclass
class AgentHealthMetrics:
    """Health metrics for an agent"""
    last_heartbeat: datetime
    response_time_ms: float
    message_count: int
    error_count: int
    uptime_percentage: float
    status: str = "healthy"  # healthy, degraded, unhealthy, offline

@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for an agent"""
    accuracy: float
    latency_ms: float
    throughput: int
    success_rate: float
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class GeniusAgentInfo:
    name: str
    type: str
    description: str
    primary_function: str
    indicator_requirements: List[str]
    update_frequency: str
    dependencies: List[str]
    performance_metrics: Dict[str, float]
    health_metrics: Optional[AgentHealthMetrics] = None
    real_time_performance: Optional[AgentPerformanceMetrics] = None
    connection_status: str = "disconnected"
    last_status_update: datetime = field(default_factory=datetime.now)
    health_metrics: AgentHealthMetrics

class Platform3GeniusAgentRegistry:
    """Enhanced central registry for all nine genius agents with real-time monitoring and communication"""
    
    def __init__(self, communication_protocol: Optional['ModelCommunicationProtocol'] = None):
        self.agents = self._initialize_all_agents()
        self.health_monitoring_active = True
        self.status_update_callbacks = []
        self.health_check_interval = 30  # seconds
        
        # Enhanced communication integration
        self.communication_protocol = communication_protocol
        self.agent_message_handlers = {}
        self.real_time_updates_enabled = COMMUNICATION_PROTOCOL_AVAILABLE and communication_protocol is not None
        
        # Real-time communication infrastructure
        self.realtime_infrastructure = {
            'health_monitoring': True,
            'message_broadcasting': True,
            'status_updates': True,
            'performance_tracking': True
        }
        
        # Shorthand for realtime access
        self.realtime = self.realtime_infrastructure
        
        if self.real_time_updates_enabled:
            self._setup_communication_integration()
    
    def set_communication_protocol(self, protocol: 'ModelCommunicationProtocol'):
        """Set the communication protocol for real-time integration"""
        self.communication_protocol = protocol
        self.real_time_updates_enabled = True
        self._setup_communication_integration()
    
    def _setup_communication_integration(self):
        """Setup integration with the communication protocol"""
        if not self.communication_protocol:
            return
        
        # Register this registry as a handler for agent status updates
        for agent_name in self.agents.keys():
            try:
                # Register each agent with the communication protocol
                asyncio.create_task(
                    self.communication_protocol.register_model(
                        agent_name,
                        self,  # Use this registry as the model instance
                        self.agents[agent_name].indicator_requirements
                    )
                )
            except Exception as e:
                print(f"Warning: Could not register agent {agent_name} with communication protocol: {e}")
    
    async def handle_message(self, message):
        """Handle messages from the communication protocol"""
        try:
            message_type = message.type.value if hasattr(message.type, 'value') else str(message.type)
            sender = message.sender
            data = message.data
            
            if message_type == 'agent_health_update':
                await self._handle_health_update(sender, data)
            elif message_type == 'agent_performance_update':
                await self._handle_performance_update(sender, data)
            elif message_type == 'agent_status_change':
                await self._handle_status_change(sender, data)
            elif message_type == 'coordination_sync':
                await self._handle_coordination_sync(data)
            
        except Exception as e:
            print(f"Error handling message in agent registry: {e}")
    
    async def _handle_health_update(self, agent_name: str, health_data: Dict[str, Any]):
        """Handle health updates from agents"""
        if agent_name in self.agents:
            health_metrics = AgentHealthMetrics(
                last_heartbeat=datetime.now(),
                response_time_ms=health_data.get('response_time_ms', 0.0),
                message_count=health_data.get('message_count', 0),
                error_count=health_data.get('error_count', 0),
                uptime_percentage=health_data.get('uptime_percentage', 100.0),
                status=health_data.get('status', 'healthy')
            )
            await self.update_agent_health(agent_name, health_metrics)
    
    async def _handle_performance_update(self, agent_name: str, performance_data: Dict[str, Any]):
        """Handle performance updates from agents"""
        if agent_name in self.agents:
            performance_metrics = AgentPerformanceMetrics(
                accuracy=performance_data.get('accuracy', 0.0),
                latency_ms=performance_data.get('latency_ms', 0.0),
                throughput=performance_data.get('throughput', 0),
                success_rate=performance_data.get('success_rate', 0.0)
            )
            await self.update_agent_performance(agent_name, performance_metrics)
    
    async def _handle_status_change(self, agent_name: str, status_data: Dict[str, Any]):
        """Handle status changes from agents"""
        new_status = status_data.get('status', 'unknown')
        await self.update_agent_connection_status(agent_name, new_status)
    
    async def _handle_coordination_sync(self, sync_data: Dict[str, Any]):
        """Handle coordination synchronization messages"""
        action = sync_data.get('action')
        if action == 'agent_registry_sync':
            # Synchronize with other registry instances
            await self._sync_with_remote_registry(sync_data)
        elif action == 'health_check_request':
            # Respond with current health status
            await self._respond_to_health_check()
    
    async def broadcast_agent_status_update(self, agent_name: str, status_type: str, data: Dict[str, Any]):
        """Broadcast agent status updates via communication protocol"""
        if not self.real_time_updates_enabled:
            return
        
        try:
            await self.communication_protocol.broadcast_message(
                MessageType.COORDINATION_SYNC,
                {
                    'action': 'agent_status_broadcast',
                    'agent_name': agent_name,
                    'status_type': status_type,
                    'data': data,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'agent_registry'
                },
                priority=MessagePriority.MEDIUM,
                sender="agent_registry"
            )
        except Exception as e:
            print(f"Error broadcasting agent status: {e}")
    
    async def request_agent_health_check(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Request immediate health check from a specific agent"""
        if not self.real_time_updates_enabled or agent_name not in self.agents:
            return None
        
        try:
            # Send health check request
            success = await self.communication_protocol.send_message_to_model(
                agent_name,
                MessageType.COORDINATION_SYNC,
                {
                    'action': 'health_check_request',
                    'timestamp': datetime.now().isoformat(),
                    'requested_by': 'agent_registry'
                },
                priority=MessagePriority.HIGH,
                sender="agent_registry"
            )
            
            if success:
                # Wait for response (in a real implementation, you'd use a callback)
                await asyncio.sleep(1)  # Give time for response
                return self._get_agent_health_summary(agent_name)
            
        except Exception as e:
            print(f"Error requesting health check for {agent_name}: {e}")
        
        return None
    
    def _get_agent_health_summary(self, agent_name: str) -> Dict[str, Any]:
        """Get comprehensive health summary for an agent"""
        if agent_name not in self.agents:
            return {}
        
        agent = self.agents[agent_name]
        return {
            'agent_name': agent_name,
            'connection_status': agent.connection_status,
            'health_status': agent.health_metrics.status if agent.health_metrics else 'unknown',
            'last_heartbeat': agent.health_metrics.last_heartbeat.isoformat() if agent.health_metrics else None,
            'response_time_ms': agent.health_metrics.response_time_ms if agent.health_metrics else 0,
            'error_count': agent.health_metrics.error_count if agent.health_metrics else 0,
            'uptime_percentage': agent.health_metrics.uptime_percentage if agent.health_metrics else 0,
            'performance': {
                'accuracy': agent.real_time_performance.accuracy if agent.real_time_performance else 0,
                'latency_ms': agent.real_time_performance.latency_ms if agent.real_time_performance else 0,
                'throughput': agent.real_time_performance.throughput if agent.real_time_performance else 0,
                'success_rate': agent.real_time_performance.success_rate if agent.real_time_performance else 0
            } if agent.real_time_performance else None,
            'dependencies_status': self.get_agent_dependencies_status(agent_name)
        }
        
    def _initialize_all_agents(self) -> Dict[str, GeniusAgentInfo]:
        return {
            "risk_genius": GeniusAgentInfo(
                name="Risk Genius",
                type="risk_management",
                description="Advanced risk assessment and portfolio protection",
                primary_function="Analyze and mitigate trading risks in real-time",
                indicator_requirements=[
                    "correlation_analysis", "beta_coefficient", "var_calculator",
                    "volatility_indicators", "drawdown_analyzer"
                ],
                update_frequency="100ms",
                dependencies=[],
                performance_metrics={"accuracy": 0.97, "latency_ms": 0.8},
                health_metrics=AgentHealthMetrics(
                    last_heartbeat=datetime.now(),
                    response_time_ms=0.8,
                    message_count=0,
                    error_count=0,
                    uptime_percentage=100.0
                )
            ),
            
            "session_expert": GeniusAgentInfo(
                name="Session Expert",
                type="temporal_analysis",
                description="Trading session optimization and timing",
                primary_function="Optimize trading based on market sessions",
                indicator_requirements=[
                    "session_indicators", "time_based_patterns", "volatility_by_hour"
                ],
                update_frequency="500ms",
                dependencies=[],
                performance_metrics={"accuracy": 0.95, "latency_ms": 0.9},
                health_metrics=AgentHealthMetrics(
                    last_heartbeat=datetime.now(),
                    response_time_ms=0.9,
                    message_count=0,
                    error_count=0,
                    uptime_percentage=100.0
                )
            ),
            
            "pattern_master": GeniusAgentInfo(
                name="Pattern Master",
                type="pattern_recognition",
                description="Advanced pattern detection and analysis",
                primary_function="Identify and validate trading patterns",
                indicator_requirements=[
                    "pattern_recognition_ai", "harmonic_patterns", "elliott_wave",
                    "japanese_candlesticks", "chart_patterns"
                ],
                update_frequency="200ms",
                dependencies=[],
                performance_metrics={"accuracy": 0.96, "latency_ms": 0.7},
                health_metrics=AgentHealthMetrics(
                    last_heartbeat=datetime.now(),
                    response_time_ms=0.7,
                    message_count=0,
                    error_count=0,
                    uptime_percentage=100.0
                )
            ),
            
            "execution_expert": GeniusAgentInfo(
                name="Execution Expert",
                type="trade_execution",
                description="Optimal trade execution and order management",
                primary_function="Execute trades with minimal slippage",
                indicator_requirements=[
                    "market_microstructure", "liquidity_indicators", "spread_analysis"
                ],
                update_frequency="50ms",
                dependencies=["risk_genius", "pattern_master"],
                performance_metrics={"accuracy": 0.98, "latency_ms": 0.5},
                health_metrics=AgentHealthMetrics(
                    last_heartbeat=datetime.now(),
                    response_time_ms=0.5,
                    message_count=0,
                    error_count=0,
                    uptime_percentage=100.0
                )
            ),
            
            "pair_specialist": GeniusAgentInfo(
                name="Pair Specialist",
                type="currency_analysis",
                description="Currency pair specific analysis and optimization",
                primary_function="Analyze currency pair characteristics",
                indicator_requirements=[
                    "correlation_analysis", "pair_volatility", "spread_patterns"
                ],
                update_frequency="1000ms",
                dependencies=["session_expert"],
                performance_metrics={"accuracy": 0.94, "latency_ms": 0.9},
                health_metrics=AgentHealthMetrics(
                    last_heartbeat=datetime.now(),
                    response_time_ms=0.9,
                    message_count=0,
                    error_count=0,
                    uptime_percentage=100.0
                )
            ),
            
            "decision_master": GeniusAgentInfo(
                name="Decision Master",
                type="decision_making",
                description="Central decision orchestration and validation",
                primary_function="Coordinate and validate trading decisions",
                indicator_requirements=[
                    "all_model_outputs", "confidence_scores", "risk_metrics"
                ],
                update_frequency="100ms",
                dependencies=["risk_genius", "pattern_master", "execution_expert"],
                performance_metrics={"accuracy": 0.97, "latency_ms": 0.6},
                health_metrics=AgentHealthMetrics(
                    last_heartbeat=datetime.now(),
                    response_time_ms=0.6,
                    message_count=0,
                    error_count=0,
                    uptime_percentage=100.0
                )
            ),
            
            "ai_model_coordinator": GeniusAgentInfo(
                name="AI Model Coordinator",
                type="model_orchestration",
                description="Coordinate all AI models and ensure harmony",
                primary_function="Orchestrate AI model interactions",
                indicator_requirements=[
                    "model_performance_metrics", "system_health", "correlation_matrix"
                ],
                update_frequency="500ms",
                dependencies=["all_agents"],
                performance_metrics={"accuracy": 0.96, "latency_ms": 0.8},
                health_metrics=AgentHealthMetrics(
                    last_heartbeat=datetime.now(),
                    response_time_ms=0.8,
                    message_count=0,
                    error_count=0,
                    uptime_percentage=100.0
                )
            ),
            
            "market_microstructure_genius": GeniusAgentInfo(
                name="Market Microstructure Genius",
                type="microstructure_analysis",
                description="Deep market microstructure and order flow analysis",
                primary_function="Analyze order flow and market depth",
                indicator_requirements=[
                    "order_flow_imbalance", "bid_ask_analysis", "volume_profile",
                    "market_depth", "tick_data_analysis"
                ],
                update_frequency="100ms",
                dependencies=["execution_expert"],
                performance_metrics={"accuracy": 0.95, "latency_ms": 0.7},
                health_metrics=AgentHealthMetrics(
                    last_heartbeat=datetime.now(),
                    response_time_ms=0.7,
                    message_count=0,
                    error_count=0,
                    uptime_percentage=100.0
                )
            ),
            
            "sentiment_integration_genius": GeniusAgentInfo(
                name="Sentiment Integration Genius",
                type="sentiment_analysis",
                description="Market sentiment and news impact analysis",
                primary_function="Integrate sentiment data into trading decisions",
                indicator_requirements=[
                    "news_sentiment", "social_sentiment", "fear_greed_index",
                    "market_sentiment_indicators"
                ],
                update_frequency="1000ms",
                dependencies=["decision_master"],
                performance_metrics={"accuracy": 0.92, "latency_ms": 0.9},
                health_metrics=AgentHealthMetrics(
                    last_heartbeat=datetime.now(),
                    response_time_ms=0.9,
                    message_count=0,
                    error_count=0,
                    uptime_percentage=100.0
                )
            )
        }
    
    async def start_monitoring(self):
        """Start real-time health monitoring with communication integration"""
        asyncio.create_task(self._monitor_agent_health())
        
        # Start real-time communication monitoring if available
        if self.real_time_updates_enabled:
            asyncio.create_task(self._monitor_communication_health())
        
    async def update_agent_connection_status(self, agent_name: str, status: str):
        """Update agent connection status and broadcast the change"""
        if agent_name in self.agents:
            old_status = self.agents[agent_name].connection_status
            self.agents[agent_name].connection_status = status
            self.agents[agent_name].last_status_update = datetime.now()
            
            # Notify local callbacks
            await self._notify_status_change(agent_name, status)
            
            # Broadcast via communication protocol
            if self.real_time_updates_enabled and old_status != status:
                await self.broadcast_agent_status_update(
                    agent_name,
                    'connection_status',
                    {
                        'old_status': old_status,
                        'new_status': status,
                        'timestamp': datetime.now().isoformat()
                    }
                )
    
    async def update_agent_health(self, agent_name: str, health_metrics: AgentHealthMetrics):
        """Update agent health metrics and broadcast if critical"""
        if agent_name in self.agents:
            old_health = self.agents[agent_name].health_metrics
            self.agents[agent_name].health_metrics = health_metrics
            
            await self._check_agent_health_status(agent_name)
            
            # Broadcast critical health changes
            if (self.real_time_updates_enabled and 
                health_metrics.status in ['unhealthy', 'offline'] and
                old_health and old_health.status != health_metrics.status):
                
                await self.broadcast_agent_status_update(
                    agent_name,
                    'health_critical',
                    {
                        'status': health_metrics.status,
                        'response_time_ms': health_metrics.response_time_ms,
                        'error_count': health_metrics.error_count,
                        'uptime_percentage': health_metrics.uptime_percentage
                    }
                )
    
    async def update_agent_performance(self, agent_name: str, performance_metrics: AgentPerformanceMetrics):
        """Update agent performance metrics and broadcast significant changes"""
        if agent_name in self.agents:
            old_performance = self.agents[agent_name].real_time_performance
            self.agents[agent_name].real_time_performance = performance_metrics
            
            # Broadcast significant performance changes
            if (self.real_time_updates_enabled and old_performance and
                abs(performance_metrics.accuracy - old_performance.accuracy) > 0.05):  # 5% change
                
                await self.broadcast_agent_status_update(
                    agent_name,
                    'performance_change',
                    {
                        'accuracy': performance_metrics.accuracy,
                        'latency_ms': performance_metrics.latency_ms,
                        'success_rate': performance_metrics.success_rate,
                        'accuracy_change': performance_metrics.accuracy - old_performance.accuracy
                    }
                )
    
    def register_status_callback(self, callback: Callable[[str, str], None]):
        """Register callback for status changes"""
        self.status_update_callbacks.append(callback)
    
    async def _notify_status_change(self, agent_name: str, new_status: str):
        """Notify all callbacks of status changes"""
        for callback in self.status_update_callbacks:
            try:
                await callback(agent_name, new_status)
            except Exception as e:
                print(f"Error in status callback: {e}")
    
    async def _check_agent_health_status(self, agent_name: str):
        """Check and update agent health status"""
        agent = self.agents[agent_name]
        if not agent.health_metrics:
            return
            
        current_time = datetime.now()
        time_since_heartbeat = (current_time - agent.health_metrics.last_heartbeat).total_seconds()
        
        # Determine health status
        if time_since_heartbeat > 120:  # 2 minutes
            new_status = "offline"
        elif agent.health_metrics.error_count > 10 or agent.health_metrics.uptime_percentage < 0.9:
            new_status = "unhealthy"
        elif agent.health_metrics.response_time_ms > 1000:
            new_status = "degraded"
        else:
            new_status = "healthy"
        
        if agent.health_metrics.status != new_status:
            agent.health_metrics.status = new_status
            await self._notify_status_change(agent_name, new_status)
    
    async def _monitor_agent_health(self):
        """Continuously monitor agent health"""
        while self.health_monitoring_active:
            try:
                for agent_name in self.agents:
                    await self._check_agent_health_status(agent_name)
                
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                print(f"Error in health monitoring: {e}")
                await asyncio.sleep(5)
    
    def get_all_agents(self) -> Dict[str, GeniusAgentInfo]:
        """Get information about all genius agents"""
        return self.agents
    
    def get_agent_by_type(self, agent_type: str) -> Optional[GeniusAgentInfo]:
        """Get specific agent by type"""
        return self.agents.get(agent_type)
    
    def get_indicator_requirements_for_agent(self, agent_type: str) -> List[str]:
        """Get indicator requirements for specific agent"""
        agent = self.agents.get(agent_type)
        return agent.indicator_requirements if agent else []
    
    def get_agent_dependencies(self, agent_type: str) -> List[str]:
        """Get dependencies for specific agent"""
        agent = self.agents.get(agent_type)
        return agent.dependencies if agent else []
    
    def get_all_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive status of all agents"""
        status_report = {}
        
        for agent_name, agent_info in self.agents.items():
            status_report[agent_name] = {
                'name': agent_info.name,
                'type': agent_info.type,
                'connection_status': agent_info.connection_status,
                'health_status': agent_info.health_metrics.status if agent_info.health_metrics else "unknown",
                'last_update': agent_info.last_status_update.isoformat(),
                'performance': {
                    'accuracy': agent_info.real_time_performance.accuracy if agent_info.real_time_performance else 0.0,
                    'latency_ms': agent_info.real_time_performance.latency_ms if agent_info.real_time_performance else 0.0,
                    'success_rate': agent_info.real_time_performance.success_rate if agent_info.real_time_performance else 0.0
                } if agent_info.real_time_performance else None
            }
        
        return status_report
    
    def get_healthy_agents(self) -> List[str]:
        """Get list of healthy agents"""
        healthy_agents = []
        for agent_name, agent_info in self.agents.items():
            if (agent_info.connection_status == "connected" and 
                agent_info.health_metrics and 
                agent_info.health_metrics.status == "healthy"):
                healthy_agents.append(agent_name)
        return healthy_agents
    
    def get_agent_dependencies_status(self, agent_name: str) -> Dict[str, str]:
        """Get status of agent dependencies"""
        if agent_name not in self.agents:
            return {}
        
        dependencies = self.agents[agent_name].dependencies
        dependency_status = {}
        
        for dep_name in dependencies:
            if dep_name in self.agents:
                dep_agent = self.agents[dep_name]
                dependency_status[dep_name] = dep_agent.health_metrics.status if dep_agent.health_metrics else "unknown"
        
        return dependency_status

    async def _monitor_communication_health(self):
        """Monitor communication protocol health and connectivity"""
        while self.health_monitoring_active and self.real_time_updates_enabled:
            try:
                if self.communication_protocol:
                    # Get communication metrics
                    metrics = self.communication_protocol.get_communication_metrics()
                    
                    # Check for communication issues
                    if metrics.get('message_delivery_rate', 100) < 95:  # Less than 95% delivery
                        print(f"Warning: Low message delivery rate: {metrics['message_delivery_rate']:.1f}%")
                    
                    if metrics.get('connection_health_rate', 100) < 90:  # Less than 90% healthy connections
                        print(f"Warning: Low connection health: {metrics['connection_health_rate']:.1f}%")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"Error in communication health monitoring: {e}")
                await asyncio.sleep(30)
    
    async def get_enhanced_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get enhanced status including real-time communication metrics"""
        status_report = self.get_all_agent_status()
        
        # Add communication metrics if available
        if self.real_time_updates_enabled and self.communication_protocol:
            try:
                comm_metrics = self.communication_protocol.get_communication_metrics()
                real_time_status = await self.communication_protocol.get_real_time_agent_status()
                
                for agent_name in status_report:
                    if agent_name in real_time_status:
                        status_report[agent_name]['real_time_metrics'] = real_time_status[agent_name]
                
                status_report['_communication_health'] = {
                    'overall_delivery_rate': comm_metrics.get('message_delivery_rate', 0),
                    'connection_health_rate': comm_metrics.get('connection_health_rate', 0),
                    'infrastructure_status': comm_metrics.get('infrastructure_status', 'unknown'),
                    'last_updated': datetime.now().isoformat()
                }
                
            except Exception as e:
                print(f"Error getting enhanced metrics: {e}")
        
        return status_report

# Global registry instance with enhanced capabilities
genius_agent_registry = Platform3GeniusAgentRegistry()
