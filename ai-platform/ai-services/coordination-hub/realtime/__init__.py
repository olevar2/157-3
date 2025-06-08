"""
Real-time Agent Communication Infrastructure
==========================================

Enhanced real-time communication modules for Platform3 agent coordination:
- AgentHealthMonitor: Real-time health tracking and heartbeat monitoring
- WebSocketAgentServer: Bidirectional WebSocket communication for agents
- MessageQueueManager: Reliable message queuing with Redis/Kafka support

Mission: Enable seamless real-time coordination between 25+ AI trading agents
for maximum humanitarian trading profits and global poverty alleviation.
"""

from .agent_health_monitor import (
    AgentHealthMonitor,
    AgentHealthStatus,
    AgentHealthMetrics,
    PerformanceMetric
)

from .websocket_agent_server import (
    WebSocketAgentServer,
    AgentConnection,
    AgentConnectionState
)

from .message_queue_manager import (
    MessageQueueManager,
    MessageQueueType,
    MessagePriority
)

__all__ = [
    'AgentHealthMonitor',
    'AgentHealthStatus', 
    'AgentHealthMetrics',
    'PerformanceMetric',
    'WebSocketAgentServer',
    'AgentConnection',
    'AgentConnectionState',
    'MessageQueueManager',
    'MessageQueueType',
    'MessagePriority'
]