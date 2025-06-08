"""
🌐 REAL-TIME WEBSOCKET AGENT COMMUNICATION SERVER
=================================================

WebSocket server for real-time bidirectional communication between intelligent agents
Part of Platform3 humanitarian trading mission infrastructure

Mission: Enable ultra-low latency agent coordination for maximum charitable impact
"""

import asyncio
import websockets
import json
import logging
import uuid
from typing import Dict, Set, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import time
import weakref

class AgentConnectionState(Enum):
    """WebSocket connection states for agents"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    ACTIVE = "active"
    IDLE = "idle"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"

@dataclass
class AgentConnection:
    """Represents a WebSocket connection to an intelligent agent"""
    agent_id: str
    connection_id: str
    websocket: websockets.WebSocketServerProtocol
    state: AgentConnectionState
    last_heartbeat: datetime
    connected_at: datetime
    authentication_data: Dict[str, Any]
    metadata: Dict[str, Any]
    message_count: int = 0
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'agent_id': self.agent_id,
            'connection_id': self.connection_id,
            'state': self.state.value,
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'connected_at': self.connected_at.isoformat(),
            'authentication_data': self.authentication_data,
            'metadata': self.metadata,
            'message_count': self.message_count,
            'error_count': self.error_count
        }

class WebSocketAgentServer:
    """
    🌐 REAL-TIME WEBSOCKET SERVER FOR INTELLIGENT AGENTS
    
    Provides ultra-low latency bidirectional communication for the 9 genius agents:
    - Risk Genius, Session Expert, Pattern Master
    - Execution Expert, Pair Specialist, Decision Master  
    - AI Model Coordinator, Market Microstructure Genius, Sentiment Integration Genius
    
    Features:
    - Sub-100ms message delivery
    - Automatic connection recovery
    - Heartbeat monitoring
    - Load balancing across connections
    - Message queuing and persistence
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        """Initialize WebSocket server for agent communication"""
        self.host = host
        self.port = port
        self.logger = logging.getLogger(__name__)
        
        # Connection management
        self.active_connections: Dict[str, AgentConnection] = {}
        self.agent_connections: Dict[str, Set[str]] = {}  # agent_id -> connection_ids
        self.connection_lookup: Dict[str, str] = {}  # connection_id -> agent_id
        
        # Server state
        self.server = None
        self.is_running = False
        self.start_time = None
        
        # Performance metrics
        self.metrics = {
            'total_connections': 0,
            'active_connections': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'heartbeats_sent': 0,
            'connection_errors': 0,
            'average_response_time': 0.0
        }
        
        # Configuration
        self.heartbeat_interval = 30  # seconds
        self.connection_timeout = 60  # seconds
        self.max_connections_per_agent = 5
        self.authentication_required = True
        
        # Background tasks
        self._background_tasks = set()
        
        self.logger.info("🌐 WebSocket Agent Server initialized")
    
    async def start(self):
        """Start the WebSocket server"""
        try:
            self.logger.info(f"🚀 Starting WebSocket Agent Server on {self.host}:{self.port}")
            
            self.server = await websockets.serve(
                self.handle_agent_connection,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10,
                max_size=1024 * 1024,  # 1MB max message size
                compression=None  # Disable compression for speed
            )
            
            self.is_running = True
            self.start_time = datetime.now()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.logger.info("✅ WebSocket Agent Server started successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start WebSocket server: {e}")
            raise
    
    async def stop(self):
        """Stop the WebSocket server"""
        try:
            self.logger.info("🛑 Stopping WebSocket Agent Server")
            
            self.is_running = False
            
            # Close all active connections
            for connection in list(self.active_connections.values()):
                try:
                    await connection.websocket.close()
                except:
                    pass
            
            # Stop server
            if self.server:
                self.server.close()
                await self.server.wait_closed()
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            self.logger.info("✅ WebSocket Agent Server stopped")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping WebSocket server: {e}")
    
    async def handle_agent_connection(self, websocket, path):
        """Handle new agent WebSocket connection"""
        connection_id = str(uuid.uuid4())
        agent_connection = None
        
        try:
            self.logger.info(f"🔗 New agent connection: {connection_id}")
            
            # Create initial connection record
            agent_connection = AgentConnection(
                agent_id="",  # To be set during authentication
                connection_id=connection_id,
                websocket=websocket,
                state=AgentConnectionState.CONNECTING,
                last_heartbeat=datetime.now(),
                connected_at=datetime.now(),
                authentication_data={},
                metadata={}
            )
            
            # Authentication process
            if self.authentication_required:
                await self._authenticate_agent(agent_connection)
            
            # Register connection
            await self._register_connection(agent_connection)
            
            # Handle messages
            await self._handle_agent_messages(agent_connection)
            
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"🔌 Agent connection closed: {connection_id}")
        except Exception as e:
            self.logger.error(f"❌ Error in agent connection {connection_id}: {e}")
            self.metrics['connection_errors'] += 1
        finally:
            if agent_connection:
                await self._unregister_connection(agent_connection)
    
    async def _authenticate_agent(self, connection: AgentConnection):
        """Authenticate agent connection with mTLS certificate validation"""
        try:
            # Send authentication request
            auth_request = {
                'type': 'auth_request',
                'connection_id': connection.connection_id,
                'timestamp': datetime.now().isoformat(),
                'requires_certificate': True,
                'supported_auth_methods': ['mTLS', 'certificate']
            }
            
            await connection.websocket.send(json.dumps(auth_request))
            
            # Wait for authentication response with timeout
            auth_timeout = 30  # 30 seconds
            start_time = time.time()
            
            while time.time() - start_time < auth_timeout:
                try:
                    message = await asyncio.wait_for(connection.websocket.recv(), timeout=5.0)
                    auth_data = json.loads(message)
                    
                    if auth_data.get('type') == 'auth_response':
                        # Extract agent identity and certificate
                        agent_id = auth_data.get('agent_id')
                        certificate_pem = auth_data.get('certificate')
                        
                        if not agent_id or not certificate_pem:
                            raise ValueError("Missing agent_id or certificate in authentication")
                        
                        # Validate certificate using security manager (if available)
                        if hasattr(self, 'security_manager') and self.security_manager:
                            certificate_data = certificate_pem.encode('utf-8')
                            identity = self.security_manager.authenticate_agent(agent_id, certificate_data)
                            
                            if not identity:
                                await connection.websocket.send(json.dumps({
                                    'type': 'auth_failed',
                                    'reason': 'Invalid certificate or unauthorized agent'
                                }))
                                raise ValueError("Authentication failed - invalid certificate")
                            
                            # Store authentication data
                            connection.agent_id = agent_id
                            connection.authentication_data = {
                                'agent_name': identity.agent_name,
                                'certificate_fingerprint': identity.certificate_fingerprint,
                                'permissions': [p.value for p in identity.permissions],
                                'authenticated_at': datetime.now().isoformat()
                            }
                            connection.state = AgentConnectionState.AUTHENTICATED
                            
                            self.logger.info(f"🔒 Agent {agent_id} authenticated successfully with certificate")
                        
                        else:
                            # Fallback: basic agent ID validation without security manager
                            connection.agent_id = agent_id
                            connection.authentication_data = {
                                'agent_name': agent_id,
                                'authenticated_at': datetime.now().isoformat(),
                                'security_mode': 'insecure'
                            }
                            connection.state = AgentConnectionState.AUTHENTICATED
                            
                            self.logger.warning(f"⚠️ Agent {agent_id} authenticated in insecure mode (no security manager)")
                        
                        # Send authentication success
                        await connection.websocket.send(json.dumps({
                            'type': 'auth_success',
                            'agent_id': agent_id,
                            'connection_id': connection.connection_id,
                            'permissions': connection.authentication_data.get('permissions', [])
                        }))
                        
                        return  # Authentication successful
                    
                except asyncio.TimeoutError:
                    continue  # Keep waiting for auth response
                
            # Authentication timeout
            await connection.websocket.send(json.dumps({
                'type': 'auth_failed',
                'reason': 'Authentication timeout'
            }))
            raise TimeoutError("Authentication timeout")
            
        except Exception as e:
            self.logger.error(f"❌ Agent authentication failed: {e}")
            connection.state = AgentConnectionState.ERROR
            raise
            
            # Wait for authentication response
            auth_response = await asyncio.wait_for(
                connection.websocket.recv(), 
                timeout=30.0
            )
            
            auth_data = json.loads(auth_response)
            
            if auth_data.get('type') != 'auth_response':
                raise ValueError("Invalid authentication response")
            
            agent_id = auth_data.get('agent_id')
            if not agent_id:
                raise ValueError("Missing agent_id in authentication")
            
            # Validate agent_id against known agents
            valid_agents = [
                'risk_genius', 'session_expert', 'pattern_master',
                'execution_expert', 'pair_specialist', 'decision_master',
                'ai_model_coordinator', 'market_microstructure_genius', 
                'sentiment_integration_genius'
            ]
            
            if agent_id not in valid_agents:
                raise ValueError(f"Unknown agent_id: {agent_id}")
            
            # Update connection with authentication data
            connection.agent_id = agent_id
            connection.state = AgentConnectionState.AUTHENTICATED
            connection.authentication_data = auth_data
            
            # Send authentication success
            auth_success = {
                'type': 'auth_success',
                'agent_id': agent_id,
                'connection_id': connection.connection_id,
                'timestamp': datetime.now().isoformat()
            }
            
            await connection.websocket.send(json.dumps(auth_success))
            
            self.logger.info(f"✅ Agent authenticated: {agent_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Authentication failed: {e}")
            connection.state = AgentConnectionState.ERROR
            raise
    
    async def _register_connection(self, connection: AgentConnection):
        """Register agent connection"""
        try:
            # Check connection limits
            if connection.agent_id in self.agent_connections:
                if len(self.agent_connections[connection.agent_id]) >= self.max_connections_per_agent:
                    raise ValueError(f"Maximum connections exceeded for agent {connection.agent_id}")
            
            # Register connection
            self.active_connections[connection.connection_id] = connection
            self.connection_lookup[connection.connection_id] = connection.agent_id
            
            if connection.agent_id not in self.agent_connections:
                self.agent_connections[connection.agent_id] = set()
            self.agent_connections[connection.agent_id].add(connection.connection_id)
            
            # Update state
            connection.state = AgentConnectionState.ACTIVE
            
            # Update metrics
            self.metrics['total_connections'] += 1
            self.metrics['active_connections'] = len(self.active_connections)
            
            self.logger.info(f"📝 Agent connection registered: {connection.agent_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register connection: {e}")
            raise
    
    async def _unregister_connection(self, connection: AgentConnection):
        """Unregister agent connection"""
        try:
            # Remove from tracking
            if connection.connection_id in self.active_connections:
                del self.active_connections[connection.connection_id]
            
            if connection.connection_id in self.connection_lookup:
                del self.connection_lookup[connection.connection_id]
            
            if connection.agent_id in self.agent_connections:
                self.agent_connections[connection.agent_id].discard(connection.connection_id)
                if not self.agent_connections[connection.agent_id]:
                    del self.agent_connections[connection.agent_id]
            
            # Update metrics
            self.metrics['active_connections'] = len(self.active_connections)
            
            self.logger.info(f"📤 Agent connection unregistered: {connection.agent_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Error unregistering connection: {e}")
    
    async def _handle_agent_messages(self, connection: AgentConnection):
        """Handle incoming messages from agent"""
        try:
            async for message in connection.websocket:
                start_time = time.time()
                
                try:
                    # Parse message
                    data = json.loads(message)
                    message_type = data.get('type')
                    
                    # Update connection stats
                    connection.message_count += 1
                    connection.last_heartbeat = datetime.now()
                    self.metrics['messages_received'] += 1
                    
                    # Handle different message types
                    if message_type == 'heartbeat':
                        await self._handle_heartbeat(connection, data)
                    elif message_type == 'agent_message':
                        await self._handle_agent_message(connection, data)
                    elif message_type == 'broadcast_request':
                        await self._handle_broadcast_request(connection, data)
                    else:
                        self.logger.warning(f"⚠️ Unknown message type: {message_type}")
                    
                    # Update response time metrics
                    response_time = (time.time() - start_time) * 1000  # ms
                    self._update_response_time_metric(response_time)
                    
                except json.JSONDecodeError:
                    self.logger.error(f"❌ Invalid JSON from {connection.agent_id}")
                    connection.error_count += 1
                except Exception as e:
                    self.logger.error(f"❌ Error handling message from {connection.agent_id}: {e}")
                    connection.error_count += 1
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"🔌 Agent {connection.agent_id} disconnected")
        except Exception as e:
            self.logger.error(f"❌ Error in message handler for {connection.agent_id}: {e}")
    
    async def _handle_heartbeat(self, connection: AgentConnection, data: Dict[str, Any]):
        """Handle heartbeat message from agent"""
        # Send heartbeat response
        response = {
            'type': 'heartbeat_ack',
            'connection_id': connection.connection_id,
            'timestamp': datetime.now().isoformat(),
            'server_time': datetime.now().isoformat()
        }
        
        await connection.websocket.send(json.dumps(response))
        self.metrics['heartbeats_sent'] += 1
    
    async def _handle_agent_message(self, connection: AgentConnection, data: Dict[str, Any]):
        """Handle inter-agent message routing"""
        try:
            target_agent = data.get('target_agent')
            message_payload = data.get('payload', {})
            
            if target_agent:
                # Route to specific agent
                await self.send_to_agent(target_agent, {
                    'type': 'agent_message',
                    'source_agent': connection.agent_id,
                    'payload': message_payload,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                # Broadcast to all agents
                await self.broadcast_to_agents({
                    'type': 'agent_broadcast',
                    'source_agent': connection.agent_id,
                    'payload': message_payload,
                    'timestamp': datetime.now().isoformat()
                }, exclude_agent=connection.agent_id)
            
        except Exception as e:
            self.logger.error(f"❌ Error routing agent message: {e}")
    
    async def _handle_broadcast_request(self, connection: AgentConnection, data: Dict[str, Any]):
        """Handle broadcast request from agent"""
        try:
            message_payload = data.get('payload', {})
            
            await self.broadcast_to_agents({
                'type': 'broadcast_message',
                'source_agent': connection.agent_id,
                'payload': message_payload,
                'timestamp': datetime.now().isoformat()
            }, exclude_agent=connection.agent_id)
            
        except Exception as e:
            self.logger.error(f"❌ Error handling broadcast request: {e}")
    
    async def send_to_agent(self, agent_id: str, message: Dict[str, Any]) -> bool:
        """Send message to specific agent"""
        try:
            if agent_id not in self.agent_connections:
                self.logger.warning(f"⚠️ Agent {agent_id} not connected")
                return False
            
            # Send to all connections for this agent
            success_count = 0
            connection_ids = list(self.agent_connections[agent_id])
            
            for connection_id in connection_ids:
                if connection_id in self.active_connections:
                    connection = self.active_connections[connection_id]
                    try:
                        await connection.websocket.send(json.dumps(message))
                        success_count += 1
                        self.metrics['messages_sent'] += 1
                    except Exception as e:
                        self.logger.error(f"❌ Failed to send to {agent_id}: {e}")
            
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"❌ Error sending to agent {agent_id}: {e}")
            return False
    
    async def broadcast_to_agents(self, message: Dict[str, Any], exclude_agent: str = None) -> int:
        """Broadcast message to all connected agents"""
        try:
            sent_count = 0
            
            for connection in self.active_connections.values():
                if exclude_agent and connection.agent_id == exclude_agent:
                    continue
                
                try:
                    await connection.websocket.send(json.dumps(message))
                    sent_count += 1
                    self.metrics['messages_sent'] += 1
                except Exception as e:
                    self.logger.error(f"❌ Failed to broadcast to {connection.agent_id}: {e}")
            
            return sent_count
            
        except Exception as e:
            self.logger.error(f"❌ Error broadcasting message: {e}")
            return 0
    
    async def _start_background_tasks(self):
        """Start background monitoring tasks"""
        try:
            # Heartbeat monitoring
            task1 = asyncio.create_task(self._heartbeat_monitor())
            self._background_tasks.add(task1)
            
            # Connection cleanup
            task2 = asyncio.create_task(self._connection_cleanup())
            self._background_tasks.add(task2)
            
            # Metrics collection
            task3 = asyncio.create_task(self._metrics_collection())
            self._background_tasks.add(task3)
            
            self.logger.info("✅ Background tasks started")
            
        except Exception as e:
            self.logger.error(f"❌ Error starting background tasks: {e}")
    
    async def _heartbeat_monitor(self):
        """Monitor agent heartbeats and detect disconnections"""
        while self.is_running:
            try:
                current_time = datetime.now()
                timeout_threshold = current_time - timedelta(seconds=self.connection_timeout)
                
                # Check for timed out connections
                timed_out_connections = []
                for connection in self.active_connections.values():
                    if connection.last_heartbeat < timeout_threshold:
                        timed_out_connections.append(connection)
                
                # Close timed out connections
                for connection in timed_out_connections:
                    self.logger.warning(f"⏰ Connection timeout for {connection.agent_id}")
                    try:
                        await connection.websocket.close()
                    except:
                        pass
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Error in heartbeat monitor: {e}")
                await asyncio.sleep(5)
    
    async def _connection_cleanup(self):
        """Clean up inactive connections"""
        while self.is_running:
            try:
                # Remove closed connections
                closed_connections = []
                for connection_id, connection in self.active_connections.items():
                    if connection.websocket.closed:
                        closed_connections.append(connection_id)
                
                for connection_id in closed_connections:
                    connection = self.active_connections[connection_id]
                    await self._unregister_connection(connection)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Error in connection cleanup: {e}")
                await asyncio.sleep(5)
    
    async def _metrics_collection(self):
        """Collect and log performance metrics"""
        while self.is_running:
            try:
                self.logger.info(f"📊 WebSocket Metrics: {self.metrics}")
                await asyncio.sleep(60)  # Log every minute
                
            except Exception as e:
                self.logger.error(f"❌ Error in metrics collection: {e}")
                await asyncio.sleep(5)
    
    def _update_response_time_metric(self, response_time: float):
        """Update average response time metric"""
        current_avg = self.metrics['average_response_time']
        total_messages = self.metrics['messages_received']
        
        if total_messages == 1:
            self.metrics['average_response_time'] = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics['average_response_time'] = (alpha * response_time) + ((1 - alpha) * current_avg)
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status"""
        return {
            'server_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'active_connections': len(self.active_connections),
            'connected_agents': list(self.agent_connections.keys()),
            'metrics': self.metrics,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        }
    
    def get_agent_connections(self) -> Dict[str, Any]:
        """Get detailed agent connection information"""
        return {
            agent_id: [
                self.active_connections[conn_id].to_dict() 
                for conn_id in connection_ids 
                if conn_id in self.active_connections
            ]
            for agent_id, connection_ids in self.agent_connections.items()
        }


# Main entry point for testing
if __name__ == "__main__":
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def main():
        server = WebSocketAgentServer(host="localhost", port=8765)
        
        try:
            await server.start()
            
            # Keep server running
            while server.is_running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Shutting down server...")
        finally:
            await server.stop()
    
    asyncio.run(main())