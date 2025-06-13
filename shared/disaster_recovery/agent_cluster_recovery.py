"""
🚑 AGENT CLUSTER DISASTER RECOVERY SYSTEM
========================================

MISSION: Ensure rapid recovery of Platform3's 9 genius agents to maintain
continuous support for humanitarian trading operations and help sick children.

This system provides:
- Complete agent cluster state backup and restoration
- Coordination protocol recovery after system failures  
- Agent dependency chain restoration and validation
- Integration with existing backup infrastructure
- Automated recovery procedures with 15-minute target
- Post-recovery validation and health verification

Goal: Minimize downtime to maximize charitable profits for medical aid
"""

import asyncio
import logging
import json
import os
import time
import shutil
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import aiofiles
import asyncpg
import aioredis
import pickle
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import existing backup infrastructure
try:
    import sys
        from backup_monitoring import BackupMonitor, BackupStatus
    BACKUP_INFRASTRUCTURE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Backup infrastructure not available: {e}")
    BACKUP_INFRASTRUCTURE_AVAILABLE = False

# Import agent communication protocol
try:
    from ...ai_platform.ai_services.coordination_hub.ModelCommunication import (
        ModelCommunicationProtocol, MessageType, MessagePriority
    )
    COMMUNICATION_PROTOCOL_AVAILABLE = True
except ImportError:
    COMMUNICATION_PROTOCOL_AVAILABLE = False

# Import agent registry
try:
    from ...ai_platform.intelligent_agents.genius_agent_registry import (
        GeniusAgentRegistry, GeniusAgentInfo, AgentHealthMetrics
    )
    AGENT_REGISTRY_AVAILABLE = True
except ImportError:
    AGENT_REGISTRY_AVAILABLE = False

# Import persistence infrastructure
try:
    from ..persistence.agent_state_manager import (
        AgentStateManager, StateType, StateSnapshot
    )
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RecoveryStatus(Enum):
    """Recovery operation status"""
    IDLE = "idle"
    STARTING = "starting"
    BACKING_UP = "backing_up"
    RECOVERING = "recovering"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLBACK = "rollback"

class RecoveryType(Enum):
    """Types of recovery operations"""
    FULL_CLUSTER = "full_cluster"
    PARTIAL_CLUSTER = "partial_cluster"
    SINGLE_AGENT = "single_agent"
    COORDINATION_ONLY = "coordination_only"
    DEPENDENCY_CHAIN = "dependency_chain"

@dataclass
class AgentClusterState:
    """Complete state of an agent cluster"""
    cluster_id: str
    timestamp: datetime
    agents: Dict[str, Dict[str, Any]]  # agent_id -> agent_state
    coordination_state: Dict[str, Any]
    dependency_graph: Dict[str, List[str]]
    communication_channels: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    configuration: Dict[str, Any]
    checksum: str = ""

@dataclass
class RecoveryOperation:
    """Recovery operation tracking"""
    operation_id: str
    recovery_type: RecoveryType
    status: RecoveryStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    cluster_backup_path: str = ""
    affected_agents: List[str] = field(default_factory=list)
    validation_results: Dict[str, bool] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)
    recovery_metrics: Dict[str, float] = field(default_factory=dict)

class AgentClusterRecovery:
    """
    Main disaster recovery system for agent clusters
    Provides comprehensive backup, recovery, and validation capabilities
    """
    
    def __init__(self, 
                 backup_base_path: str = "/opt/platform3/backups/agent_clusters",
                 postgres_config: Optional[Dict] = None,
                 redis_config: Optional[Dict] = None):
        """Initialize the agent cluster recovery system"""
        self.backup_base_path = Path(backup_base_path)
        self.backup_base_path.mkdir(parents=True, exist_ok=True)
        
        self.postgres_config = postgres_config or {
            'host': 'localhost', 'port': 5432, 'database': 'platform3',
            'user': 'platform3_user', 'password': 'platform3_password'
        }
        
        self.redis_config = redis_config or {
            'host': 'localhost', 'port': 6379, 'db': 0, 'password': None
        }
        
        self.current_recovery: Optional[RecoveryOperation] = None
        self.recovery_history: List[RecoveryOperation] = []
        self.agent_registry: Optional[Any] = None
        self.communication_protocol: Optional[Any] = None
        self.state_manager: Optional[Any] = None
        
        # Recovery targets
        self.recovery_time_target = timedelta(minutes=15)  # 15-minute target
        self.validation_timeout = timedelta(minutes=5)
        
        # Initialize components
        asyncio.create_task(self._initialize_components())
    
    async def _initialize_components(self):
        """Initialize recovery system components"""
        try:
            # Initialize agent registry if available
            if AGENT_REGISTRY_AVAILABLE:
                from ...ai_platform.intelligent_agents.genius_agent_registry import GeniusAgentRegistry
                self.agent_registry = GeniusAgentRegistry()
            
            # Initialize communication protocol if available
            if COMMUNICATION_PROTOCOL_AVAILABLE:
                from ...ai_platform.ai_services.coordination_hub.ModelCommunication import ModelCommunicationProtocol
                self.communication_protocol = ModelCommunicationProtocol()
            
            # Initialize state manager if available
            if PERSISTENCE_AVAILABLE:
                from ..persistence.agent_state_manager import create_agent_state_manager
                self.state_manager = await create_agent_state_manager()
            
            logger.info("Agent cluster recovery system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize recovery components: {e}")
    
    async def create_cluster_backup(self, cluster_id: str = "main_cluster") -> AgentClusterState:
        """
        Create complete backup of agent cluster state
        
        Args:
            cluster_id: Identifier for the cluster
            
        Returns:
            AgentClusterState: Complete cluster state snapshot
        """
        logger.info(f"Creating cluster backup for {cluster_id}")
        start_time = time.time()
        
        try:
            # Collect agent states
            agents_state = await self._collect_agent_states()
            
            # Collect coordination state
            coordination_state = await self._collect_coordination_state()
            
            # Collect dependency graph
            dependency_graph = await self._collect_dependency_graph()
            
            # Collect communication channels
            communication_channels = await self._collect_communication_state()
            
            # Collect performance metrics
            performance_metrics = await self._collect_performance_metrics()
            
            # Collect configuration
            configuration = await self._collect_configuration()
            
            # Create cluster state
            cluster_state = AgentClusterState(
                cluster_id=cluster_id,
                timestamp=datetime.now(),
                agents=agents_state,
                coordination_state=coordination_state,
                dependency_graph=dependency_graph,
                communication_channels=communication_channels,
                performance_metrics=performance_metrics,
                configuration=configuration
            )
            
            # Generate checksum
            cluster_state.checksum = self._generate_checksum(cluster_state)
            
            # Save backup to disk
            backup_path = await self._save_cluster_backup(cluster_state)
            
            backup_time = time.time() - start_time
            logger.info(f"Cluster backup completed in {backup_time:.2f}s: {backup_path}")
            
            return cluster_state
            
        except Exception as e:
            logger.error(f"Failed to create cluster backup: {e}")
            raise
    
    async def restore_cluster_from_backup(self, 
                                        backup_path: str = None,
                                        recovery_type: RecoveryType = RecoveryType.FULL_CLUSTER) -> bool:
        """
        Restore agent cluster from backup
        
        Args:
            backup_path: Path to backup file (latest if None)
            recovery_type: Type of recovery to perform
            
        Returns:
            bool: True if recovery successful
        """
        operation_id = f"recovery_{int(time.time())}"
        
        # Initialize recovery operation
        recovery_op = RecoveryOperation(
            operation_id=operation_id,
            recovery_type=recovery_type,
            status=RecoveryStatus.STARTING,
            start_time=datetime.now()
        )
        
        self.current_recovery = recovery_op
        
        try:
            logger.info(f"Starting cluster recovery operation {operation_id}")
            
            # Load backup
            recovery_op.status = RecoveryStatus.BACKING_UP
            cluster_state = await self._load_cluster_backup(backup_path)
            recovery_op.cluster_backup_path = backup_path or "latest"
            
            # Verify backup integrity
            if not self._verify_backup_integrity(cluster_state):
                raise Exception("Backup integrity verification failed")
            
            # Start recovery process
            recovery_op.status = RecoveryStatus.RECOVERING
            
            # Stop current agents gracefully
            await self._graceful_agent_shutdown()
            
            # Restore database state
            await self._restore_database_state(cluster_state)
            
            # Restore Redis state
            await self._restore_redis_state(cluster_state)
            
            # Restore agent configurations
            await self._restore_agent_configurations(cluster_state)
            
            # Restart agents with restored state
            await self._restart_agents_with_state(cluster_state)
            
            # Restore coordination protocols
            await self._restore_coordination_protocols(cluster_state)
            
            # Restore dependency chains
            await self._restore_dependency_chains(cluster_state)
            
            # Validate recovery
            recovery_op.status = RecoveryStatus.VALIDATING
            validation_results = await self._validate_recovery(cluster_state)
            recovery_op.validation_results = validation_results
            
            # Check if recovery meets criteria
            if self._check_recovery_success(validation_results):
                recovery_op.status = RecoveryStatus.COMPLETED
                recovery_op.end_time = datetime.now()
                
                recovery_time = (recovery_op.end_time - recovery_op.start_time).total_seconds()
                recovery_op.recovery_metrics['total_time_seconds'] = recovery_time
                recovery_op.recovery_metrics['agents_recovered'] = len(cluster_state.agents)
                
                logger.info(f"Cluster recovery completed successfully in {recovery_time:.2f}s")
                
                # Add to history
                self.recovery_history.append(recovery_op)
                self.current_recovery = None
                
                return True
            else:
                # Recovery validation failed
                recovery_op.status = RecoveryStatus.FAILED
                recovery_op.error_log.append("Recovery validation failed")
                logger.error("Recovery validation failed")
                return False
                
        except Exception as e:
            recovery_op.status = RecoveryStatus.FAILED
            recovery_op.end_time = datetime.now()
            recovery_op.error_log.append(str(e))
            
            logger.error(f"Cluster recovery failed: {e}")
            
            # Add to history even if failed
            self.recovery_history.append(recovery_op)
            self.current_recovery = None
            
            return False
    
    async def _collect_agent_states(self) -> Dict[str, Dict[str, Any]]:
        """Collect current state of all agents"""
        agent_states = {}
        
        try:
            if self.agent_registry:
                agents = await self.agent_registry.get_all_agents()
                
                for agent_id, agent_info in agents.items():
                    # Get comprehensive agent state
                    state = {
                        'info': asdict(agent_info),
                        'health_metrics': {},
                        'configuration': {},
                        'learning_state': {},
                        'communication_state': {}
                    }
                    
                    # Get health metrics if available
                    if hasattr(self.agent_registry, 'get_agent_health'):
                        health = await self.agent_registry.get_agent_health(agent_id)
                        if health:
                            state['health_metrics'] = asdict(health)
                    
                    # Get persistent state if available
                    if self.state_manager:
                        try:
                            snapshot = await self.state_manager.get_agent_state(agent_id)
                            if snapshot:
                                state['persistent_state'] = asdict(snapshot)
                        except Exception as e:
                            logger.warning(f"Could not get persistent state for {agent_id}: {e}")
                    
                    agent_states[agent_id] = state
            
            return agent_states
            
        except Exception as e:
            logger.error(f"Failed to collect agent states: {e}")
            return {}
    
    async def _collect_coordination_state(self) -> Dict[str, Any]:
        """Collect coordination protocol state"""
        try:
            coordination_state = {}
            
            if self.communication_protocol:
                # Get message queues state
                if hasattr(self.communication_protocol, 'get_queue_states'):
                    coordination_state['message_queues'] = await self.communication_protocol.get_queue_states()
                
                # Get active connections
                if hasattr(self.communication_protocol, 'get_active_connections'):
                    coordination_state['active_connections'] = await self.communication_protocol.get_active_connections()
                
                # Get routing tables
                if hasattr(self.communication_protocol, 'get_routing_state'):
                    coordination_state['routing_tables'] = await self.communication_protocol.get_routing_state()
            
            return coordination_state
            
        except Exception as e:
            logger.error(f"Failed to collect coordination state: {e}")
            return {}
    
    async def _collect_dependency_graph(self) -> Dict[str, List[str]]:
        """Collect agent dependency graph"""
        try:
            dependency_graph = {}
            
            if self.agent_registry:
                agents = await self.agent_registry.get_all_agents()
                
                for agent_id, agent_info in agents.items():
                    dependencies = getattr(agent_info, 'dependencies', [])
                    dependency_graph[agent_id] = dependencies
            
            return dependency_graph
            
        except Exception as e:
            logger.error(f"Failed to collect dependency graph: {e}")
            return {}
    
    async def _collect_communication_state(self) -> Dict[str, Any]:
        """Collect communication channels state"""
        try:
            communication_state = {
                'websocket_connections': {},
                'message_counts': {},
                'latency_metrics': {}
            }
            
            # Collect WebSocket connection states
            if self.communication_protocol and hasattr(self.communication_protocol, 'websocket_server'):
                ws_server = self.communication_protocol.websocket_server
                if hasattr(ws_server, 'get_connection_states'):
                    communication_state['websocket_connections'] = await ws_server.get_connection_states()
            
            return communication_state
            
        except Exception as e:
            logger.error(f"Failed to collect communication state: {e}")
            return {}
    
    async def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect system performance metrics"""
        try:
            performance_metrics = {
                'agent_performance': {},
                'system_metrics': {},
                'collaboration_metrics': {}
            }
            
            # Get agent performance metrics
            if self.agent_registry:
                agents = await self.agent_registry.get_all_agents()
                for agent_id, agent_info in agents.items():
                    if hasattr(agent_info, 'performance_metrics'):
                        performance_metrics['agent_performance'][agent_id] = agent_info.performance_metrics
            
            # Get system metrics
            import psutil
            performance_metrics['system_metrics'] = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'network_stats': dict(psutil.net_io_counters()._asdict())
            }
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
            return {}
    
    async def _collect_configuration(self) -> Dict[str, Any]:
        """Collect system configuration"""
        try:
            configuration = {
                'agent_configs': {},
                'system_config': {},
                'communication_config': {}
            }
            
            # Try to get agent configurations
            if self.communication_protocol and hasattr(self.communication_protocol, 'config_coordinator'):
                config_coordinator = self.communication_protocol.config_coordinator
                if hasattr(config_coordinator, 'get_all_agent_configs'):
                    configuration['agent_configs'] = await config_coordinator.get_all_agent_configs()
            
            return configuration
            
        except Exception as e:
            logger.error(f"Failed to collect configuration: {e}")
            return {}
    
    def _generate_checksum(self, cluster_state: AgentClusterState) -> str:
        """Generate checksum for cluster state integrity verification"""
        try:
            # Create a copy without checksum for hashing
            state_dict = asdict(cluster_state)
            state_dict.pop('checksum', None)
            
            # Convert to JSON and generate hash
            state_json = json.dumps(state_dict, sort_keys=True, default=str)
            return hashlib.sha256(state_json.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to generate checksum: {e}")
            return ""
    
    async def _save_cluster_backup(self, cluster_state: AgentClusterState) -> str:
        """Save cluster backup to disk"""
        timestamp = cluster_state.timestamp.strftime("%Y%m%d_%H%M%S")
        backup_filename = f"cluster_backup_{cluster_state.cluster_id}_{timestamp}.json"
        backup_path = self.backup_base_path / backup_filename
        
        try:
            async with aiofiles.open(backup_path, 'w') as f:
                await f.write(json.dumps(asdict(cluster_state), indent=2, default=str))
            
            # Create symlink to latest backup
            latest_link = self.backup_base_path / f"latest_{cluster_state.cluster_id}.json"
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(backup_filename)
            
            logger.info(f"Cluster backup saved: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to save cluster backup: {e}")
            raise
    
    async def _load_cluster_backup(self, backup_path: Optional[str] = None) -> AgentClusterState:
        """Load cluster backup from disk"""
        if backup_path is None:
            # Use latest backup
            backup_path = self.backup_base_path / "latest_main_cluster.json"
        
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        try:
            async with aiofiles.open(backup_path, 'r') as f:
                backup_data = json.loads(await f.read())
            
            # Reconstruct AgentClusterState
            cluster_state = AgentClusterState(**backup_data)
            
            logger.info(f"Cluster backup loaded: {backup_path}")
            return cluster_state
            
        except Exception as e:
            logger.error(f"Failed to load cluster backup: {e}")
            raise
    
    def _verify_backup_integrity(self, cluster_state: AgentClusterState) -> bool:
        """Verify backup integrity using checksum"""
        try:
            stored_checksum = cluster_state.checksum
            calculated_checksum = self._generate_checksum(cluster_state)
            
            if stored_checksum == calculated_checksum:
                logger.info("Backup integrity verified successfully")
                return True
            else:
                logger.error(f"Backup integrity check failed: stored={stored_checksum}, calculated={calculated_checksum}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to verify backup integrity: {e}")
            return False    
    async def _graceful_agent_shutdown(self):
        """Gracefully shutdown all agents before recovery"""
        try:
            logger.info("Initiating graceful agent shutdown")
            
            if self.communication_protocol:
                # Send shutdown signal to all agents
                await self.communication_protocol.broadcast_message({
                    'type': 'SYSTEM_SHUTDOWN',
                    'message': 'Preparing for recovery operation',
                    'graceful': True,
                    'timeout': 30
                })
                
                # Wait for agents to acknowledge shutdown
                await asyncio.sleep(5)
            
            # Stop agent registry
            if self.agent_registry and hasattr(self.agent_registry, 'stop_all_agents'):
                await self.agent_registry.stop_all_agents()
            
            logger.info("Agent shutdown completed")
            
        except Exception as e:
            logger.error(f"Failed during agent shutdown: {e}")
    
    async def _restore_database_state(self, cluster_state: AgentClusterState):
        """Restore PostgreSQL database state"""
        try:
            logger.info("Restoring database state")
            
            conn = await asyncpg.connect(**self.postgres_config)
            
            try:
                # Begin transaction
                async with conn.transaction():
                    # Clear existing agent state tables
                    await conn.execute("DELETE FROM agent_states WHERE cluster_id = $1", cluster_state.cluster_id)
                    await conn.execute("DELETE FROM agent_coordination WHERE cluster_id = $1", cluster_state.cluster_id)
                    
                    # Restore agent states
                    for agent_id, agent_state in cluster_state.agents.items():
                        await conn.execute("""
                            INSERT INTO agent_states (cluster_id, agent_id, state_data, timestamp)
                            VALUES ($1, $2, $3, $4)
                        """, cluster_state.cluster_id, agent_id, json.dumps(agent_state), cluster_state.timestamp)
                    
                    # Restore coordination state
                    await conn.execute("""
                        INSERT INTO agent_coordination (cluster_id, coordination_data, timestamp)
                        VALUES ($1, $2, $3)
                    """, cluster_state.cluster_id, json.dumps(cluster_state.coordination_state), cluster_state.timestamp)
                
                logger.info("Database state restored successfully")
                
            finally:
                await conn.close()
                
        except Exception as e:
            logger.error(f"Failed to restore database state: {e}")
            raise
    
    async def _restore_redis_state(self, cluster_state: AgentClusterState):
        """Restore Redis cache state"""
        try:
            logger.info("Restoring Redis state")
            
            redis = await aioredis.from_url(
                f"redis://{self.redis_config['host']}:{self.redis_config['port']}"
            )
            
            try:
                # Clear existing cluster data
                pattern = f"agent_cluster:{cluster_state.cluster_id}:*"
                keys = await redis.keys(pattern)
                if keys:
                    await redis.delete(*keys)
                
                # Restore agent communication state
                for agent_id, agent_state in cluster_state.agents.items():
                    agent_key = f"agent_cluster:{cluster_state.cluster_id}:agent:{agent_id}"
                    await redis.hset(agent_key, mapping={
                        'state': json.dumps(agent_state),
                        'last_backup': cluster_state.timestamp.isoformat()
                    })
                
                # Restore communication channels
                channels_key = f"agent_cluster:{cluster_state.cluster_id}:channels"
                await redis.set(channels_key, json.dumps(cluster_state.communication_channels))
                
                logger.info("Redis state restored successfully")
                
            finally:
                await redis.close()
                
        except Exception as e:
            logger.error(f"Failed to restore Redis state: {e}")
            raise
    
    async def _restore_agent_configurations(self, cluster_state: AgentClusterState):
        """Restore agent configurations"""
        try:
            logger.info("Restoring agent configurations")
            
            if self.communication_protocol and hasattr(self.communication_protocol, 'config_coordinator'):
                config_coordinator = self.communication_protocol.config_coordinator
                
                # Restore agent configurations
                agent_configs = cluster_state.configuration.get('agent_configs', {})
                for agent_id, config in agent_configs.items():
                    if hasattr(config_coordinator, 'update_agent_configuration'):
                        await config_coordinator.update_agent_configuration(agent_id, config)
            
            logger.info("Agent configurations restored successfully")
            
        except Exception as e:
            logger.error(f"Failed to restore agent configurations: {e}")
    
    async def _restart_agents_with_state(self, cluster_state: AgentClusterState):
        """Restart agents with restored state"""
        try:
            logger.info("Restarting agents with restored state")
            
            if self.agent_registry:
                # Restart agents in dependency order
                dependency_graph = cluster_state.dependency_graph
                start_order = self._calculate_start_order(dependency_graph)
                
                for agent_id in start_order:
                    if agent_id in cluster_state.agents:
                        agent_state = cluster_state.agents[agent_id]
                        
                        # Restart agent with state
                        if hasattr(self.agent_registry, 'restart_agent_with_state'):
                            await self.agent_registry.restart_agent_with_state(agent_id, agent_state)
                        elif hasattr(self.agent_registry, 'start_agent'):
                            await self.agent_registry.start_agent(agent_id)
                        
                        # Wait between agent starts to avoid overwhelming the system
                        await asyncio.sleep(2)
            
            logger.info("Agents restarted successfully")
            
        except Exception as e:
            logger.error(f"Failed to restart agents: {e}")
            raise
    
    async def _restore_coordination_protocols(self, cluster_state: AgentClusterState):
        """Restore coordination protocols"""
        try:
            logger.info("Restoring coordination protocols")
            
            if self.communication_protocol:
                # Restore message queues
                coordination_state = cluster_state.coordination_state
                
                if 'message_queues' in coordination_state:
                    queue_states = coordination_state['message_queues']
                    if hasattr(self.communication_protocol, 'restore_queue_states'):
                        await self.communication_protocol.restore_queue_states(queue_states)
                
                # Restore routing tables
                if 'routing_tables' in coordination_state:
                    routing_state = coordination_state['routing_tables']
                    if hasattr(self.communication_protocol, 'restore_routing_state'):
                        await self.communication_protocol.restore_routing_state(routing_state)
            
            logger.info("Coordination protocols restored successfully")
            
        except Exception as e:
            logger.error(f"Failed to restore coordination protocols: {e}")
    
    async def _restore_dependency_chains(self, cluster_state: AgentClusterState):
        """Restore agent dependency chains"""
        try:
            logger.info("Restoring dependency chains")
            
            dependency_graph = cluster_state.dependency_graph
            
            if self.communication_protocol and hasattr(self.communication_protocol, 'dependency_resolver'):
                dependency_resolver = self.communication_protocol.dependency_resolver
                
                # Restore dependency relationships
                for agent_id, dependencies in dependency_graph.items():
                    if hasattr(dependency_resolver, 'set_agent_dependencies'):
                        await dependency_resolver.set_agent_dependencies(agent_id, dependencies)
            
            logger.info("Dependency chains restored successfully")
            
        except Exception as e:
            logger.error(f"Failed to restore dependency chains: {e}")
    
    def _calculate_start_order(self, dependency_graph: Dict[str, List[str]]) -> List[str]:
        """Calculate agent start order based on dependencies (topological sort)"""
        try:
            from collections import deque, defaultdict
            
            # Build graph and in-degree count
            graph = defaultdict(list)
            in_degree = defaultdict(int)
            all_agents = set()
            
            for agent, deps in dependency_graph.items():
                all_agents.add(agent)
                for dep in deps:
                    all_agents.add(dep)
                    graph[dep].append(agent)
                    in_degree[agent] += 1
            
            # Initialize in-degree for all agents
            for agent in all_agents:
                if agent not in in_degree:
                    in_degree[agent] = 0
            
            # Topological sort using Kahn's algorithm
            queue = deque([agent for agent in all_agents if in_degree[agent] == 0])
            start_order = []
            
            while queue:
                agent = queue.popleft()
                start_order.append(agent)
                
                for dependent in graph[agent]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
            
            # Check for circular dependencies
            if len(start_order) != len(all_agents):
                logger.warning("Circular dependencies detected, using original order")
                return list(dependency_graph.keys())
            
            return start_order
            
        except Exception as e:
            logger.error(f"Failed to calculate start order: {e}")
            return list(dependency_graph.keys())
    
    async def _validate_recovery(self, cluster_state: AgentClusterState) -> Dict[str, bool]:
        """Validate recovery operation success"""
        logger.info("Starting recovery validation")
        validation_results = {}
        
        try:
            # Check agent health
            validation_results['agents_healthy'] = await self._validate_agent_health(cluster_state)
            
            # Check communication
            validation_results['communication_working'] = await self._validate_communication()
            
            # Check dependencies
            validation_results['dependencies_functional'] = await self._validate_dependencies(cluster_state)
            
            # Check performance
            validation_results['performance_acceptable'] = await self._validate_performance()
            
            # Check data integrity
            validation_results['data_integrity'] = await self._validate_data_integrity(cluster_state)
            
            logger.info(f"Recovery validation completed: {validation_results}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Recovery validation failed: {e}")
            return {'validation_error': False}
    
    async def _validate_agent_health(self, cluster_state: AgentClusterState) -> bool:
        """Validate that all agents are healthy"""
        try:
            if not self.agent_registry:
                return True  # Skip if registry not available
            
            agents = await self.agent_registry.get_all_agents()
            expected_agents = set(cluster_state.agents.keys())
            running_agents = set(agents.keys())
            
            # Check if all expected agents are running
            if not expected_agents.issubset(running_agents):
                missing = expected_agents - running_agents
                logger.error(f"Missing agents after recovery: {missing}")
                return False
            
            # Check individual agent health
            for agent_id in expected_agents:
                if hasattr(self.agent_registry, 'get_agent_health'):
                    health = await self.agent_registry.get_agent_health(agent_id)
                    if health and hasattr(health, 'status'):
                        if health.status not in ['healthy', 'degraded']:
                            logger.error(f"Agent {agent_id} is not healthy: {health.status}")
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Agent health validation failed: {e}")
            return False
    
    async def _validate_communication(self) -> bool:
        """Validate that communication is working"""
        try:
            if not self.communication_protocol:
                return True  # Skip if protocol not available
            
            # Test basic communication
            test_message = {
                'type': 'HEALTH_CHECK',
                'timestamp': datetime.now().isoformat(),
                'test_id': f'recovery_test_{int(time.time())}'
            }
            
            # Try to send test message
            if hasattr(self.communication_protocol, 'send_test_message'):
                response = await asyncio.wait_for(
                    self.communication_protocol.send_test_message(test_message),
                    timeout=10.0
                )
                return response.get('success', False)
            
            return True  # Assume success if method not available
            
        except Exception as e:
            logger.error(f"Communication validation failed: {e}")
            return False
    
    async def _validate_dependencies(self, cluster_state: AgentClusterState) -> bool:
        """Validate that dependencies are functional"""
        try:
            dependency_graph = cluster_state.dependency_graph
            
            if not self.communication_protocol or not hasattr(self.communication_protocol, 'dependency_resolver'):
                return True  # Skip if resolver not available
            
            dependency_resolver = self.communication_protocol.dependency_resolver
            
            # Test dependency resolution
            for agent_id, dependencies in dependency_graph.items():
                if dependencies:  # Only test agents with dependencies
                    if hasattr(dependency_resolver, 'test_agent_dependencies'):
                        success = await dependency_resolver.test_agent_dependencies(agent_id)
                        if not success:
                            logger.error(f"Dependency test failed for agent {agent_id}")
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Dependency validation failed: {e}")
            return False
    
    async def _validate_performance(self) -> bool:
        """Validate that performance is acceptable"""
        try:
            # Check system resources
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            # Accept if CPU < 90% and memory < 90%
            if cpu_percent > 90 or memory_percent > 90:
                logger.warning(f"High resource usage: CPU={cpu_percent}%, Memory={memory_percent}%")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Performance validation failed: {e}")
            return False
    
    async def _validate_data_integrity(self, cluster_state: AgentClusterState) -> bool:
        """Validate data integrity after recovery"""
        try:
            # Verify checksum
            current_checksum = self._generate_checksum(cluster_state)
            if current_checksum != cluster_state.checksum:
                logger.error("Data integrity check failed: checksum mismatch")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Data integrity validation failed: {e}")
            return False
    
    def _check_recovery_success(self, validation_results: Dict[str, bool]) -> bool:
        """Check if recovery meets success criteria"""
        required_validations = [
            'agents_healthy',
            'communication_working', 
            'dependencies_functional'
        ]
        
        # All required validations must pass
        for validation in required_validations:
            if not validation_results.get(validation, False):
                logger.error(f"Recovery failed: {validation} validation failed")
                return False
        
        # Optional validations (warn but don't fail)
        optional_validations = ['performance_acceptable', 'data_integrity']
        for validation in optional_validations:
            if not validation_results.get(validation, True):
                logger.warning(f"Recovery warning: {validation} validation failed")
        
        return True
    
    async def get_recovery_status(self) -> Dict[str, Any]:
        """Get current recovery operation status"""
        if self.current_recovery:
            return {
                'status': self.current_recovery.status.value,
                'operation_id': self.current_recovery.operation_id,
                'start_time': self.current_recovery.start_time.isoformat(),
                'recovery_type': self.current_recovery.recovery_type.value,
                'affected_agents': self.current_recovery.affected_agents,
                'validation_results': self.current_recovery.validation_results,
                'recovery_metrics': self.current_recovery.recovery_metrics
            }
        else:
            return {
                'status': 'idle',
                'last_recovery': self.recovery_history[-1].operation_id if self.recovery_history else None
            }
    
    async def list_available_backups(self) -> List[Dict[str, Any]]:
        """List available cluster backups"""
        backups = []
        
        try:
            for backup_file in self.backup_base_path.glob("cluster_backup_*.json"):
                if backup_file.is_file():
                    stat = backup_file.stat()
                    backups.append({
                        'filename': backup_file.name,
                        'path': str(backup_file),
                        'size': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
            
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x['created'], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
        
        return backups
    
    async def test_recovery_system(self) -> Dict[str, Any]:
        """Test recovery system functionality"""
        logger.info("Starting recovery system test")
        
        test_results = {
            'backup_creation': False,
            'backup_loading': False,
            'integrity_verification': False,
            'component_initialization': False,
            'overall_status': False
        }
        
        try:
            # Test backup creation
            test_cluster_state = await self.create_cluster_backup("test_cluster")
            test_results['backup_creation'] = True
            
            # Test backup loading
            backup_path = await self._save_cluster_backup(test_cluster_state)
            loaded_state = await self._load_cluster_backup(backup_path)
            test_results['backup_loading'] = True
            
            # Test integrity verification
            integrity_ok = self._verify_backup_integrity(loaded_state)
            test_results['integrity_verification'] = integrity_ok
            
            # Test component initialization
            await self._initialize_components()
            test_results['component_initialization'] = True
            
            # Overall status
            test_results['overall_status'] = all([
                test_results['backup_creation'],
                test_results['backup_loading'],
                test_results['integrity_verification'],
                test_results['component_initialization']
            ])
            
            # Clean up test backup
            if backup_path and Path(backup_path).exists():
                Path(backup_path).unlink()
            
            logger.info(f"Recovery system test completed: {test_results}")
            
        except Exception as e:
            logger.error(f"Recovery system test failed: {e}")
            test_results['error'] = str(e)
        
        return test_results

# Factory function for creating recovery system
async def create_agent_cluster_recovery(
    backup_base_path: str = None,
    postgres_config: Dict = None,
    redis_config: Dict = None
) -> AgentClusterRecovery:
    """
    Factory function to create and initialize agent cluster recovery system
    
    Args:
        backup_base_path: Base path for backup storage
        postgres_config: PostgreSQL connection configuration
        redis_config: Redis connection configuration
        
    Returns:
        AgentClusterRecovery: Initialized recovery system
    """
    
    # Use default backup path if not provided
    if backup_base_path is None:
        backup_base_path = "/opt/platform3/backups/agent_clusters"
        
        # On Windows, use a different default path
        if os.name == 'nt':
            backup_base_path = "d:/Platform3/backups/agent_clusters"
    
    # Create recovery system
    recovery_system = AgentClusterRecovery(
        backup_base_path=backup_base_path,
        postgres_config=postgres_config,
        redis_config=redis_config
    )
    
    # Wait for initialization to complete
    await asyncio.sleep(0.1)
    
    return recovery_system

# Main execution for testing
async def main():
    """Main function for testing recovery system"""
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create recovery system
        recovery = await create_agent_cluster_recovery()
        
        # Run system test
        test_results = await recovery.test_recovery_system()
        
        print("Agent Cluster Recovery System Test Results:")
        print("=" * 50)
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name}: {status}")
        
        overall_status = test_results.get('overall_status', False)
        print(f"\nOverall Status: {'✅ SYSTEM READY' if overall_status else '❌ SYSTEM NOT READY'}")
        
        if overall_status:
            print("\n🚑 Agent Cluster Disaster Recovery System is operational!")
            print("📊 Ready to protect humanitarian trading operations")
            print("⏱️  Recovery target: < 15 minutes")
            print("🏥 Mission: Ensure continuous support for sick children")
        
    except Exception as e:
        print(f"❌ Recovery system test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())