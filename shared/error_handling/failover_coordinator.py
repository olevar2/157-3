"""
🔄 FAILOVER COORDINATION SYSTEM
===============================

Advanced failover coordination system for Platform3 agent ecosystem.
Manages failover sequences, backup activation, and service continuity.

Mission: Ensure zero-downtime operation through intelligent failover
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
from collections import defaultdict, deque

class FailoverStrategy(Enum):
    """Failover strategies for different scenarios"""
    IMMEDIATE = "immediate"          # Immediate failover without delay
    GRACEFUL = "graceful"            # Graceful shutdown then failover
    HOT_STANDBY = "hot_standby"      # Hot standby activation
    COLD_STANDBY = "cold_standby"    # Cold standby with initialization
    LOAD_BALANCING = "load_balancing" # Redistribute load to healthy agents
    DEGRADED_MODE = "degraded_mode"  # Continue with reduced functionality

class FailoverTrigger(Enum):
    """Triggers that initiate failover procedures"""
    HEALTH_CHECK_FAILED = "health_check_failed"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    UNRESPONSIVE = "unresponsive"
    MANUAL_TRIGGER = "manual_trigger"
    SCHEDULED_MAINTENANCE = "scheduled_maintenance"
    CASCADE_PREVENTION = "cascade_prevention"

class AgentType(Enum):
    """Agent types for specialized failover handling"""
    CRITICAL_SERVICE = "critical_service"    # Must have immediate backup
    STATEFUL_SERVICE = "stateful_service"    # Requires state transfer
    STATELESS_SERVICE = "stateless_service"  # Can be replaced easily
    COORDINATION_HUB = "coordination_hub"    # Central coordination agent
    DATA_PROCESSOR = "data_processor"        # Processing pipeline agent
    EXTERNAL_INTERFACE = "external_interface" # External API interface

@dataclass
class FailoverConfig:
    """Configuration for agent failover behavior"""
    agent_id: str
    agent_type: AgentType
    strategy: FailoverStrategy
    backup_agents: List[str] = field(default_factory=list)
    max_failover_time: int = 300  # Maximum time for failover in seconds
    health_check_interval: int = 30  # Health check interval in seconds
    retry_attempts: int = 3
    graceful_shutdown_timeout: int = 60
    state_sync_required: bool = False
    dependencies: List[str] = field(default_factory=list)
    load_balancing_weights: Dict[str, float] = field(default_factory=dict)
    custom_failover_script: Optional[str] = None

@dataclass
class FailoverEvent:
    """Represents a failover event"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    failed_agent: str = ""
    trigger: FailoverTrigger = FailoverTrigger.HEALTH_CHECK_FAILED
    strategy: FailoverStrategy = FailoverStrategy.GRACEFUL
    backup_agent: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    completion_time: Optional[datetime] = None
    success: bool = False
    state_transferred: bool = False
    rollback_completed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentStatus:
    """Current status of an agent in the failover system"""
    agent_id: str
    is_primary: bool = True
    is_backup: bool = False
    is_active: bool = True
    is_healthy: bool = True
    last_health_check: Optional[datetime] = None
    current_load: float = 0.0
    max_capacity: float = 100.0
    state_snapshot: Optional[Dict[str, Any]] = None
    backup_for: List[str] = field(default_factory=list)
    primary_for: List[str] = field(default_factory=list)

class FailoverCoordinator:
    """
    🔄 FAILOVER COORDINATION ENGINE
    
    Manages failover procedures, backup activation, and service continuity
    across the Platform3 agent ecosystem.
    
    Key Features:
    - Multiple failover strategies (hot/cold standby, load balancing)
    - State synchronization for stateful services
    - Graceful shutdown and startup procedures
    - Rollback mechanisms for failed failovers
    - Health monitoring and automatic trigger detection
    - Custom failover scripts for complex scenarios
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Failover Coordinator"""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Core state
        self.failover_configs: Dict[str, FailoverConfig] = {}
        self.agent_statuses: Dict[str, AgentStatus] = {}
        self.active_failovers: Dict[str, FailoverEvent] = {}
        self.failover_history: deque = deque(maxlen=1000)
        
        # Health monitoring
        self.health_monitors: Dict[str, asyncio.Task] = {}
        self.health_check_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Failover callbacks
        self.failover_callbacks: List[Callable] = []
        self.state_sync_handlers: Dict[str, Callable] = {}
        
        # Coordination with other systems
        self.error_propagation_manager = None  # Will be injected
        self.agent_registry = None  # Will be injected
        
        # Initialize default configurations
        self._setup_default_configs()
        
        # Start background tasks
        self._start_background_tasks()
        
        self.logger.info("🔄 Failover Coordinator initialized")
    
    def _setup_default_configs(self):
        """Setup default failover configurations for known agents"""
        
        # Critical services with hot standby
        critical_agents = {
            "risk_genius": {
                "type": AgentType.CRITICAL_SERVICE,
                "strategy": FailoverStrategy.HOT_STANDBY,
                "backups": ["risk_genius_backup", "decision_master"],
                "state_sync": True
            },
            "decision_master": {
                "type": AgentType.COORDINATION_HUB,
                "strategy": FailoverStrategy.GRACEFUL,
                "backups": ["decision_master_backup"],
                "state_sync": True
            },
            "execution_expert": {
                "type": AgentType.CRITICAL_SERVICE,
                "strategy": FailoverStrategy.IMMEDIATE,
                "backups": ["execution_expert_backup"],
                "state_sync": False
            }
        }
        
        for agent_id, config_data in critical_agents.items():
            config = FailoverConfig(
                agent_id=agent_id,
                agent_type=config_data["type"],
                strategy=config_data["strategy"],
                backup_agents=config_data["backups"],
                state_sync_required=config_data["state_sync"],
                max_failover_time=120,  # 2 minutes for critical services
                health_check_interval=15  # Check every 15 seconds
            )
            self.failover_configs[agent_id] = config
        
        # Load balancing services
        load_balanced_agents = {
            "pattern_master": {
                "type": AgentType.DATA_PROCESSOR,
                "strategy": FailoverStrategy.LOAD_BALANCING,
                "backups": ["pattern_analyzer_1", "pattern_analyzer_2", "pattern_analyzer_3"]
            },
            "sentiment_analyzer": {
                "type": AgentType.STATELESS_SERVICE,
                "strategy": FailoverStrategy.COLD_STANDBY,
                "backups": ["sentiment_backup"]
            }
        }
        
        for agent_id, config_data in load_balanced_agents.items():
            config = FailoverConfig(
                agent_id=agent_id,
                agent_type=config_data["type"],
                strategy=config_data["strategy"],
                backup_agents=config_data["backups"],
                state_sync_required=False,
                max_failover_time=60,
                health_check_interval=30
            )
            self.failover_configs[agent_id] = config
        
        self.logger.info(f"Setup {len(self.failover_configs)} default failover configurations")
    
    def _start_background_tasks(self):
        """Start background monitoring and coordination tasks"""
        asyncio.create_task(self._monitor_agent_health())
        asyncio.create_task(self._manage_active_failovers())
        asyncio.create_task(self._sync_agent_states())
        asyncio.create_task(self._cleanup_old_events())
    
    async def register_agent(self, agent_id: str, config: FailoverConfig):
        """Register an agent with failover configuration"""
        self.failover_configs[agent_id] = config
        
        # Initialize agent status
        status = AgentStatus(
            agent_id=agent_id,
            is_primary=True,
            is_active=True,
            is_healthy=True,
            last_health_check=datetime.now()
        )
        self.agent_statuses[agent_id] = status
        
        # Setup backup relationships
        for backup_id in config.backup_agents:
            if backup_id not in self.agent_statuses:
                backup_status = AgentStatus(
                    agent_id=backup_id,
                    is_primary=False,
                    is_backup=True,
                    is_active=False,
                    is_healthy=True
                )
                self.agent_statuses[backup_id] = backup_status
            
            self.agent_statuses[backup_id].backup_for.append(agent_id)
        
        # Start health monitoring
        await self._start_health_monitoring(agent_id)
        
        self.logger.info(f"🔄 Registered agent {agent_id} for failover")
    
    async def trigger_failover(self, agent_id: str, trigger: FailoverTrigger, 
                              force: bool = False) -> bool:
        """
        Trigger failover for a specific agent
        
        Args:
            agent_id: Agent to failover
            trigger: What triggered the failover
            force: Force failover even if agent appears healthy
            
        Returns:
            bool: True if failover initiated successfully
        """
        try:
            if agent_id not in self.failover_configs:
                self.logger.error(f"❌ No failover config for agent {agent_id}")
                return False
            
            if agent_id in self.active_failovers and not force:
                self.logger.warning(f"⚠️ Failover already in progress for {agent_id}")
                return False
            
            config = self.failover_configs[agent_id]
            
            # Create failover event
            failover_event = FailoverEvent(
                failed_agent=agent_id,
                trigger=trigger,
                strategy=config.strategy,
                metadata={
                    'force': force,
                    'config': config.agent_id,
                    'backup_count': len(config.backup_agents)
                }
            )
            
            self.active_failovers[agent_id] = failover_event
            
            self.logger.warning(f"🔄 Starting failover for {agent_id} (trigger: {trigger.value})")
            
            # Execute failover strategy
            success = await self._execute_failover(failover_event, config)
            
            # Complete failover event
            failover_event.completion_time = datetime.now()
            failover_event.success = success
            
            if success:
                self.logger.info(f"✅ Failover completed successfully for {agent_id}")
            else:
                self.logger.error(f"❌ Failover failed for {agent_id}")
                # Attempt rollback
                await self._attempt_rollback(failover_event, config)
            
            # Move to history and cleanup
            self.failover_history.append(failover_event)
            if agent_id in self.active_failovers:
                del self.active_failovers[agent_id]
            
            # Notify callbacks
            for callback in self.failover_callbacks:
                try:
                    await callback(failover_event)
                except Exception as e:
                    self.logger.error(f"Failover callback error: {e}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error during failover for {agent_id}: {e}")
            return False
    
    async def _execute_failover(self, event: FailoverEvent, config: FailoverConfig) -> bool:
        """Execute the appropriate failover strategy"""
        try:
            if config.strategy == FailoverStrategy.IMMEDIATE:
                return await self._immediate_failover(event, config)
            elif config.strategy == FailoverStrategy.GRACEFUL:
                return await self._graceful_failover(event, config)
            elif config.strategy == FailoverStrategy.HOT_STANDBY:
                return await self._hot_standby_failover(event, config)
            elif config.strategy == FailoverStrategy.COLD_STANDBY:
                return await self._cold_standby_failover(event, config)
            elif config.strategy == FailoverStrategy.LOAD_BALANCING:
                return await self._load_balancing_failover(event, config)
            elif config.strategy == FailoverStrategy.DEGRADED_MODE:
                return await self._degraded_mode_failover(event, config)
            else:
                self.logger.error(f"❌ Unknown failover strategy: {config.strategy}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failover execution error: {e}")
            return False
    
    async def _immediate_failover(self, event: FailoverEvent, config: FailoverConfig) -> bool:
        """Execute immediate failover - switch to backup immediately"""
        backup_agent = await self._select_best_backup(config)
        if not backup_agent:
            return False
        
        event.backup_agent = backup_agent
        
        # Immediately activate backup
        if await self._activate_backup_agent(backup_agent, event.failed_agent):
            # Deactivate failed agent
            await self._deactivate_agent(event.failed_agent)
            return True
        
        return False
    
    async def _graceful_failover(self, event: FailoverEvent, config: FailoverConfig) -> bool:
        """Execute graceful failover - proper shutdown then backup activation"""
        backup_agent = await self._select_best_backup(config)
        if not backup_agent:
            return False
        
        event.backup_agent = backup_agent
        
        # Step 1: Graceful shutdown of failed agent
        shutdown_success = await self._graceful_shutdown(event.failed_agent, config.graceful_shutdown_timeout)
        
        # Step 2: Transfer state if required
        if config.state_sync_required:
            state_transferred = await self._transfer_agent_state(event.failed_agent, backup_agent)
            event.state_transferred = state_transferred
        
        # Step 3: Activate backup
        if await self._activate_backup_agent(backup_agent, event.failed_agent):
            return True
        
        return False
    
    async def _hot_standby_failover(self, event: FailoverEvent, config: FailoverConfig) -> bool:
        """Execute hot standby failover - backup is already running"""
        # Find active hot standby
        hot_standby = None
        for backup_id in config.backup_agents:
            status = self.agent_statuses.get(backup_id)
            if status and status.is_active and status.is_healthy:
                hot_standby = backup_id
                break
        
        if not hot_standby:
            # No hot standby available, fall back to graceful
            return await self._graceful_failover(event, config)
        
        event.backup_agent = hot_standby
        
        # Simply switch traffic to hot standby
        await self._switch_traffic(event.failed_agent, hot_standby)
        
        # Update statuses
        self._update_agent_status(event.failed_agent, is_primary=False, is_active=False)
        self._update_agent_status(hot_standby, is_primary=True, is_active=True)
        
        return True
    
    async def _cold_standby_failover(self, event: FailoverEvent, config: FailoverConfig) -> bool:
        """Execute cold standby failover - initialize and start backup"""
        backup_agent = await self._select_best_backup(config)
        if not backup_agent:
            return False
        
        event.backup_agent = backup_agent
        
        # Initialize cold standby
        if await self._initialize_cold_standby(backup_agent, event.failed_agent):
            # Transfer state if required
            if config.state_sync_required:
                state_transferred = await self._transfer_agent_state(event.failed_agent, backup_agent)
                event.state_transferred = state_transferred
            
            # Activate backup
            if await self._activate_backup_agent(backup_agent, event.failed_agent):
                return True
        
        return False
    
    async def _load_balancing_failover(self, event: FailoverEvent, config: FailoverConfig) -> bool:
        """Execute load balancing failover - redistribute load to healthy agents"""
        healthy_backups = []
        for backup_id in config.backup_agents:
            status = self.agent_statuses.get(backup_id)
            if status and status.is_healthy and status.current_load < status.max_capacity * 0.8:
                healthy_backups.append(backup_id)
        
        if not healthy_backups:
            return False
        
        # Calculate load redistribution
        failed_agent_load = self.agent_statuses.get(event.failed_agent, AgentStatus(event.failed_agent)).current_load
        load_per_backup = failed_agent_load / len(healthy_backups)
        
        # Redistribute load
        for backup_id in healthy_backups:
            await self._redistribute_load(backup_id, load_per_backup)
        
        # Update failed agent status
        self._update_agent_status(event.failed_agent, is_active=False, current_load=0.0)
        
        event.backup_agent = ",".join(healthy_backups)  # Multiple backups
        return True
    
    async def _degraded_mode_failover(self, event: FailoverEvent, config: FailoverConfig) -> bool:
        """Execute degraded mode failover - continue with reduced functionality"""
        # Mark agent as degraded but keep it running
        self._update_agent_status(event.failed_agent, is_active=True, current_load=50.0)
        
        # Notify system of degraded mode
        await self._notify_degraded_mode(event.failed_agent)
        
        event.backup_agent = "degraded_mode"
        return True
    
    async def _select_best_backup(self, config: FailoverConfig) -> Optional[str]:
        """Select the best available backup agent"""
        available_backups = []
        
        for backup_id in config.backup_agents:
            status = self.agent_statuses.get(backup_id)
            if status and status.is_healthy and not status.is_active:
                # Calculate backup score based on various factors
                score = self._calculate_backup_score(backup_id, status)
                available_backups.append((backup_id, score))
        
        if not available_backups:
            self.logger.error(f"❌ No healthy backup agents available for {config.agent_id}")
            return None
        
        # Sort by score (higher is better) and return best
        available_backups.sort(key=lambda x: x[1], reverse=True)
        return available_backups[0][0]
    
    def _calculate_backup_score(self, backup_id: str, status: AgentStatus) -> float:
        """Calculate score for backup agent selection"""
        score = 100.0  # Base score
        
        # Prefer agents with lower current load
        score -= status.current_load
        
        # Prefer agents with recent health checks
        if status.last_health_check:
            minutes_since_check = (datetime.now() - status.last_health_check).total_seconds() / 60
            score -= minutes_since_check  # Penalize stale health checks
        
        # Prefer agents that aren't backup for many others
        score -= len(status.backup_for) * 10  # Penalize overloaded backups
        
        return max(0.0, score)
    
    async def _activate_backup_agent(self, backup_id: str, failed_agent_id: str) -> bool:
        """Activate a backup agent"""
        try:
            self.logger.info(f"🔄 Activating backup agent {backup_id} for {failed_agent_id}")
            
            # Update status
            self._update_agent_status(backup_id, is_primary=True, is_active=True)
            
            # Execute custom activation script if available
            config = self.failover_configs.get(failed_agent_id)
            if config and config.custom_failover_script:
                await self._execute_custom_script(config.custom_failover_script, {
                    'action': 'activate',
                    'backup_agent': backup_id,
                    'failed_agent': failed_agent_id
                })
            
            # Notify agent registry if available
            if self.agent_registry:
                await self.agent_registry.update_agent_status(backup_id, 'active')
                await self.agent_registry.update_agent_status(failed_agent_id, 'inactive')
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to activate backup agent {backup_id}: {e}")
            return False
    
    async def _deactivate_agent(self, agent_id: str) -> bool:
        """Deactivate a failed agent"""
        try:
            self.logger.info(f"🔄 Deactivating agent {agent_id}")
            
            # Update status
            self._update_agent_status(agent_id, is_primary=False, is_active=False)
            
            # Stop health monitoring temporarily
            if agent_id in self.health_monitors:
                self.health_monitors[agent_id].cancel()
                del self.health_monitors[agent_id]
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to deactivate agent {agent_id}: {e}")
            return False
    
    async def _graceful_shutdown(self, agent_id: str, timeout: int) -> bool:
        """Perform graceful shutdown of an agent"""
        try:
            self.logger.info(f"🔄 Graceful shutdown of agent {agent_id}")
            
            # Send shutdown signal and wait for completion
            # This would integrate with actual agent management system
            await asyncio.sleep(min(timeout, 10))  # Simulate graceful shutdown
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Graceful shutdown failed for {agent_id}: {e}")
            return False
    
    async def _transfer_agent_state(self, source_agent: str, target_agent: str) -> bool:
        """Transfer state from source to target agent"""
        try:
            self.logger.info(f"🔄 Transferring state from {source_agent} to {target_agent}")
            
            # Get state snapshot
            source_status = self.agent_statuses.get(source_agent)
            if source_status and source_status.state_snapshot:
                # Transfer state using registered handler
                if source_agent in self.state_sync_handlers:
                    handler = self.state_sync_handlers[source_agent]
                    success = await handler(source_status.state_snapshot, target_agent)
                    return success
            
            return True  # Default success if no state to transfer
            
        except Exception as e:
            self.logger.error(f"❌ State transfer failed: {e}")
            return False
    
    async def _switch_traffic(self, from_agent: str, to_agent: str):
        """Switch traffic from one agent to another"""
        self.logger.info(f"🔄 Switching traffic from {from_agent} to {to_agent}")
        # This would integrate with load balancer or routing system
        # For now, just update internal routing
        pass
    
    async def _initialize_cold_standby(self, backup_id: str, failed_agent_id: str) -> bool:
        """Initialize a cold standby agent"""
        try:
            self.logger.info(f"🔄 Initializing cold standby {backup_id}")
            
            # Simulate initialization process
            await asyncio.sleep(5)  # Cold start delay
            
            # Update status
            self._update_agent_status(backup_id, is_active=True, is_healthy=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Cold standby initialization failed: {e}")
            return False
    
    async def _redistribute_load(self, agent_id: str, additional_load: float):
        """Redistribute load to an agent"""
        status = self.agent_statuses.get(agent_id)
        if status:
            status.current_load += additional_load
            self.logger.info(f"🔄 Redistributed {additional_load} load to {agent_id}")
    
    async def _notify_degraded_mode(self, agent_id: str):
        """Notify system that agent is in degraded mode"""
        self.logger.warning(f"⚠️ Agent {agent_id} operating in degraded mode")
    
    def _update_agent_status(self, agent_id: str, **kwargs):
        """Update agent status"""
        if agent_id not in self.agent_statuses:
            self.agent_statuses[agent_id] = AgentStatus(agent_id=agent_id)
        
        status = self.agent_statuses[agent_id]
        for key, value in kwargs.items():
            if hasattr(status, key):
                setattr(status, key, value)
    
    async def _attempt_rollback(self, event: FailoverEvent, config: FailoverConfig):
        """Attempt to rollback failed failover"""
        try:
            self.logger.warning(f"🔄 Attempting rollback for {event.failed_agent}")
            
            if event.backup_agent:
                # Deactivate backup
                await self._deactivate_agent(event.backup_agent)
                
                # Try to reactivate original agent
                if await self._activate_backup_agent(event.failed_agent, event.backup_agent):
                    event.rollback_completed = True
                    self.logger.info(f"✅ Rollback completed for {event.failed_agent}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Rollback failed: {e}")
            return False
    
    async def _execute_custom_script(self, script_path: str, context: Dict[str, Any]) -> bool:
        """Execute custom failover script"""
        try:
            # This would execute custom failover logic
            self.logger.info(f"🔄 Executing custom script: {script_path}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Custom script execution failed: {e}")
            return False
    
    async def _monitor_agent_health(self):
        """Background health monitoring"""
        while True:
            try:
                for agent_id, config in self.failover_configs.items():
                    await self._check_agent_health(agent_id, config)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Health monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _check_agent_health(self, agent_id: str, config: FailoverConfig):
        """Check health of a specific agent"""
        try:
            # Call health check callbacks
            if agent_id in self.health_check_callbacks:
                for callback in self.health_check_callbacks[agent_id]:
                    is_healthy = await callback(agent_id)
                    
                    status = self.agent_statuses.get(agent_id)
                    if status:
                        previous_health = status.is_healthy
                        status.is_healthy = is_healthy
                        status.last_health_check = datetime.now()
                        
                        # Trigger failover if health degraded
                        if previous_health and not is_healthy:
                            await self.trigger_failover(agent_id, FailoverTrigger.HEALTH_CHECK_FAILED)
            
        except Exception as e:
            self.logger.error(f"❌ Health check error for {agent_id}: {e}")
    
    async def _start_health_monitoring(self, agent_id: str):
        """Start health monitoring for an agent"""
        if agent_id not in self.health_monitors:
            task = asyncio.create_task(self._individual_health_monitor(agent_id))
            self.health_monitors[agent_id] = task
    
    async def _individual_health_monitor(self, agent_id: str):
        """Individual health monitor for a specific agent"""
        config = self.failover_configs.get(agent_id)
        if not config:
            return
        
        while True:
            try:
                await self._check_agent_health(agent_id, config)
                await asyncio.sleep(config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Individual health monitor error for {agent_id}: {e}")
                await asyncio.sleep(60)
    
    async def _manage_active_failovers(self):
        """Manage active failover processes"""
        while True:
            try:
                current_time = datetime.now()
                
                for agent_id, event in list(self.active_failovers.items()):
                    # Check for timeout
                    config = self.failover_configs.get(agent_id)
                    if config:
                        elapsed = (current_time - event.start_time).total_seconds()
                        if elapsed > config.max_failover_time:
                            self.logger.error(f"❌ Failover timeout for {agent_id}")
                            event.success = False
                            event.completion_time = current_time
                            self.failover_history.append(event)
                            del self.active_failovers[agent_id]
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Failover management error: {e}")
                await asyncio.sleep(30)
    
    async def _sync_agent_states(self):
        """Synchronize agent states periodically"""
        while True:
            try:
                for agent_id, status in self.agent_statuses.items():
                    if status.is_active and agent_id in self.state_sync_handlers:
                        # Capture current state snapshot
                        handler = self.state_sync_handlers[agent_id]
                        state_snapshot = await handler(None, "capture_state")
                        status.state_snapshot = state_snapshot
                
                await asyncio.sleep(300)  # Sync every 5 minutes
                
            except Exception as e:
                self.logger.error(f"❌ State sync error: {e}")
                await asyncio.sleep(600)
    
    async def _cleanup_old_events(self):
        """Clean up old failover events"""
        while True:
            try:
                # Keep only last 24 hours of events
                cutoff_time = datetime.now() - timedelta(hours=24)
                
                # Filter history
                self.failover_history = deque([
                    event for event in self.failover_history
                    if event.start_time > cutoff_time
                ], maxlen=1000)
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                self.logger.error(f"❌ Cleanup error: {e}")
                await asyncio.sleep(1800)
    
    # Public API methods
    
    def register_health_check_callback(self, agent_id: str, callback: Callable):
        """Register health check callback for an agent"""
        self.health_check_callbacks[agent_id].append(callback)
    
    def register_state_sync_handler(self, agent_id: str, handler: Callable):
        """Register state synchronization handler"""
        self.state_sync_handlers[agent_id] = handler
    
    def register_failover_callback(self, callback: Callable):
        """Register callback for failover events"""
        self.failover_callbacks.append(callback)
    
    def get_agent_status(self, agent_id: str) -> Optional[AgentStatus]:
        """Get current status of an agent"""
        return self.agent_statuses.get(agent_id)
    
    def get_failover_statistics(self) -> Dict[str, Any]:
        """Get failover statistics"""
        now = datetime.now()
        recent_failovers = [
            event for event in self.failover_history
            if (now - event.start_time).total_seconds() < 3600  # Last hour
        ]
        
        successful_failovers = [event for event in recent_failovers if event.success]
        
        return {
            'total_agents_configured': len(self.failover_configs),
            'active_failovers': len(self.active_failovers),
            'recent_failovers_count': len(recent_failovers),
            'recent_success_rate': len(successful_failovers) / max(1, len(recent_failovers)),
            'agents_with_backups': len([
                config for config in self.failover_configs.values()
                if config.backup_agents
            ]),
            'healthy_agents': len([
                status for status in self.agent_statuses.values()
                if status.is_healthy
            ]),
            'active_agents': len([
                status for status in self.agent_statuses.values()
                if status.is_active
            ])
        }
    
    async def force_agent_failover(self, agent_id: str) -> bool:
        """Force immediate failover of an agent"""
        return await self.trigger_failover(agent_id, FailoverTrigger.MANUAL_TRIGGER, force=True)
    
    async def restore_agent(self, agent_id: str) -> bool:
        """Restore agent from failover state"""
        try:
            status = self.agent_statuses.get(agent_id)
            if not status:
                return False
            
            # Reactivate agent
            status.is_active = True
            status.is_healthy = True
            status.is_primary = True
            
            # Restart health monitoring
            await self._start_health_monitoring(agent_id)
            
            self.logger.info(f"🔄 Restored agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Agent restoration failed: {e}")
            return False
    
    def set_error_propagation_manager(self, manager):
        """Set reference to error propagation manager"""
        self.error_propagation_manager = manager
    
    def set_agent_registry(self, registry):
        """Set reference to agent registry"""
        self.agent_registry = registry


# Global instance for easy access
failover_coordinator = FailoverCoordinator()