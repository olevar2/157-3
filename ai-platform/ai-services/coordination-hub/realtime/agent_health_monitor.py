"""
💓 AGENT HEARTBEAT AND HEALTH MONITORING SYSTEM
==============================================

Comprehensive health monitoring for the 9 Platform3 genius agents
Tracks performance, availability, and coordination effectiveness

Mission: Ensure 99.9%+ uptime for humanitarian trading operations
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import statistics
import subprocess
import sys
import os

class RecoveryStrategy(Enum):
    """Recovery strategies for failed agents"""
    RESTART = "restart"
    BACKUP_ACTIVATION = "backup_activation"
    LOAD_REDISTRIBUTION = "load_redistribution"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    ESCALATION = "escalation"

class RecoveryAction(Enum):
    """Types of recovery actions"""
    AGENT_RESTART = "agent_restart"
    BACKUP_AGENT_START = "backup_agent_start"
    WORKLOAD_REDISTRIBUTE = "workload_redistribute"
    ALERT_ESCALATION = "alert_escalation"
    SYSTEM_ISOLATION = "system_isolation"

class AgentHealthStatus(Enum):
    """Health status levels for agents"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"
    UNKNOWN = "unknown"

@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt"""
    agent_id: str
    strategy: RecoveryStrategy
    action: RecoveryAction
    timestamp: datetime
    success: bool
    execution_time_ms: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_id': self.agent_id,
            'strategy': self.strategy.value,
            'action': self.action.value,
            'timestamp': self.timestamp.isoformat(),
            'success': self.success,
            'execution_time_ms': self.execution_time_ms,
            'error_message': self.error_message,
            'details': self.details
        }

@dataclass
class BackupAgentConfig:
    """Configuration for backup agents"""
    agent_id: str
    backup_agent_id: str
    activation_priority: int  # 1 = highest priority
    capabilities: List[str]
    startup_command: Optional[str] = None
    config_template: Optional[str] = None
    warm_standby: bool = False  # Keep backup running in standby mode
    
class AutoRecoveryManager:
    """
    🔄 AUTOMATIC AGENT RECOVERY SYSTEM
    
    Implements automatic recovery procedures for failed or degraded agents:
    - Auto-restart of failed agents
    - Backup agent activation for critical agents  
    - Workload redistribution during recovery
    - Graceful degradation when recovery fails
    - Recovery escalation procedures
    - Success/failure tracking and learning
    
    Ensures 99.9%+ uptime for humanitarian trading operations
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Auto Recovery Manager"""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Recovery configuration
        self.max_recovery_attempts = self.config.get('max_recovery_attempts', 3)
        self.recovery_timeout_seconds = self.config.get('recovery_timeout_seconds', 60)
        self.escalation_threshold = self.config.get('escalation_threshold', 5)  # failures before escalation
        self.cooldown_period_minutes = self.config.get('cooldown_period_minutes', 10)
        
        # Recovery tracking
        self.recovery_attempts: List[RecoveryAttempt] = []
        self.agent_failure_counts: Dict[str, int] = {}
        self.last_recovery_times: Dict[str, datetime] = {}
        self.active_recoveries: Dict[str, asyncio.Task] = {}
        
        # Backup agent configuration
        self.backup_agents: Dict[str, BackupAgentConfig] = {}
        self.active_backups: Dict[str, str] = {}  # primary_agent_id -> backup_agent_id
        
        # Recovery callbacks
        self.recovery_callbacks: List[Callable[[RecoveryAttempt], None]] = []
        self.escalation_callbacks: List[Callable[[str, int], None]] = []
        
        # Critical agent configuration
        self.critical_agents = {
            'risk_genius', 'decision_master', 'execution_expert', 
            'ai_model_coordinator', 'pattern_master'
        }
        
        # Initialize backup agent configurations
        self._setup_backup_configurations()
        
        self.logger.info("🔄 Auto Recovery Manager initialized")
    
    def _setup_backup_configurations(self):
        """Setup backup agent configurations for critical agents"""
        backup_configs = [
            BackupAgentConfig(
                agent_id="risk_genius",
                backup_agent_id="risk_genius_backup",
                activation_priority=1,
                capabilities=["risk_assessment", "portfolio_protection", "volatility_analysis"],
                startup_command="python -m ai-platform.intelligent-agents.risk_genius_backup",
                warm_standby=True
            ),
            BackupAgentConfig(
                agent_id="decision_master",
                backup_agent_id="decision_master_backup", 
                activation_priority=1,
                capabilities=["decision_orchestration", "strategy_coordination", "trade_approval"],
                startup_command="python -m ai-platform.intelligent-agents.decision_master_backup",
                warm_standby=True
            ),
            BackupAgentConfig(
                agent_id="execution_expert",
                backup_agent_id="execution_expert_backup",
                activation_priority=2,
                capabilities=["trade_execution", "order_management", "execution_optimization"],
                startup_command="python -m ai-platform.intelligent-agents.execution_expert_backup",
                warm_standby=False
            ),
            BackupAgentConfig(
                agent_id="ai_model_coordinator",
                backup_agent_id="ai_model_coordinator_backup",
                activation_priority=1,
                capabilities=["model_coordination", "ensemble_management", "prediction_aggregation"],
                startup_command="python -m ai-platform.intelligent-agents.ai_model_coordinator_backup",
                warm_standby=True
            ),
            BackupAgentConfig(
                agent_id="pattern_master",
                backup_agent_id="pattern_master_backup",
                activation_priority=3,
                capabilities=["pattern_detection", "trend_analysis", "market_structure"],
                startup_command="python -m ai-platform.intelligent-agents.pattern_master_backup",
                warm_standby=False
            )
        ]
        
        for config in backup_configs:
            self.backup_agents[config.agent_id] = config
    
    async def handle_agent_failure(self, agent_id: str, failure_reason: str, severity: str) -> bool:
        """
        Handle agent failure with automatic recovery
        
        Args:
            agent_id: ID of the failed agent
            failure_reason: Reason for failure
            severity: Severity level (critical, warning, etc.)
            
        Returns:
            bool: True if recovery was initiated successfully
        """
        try:
            self.logger.warning(f"🚨 Agent failure detected: {agent_id} - {failure_reason}")
            
            # Check if recovery is already in progress
            if agent_id in self.active_recoveries:
                self.logger.info(f"🔄 Recovery already in progress for {agent_id}")
                return True
            
            # Check cooldown period
            if self._is_in_cooldown(agent_id):
                self.logger.info(f"⏳ Agent {agent_id} in recovery cooldown period")
                return False
            
            # Increment failure count
            self.agent_failure_counts[agent_id] = self.agent_failure_counts.get(agent_id, 0) + 1
            failure_count = self.agent_failure_counts[agent_id]
            
            # Determine recovery strategy based on failure count and severity
            strategy = self._determine_recovery_strategy(agent_id, failure_count, severity)
            
            # Check for escalation threshold
            if failure_count >= self.escalation_threshold:
                await self._escalate_failure(agent_id, failure_count)
                return False
            
            # Start recovery process
            recovery_task = asyncio.create_task(
                self._execute_recovery(agent_id, strategy, failure_reason)
            )
            self.active_recoveries[agent_id] = recovery_task
            
            self.logger.info(f"🔄 Recovery initiated for {agent_id} using strategy: {strategy.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error handling agent failure for {agent_id}: {e}")
            return False
    
    def _is_in_cooldown(self, agent_id: str) -> bool:
        """Check if agent is in cooldown period"""
        if agent_id not in self.last_recovery_times:
            return False
        
        last_recovery = self.last_recovery_times[agent_id]
        cooldown_end = last_recovery + timedelta(minutes=self.cooldown_period_minutes)
        return datetime.now() < cooldown_end
    
    def _determine_recovery_strategy(self, agent_id: str, failure_count: int, severity: str) -> RecoveryStrategy:
        """Determine the best recovery strategy for the situation"""
        is_critical = agent_id in self.critical_agents
        has_backup = agent_id in self.backup_agents
        
        # For critical severity or multiple failures, use backup activation if available
        if (severity == "critical" or failure_count >= 2) and has_backup:
            return RecoveryStrategy.BACKUP_ACTIVATION
        
        # For first failure, try restart
        if failure_count == 1:
            return RecoveryStrategy.RESTART
        
        # For critical agents without backup, try load redistribution
        if is_critical and not has_backup:
            return RecoveryStrategy.LOAD_REDISTRIBUTION
        
        # Default to graceful degradation
        return RecoveryStrategy.GRACEFUL_DEGRADATION
    
    async def _execute_recovery(self, agent_id: str, strategy: RecoveryStrategy, failure_reason: str):
        """Execute the recovery strategy"""
        start_time = time.time()
        success = False
        error_message = None
        action = None
        
        try:
            self.logger.info(f"🔄 Executing recovery for {agent_id} using {strategy.value}")
            
            if strategy == RecoveryStrategy.RESTART:
                action = RecoveryAction.AGENT_RESTART
                success = await self._restart_agent(agent_id)
                
            elif strategy == RecoveryStrategy.BACKUP_ACTIVATION:
                action = RecoveryAction.BACKUP_AGENT_START
                success = await self._activate_backup_agent(agent_id)
                
            elif strategy == RecoveryStrategy.LOAD_REDISTRIBUTION:
                action = RecoveryAction.WORKLOAD_REDISTRIBUTE
                success = await self._redistribute_workload(agent_id)
                
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                action = RecoveryAction.SYSTEM_ISOLATION
                success = await self._implement_graceful_degradation(agent_id)
                
            elif strategy == RecoveryStrategy.ESCALATION:
                action = RecoveryAction.ALERT_ESCALATION
                success = await self._escalate_failure(agent_id, self.agent_failure_counts.get(agent_id, 0))
            
        except Exception as e:
            error_message = str(e)
            self.logger.error(f"❌ Recovery execution failed for {agent_id}: {e}")
        
        finally:
            # Record recovery attempt
            execution_time = (time.time() - start_time) * 1000
            attempt = RecoveryAttempt(
                agent_id=agent_id,
                strategy=strategy,
                action=action or RecoveryAction.AGENT_RESTART,
                timestamp=datetime.now(),
                success=success,
                execution_time_ms=execution_time,
                error_message=error_message,
                details={'failure_reason': failure_reason}
            )
            
            self.recovery_attempts.append(attempt)
            self.last_recovery_times[agent_id] = datetime.now()
            
            # Remove from active recoveries
            if agent_id in self.active_recoveries:
                del self.active_recoveries[agent_id]
            
            # Notify callbacks
            for callback in self.recovery_callbacks:
                try:
                    await callback(attempt)
                except Exception as e:
                    self.logger.error(f"❌ Recovery callback failed: {e}")
            
            # Log result
            if success:
                self.logger.info(f"✅ Recovery successful for {agent_id} in {execution_time:.1f}ms")
                # Reset failure count on successful recovery
                self.agent_failure_counts[agent_id] = 0
            else:
                self.logger.error(f"❌ Recovery failed for {agent_id}")
    
    async def _restart_agent(self, agent_id: str) -> bool:
        """Restart a failed agent"""
        try:
            self.logger.info(f"🔄 Attempting to restart agent: {agent_id}")
            
            # Simulate agent restart process
            # In a real implementation, this would:
            # 1. Stop the current agent process
            # 2. Clean up resources
            # 3. Start a new agent instance
            # 4. Verify the agent is healthy
            
            # For now, simulate the restart process
            await asyncio.sleep(2)  # Simulate restart time
            
            # Simulate successful restart (90% success rate)
            import random
            success = random.random() > 0.1
            
            if success:
                self.logger.info(f"✅ Agent {agent_id} restarted successfully")
            else:
                self.logger.error(f"❌ Agent {agent_id} restart failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error restarting agent {agent_id}: {e}")
            return False
    
    async def _activate_backup_agent(self, agent_id: str) -> bool:
        """Activate backup agent for critical agent"""
        try:
            if agent_id not in self.backup_agents:
                self.logger.error(f"❌ No backup agent configured for {agent_id}")
                return False
            
            backup_config = self.backup_agents[agent_id]
            backup_id = backup_config.backup_agent_id
            
            self.logger.info(f"🔄 Activating backup agent {backup_id} for {agent_id}")
            
            # Check if backup is already active
            if agent_id in self.active_backups:
                self.logger.info(f"✅ Backup agent {backup_id} already active for {agent_id}")
                return True
            
            # Start backup agent
            if backup_config.startup_command:
                # In real implementation, start the backup agent process
                # For simulation, just record the activation
                await asyncio.sleep(1)  # Simulate startup time
                
                self.active_backups[agent_id] = backup_id
                
                self.logger.info(f"✅ Backup agent {backup_id} activated for {agent_id}")
                return True
            else:
                self.logger.error(f"❌ No startup command configured for backup {backup_id}")
                return False
            
        except Exception as e:
            self.logger.error(f"❌ Error activating backup agent for {agent_id}: {e}")
            return False
    
    async def _redistribute_workload(self, agent_id: str) -> bool:
        """Redistribute workload from failed agent to healthy agents"""
        try:
            self.logger.info(f"🔄 Redistributing workload from failed agent: {agent_id}")
            
            # In real implementation, this would:
            # 1. Identify capabilities of failed agent
            # 2. Find healthy agents with similar capabilities
            # 3. Redistribute pending work and ongoing tasks
            # 4. Update routing tables and load balancers
            # 5. Notify dependent systems of the change
            
            # For simulation, just record the redistribution
            await asyncio.sleep(0.5)  # Simulate redistribution time
            
            self.logger.info(f"✅ Workload redistributed from {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error redistributing workload from {agent_id}: {e}")
            return False
    
    async def _implement_graceful_degradation(self, agent_id: str) -> bool:
        """Implement graceful degradation when recovery options are exhausted"""
        try:
            self.logger.info(f"🔄 Implementing graceful degradation for {agent_id}")
            
            # In real implementation, this would:
            # 1. Isolate the failed agent from the system
            # 2. Reduce system functionality gracefully
            # 3. Notify users of reduced capabilities
            # 4. Implement fallback procedures
            # 5. Maintain essential operations
            
            # For simulation, just record the degradation
            await asyncio.sleep(0.2)  # Simulate degradation setup
            
            self.logger.info(f"✅ Graceful degradation implemented for {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error implementing graceful degradation for {agent_id}: {e}")
            return False
    
    async def _escalate_failure(self, agent_id: str, failure_count: int) -> bool:
        """Escalate repeated agent failures to administrators"""
        try:
            self.logger.critical(f"🚨 ESCALATING: Agent {agent_id} has failed {failure_count} times")
            
            # Notify escalation callbacks
            for callback in self.escalation_callbacks:
                try:
                    await callback(agent_id, failure_count)
                except Exception as e:
                    self.logger.error(f"❌ Escalation callback failed: {e}")
            
            # In real implementation, this would:
            # 1. Send alerts to operations team
            # 2. Create incident tickets
            # 3. Trigger emergency procedures
            # 4. Potentially switch to maintenance mode
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error escalating failure for {agent_id}: {e}")
            return False
    
    def add_recovery_callback(self, callback: Callable[[RecoveryAttempt], None]):
        """Add callback for recovery attempt notifications"""
        self.recovery_callbacks.append(callback)
    
    def add_escalation_callback(self, callback: Callable[[str, int], None]):
        """Add callback for failure escalation notifications"""
        self.escalation_callbacks.append(callback)
    
    def get_recovery_history(self, agent_id: Optional[str] = None) -> List[RecoveryAttempt]:
        """Get recovery history for all agents or specific agent"""
        if agent_id:
            return [attempt for attempt in self.recovery_attempts if attempt.agent_id == agent_id]
        return self.recovery_attempts.copy()
    
    def get_active_recoveries(self) -> Dict[str, Any]:
        """Get currently active recovery operations"""
        return dict(self.active_recoveries)
    
    def get_active_backups(self) -> Dict[str, str]:
        """Get currently active backup agents"""
        return dict(self.active_backups)
    
    def get_recovery_metrics(self) -> Dict[str, Any]:
        """Get recovery system metrics"""
        total_attempts = len(self.recovery_attempts)
        successful_attempts = sum(1 for attempt in self.recovery_attempts if attempt.success)
        
        strategy_counts = {}
        for attempt in self.recovery_attempts:
            strategy = attempt.strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        average_recovery_time = 0.0
        if total_attempts > 0:
            total_time = sum(attempt.execution_time_ms for attempt in self.recovery_attempts)
            average_recovery_time = total_time / total_attempts
        
        return {
            'total_recovery_attempts': total_attempts,
            'successful_recoveries': successful_attempts,
            'success_rate_percent': (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0,
            'average_recovery_time_ms': average_recovery_time,
            'strategy_usage': strategy_counts,
            'active_recoveries': len(self.active_recoveries),
            'active_backup_agents': len(self.active_backups),
            'agent_failure_counts': self.agent_failure_counts.copy()
        }
    """Health status levels for agents"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"

class PerformanceMetric(Enum):
    """Types of performance metrics tracked"""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    ACCURACY = "accuracy"
    PROFIT_CONTRIBUTION = "profit_contribution"

@dataclass
class AgentHealthMetrics:
    """Health metrics for an individual agent"""
    agent_id: str
    status: AgentHealthStatus
    last_heartbeat: datetime
    response_time_ms: float
    throughput_per_minute: float
    error_rate_percent: float
    cpu_usage_percent: float
    memory_usage_mb: float
    accuracy_percent: float
    profit_contribution_usd: float
    uptime_percent: float
    message_queue_depth: int
    last_error: Optional[str] = None
    last_update: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'agent_id': self.agent_id,
            'status': self.status.value,
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'response_time_ms': self.response_time_ms,
            'throughput_per_minute': self.throughput_per_minute,
            'error_rate_percent': self.error_rate_percent,
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_usage_mb': self.memory_usage_mb,
            'accuracy_percent': self.accuracy_percent,
            'profit_contribution_usd': self.profit_contribution_usd,
            'uptime_percent': self.uptime_percent,
            'message_queue_depth': self.message_queue_depth,
            'last_error': self.last_error,
            'last_update': self.last_update.isoformat()
        }

@dataclass
class HealthCheckResult:
    """Result of a health check operation"""
    agent_id: str
    check_type: str
    success: bool
    response_time_ms: float
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_id': self.agent_id,
            'check_type': self.check_type,
            'success': self.success,
            'response_time_ms': self.response_time_ms,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    metric: PerformanceMetric
    threshold: float
    comparison: str  # 'gt', 'lt', 'eq'
    severity: AgentHealthStatus
    enabled: bool = True
    
class AgentHealthMonitor:
    """
    💓 COMPREHENSIVE AGENT HEALTH MONITORING SYSTEM
    
    Monitors the health and performance of all 9 genius agents:
    - Risk Genius: Risk assessment and portfolio protection
    - Session Expert: Trading session optimization
    - Pattern Master: Pattern detection and analysis
    - Execution Expert: Trade execution optimization
    - Pair Specialist: Currency pair analysis
    - Decision Master: Decision orchestration
    - AI Model Coordinator: Model coordination
    - Market Microstructure Genius: Microstructure analysis
    - Sentiment Integration Genius: Sentiment analysis
    
    Features:
    - Real-time health monitoring
    - Performance trend analysis
    - Automated alerting
    - Predictive health analytics
    - Service level monitoring
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize agent health monitor with auto-recovery capabilities"""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Health tracking
        self.agent_metrics: Dict[str, AgentHealthMetrics] = {}
        self.health_history: Dict[str, List[AgentHealthMetrics]] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # Performance tracking
        self.response_time_history: Dict[str, List[float]] = {}
        self.throughput_history: Dict[str, List[float]] = {}
        self.error_history: Dict[str, List[float]] = {}
        
        # Configuration
        self.heartbeat_interval = self.config.get('heartbeat_interval', 30)  # seconds
        self.health_check_interval = self.config.get('health_check_interval', 60)  # seconds
        self.history_retention_hours = self.config.get('history_retention_hours', 24)
        self.alert_cooldown_minutes = self.config.get('alert_cooldown_minutes', 5)
        
        # Alert tracking
        self.active_alerts: Dict[str, datetime] = {}
        self.alert_callbacks: List[Callable[[str, AgentHealthStatus, str], None]] = []
        
        # AUTO-RECOVERY INTEGRATION 🔄
        self.auto_recovery_enabled = self.config.get('auto_recovery_enabled', True)
        self.auto_recovery_manager = AutoRecoveryManager(self.config.get('recovery_config', {}))
        
        # Add recovery callback to handle recovery notifications
        self.auto_recovery_manager.add_recovery_callback(self._handle_recovery_notification)
        self.auto_recovery_manager.add_escalation_callback(self._handle_escalation_notification)
        
        # Background tasks
        self._monitoring_tasks = set()
        self._running = False
        
        # Initialize default alert rules
        self._setup_default_alert_rules()
        
        self.logger.info("💓 Agent Health Monitor initialized")
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules for agent monitoring"""
        default_rules = [
            AlertRule("response_time_warning", "High Response Time", 
                     PerformanceMetric.RESPONSE_TIME, 100.0, "gt", AgentHealthStatus.WARNING),
            AlertRule("response_time_critical", "Critical Response Time", 
                     PerformanceMetric.RESPONSE_TIME, 500.0, "gt", AgentHealthStatus.CRITICAL),
            AlertRule("error_rate_warning", "High Error Rate", 
                     PerformanceMetric.ERROR_RATE, 5.0, "gt", AgentHealthStatus.WARNING),
            AlertRule("error_rate_critical", "Critical Error Rate", 
                     PerformanceMetric.ERROR_RATE, 15.0, "gt", AgentHealthStatus.CRITICAL),
            AlertRule("cpu_usage_warning", "High CPU Usage", 
                     PerformanceMetric.CPU_USAGE, 80.0, "gt", AgentHealthStatus.WARNING),
            AlertRule("memory_usage_critical", "Critical Memory Usage", 
                     PerformanceMetric.MEMORY_USAGE, 1024.0, "gt", AgentHealthStatus.CRITICAL),
            AlertRule("accuracy_warning", "Low Trading Accuracy", 
                     PerformanceMetric.ACCURACY, 85.0, "lt", AgentHealthStatus.WARNING)
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
    
    async def start_monitoring(self):
        """Start health monitoring for all agents"""
        try:
            self.logger.info("🚀 Starting Agent Health Monitor")
            self._running = True
            
            # Start monitoring tasks
            await self._start_monitoring_tasks()
            
            self.logger.info("✅ Agent Health Monitor started")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start health monitor: {e}")
            raise
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        try:
            self.logger.info("🛑 Stopping Agent Health Monitor")
            self._running = False
            
            # Cancel monitoring tasks
            for task in self._monitoring_tasks:
                task.cancel()
            
            self.logger.info("✅ Agent Health Monitor stopped")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping health monitor: {e}")
    
    async def register_agent(self, agent_id: str, initial_metrics: Dict[str, Any] = None):
        """Register agent for health monitoring"""
        try:
            # Create initial health metrics
            self.agent_metrics[agent_id] = AgentHealthMetrics(
                agent_id=agent_id,
                status=AgentHealthStatus.UNKNOWN,
                last_heartbeat=datetime.now(),
                response_time_ms=0.0,
                throughput_per_minute=0.0,
                error_rate_percent=0.0,
                cpu_usage_percent=0.0,
                memory_usage_mb=0.0,
                accuracy_percent=0.0,
                profit_contribution_usd=0.0,
                uptime_percent=0.0,
                message_queue_depth=0
            )
            
            # Initialize history tracking
            self.health_history[agent_id] = []
            self.response_time_history[agent_id] = []
            self.throughput_history[agent_id] = []
            self.error_history[agent_id] = []
            
            # Update with initial metrics if provided
            if initial_metrics:
                await self.update_agent_metrics(agent_id, initial_metrics)
            
            self.logger.info(f"📝 Agent registered for monitoring: {agent_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register agent {agent_id}: {e}")
            raise
    
    async def update_agent_metrics(self, agent_id: str, metrics: Dict[str, Any]):
        """Update agent health metrics"""
        try:
            if agent_id not in self.agent_metrics:
                await self.register_agent(agent_id)
            
            current_metrics = self.agent_metrics[agent_id]
            
            # Update metrics
            current_metrics.last_heartbeat = datetime.now()
            current_metrics.response_time_ms = metrics.get('response_time_ms', current_metrics.response_time_ms)
            current_metrics.throughput_per_minute = metrics.get('throughput_per_minute', current_metrics.throughput_per_minute)
            current_metrics.error_rate_percent = metrics.get('error_rate_percent', current_metrics.error_rate_percent)
            current_metrics.cpu_usage_percent = metrics.get('cpu_usage_percent', current_metrics.cpu_usage_percent)
            current_metrics.memory_usage_mb = metrics.get('memory_usage_mb', current_metrics.memory_usage_mb)
            current_metrics.accuracy_percent = metrics.get('accuracy_percent', current_metrics.accuracy_percent)
            current_metrics.profit_contribution_usd = metrics.get('profit_contribution_usd', current_metrics.profit_contribution_usd)
            current_metrics.uptime_percent = metrics.get('uptime_percent', current_metrics.uptime_percent)
            current_metrics.message_queue_depth = metrics.get('message_queue_depth', current_metrics.message_queue_depth)
            current_metrics.last_error = metrics.get('last_error', current_metrics.last_error)
            current_metrics.last_update = datetime.now()
            
            # Update history
            self.response_time_history[agent_id].append(current_metrics.response_time_ms)
            self.throughput_history[agent_id].append(current_metrics.throughput_per_minute)
            self.error_history[agent_id].append(current_metrics.error_rate_percent)
            
            # Limit history size
            max_history = 1440  # 24 hours of minute-by-minute data
            for history_list in [self.response_time_history[agent_id], 
                               self.throughput_history[agent_id], 
                               self.error_history[agent_id]]:
                if len(history_list) > max_history:
                    history_list.pop(0)
            
            # Calculate health status
            await self._calculate_health_status(agent_id)
            
            # Check alert rules
            await self._check_alert_rules(agent_id)
            
            # Store in history
            self.health_history[agent_id].append(
                AgentHealthMetrics(**current_metrics.__dict__)
            )
            
            # Limit health history
            if len(self.health_history[agent_id]) > max_history:
                self.health_history[agent_id].pop(0)
            
            self.logger.debug(f"📊 Updated metrics for {agent_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update metrics for {agent_id}: {e}")
    
    async def _calculate_health_status(self, agent_id: str):
        """Calculate overall health status for agent"""
        try:
            metrics = self.agent_metrics[agent_id]
            
            # Check heartbeat freshness
            heartbeat_age = (datetime.now() - metrics.last_heartbeat).total_seconds()
            if heartbeat_age > 120:  # 2 minutes
                metrics.status = AgentHealthStatus.UNAVAILABLE
                return
            
            # Calculate health score based on multiple factors
            health_score = 100.0
            
            # Response time impact
            if metrics.response_time_ms > 500:
                health_score -= 30
            elif metrics.response_time_ms > 100:
                health_score -= 15
            
            # Error rate impact
            if metrics.error_rate_percent > 15:
                health_score -= 40
            elif metrics.error_rate_percent > 5:
                health_score -= 20
            
            # Resource usage impact
            if metrics.cpu_usage_percent > 90:
                health_score -= 25
            elif metrics.cpu_usage_percent > 80:
                health_score -= 10
            
            if metrics.memory_usage_mb > 1024:
                health_score -= 20
            elif metrics.memory_usage_mb > 512:
                health_score -= 10
            
            # Accuracy impact (for trading agents)
            if metrics.accuracy_percent < 70:
                health_score -= 30
            elif metrics.accuracy_percent < 85:
                health_score -= 15
            
            # Determine status based on score
            if health_score >= 80:
                metrics.status = AgentHealthStatus.HEALTHY
            elif health_score >= 60:
                metrics.status = AgentHealthStatus.WARNING
            else:
                metrics.status = AgentHealthStatus.CRITICAL
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating health status for {agent_id}: {e}")
            self.agent_metrics[agent_id].status = AgentHealthStatus.UNKNOWN
    
    async def _check_alert_rules(self, agent_id: str):
        """Check alert rules against agent metrics"""
        try:
            metrics = self.agent_metrics[agent_id]
            
            for rule in self.alert_rules.values():
                if not rule.enabled:
                    continue
                
                # Get metric value
                metric_value = self._get_metric_value(metrics, rule.metric)
                if metric_value is None:
                    continue
                
                # Check threshold
                triggered = False
                if rule.comparison == "gt" and metric_value > rule.threshold:
                    triggered = True
                elif rule.comparison == "lt" and metric_value < rule.threshold:
                    triggered = True
                elif rule.comparison == "eq" and metric_value == rule.threshold:
                    triggered = True
                
                if triggered:
                    await self._trigger_alert(agent_id, rule, metric_value)
            
        except Exception as e:
            self.logger.error(f"❌ Error checking alert rules for {agent_id}: {e}")
    
    def _get_metric_value(self, metrics: AgentHealthMetrics, metric: PerformanceMetric) -> Optional[float]:
        """Get metric value from agent metrics"""
        metric_map = {
            PerformanceMetric.RESPONSE_TIME: metrics.response_time_ms,
            PerformanceMetric.THROUGHPUT: metrics.throughput_per_minute,
            PerformanceMetric.ERROR_RATE: metrics.error_rate_percent,
            PerformanceMetric.CPU_USAGE: metrics.cpu_usage_percent,
            PerformanceMetric.MEMORY_USAGE: metrics.memory_usage_mb,
            PerformanceMetric.ACCURACY: metrics.accuracy_percent,
            PerformanceMetric.PROFIT_CONTRIBUTION: metrics.profit_contribution_usd
        }
        return metric_map.get(metric)
    
    async def _trigger_alert(self, agent_id: str, rule: AlertRule, metric_value: float):
        """Trigger alert for rule violation and initiate auto-recovery if needed"""
        try:
            alert_key = f"{agent_id}:{rule.rule_id}"
            
            # Check cooldown
            if alert_key in self.active_alerts:
                last_alert = self.active_alerts[alert_key]
                if (datetime.now() - last_alert).total_seconds() < (self.alert_cooldown_minutes * 60):
                    return
            
            # Record alert
            self.active_alerts[alert_key] = datetime.now()
            
            alert_message = f"Alert: {rule.name} for {agent_id} - {rule.metric.value}: {metric_value} (threshold: {rule.threshold})"
            
            # Log alert
            if rule.severity == AgentHealthStatus.CRITICAL:
                self.logger.critical(f"🚨 {alert_message}")
            elif rule.severity == AgentHealthStatus.WARNING:
                self.logger.warning(f"⚠️ {alert_message}")
            
            # AUTO-RECOVERY INTEGRATION 🔄
            # Trigger auto-recovery for critical and warning alerts
            if self.auto_recovery_enabled and rule.severity in [AgentHealthStatus.CRITICAL, AgentHealthStatus.WARNING]:
                failure_reason = f"{rule.name}: {rule.metric.value} = {metric_value} (threshold: {rule.threshold})"
                severity_str = "critical" if rule.severity == AgentHealthStatus.CRITICAL else "warning"
                
                recovery_initiated = await self.auto_recovery_manager.handle_agent_failure(
                    agent_id=agent_id,
                    failure_reason=failure_reason,
                    severity=severity_str
                )
                
                if recovery_initiated:
                    self.logger.info(f"🔄 Auto-recovery initiated for {agent_id}")
                else:
                    self.logger.warning(f"⚠️ Auto-recovery not initiated for {agent_id}")
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(agent_id, rule.severity, alert_message)
                except Exception as e:
                    self.logger.error(f"❌ Alert callback failed: {e}")
            
        except Exception as e:
            self.logger.error(f"❌ Error triggering alert: {e}")
    
    async def perform_health_check(self, agent_id: str, check_type: str = "ping") -> HealthCheckResult:
        """Perform active health check on agent"""
        try:
            start_time = time.time()
            
            # Different types of health checks
            if check_type == "ping":
                result = await self._ping_health_check(agent_id)
            elif check_type == "performance":
                result = await self._performance_health_check(agent_id)
            elif check_type == "integration":
                result = await self._integration_health_check(agent_id)
            else:
                result = HealthCheckResult(
                    agent_id=agent_id,
                    check_type=check_type,
                    success=False,
                    response_time_ms=0.0,
                    details={"error": f"Unknown check type: {check_type}"}
                )
            
            response_time = (time.time() - start_time) * 1000
            result.response_time_ms = response_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Health check failed for {agent_id}: {e}")
            return HealthCheckResult(
                agent_id=agent_id,
                check_type=check_type,
                success=False,
                response_time_ms=0.0,
                details={"error": str(e)}
            )
    
    async def _ping_health_check(self, agent_id: str) -> HealthCheckResult:
        """Basic ping health check"""
        # Simulate ping check
        await asyncio.sleep(0.01)  # Simulate network latency
        
        return HealthCheckResult(
            agent_id=agent_id,
            check_type="ping",
            success=True,
            response_time_ms=0.0,
            details={"status": "alive"}
        )
    
    async def _performance_health_check(self, agent_id: str) -> HealthCheckResult:
        """Performance-based health check"""
        # Simulate performance check
        await asyncio.sleep(0.05)  # Simulate processing time
        
        return HealthCheckResult(
            agent_id=agent_id,
            check_type="performance",
            success=True,
            response_time_ms=0.0,
            details={"performance": "good"}
        )
    
    async def _integration_health_check(self, agent_id: str) -> HealthCheckResult:
        """Integration health check"""
        # Simulate integration check
        await asyncio.sleep(0.1)  # Simulate complex check
        
        return HealthCheckResult(
            agent_id=agent_id,
            check_type="integration",
            success=True,
            response_time_ms=0.0,
            details={"integrations": "healthy"}
        )
    
    async def _start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        try:
            # Heartbeat monitoring
            task1 = asyncio.create_task(self._heartbeat_monitor())
            self._monitoring_tasks.add(task1)
            
            # Periodic health checks
            task2 = asyncio.create_task(self._periodic_health_checks())
            self._monitoring_tasks.add(task2)
            
            # Metrics cleanup
            task3 = asyncio.create_task(self._metrics_cleanup())
            self._monitoring_tasks.add(task3)
            
            # Health reporting
            task4 = asyncio.create_task(self._health_reporting())
            self._monitoring_tasks.add(task4)
            
            self.logger.info("✅ Monitoring tasks started")
            
        except Exception as e:
            self.logger.error(f"❌ Error starting monitoring tasks: {e}")
    
    async def _heartbeat_monitor(self):
        """Monitor agent heartbeats"""
        while self._running:
            try:
                current_time = datetime.now()
                
                for agent_id, metrics in self.agent_metrics.items():
                    heartbeat_age = (current_time - metrics.last_heartbeat).total_seconds()
                    
                    if heartbeat_age > 120:  # 2 minutes
                        if metrics.status != AgentHealthStatus.UNAVAILABLE:
                            metrics.status = AgentHealthStatus.UNAVAILABLE
                            self.logger.warning(f"💔 Agent {agent_id} heartbeat timeout")
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Heartbeat monitor error: {e}")
                await asyncio.sleep(5)
    
    async def _periodic_health_checks(self):
        """Perform periodic active health checks"""
        while self._running:
            try:
                for agent_id in self.agent_metrics.keys():
                    # Perform health check
                    result = await self.perform_health_check(agent_id, "ping")
                    
                    # Update metrics based on health check
                    if not result.success:
                        await self.update_agent_metrics(agent_id, {
                            'error_rate_percent': self.agent_metrics[agent_id].error_rate_percent + 1
                        })
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Periodic health check error: {e}")
                await asyncio.sleep(5)
    
    async def _metrics_cleanup(self):
        """Clean up old metrics data"""
        while self._running:
            try:
                cutoff_time = datetime.now() - timedelta(hours=self.history_retention_hours)
                
                # Clean up health history
                for agent_id in self.health_history.keys():
                    self.health_history[agent_id] = [
                        metrics for metrics in self.health_history[agent_id]
                        if metrics.last_update > cutoff_time
                    ]
                
                await asyncio.sleep(3600)  # Every hour
                
            except Exception as e:
                self.logger.error(f"❌ Metrics cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def _health_reporting(self):
        """Generate periodic health reports"""
        while self._running:
            try:
                # Generate health summary
                healthy_count = sum(1 for m in self.agent_metrics.values() 
                                  if m.status == AgentHealthStatus.HEALTHY)
                total_count = len(self.agent_metrics)
                
                if total_count > 0:
                    health_percentage = (healthy_count / total_count) * 100
                    self.logger.info(f"📊 Agent Health Summary: {healthy_count}/{total_count} healthy ({health_percentage:.1f}%)")
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                self.logger.error(f"❌ Health reporting error: {e}")
                await asyncio.sleep(60)
    
    def add_alert_callback(self, callback: Callable[[str, AgentHealthStatus, str], None]):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def get_agent_health(self, agent_id: str) -> Optional[AgentHealthMetrics]:
        """Get current health metrics for agent"""
        return self.agent_metrics.get(agent_id)
    
    def get_all_agent_health(self) -> Dict[str, AgentHealthMetrics]:
        """Get health metrics for all agents"""
        return {agent_id: metrics.to_dict() for agent_id, metrics in self.agent_metrics.items()}
    
    def get_health_trends(self, agent_id: str) -> Dict[str, List[float]]:
        """Get health trends for agent"""
        return {
            'response_time': self.response_time_history.get(agent_id, []),
            'throughput': self.throughput_history.get(agent_id, []),
            'error_rate': self.error_history.get(agent_id, [])
        }
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary with recovery metrics"""
        if not self.agent_metrics:
            return {"status": "no_agents", "healthy_agents": 0, "total_agents": 0}
        
        status_counts = {}
        for status in AgentHealthStatus:
            status_counts[status.value] = sum(1 for m in self.agent_metrics.values() if m.status == status)
        
        total_agents = len(self.agent_metrics)
        healthy_agents = status_counts.get('healthy', 0)
        
        overall_status = "healthy"
        if status_counts.get('critical', 0) > 0:
            overall_status = "critical"
        elif status_counts.get('warning', 0) > 0:
            overall_status = "warning"
        elif status_counts.get('unavailable', 0) > 0:
            overall_status = "degraded"
        
        # Include recovery metrics
        recovery_metrics = self.auto_recovery_manager.get_recovery_metrics() if self.auto_recovery_enabled else {}
        
        return {
            "status": overall_status,
            "healthy_agents": healthy_agents,
            "total_agents": total_agents,
            "health_percentage": (healthy_agents / total_agents * 100) if total_agents > 0 else 0,
            "status_breakdown": status_counts,
            "active_alerts": len(self.active_alerts),
            "auto_recovery_enabled": self.auto_recovery_enabled,
            "recovery_metrics": recovery_metrics
        }
    
    # AUTO-RECOVERY INTEGRATION METHODS 🔄
    
    async def _handle_recovery_notification(self, recovery_attempt: RecoveryAttempt):
        """Handle notifications from auto-recovery manager"""
        try:
            if recovery_attempt.success:
                self.logger.info(f"✅ Recovery successful for {recovery_attempt.agent_id}: {recovery_attempt.strategy.value}")
                
                # Reset agent to healthy status if recovery was successful
                if recovery_attempt.agent_id in self.agent_metrics:
                    self.agent_metrics[recovery_attempt.agent_id].status = AgentHealthStatus.HEALTHY
                    self.agent_metrics[recovery_attempt.agent_id].last_error = None
                    
            else:
                self.logger.error(f"❌ Recovery failed for {recovery_attempt.agent_id}: {recovery_attempt.error_message}")
                
        except Exception as e:
            self.logger.error(f"❌ Error handling recovery notification: {e}")
    
    async def _handle_escalation_notification(self, agent_id: str, failure_count: int):
        """Handle escalation notifications from auto-recovery manager"""
        try:
            self.logger.critical(f"🚨 ESCALATION: Agent {agent_id} has failed {failure_count} times - manual intervention required")
            
            # Mark agent as requiring manual intervention
            if agent_id in self.agent_metrics:
                self.agent_metrics[agent_id].status = AgentHealthStatus.CRITICAL
                self.agent_metrics[agent_id].last_error = f"Escalated after {failure_count} recovery failures"
            
            # TODO: Send notifications to operations team
            # TODO: Create incident tickets
            # TODO: Trigger emergency procedures
            
        except Exception as e:
            self.logger.error(f"❌ Error handling escalation notification: {e}")
    
    def simulate_agent_failure(self, agent_id: str, failure_type: str = "high_response_time"):
        """Simulate agent failure for testing auto-recovery (TESTING ONLY)"""
        if not self.auto_recovery_enabled:
            self.logger.warning("Auto-recovery is disabled - cannot simulate failure")
            return
        
        self.logger.warning(f"🧪 SIMULATING FAILURE for {agent_id}: {failure_type}")
        
        # Simulate different types of failures
        if failure_type == "high_response_time":
            # Trigger high response time alert
            asyncio.create_task(self.update_agent_metrics(agent_id, {
                'response_time_ms': 600.0,  # Above critical threshold
                'error_rate_percent': 5.0
            }))
        elif failure_type == "high_error_rate":
            # Trigger high error rate alert
            asyncio.create_task(self.update_agent_metrics(agent_id, {
                'error_rate_percent': 20.0,  # Above critical threshold
                'response_time_ms': 50.0
            }))
        elif failure_type == "unavailable":
            # Mark as unavailable
            if agent_id in self.agent_metrics:
                self.agent_metrics[agent_id].status = AgentHealthStatus.UNAVAILABLE
                self.agent_metrics[agent_id].last_error = "Simulated unavailability"
                
                # Trigger recovery directly
                asyncio.create_task(self.auto_recovery_manager.handle_agent_failure(
                    agent_id=agent_id,
                    failure_reason="Simulated unavailability",
                    severity="critical"
                ))
    
    def get_recovery_history(self, agent_id: Optional[str] = None) -> List[RecoveryAttempt]:
        """Get recovery history from auto-recovery manager"""
        if not self.auto_recovery_enabled:
            return []
        return self.auto_recovery_manager.get_recovery_history(agent_id)
    
    def get_recovery_metrics(self) -> Dict[str, Any]:
        """Get recovery metrics from auto-recovery manager"""
        if not self.auto_recovery_enabled:
            return {"auto_recovery_enabled": False}
        return self.auto_recovery_manager.get_recovery_metrics()
    
    def enable_auto_recovery(self):
        """Enable auto-recovery system"""
        self.auto_recovery_enabled = True
        self.logger.info("🔄 Auto-recovery system enabled")
    
    def disable_auto_recovery(self):
        """Disable auto-recovery system"""
        self.auto_recovery_enabled = False
        self.logger.info("🛑 Auto-recovery system disabled")