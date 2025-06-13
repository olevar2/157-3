"""
🚨 ERROR PROPAGATION & COORDINATION SYSTEM
==========================================

Comprehensive error propagation framework for Platform3 agent coordination.
Manages error cascading, propagation chains, and cross-agent failure handling.

Mission: Prevent cascading failures and ensure system resilience
"""

import asyncio
import logging
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class ErrorSeverity(Enum):
    """Error severity levels for propagation decisions"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class PropagationStrategy(Enum):
    """Strategies for error propagation"""

    IMMEDIATE = "immediate"  # Propagate immediately
    DELAYED = "delayed"  # Delay propagation to prevent cascading
    SELECTIVE = "selective"  # Only propagate to specific agents
    SUPPRESSED = "suppressed"  # Suppress propagation to prevent cascade
    ESCALATED = "escalated"  # Escalate to higher level coordination


class ErrorCategory(Enum):
    """Categories of errors for targeted handling"""

    NETWORK = "network"
    COMPUTATION = "computation"
    DATA = "data"
    RESOURCE = "resource"
    COORDINATION = "coordination"
    EXTERNAL_SERVICE = "external_service"
    AUTHENTICATION = "authentication"
    CONFIGURATION = "configuration"


@dataclass
class ErrorEvent:
    """Represents an error event in the system"""

    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_agent: str = ""
    error_type: str = ""
    severity: ErrorSeverity = ErrorSeverity.WARNING
    category: ErrorCategory = ErrorCategory.COMPUTATION
    message: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    stack_trace: Optional[str] = None
    affected_operations: List[str] = field(default_factory=list)
    propagation_chain: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PropagationRule:
    """Rules for error propagation between agents"""

    source_agent: str
    target_agents: List[str]
    error_categories: List[ErrorCategory]
    severity_threshold: ErrorSeverity
    strategy: PropagationStrategy
    delay_seconds: int = 0
    conditions: Dict[str, Any] = field(default_factory=dict)
    suppression_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentErrorState:
    """Current error state of an agent"""

    agent_id: str
    current_errors: List[ErrorEvent] = field(default_factory=list)
    error_count_last_hour: int = 0
    last_error_time: Optional[datetime] = None
    consecutive_failures: int = 0
    is_failing: bool = False
    is_degraded: bool = False
    quarantine_until: Optional[datetime] = None
    circuit_breaker_open: bool = False


class ErrorPropagationManager:
    """
    🚨 CORE ERROR PROPAGATION ENGINE

    Manages error propagation chains, prevents cascading failures,
    and coordinates error responses across the agent ecosystem.

    Key Features:
    - Intelligent error propagation with delay mechanisms
    - Circuit breaker integration to prevent error cascading
    - Agent quarantine for failing components
    - Error correlation and root cause analysis
    - Dynamic propagation rule adjustment
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Error Propagation Manager"""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Core state
        self.agent_states: Dict[str, AgentErrorState] = {}
        self.propagation_rules: List[PropagationRule] = []
        self.error_history: deque = deque(maxlen=10000)  # Last 10k errors
        self.active_propagations: Dict[str, asyncio.Task] = {}

        # Circuit breaker configuration
        self.circuit_breaker_thresholds = {
            ErrorSeverity.CRITICAL: 3,  # 3 critical errors open circuit
            ErrorSeverity.ERROR: 5,  # 5 errors open circuit
            ErrorSeverity.WARNING: 10,  # 10 warnings open circuit
        }

        # Propagation timing
        self.propagation_delays = {
            ErrorSeverity.EMERGENCY: 0,  # Immediate
            ErrorSeverity.CRITICAL: 1,  # 1 second delay
            ErrorSeverity.ERROR: 3,  # 3 second delay
            ErrorSeverity.WARNING: 10,  # 10 second delay
            ErrorSeverity.INFO: 30,  # 30 second delay
        }

        # Callbacks
        self.error_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.propagation_callbacks: List[Callable] = []
        self.quarantine_callbacks: List[Callable] = []

        # Initialize default propagation rules
        self._setup_default_propagation_rules()

        # Start background tasks
        self._start_background_tasks()

        self.logger.info("🚨 Error Propagation Manager initialized")

    def _setup_default_propagation_rules(self):
        """Setup default error propagation rules between agents"""

        # Critical agent failure propagation rules
        critical_agents = [
            "risk_genius",
            "decision_master",
            "execution_expert",
            "ai_model_coordinator",
            "pattern_master",
        ]

        # Rule 1: Critical agent failures propagate to all dependent agents
        for critical_agent in critical_agents:
            dependent_agents = self._get_dependent_agents(critical_agent)
            if dependent_agents:
                rule = PropagationRule(
                    source_agent=critical_agent,
                    target_agents=dependent_agents,
                    error_categories=[
                        ErrorCategory.COORDINATION,
                        ErrorCategory.COMPUTATION,
                    ],
                    severity_threshold=ErrorSeverity.ERROR,
                    strategy=PropagationStrategy.DELAYED,
                    delay_seconds=2,
                    conditions={"failure_count_threshold": 2},
                )
                self.propagation_rules.append(rule)

        # Rule 2: Network errors propagate selectively to prevent cascade
        network_rule = PropagationRule(
            source_agent="*",  # Any agent
            target_agents=["coordination_hub", "health_monitor"],
            error_categories=[ErrorCategory.NETWORK],
            severity_threshold=ErrorSeverity.WARNING,
            strategy=PropagationStrategy.SELECTIVE,
            delay_seconds=5,
        )
        self.propagation_rules.append(network_rule)

        # Rule 3: Resource exhaustion errors are escalated but not propagated
        resource_rule = PropagationRule(
            source_agent="*",
            target_agents=["resource_manager"],
            error_categories=[ErrorCategory.RESOURCE],
            severity_threshold=ErrorSeverity.ERROR,
            strategy=PropagationStrategy.ESCALATED,
            delay_seconds=0,
        )
        self.propagation_rules.append(resource_rule)

        # Rule 4: Authentication errors are immediately escalated
        auth_rule = PropagationRule(
            source_agent="*",
            target_agents=["security_manager", "audit_logger"],
            error_categories=[ErrorCategory.AUTHENTICATION],
            severity_threshold=ErrorSeverity.WARNING,
            strategy=PropagationStrategy.IMMEDIATE,
            delay_seconds=0,
        )
        self.propagation_rules.append(auth_rule)

        self.logger.info(
            f"Setup {len(self.propagation_rules)} default propagation rules"
        )

    def _get_dependent_agents(self, agent_id: str) -> List[str]:
        """Get list of agents dependent on the given agent"""
        dependency_map = {
            "risk_genius": ["decision_master", "execution_expert", "portfolio_manager"],
            "decision_master": ["execution_expert", "coordination_hub"],
            "execution_expert": ["trade_monitor", "settlement_manager"],
            "ai_model_coordinator": [
                "pattern_master",
                "sentiment_analyzer",
                "prediction_engine",
            ],
            "pattern_master": ["trend_analyzer", "signal_generator"],
            "market_data_agent": [
                "risk_genius",
                "pattern_master",
                "sentiment_analyzer",
            ],
            "coordination_hub": ["health_monitor", "performance_tracker"],
        }
        return dependency_map.get(agent_id, [])

    def _start_background_tasks(self):
        """Start background monitoring and cleanup tasks"""
        asyncio.create_task(self._monitor_agent_health())
        asyncio.create_task(self._cleanup_old_errors())
        asyncio.create_task(self._adjust_propagation_rules())

    async def handle_error_event(self, error_event: ErrorEvent) -> bool:
        """
        Handle incoming error event and manage propagation

        Args:
            error_event: The error event to handle

        Returns:
            bool: True if handled successfully
        """
        try:
            self.logger.warning(
                f"🚨 Error event: {error_event.source_agent} - {error_event.message}"
            )

            # Record error in history
            self.error_history.append(error_event)

            # Update agent error state
            await self._update_agent_error_state(error_event)

            # Check circuit breaker conditions
            await self._check_circuit_breaker(
                error_event.source_agent, error_event.severity
            )

            # Determine propagation strategy
            propagation_decisions = await self._determine_propagation(error_event)

            # Execute propagation
            for decision in propagation_decisions:
                await self._execute_propagation(error_event, decision)

            # Call error handlers
            await self._call_error_handlers(error_event)

            return True

        except Exception as e:
            self.logger.error(f"❌ Error handling error event: {e}")
            return False

    async def _update_agent_error_state(self, error_event: ErrorEvent):
        """Update the error state for the source agent"""
        agent_id = error_event.source_agent

        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = AgentErrorState(agent_id=agent_id)

        state = self.agent_states[agent_id]
        state.current_errors.append(error_event)
        state.last_error_time = error_event.timestamp

        # Count errors in last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        state.error_count_last_hour = len(
            [e for e in state.current_errors if e.timestamp > one_hour_ago]
        )

        # Update failure indicators
        if error_event.severity in [
            ErrorSeverity.ERROR,
            ErrorSeverity.CRITICAL,
            ErrorSeverity.EMERGENCY,
        ]:
            state.consecutive_failures += 1

            if state.consecutive_failures >= 3:
                state.is_failing = True
                await self._consider_quarantine(agent_id)
        else:
            # Reset on successful operation
            if (
                error_event.severity == ErrorSeverity.INFO
                and "success" in error_event.message.lower()
            ):
                state.consecutive_failures = 0
                state.is_failing = False

        # Degradation detection
        if state.error_count_last_hour >= 5:
            state.is_degraded = True
        elif state.error_count_last_hour <= 2:
            state.is_degraded = False

        # Keep only recent errors (last 100)
        state.current_errors = state.current_errors[-100:]

    async def _check_circuit_breaker(self, agent_id: str, severity: ErrorSeverity):
        """Check and update circuit breaker status"""
        state = self.agent_states.get(agent_id)
        if not state:
            return

        # Count recent errors by severity
        now = datetime.now()
        recent_errors = [
            e
            for e in state.current_errors
            if (now - e.timestamp).total_seconds() < 300  # Last 5 minutes
        ]

        severity_counts = defaultdict(int)
        for error in recent_errors:
            severity_counts[error.severity] += 1

        # Check thresholds
        should_open = False
        for sev, threshold in self.circuit_breaker_thresholds.items():
            if severity_counts[sev] >= threshold:
                should_open = True
                break

        if should_open and not state.circuit_breaker_open:
            state.circuit_breaker_open = True
            self.logger.warning(f"🔴 Circuit breaker OPENED for {agent_id}")

            # Notify callbacks
            for callback in self.quarantine_callbacks:
                try:
                    await callback(agent_id, "circuit_breaker_open")
                except Exception as e:
                    self.logger.error(f"Callback error: {e}")

        elif not should_open and state.circuit_breaker_open:
            # Check if we can close the circuit breaker
            if (now - state.last_error_time).total_seconds() > 60:  # 1 minute recovery
                state.circuit_breaker_open = False
                self.logger.info(f"🟢 Circuit breaker CLOSED for {agent_id}")

    async def _determine_propagation(
        self, error_event: ErrorEvent
    ) -> List[Dict[str, Any]]:
        """Determine how to propagate this error event"""
        decisions = []

        for rule in self.propagation_rules:
            # Check if rule applies
            if not self._rule_matches(rule, error_event):
                continue

            # Check suppression conditions
            if self._should_suppress_propagation(rule, error_event):
                continue

            # Create propagation decision
            decision = {
                "rule": rule,
                "target_agents": rule.target_agents.copy(),
                "strategy": rule.strategy,
                "delay": rule.delay_seconds,
                "priority": self._calculate_priority(error_event.severity),
            }

            # Apply conditions
            if rule.conditions:
                decision = self._apply_conditions(
                    decision, rule.conditions, error_event
                )

            if decision:  # Only add if not filtered out by conditions
                decisions.append(decision)

        return decisions

    def _rule_matches(self, rule: PropagationRule, error_event: ErrorEvent) -> bool:
        """Check if propagation rule matches the error event"""
        # Check source agent
        if rule.source_agent != "*" and rule.source_agent != error_event.source_agent:
            return False

        # Check error category
        if error_event.category not in rule.error_categories:
            return False

        # Check severity threshold
        severity_order = [
            ErrorSeverity.INFO,
            ErrorSeverity.WARNING,
            ErrorSeverity.ERROR,
            ErrorSeverity.CRITICAL,
            ErrorSeverity.EMERGENCY,
        ]

        event_level = severity_order.index(error_event.severity)
        threshold_level = severity_order.index(rule.severity_threshold)

        if event_level < threshold_level:
            return False

        return True

    def _should_suppress_propagation(
        self, rule: PropagationRule, error_event: ErrorEvent
    ) -> bool:
        """Check if propagation should be suppressed"""
        # Check for recent similar errors (debouncing)
        recent_threshold = datetime.now() - timedelta(minutes=5)
        similar_errors = [
            e
            for e in self.error_history
            if (
                e.source_agent == error_event.source_agent
                and e.category == error_event.category
                and e.severity == error_event.severity
                and e.timestamp > recent_threshold
            )
        ]

        if len(similar_errors) > 3:  # Too many similar errors recently
            return True

        # Check if target agents are already in error state
        failing_targets = [
            agent
            for agent in rule.target_agents
            if agent in self.agent_states and self.agent_states[agent].is_failing
        ]

        if (
            len(failing_targets) >= len(rule.target_agents) * 0.5
        ):  # More than 50% already failing
            return True

        # Check suppression conditions from rule
        if rule.suppression_conditions:
            for condition, value in rule.suppression_conditions.items():
                if condition == "cascade_prevention" and value:
                    # Check for potential cascade
                    if self._detect_cascade_risk(error_event):
                        return True

        return False

    def _detect_cascade_risk(self, error_event: ErrorEvent) -> bool:
        """Detect if propagating this error could cause a cascade"""
        # Count agents currently in error state
        failing_agents = sum(
            1 for state in self.agent_states.values() if state.is_failing
        )
        total_agents = len(self.agent_states)

        if (
            total_agents > 0 and failing_agents / total_agents > 0.3
        ):  # More than 30% failing
            return True

        # Check for rapid error rate increase
        recent_errors = [
            e
            for e in self.error_history
            if (datetime.now() - e.timestamp).total_seconds() < 60  # Last minute
        ]

        if len(recent_errors) > 20:  # Too many errors too quickly
            return True

        return False

    async def _execute_propagation(
        self, error_event: ErrorEvent, decision: Dict[str, Any]
    ):
        """Execute the propagation decision"""
        rule = decision["rule"]
        strategy = decision["strategy"]
        delay = decision["delay"]

        if delay > 0:
            await asyncio.sleep(delay)

        if strategy == PropagationStrategy.IMMEDIATE:
            await self._propagate_immediate(error_event, decision["target_agents"])
        elif strategy == PropagationStrategy.SELECTIVE:
            await self._propagate_selective(error_event, decision["target_agents"])
        elif strategy == PropagationStrategy.ESCALATED:
            await self._propagate_escalated(error_event, decision["target_agents"])
        elif strategy == PropagationStrategy.DELAYED:
            await self._propagate_delayed(error_event, decision["target_agents"], delay)

        # Record propagation
        for callback in self.propagation_callbacks:
            try:
                await callback(error_event, decision)
            except Exception as e:
                self.logger.error(f"Propagation callback error: {e}")

    async def _propagate_immediate(
        self, error_event: ErrorEvent, target_agents: List[str]
    ):
        """Immediately propagate error to target agents"""
        propagated_event = ErrorEvent(
            source_agent=error_event.source_agent,
            error_type=f"propagated_{error_event.error_type}",
            severity=error_event.severity,
            category=error_event.category,
            message=f"Propagated: {error_event.message}",
            context=error_event.context.copy(),
            propagation_chain=error_event.propagation_chain
            + [error_event.source_agent],
        )

        for agent in target_agents:
            if agent in self.error_handlers:
                for handler in self.error_handlers[agent]:
                    try:
                        await handler(propagated_event)
                    except Exception as e:
                        self.logger.error(f"Error handler failed for {agent}: {e}")

    async def _propagate_selective(
        self, error_event: ErrorEvent, target_agents: List[str]
    ):
        """Selectively propagate to healthy target agents only"""
        healthy_targets = [
            agent
            for agent in target_agents
            if agent not in self.agent_states or not self.agent_states[agent].is_failing
        ]

        if healthy_targets:
            await self._propagate_immediate(error_event, healthy_targets)

    async def _propagate_escalated(
        self, error_event: ErrorEvent, target_agents: List[str]
    ):
        """Escalate error to management/coordination agents"""
        escalated_event = ErrorEvent(
            source_agent=error_event.source_agent,
            error_type=f"escalated_{error_event.error_type}",
            severity=(
                ErrorSeverity.CRITICAL
                if error_event.severity != ErrorSeverity.EMERGENCY
                else ErrorSeverity.EMERGENCY
            ),
            category=error_event.category,
            message=f"ESCALATED: {error_event.message}",
            context=error_event.context.copy(),
            propagation_chain=error_event.propagation_chain
            + [error_event.source_agent],
            metadata={
                "escalation_reason": "automatic_escalation",
                "original_severity": error_event.severity.value,
            },
        )

        await self._propagate_immediate(escalated_event, target_agents)

    async def _propagate_delayed(
        self, error_event: ErrorEvent, target_agents: List[str], delay: int
    ):
        """Propagate error after additional delay"""
        await asyncio.sleep(delay)
        await self._propagate_selective(error_event, target_agents)

    async def _consider_quarantine(self, agent_id: str):
        """Consider placing agent in quarantine"""
        state = self.agent_states.get(agent_id)
        if not state or state.quarantine_until:
            return  # Already quarantined

        if state.consecutive_failures >= 5 or state.error_count_last_hour >= 10:
            # Quarantine for 10 minutes
            state.quarantine_until = datetime.now() + timedelta(minutes=10)

            self.logger.warning(
                f"🚧 Agent {agent_id} placed in quarantine until {state.quarantine_until}"
            )

            # Notify quarantine callbacks
            for callback in self.quarantine_callbacks:
                try:
                    await callback(agent_id, "quarantined")
                except Exception as e:
                    self.logger.error(f"Quarantine callback error: {e}")

    def _calculate_priority(self, severity: ErrorSeverity) -> int:
        """Calculate propagation priority based on severity"""
        priority_map = {
            ErrorSeverity.EMERGENCY: 0,
            ErrorSeverity.CRITICAL: 1,
            ErrorSeverity.ERROR: 2,
            ErrorSeverity.WARNING: 3,
            ErrorSeverity.INFO: 4,
        }
        return priority_map.get(severity, 3)

    def _apply_conditions(
        self,
        decision: Dict[str, Any],
        conditions: Dict[str, Any],
        error_event: ErrorEvent,
    ) -> Optional[Dict[str, Any]]:
        """Apply conditional logic to propagation decision"""
        if "failure_count_threshold" in conditions:
            threshold = conditions["failure_count_threshold"]
            state = self.agent_states.get(error_event.source_agent)

            if not state or state.consecutive_failures < threshold:
                return None  # Don't propagate if threshold not met

        if "time_window_minutes" in conditions:
            window_minutes = conditions["time_window_minutes"]
            cutoff_time = datetime.now() - timedelta(minutes=window_minutes)

            # Only propagate if error is within time window
            if error_event.timestamp < cutoff_time:
                return None

        return decision

    async def _call_error_handlers(self, error_event: ErrorEvent):
        """Call registered error handlers"""
        # Call global handlers
        if "*" in self.error_handlers:
            for handler in self.error_handlers["*"]:
                try:
                    await handler(error_event)
                except Exception as e:
                    self.logger.error(f"Global error handler failed: {e}")

        # Call agent-specific handlers
        if error_event.source_agent in self.error_handlers:
            for handler in self.error_handlers[error_event.source_agent]:
                try:
                    await handler(error_event)
                except Exception as e:
                    self.logger.error(f"Agent error handler failed: {e}")

    async def _monitor_agent_health(self):
        """Background task to monitor agent health"""
        while True:
            try:
                for agent_id, state in self.agent_states.items():
                    # Check quarantine expiry
                    if (
                        state.quarantine_until
                        and datetime.now() > state.quarantine_until
                    ):
                        state.quarantine_until = None
                        state.consecutive_failures = 0
                        state.is_failing = False
                        self.logger.info(
                            f"🟢 Agent {agent_id} released from quarantine"
                        )

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)

    async def _cleanup_old_errors(self):
        """Background task to clean up old error records"""
        while True:
            try:
                cutoff_time = datetime.now() - timedelta(hours=24)

                # Clean up agent error states
                for state in self.agent_states.values():
                    state.current_errors = [
                        e for e in state.current_errors if e.timestamp > cutoff_time
                    ]

                await asyncio.sleep(3600)  # Clean up every hour

            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(1800)

    async def _adjust_propagation_rules(self):
        """Background task to dynamically adjust propagation rules"""
        while True:
            try:
                # Analyze error patterns and adjust rules
                await self._analyze_error_patterns()
                await asyncio.sleep(1800)  # Adjust every 30 minutes

            except Exception as e:
                self.logger.error(f"Rule adjustment error: {e}")
                await asyncio.sleep(1800)

    async def _analyze_error_patterns(self):
        """Analyze error patterns and adjust propagation rules"""
        if len(self.error_history) < 100:
            return  # Not enough data

        # Analyze recent error patterns
        recent_errors = [
            e
            for e in self.error_history
            if (datetime.now() - e.timestamp).total_seconds() < 3600  # Last hour
        ]

        # Count errors by agent and category
        agent_error_counts = defaultdict(int)
        category_counts = defaultdict(int)

        for error in recent_errors:
            agent_error_counts[error.source_agent] += 1
            category_counts[error.category] += 1

        # Adjust rules based on patterns
        if len(recent_errors) > 50:  # High error rate
            # Increase propagation delays to prevent cascading
            for rule in self.propagation_rules:
                if rule.strategy == PropagationStrategy.IMMEDIATE:
                    rule.strategy = PropagationStrategy.DELAYED
                    rule.delay_seconds = max(rule.delay_seconds, 5)

        elif len(recent_errors) < 10:  # Low error rate
            # Reduce delays for faster propagation
            for rule in self.propagation_rules:
                if (
                    rule.strategy == PropagationStrategy.DELAYED
                    and rule.delay_seconds > 2
                ):
                    rule.delay_seconds = max(2, rule.delay_seconds - 1)

    # Public API methods

    def register_error_handler(self, agent_id: str, handler: Callable):
        """Register an error handler for an agent"""
        self.error_handlers[agent_id].append(handler)

    def register_propagation_callback(self, callback: Callable):
        """Register a callback for propagation events"""
        self.propagation_callbacks.append(callback)

    def register_quarantine_callback(self, callback: Callable):
        """Register a callback for quarantine events"""
        self.quarantine_callbacks.append(callback)

    def add_propagation_rule(self, rule: PropagationRule):
        """Add a custom propagation rule"""
        self.propagation_rules.append(rule)

    def get_agent_error_state(self, agent_id: str) -> Optional[AgentErrorState]:
        """Get current error state for an agent"""
        return self.agent_states.get(agent_id)

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and metrics"""
        now = datetime.now()
        recent_errors = [
            e for e in self.error_history if (now - e.timestamp).total_seconds() < 3600
        ]

        return {
            "total_errors_last_hour": len(recent_errors),
            "agents_in_quarantine": len(
                [
                    s
                    for s in self.agent_states.values()
                    if s.quarantine_until and s.quarantine_until > now
                ]
            ),
            "agents_failing": len(
                [s for s in self.agent_states.values() if s.is_failing]
            ),
            "circuit_breakers_open": len(
                [s for s in self.agent_states.values() if s.circuit_breaker_open]
            ),
            "propagation_rules_active": len(self.propagation_rules),
            "active_propagations": len(self.active_propagations),
        }

    async def force_agent_recovery(self, agent_id: str) -> bool:
        """Force recovery of an agent from error state"""
        if agent_id in self.agent_states:
            state = self.agent_states[agent_id]
            state.consecutive_failures = 0
            state.is_failing = False
            state.is_degraded = False
            state.circuit_breaker_open = False
            state.quarantine_until = None
            state.current_errors.clear()

            self.logger.info(f"🔄 Forced recovery for agent {agent_id}")
            return True

        return False

    async def simulate_error_cascade(
        self, initial_agent: str, error_type: str
    ) -> Dict[str, Any]:
        """Simulate error cascade for testing purposes"""
        initial_error = ErrorEvent(
            source_agent=initial_agent,
            error_type=error_type,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.COORDINATION,
            message=f"Simulated cascade error: {error_type}",
            context={"simulation": True},
        )

        result = await self.handle_error_event(initial_error)

        return {
            "initial_error": initial_error.error_id,
            "propagation_successful": result,
            "agents_affected": len(
                [
                    s
                    for s in self.agent_states.values()
                    if s.last_error_time
                    and (datetime.now() - s.last_error_time).total_seconds() < 60
                ]
            ),
        }


# Global instance for easy access
error_propagation_manager = ErrorPropagationManager()
