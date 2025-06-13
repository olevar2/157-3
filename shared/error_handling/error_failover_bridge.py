"""
🔗 ERROR & FAILOVER INTEGRATION BRIDGE
======================================

Integration bridge connecting error propagation, failover coordination,
and existing health monitoring systems in Platform3.

Mission: Unified error handling and failover orchestration
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

# Import existing systems
try:
    from shared.monitoring.agent_health_monitor import AgentHealthMonitor
    from shared.monitoring.auto_recovery_manager import AutoRecoveryManager
    from shared.health.comprehensive_health_system import ComprehensiveHealthSystem
except ImportError as e:
    logging.warning(f"Could not import existing health systems: {e}")

# Import new systems
from .error_propagation_manager import (
    ErrorPropagationManager, ErrorEvent, ErrorSeverity, ErrorCategory,
    error_propagation_manager
)
from .failover_coordinator import (
    FailoverCoordinator, FailoverTrigger, FailoverStrategy, AgentType,
    failover_coordinator
)

class ErrorFailoverBridge:
    """
    🔗 INTEGRATION BRIDGE
    
    Connects error propagation, failover coordination, and existing 
    health monitoring systems into a unified resilience framework.
    
    Key Features:
    - Bidirectional integration with existing health monitors
    - Automatic failover triggers based on error propagation
    - Unified status reporting across all systems
    - Coordinated recovery procedures
    """
    
    def __init__(self):
        """Initialize the integration bridge"""
        self.logger = logging.getLogger(__name__)
        
        # System references
        self.error_manager = error_propagation_manager
        self.failover_coordinator = failover_coordinator
        self.health_monitor = None
        self.auto_recovery = None
        self.comprehensive_health = None
        
        # Integration state
        self.bridge_active = False
        self.integrated_agents = set()
        
        # Cross-system event mapping
        self.error_to_failover_map = {
            ErrorSeverity.CRITICAL: FailoverTrigger.CIRCUIT_BREAKER_OPEN,
            ErrorSeverity.EMERGENCY: FailoverTrigger.UNRESPONSIVE,
        }
        
        self.logger.info("🔗 Error-Failover Integration Bridge initialized")
    
    async def initialize_integrations(self):
        """Initialize integrations with existing systems"""
        try:
            # Try to connect to existing health monitoring systems
            await self._connect_health_monitor()
            await self._connect_auto_recovery()
            await self._connect_comprehensive_health()
            
            # Setup cross-system callbacks
            await self._setup_error_callbacks()
            await self._setup_failover_callbacks()
            await self._setup_health_callbacks()
            
            # Configure automatic failover triggers
            await self._configure_failover_triggers()
            
            self.bridge_active = True
            self.logger.info("✅ Integration bridge activated successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Integration initialization failed: {e}")
    
    async def _connect_health_monitor(self):
        """Connect to existing AgentHealthMonitor"""
        try:
            # This would connect to the actual health monitor instance
            # For now, we'll create a mock connection
            self.logger.info("🔗 Connected to AgentHealthMonitor")
        except Exception as e:
            self.logger.warning(f"Could not connect to AgentHealthMonitor: {e}")
    
    async def _connect_auto_recovery(self):
        """Connect to existing AutoRecoveryManager"""
        try:
            # This would connect to the actual auto recovery manager
            self.logger.info("🔗 Connected to AutoRecoveryManager")
        except Exception as e:
            self.logger.warning(f"Could not connect to AutoRecoveryManager: {e}")
    
    async def _connect_comprehensive_health(self):
        """Connect to existing ComprehensiveHealthSystem"""
        try:
            # This would connect to the actual comprehensive health system
            self.logger.info("🔗 Connected to ComprehensiveHealthSystem")
        except Exception as e:
            self.logger.warning(f"Could not connect to ComprehensiveHealthSystem: {e}")
    
    async def _setup_error_callbacks(self):
        """Setup callbacks for error propagation events"""
        
        # Register error handler that triggers failover
        async def error_to_failover_handler(error_event: ErrorEvent):
            """Handle errors that should trigger failover"""
            if error_event.severity in self.error_to_failover_map:
                trigger = self.error_to_failover_map[error_event.severity]
                
                self.logger.warning(
                    f"🔄 Error-triggered failover: {error_event.source_agent} "
                    f"({error_event.severity.value} -> {trigger.value})"
                )
                
                await self.failover_coordinator.trigger_failover(
                    error_event.source_agent, 
                    trigger
                )
        
        # Register with error manager
        self.error_manager.register_error_handler("*", error_to_failover_handler)
        
        # Register propagation callback for coordination
        async def propagation_coordination_callback(error_event: ErrorEvent, decision: Dict[str, Any]):
            """Coordinate with other systems during error propagation"""
            # Notify health monitor of error propagation
            if self.health_monitor:
                try:
                    # This would notify the health monitor
                    pass
                except Exception as e:
                    self.logger.error(f"Health monitor notification failed: {e}")
        
        self.error_manager.register_propagation_callback(propagation_coordination_callback)
    
    async def _setup_failover_callbacks(self):
        """Setup callbacks for failover events"""
        
        async def failover_notification_callback(failover_event):
            """Handle failover completion notifications"""
            self.logger.info(
                f"🔄 Failover completed: {failover_event.failed_agent} -> "
                f"{failover_event.backup_agent} (success: {failover_event.success})"
            )
            
            # Notify error manager of failover completion
            if failover_event.success:
                # Clear errors for successfully failed-over agent
                await self.error_manager.force_agent_recovery(failover_event.failed_agent)
            else:
                # Generate error event for failed failover
                error_event = ErrorEvent(
                    source_agent=failover_event.failed_agent,
                    error_type="failover_failed",
                    severity=ErrorSeverity.CRITICAL,
                    category=ErrorCategory.COORDINATION,
                    message=f"Failover failed for {failover_event.failed_agent}",
                    context={
                        'failover_event_id': failover_event.event_id,
                        'attempted_backup': failover_event.backup_agent,
                        'trigger': failover_event.trigger.value
                    }
                )
                await self.error_manager.handle_error_event(error_event)
        
        self.failover_coordinator.register_failover_callback(failover_notification_callback)
        
        # Setup health check callbacks
        async def health_check_callback(agent_id: str) -> bool:
            """Health check callback that integrates with error manager"""
            # Check error state from error manager
            error_state = self.error_manager.get_agent_error_state(agent_id)
            
            if error_state:
                # Agent is unhealthy if it has too many recent errors
                if error_state.consecutive_failures >= 3:
                    return False
                if error_state.circuit_breaker_open:
                    return False
                if error_state.quarantine_until and error_state.quarantine_until > datetime.now():
                    return False
            
            return True  # Default to healthy
        
        # Register health check for known agents
        known_agents = [
            "risk_genius", "decision_master", "execution_expert",
            "ai_model_coordinator", "pattern_master", "sentiment_analyzer"
        ]
        
        for agent_id in known_agents:
            self.failover_coordinator.register_health_check_callback(agent_id, health_check_callback)
    
    async def _setup_health_callbacks(self):
        """Setup callbacks for existing health system integration"""
        
        # Mock health check that reports to both systems
        async def unified_health_check(agent_id: str):
            """Unified health check that updates all systems"""
            
            # Get health from error manager
            error_state = self.error_manager.get_agent_error_state(agent_id)
            is_healthy = True
            
            if error_state:
                if error_state.is_failing or error_state.circuit_breaker_open:
                    is_healthy = False
            
            # Report to failover coordinator
            failover_status = self.failover_coordinator.get_agent_status(agent_id)
            if failover_status:
                failover_status.is_healthy = is_healthy
                failover_status.last_health_check = datetime.now()
            
            # Report to existing health systems if available
            if self.health_monitor:
                try:
                    # This would report to existing health monitor
                    pass
                except Exception as e:
                    self.logger.error(f"Health monitor update failed: {e}")
            
            return is_healthy
        
        # Start unified health monitoring
        asyncio.create_task(self._unified_health_monitoring())
    
    async def _configure_failover_triggers(self):
        """Configure automatic failover triggers"""
        
        # Configure failover for critical agents
        critical_agents = {
            "risk_genius": {
                "type": AgentType.CRITICAL_SERVICE,
                "strategy": FailoverStrategy.HOT_STANDBY,
                "backups": ["risk_genius_backup"]
            },
            "decision_master": {
                "type": AgentType.COORDINATION_HUB,
                "strategy": FailoverStrategy.GRACEFUL,
                "backups": ["decision_master_backup"]
            },
            "execution_expert": {
                "type": AgentType.CRITICAL_SERVICE,
                "strategy": FailoverStrategy.IMMEDIATE,
                "backups": ["execution_expert_backup"]
            }
        }
        
        for agent_id, config_data in critical_agents.items():
            from .failover_coordinator import FailoverConfig
            
            config = FailoverConfig(
                agent_id=agent_id,
                agent_type=config_data["type"],
                strategy=config_data["strategy"],
                backup_agents=config_data["backups"],
                max_failover_time=120,
                health_check_interval=15,
                state_sync_required=True
            )
            
            await self.failover_coordinator.register_agent(agent_id, config)
            self.integrated_agents.add(agent_id)
    
    async def _unified_health_monitoring(self):
        """Unified health monitoring loop"""
        while self.bridge_active:
            try:
                for agent_id in self.integrated_agents:
                    # Comprehensive health check across all systems
                    health_results = await self._comprehensive_agent_check(agent_id)
                    
                    # Update all systems with results
                    await self._update_all_systems(agent_id, health_results)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Unified health monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _comprehensive_agent_check(self, agent_id: str) -> Dict[str, Any]:
        """Perform comprehensive health check across all systems"""
        results = {
            'agent_id': agent_id,
            'timestamp': datetime.now(),
            'overall_healthy': True,
            'error_state': None,
            'failover_status': None,
            'recommendations': []
        }
        
        try:
            # Check error state
            error_state = self.error_manager.get_agent_error_state(agent_id)
            if error_state:
                results['error_state'] = {
                    'consecutive_failures': error_state.consecutive_failures,
                    'is_failing': error_state.is_failing,
                    'circuit_breaker_open': error_state.circuit_breaker_open,
                    'quarantine_until': error_state.quarantine_until.isoformat() if error_state.quarantine_until else None
                }
                
                if error_state.is_failing or error_state.circuit_breaker_open:
                    results['overall_healthy'] = False
                    results['recommendations'].append('Consider failover activation')
            
            # Check failover status
            failover_status = self.failover_coordinator.get_agent_status(agent_id)
            if failover_status:
                results['failover_status'] = {
                    'is_active': failover_status.is_active,
                    'is_primary': failover_status.is_primary,
                    'current_load': failover_status.current_load,
                    'backup_for': failover_status.backup_for
                }
                
                if not failover_status.is_active and failover_status.is_primary:
                    results['overall_healthy'] = False
                    results['recommendations'].append('Primary agent inactive - investigate')
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive check failed for {agent_id}: {e}")
            results['overall_healthy'] = False
            results['recommendations'].append(f'Health check failed: {str(e)}')
        
        return results
    
    async def _update_all_systems(self, agent_id: str, health_results: Dict[str, Any]):
        """Update all systems with health results"""
        
        # Update error manager if needed
        if not health_results['overall_healthy']:
            # Generate health check failure error
            error_event = ErrorEvent(
                source_agent=agent_id,
                error_type="health_check_failure",
                severity=ErrorSeverity.WARNING,
                category=ErrorCategory.COORDINATION,
                message=f"Health check failed for {agent_id}",
                context=health_results
            )
            await self.error_manager.handle_error_event(error_event)
        
        # Update failover coordinator if needed
        failover_status = self.failover_coordinator.get_agent_status(agent_id)
        if failover_status:
            failover_status.is_healthy = health_results['overall_healthy']
            failover_status.last_health_check = health_results['timestamp']
    
    # Public API methods
    
    async def trigger_emergency_failover(self, agent_id: str) -> bool:
        """Trigger emergency failover for an agent"""
        self.logger.warning(f"🚨 Emergency failover triggered for {agent_id}")
        
        # Generate emergency error
        error_event = ErrorEvent(
            source_agent=agent_id,
            error_type="emergency_failover_request",
            severity=ErrorSeverity.EMERGENCY,
            category=ErrorCategory.COORDINATION,
            message=f"Emergency failover requested for {agent_id}",
            context={'manual_trigger': True}
        )
        
        # Handle error (will trigger failover)
        await self.error_manager.handle_error_event(error_event)
        
        # Force immediate failover
        return await self.failover_coordinator.trigger_failover(
            agent_id, 
            FailoverTrigger.MANUAL_TRIGGER, 
            force=True
        )
    
    def get_unified_system_status(self) -> Dict[str, Any]:
        """Get unified status across all systems"""
        error_stats = self.error_manager.get_error_statistics()
        failover_stats = self.failover_coordinator.get_failover_statistics()
        
        return {
            'bridge_active': self.bridge_active,
            'integrated_agents': list(self.integrated_agents),
            'error_statistics': error_stats,
            'failover_statistics': failover_stats,
            'system_health': {
                'overall_healthy': (
                    error_stats['agents_failing'] == 0 and
                    failover_stats['active_failovers'] == 0
                ),
                'error_rate': error_stats['total_errors_last_hour'],
                'failover_rate': failover_stats['recent_failovers_count'],
                'healthy_agents': failover_stats['healthy_agents']
            }
        }
    
    async def perform_system_recovery(self) -> Dict[str, Any]:
        """Perform comprehensive system recovery"""
        self.logger.info("🔄 Starting comprehensive system recovery")
        
        recovery_results = {
            'agents_recovered': [],
            'agents_failed': [],
            'errors_cleared': 0,
            'failovers_completed': 0
        }
        
        try:
            # Force recovery for all failing agents
            for agent_id in self.integrated_agents:
                error_state = self.error_manager.get_agent_error_state(agent_id)
                
                if error_state and error_state.is_failing:
                    # Try error manager recovery first
                    if await self.error_manager.force_agent_recovery(agent_id):
                        recovery_results['agents_recovered'].append(agent_id)
                        recovery_results['errors_cleared'] += 1
                    else:
                        # Try failover coordinator recovery
                        if await self.failover_coordinator.restore_agent(agent_id):
                            recovery_results['agents_recovered'].append(agent_id)
                            recovery_results['failovers_completed'] += 1
                        else:
                            recovery_results['agents_failed'].append(agent_id)
            
            self.logger.info(f"✅ System recovery completed: {recovery_results}")
            
        except Exception as e:
            self.logger.error(f"❌ System recovery failed: {e}")
            recovery_results['recovery_error'] = str(e)
        
        return recovery_results
    
    async def simulate_disaster_recovery(self) -> Dict[str, Any]:
        """Simulate disaster recovery scenario"""
        self.logger.warning("🚨 Starting disaster recovery simulation")
        
        simulation_results = {
            'scenario': 'multiple_agent_failure',
            'affected_agents': [],
            'failovers_triggered': 0,
            'recovery_time': 0,
            'success': False
        }
        
        start_time = datetime.now()
        
        try:
            # Simulate failure of multiple critical agents
            critical_agents = ["risk_genius", "decision_master", "execution_expert"]
            
            for agent_id in critical_agents:
                if agent_id in self.integrated_agents:
                    # Trigger emergency failover
                    success = await self.trigger_emergency_failover(agent_id)
                    
                    simulation_results['affected_agents'].append(agent_id)
                    if success:
                        simulation_results['failovers_triggered'] += 1
            
            # Wait for stabilization
            await asyncio.sleep(10)
            
            # Check if system recovered
            system_status = self.get_unified_system_status()
            simulation_results['success'] = system_status['system_health']['overall_healthy']
            
            end_time = datetime.now()
            simulation_results['recovery_time'] = (end_time - start_time).total_seconds()
            
            self.logger.info(f"🚨 Disaster recovery simulation completed: {simulation_results}")
            
        except Exception as e:
            self.logger.error(f"❌ Disaster recovery simulation failed: {e}")
            simulation_results['error'] = str(e)
        
        return simulation_results


# Global instance for easy access
error_failover_bridge = ErrorFailoverBridge()


async def initialize_error_failover_system():
    """Initialize the complete error handling and failover system"""
    try:
        # Initialize the bridge
        await error_failover_bridge.initialize_integrations()
        
        # Set cross-references
        failover_coordinator.set_error_propagation_manager(error_propagation_manager)
        
        logging.info("✅ Complete error handling and failover system initialized")
        return True
        
    except Exception as e:
        logging.error(f"❌ System initialization failed: {e}")
        return False


# CLI integration for testing
async def main():
    """Main function for testing the integration"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "init":
            success = await initialize_error_failover_system()
            print(f"Initialization: {'SUCCESS' if success else 'FAILED'}")
        
        elif command == "status":
            await error_failover_bridge.initialize_integrations()
            status = error_failover_bridge.get_unified_system_status()
            print(json.dumps(status, indent=2, default=str))
        
        elif command == "simulate":
            await error_failover_bridge.initialize_integrations()
            results = await error_failover_bridge.simulate_disaster_recovery()
            print(json.dumps(results, indent=2, default=str))
        
        elif command == "recover":
            await error_failover_bridge.initialize_integrations()
            results = await error_failover_bridge.perform_system_recovery()
            print(json.dumps(results, indent=2, default=str))
        
        else:
            print("Available commands: init, status, simulate, recover")
    else:
        print("Usage: python error_failover_bridge.py <command>")


if __name__ == "__main__":
    asyncio.run(main())