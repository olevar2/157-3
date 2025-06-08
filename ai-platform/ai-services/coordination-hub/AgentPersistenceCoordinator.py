#!/usr/bin/env python3
"""
Platform3 Agent Persistence Coordinator
Cross-agent persistence coordination and transaction management
"""

import os
import sys
import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import uuid

# Add shared modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'shared'))
from shared.persistence.agent_state_manager import (
    AgentStateManager, StateType, StateSnapshot, create_agent_state_manager
)
from shared.platform3_logging.platform3_logger import get_logger
from shared.error_handling.platform3_error_system import BaseService, ServiceError


class TransactionType(Enum):
    """Types of cross-agent transactions"""
    COORDINATED_SAVE = "coordinated_save"
    CROSS_AGENT_SYNC = "cross_agent_sync"
    DEPENDENCY_UPDATE = "dependency_update"
    DISTRIBUTED_SNAPSHOT = "distributed_snapshot"
    RECOVERY_COORDINATION = "recovery_coordination"


class TransactionStatus(Enum):
    """Status of cross-agent transactions"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class AgentTransaction:
    """Cross-agent transaction definition"""
    transaction_id: str
    transaction_type: TransactionType
    participating_agents: List[str]
    coordinator_agent: str
    start_time: datetime
    timeout_seconds: int = 300
    status: TransactionStatus = TransactionStatus.PENDING
    operations: List[Dict[str, Any]] = field(default_factory=list)
    rollback_data: Dict[str, Any] = field(default_factory=dict)
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class CrossAgentSyncPlan:
    """Plan for synchronizing states across multiple agents"""
    sync_id: str
    primary_agent: str
    dependent_agents: List[str]
    state_types: List[StateType]
    sync_direction: str  # "primary_to_dependents", "dependents_to_primary", "bidirectional"
    conflict_resolution: str  # "primary_wins", "latest_wins", "manual"
    created_at: datetime


class AgentPersistenceCoordinator(BaseService):
    """
    Coordinates persistence operations across multiple agents
    Manages cross-agent transactions and state synchronization
    """
    
    def __init__(self, state_manager: Optional[AgentStateManager] = None):
        super().__init__(service_name="agent_persistence_coordinator")
        
        self.logger = get_logger("platform3.persistence_coordinator")
        self.state_manager = state_manager
        
        # Transaction management
        self.active_transactions: Dict[str, AgentTransaction] = {}
        self.transaction_history: List[AgentTransaction] = []
        self.max_transaction_history = 1000
        
        # Synchronization management
        self.active_syncs: Dict[str, CrossAgentSyncPlan] = {}
        self.sync_locks: Set[str] = set()  # Agents currently locked for sync
        
        # Agent relationships
        self.agent_dependencies = {
            "decision_master": ["risk_genius", "pattern_master", "execution_expert"],
            "ai_model_coordinator": ["all"],  # Coordinates all agents
            "execution_expert": ["risk_genius", "pattern_master"],
            "portfolio_genius": ["risk_genius", "market_genius"],
            "strategy_architect": ["risk_genius", "pattern_master"],
            "trading_genius": ["execution_expert", "risk_genius"],
            "pattern_master": ["market_genius"],
            "market_genius": [],
            "risk_genius": []
        }
        
        # Performance metrics
        self.transaction_metrics = {
            'total_transactions': 0,
            'successful_transactions': 0,
            'failed_transactions': 0,
            'rolled_back_transactions': 0,
            'avg_transaction_duration': 0.0,
            'sync_operations': 0,
            'successful_syncs': 0
        }
        
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the persistence coordinator"""
        try:
            # Initialize state manager if not provided
            if not self.state_manager:
                self.state_manager = await create_agent_state_manager()
            
            # Start background tasks
            asyncio.create_task(self._transaction_monitor())
            asyncio.create_task(self._periodic_sync_check())
            
            self.is_initialized = True
            self.logger.info("Agent Persistence Coordinator initialized successfully")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize Agent Persistence Coordinator: {str(e)}"
            self.logger.error(error_msg)
            self.emit_error(ServiceError(
                message=error_msg,
                error_code="COORDINATOR_INIT_ERROR",
                service_context="agent_persistence_coordinator"
            ))
            return False
    
    async def begin_cross_agent_transaction(self,
                                          transaction_type: TransactionType,
                                          participating_agents: List[str],
                                          coordinator_agent: str,
                                          timeout_seconds: int = 300) -> str:
        """Begin a cross-agent transaction"""
        try:
            transaction_id = str(uuid.uuid4())
            
            # Check for conflicts with existing transactions
            for agent in participating_agents:
                if any(agent in t.participating_agents for t in self.active_transactions.values()):
                    raise ServiceError(
                        f"Agent {agent} is already participating in another transaction",
                        "TRANSACTION_CONFLICT",
                        "agent_persistence_coordinator"
                    )
            
            transaction = AgentTransaction(
                transaction_id=transaction_id,
                transaction_type=transaction_type,
                participating_agents=participating_agents,
                coordinator_agent=coordinator_agent,
                start_time=datetime.now(),
                timeout_seconds=timeout_seconds
            )
            
            self.active_transactions[transaction_id] = transaction
            self.transaction_metrics['total_transactions'] += 1
            
            self.logger.info(f"Started cross-agent transaction {transaction_id}", meta={
                "transaction_type": transaction_type.value,
                "participating_agents": participating_agents,
                "coordinator": coordinator_agent
            })
            
            return transaction_id
            
        except Exception as e:
            error_msg = f"Failed to begin cross-agent transaction: {str(e)}"
            self.logger.error(error_msg)
            raise ServiceError(error_msg, "TRANSACTION_START_ERROR", "agent_persistence_coordinator")
    
    async def add_transaction_operation(self,
                                      transaction_id: str,
                                      agent_name: str,
                                      operation_type: str,
                                      operation_data: Dict[str, Any]) -> bool:
        """Add an operation to a transaction"""
        try:
            if transaction_id not in self.active_transactions:
                raise ValueError(f"Transaction {transaction_id} not found")
            
            transaction = self.active_transactions[transaction_id]
            
            if transaction.status != TransactionStatus.PENDING:
                raise ValueError(f"Cannot add operations to transaction in status {transaction.status}")
            
            operation = {
                'agent_name': agent_name,
                'operation_type': operation_type,
                'operation_data': operation_data,
                'timestamp': datetime.now().isoformat(),
                'operation_id': str(uuid.uuid4())
            }
            
            transaction.operations.append(operation)
            
            self.logger.debug(f"Added operation to transaction {transaction_id}", meta={
                "agent_name": agent_name,
                "operation_type": operation_type
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add transaction operation: {e}")
            return False
    
    async def commit_transaction(self, transaction_id: str) -> bool:
        """Commit a cross-agent transaction"""
        try:
            if transaction_id not in self.active_transactions:
                raise ValueError(f"Transaction {transaction_id} not found")
            
            transaction = self.active_transactions[transaction_id]
            transaction.status = TransactionStatus.IN_PROGRESS
            
            # Execute all operations in the transaction
            executed_operations = []
            rollback_needed = False
            
            for operation in transaction.operations:
                try:
                    success = await self._execute_transaction_operation(operation)
                    if success:
                        executed_operations.append(operation)
                    else:
                        rollback_needed = True
                        break
                        
                except Exception as e:
                    self.logger.error(f"Operation failed during transaction commit: {e}")
                    rollback_needed = True
                    break
            
            if rollback_needed:
                # Rollback executed operations
                await self._rollback_transaction_operations(executed_operations)
                transaction.status = TransactionStatus.ROLLED_BACK
                transaction.error_message = "One or more operations failed"
                self.transaction_metrics['rolled_back_transactions'] += 1
                return False
            else:
                # All operations successful
                transaction.status = TransactionStatus.COMMITTED
                transaction.end_time = datetime.now()
                self.transaction_metrics['successful_transactions'] += 1
                
                # Update metrics
                duration = (transaction.end_time - transaction.start_time).total_seconds()
                self.transaction_metrics['avg_transaction_duration'] = (
                    (self.transaction_metrics['avg_transaction_duration'] * 
                     (self.transaction_metrics['successful_transactions'] - 1) + duration)
                    / self.transaction_metrics['successful_transactions']
                )
                
                self.logger.info(f"Successfully committed transaction {transaction_id}", meta={
                    "duration_seconds": duration,
                    "operations_count": len(transaction.operations)
                })
                
                return True
            
        except Exception as e:
            transaction = self.active_transactions.get(transaction_id)
            if transaction:
                transaction.status = TransactionStatus.FAILED
                transaction.error_message = str(e)
                transaction.end_time = datetime.now()
            
            self.transaction_metrics['failed_transactions'] += 1
            error_msg = f"Failed to commit transaction: {str(e)}"
            self.logger.error(error_msg)
            raise ServiceError(error_msg, "TRANSACTION_COMMIT_ERROR", "agent_persistence_coordinator")
        
        finally:
            # Move transaction to history and remove from active
            if transaction_id in self.active_transactions:
                transaction = self.active_transactions.pop(transaction_id)
                self.transaction_history.append(transaction)
                
                # Maintain history size limit
                if len(self.transaction_history) > self.max_transaction_history:
                    self.transaction_history = self.transaction_history[-self.max_transaction_history:]
    
    async def _execute_transaction_operation(self, operation: Dict[str, Any]) -> bool:
        """Execute a single transaction operation"""
        try:
            agent_name = operation['agent_name']
            operation_type = operation['operation_type']
            operation_data = operation['operation_data']
            
            if operation_type == "save_state":
                state_id = await self.state_manager.save_agent_state(
                    agent_name=agent_name,
                    state_type=StateType(operation_data['state_type']),
                    state_data=operation_data['state_data'],
                    version=operation_data.get('version', '1.0'),
                    tags=operation_data.get('tags', [])
                )
                operation['result'] = {'state_id': state_id}
                return True
                
            elif operation_type == "create_snapshot":
                snapshot_id = await self.state_manager.create_agent_snapshot(agent_name)
                operation['result'] = {'snapshot_id': snapshot_id}
                return True
                
            elif operation_type == "restore_from_snapshot":
                success = await self.state_manager.restore_agent_from_snapshot(
                    agent_name, operation_data['snapshot_id']
                )
                operation['result'] = {'restored': success}
                return success
                
            else:
                self.logger.warning(f"Unknown operation type: {operation_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to execute operation: {e}")
            operation['error'] = str(e)
            return False
    
    async def _rollback_transaction_operations(self, operations: List[Dict[str, Any]]):
        """Rollback executed transaction operations"""
        for operation in reversed(operations):
            try:
                operation_type = operation['operation_type']
                
                if operation_type == "save_state" and 'result' in operation:
                    # Delete the saved state
                    # Note: This would require additional implementation in AgentStateManager
                    pass
                elif operation_type == "create_snapshot" and 'result' in operation:
                    # Delete the created snapshot
                    # Note: This would require additional implementation in AgentStateManager
                    pass
                    
            except Exception as e:
                self.logger.warning(f"Failed to rollback operation: {e}")
    
    async def synchronize_agent_states(self,
                                     primary_agent: str,
                                     dependent_agents: List[str],
                                     state_types: List[StateType],
                                     sync_direction: str = "primary_to_dependents") -> str:
        """Synchronize states between agents"""
        try:
            sync_id = str(uuid.uuid4())
            
            # Check for agent locks
            all_agents = [primary_agent] + dependent_agents
            locked_agents = [agent for agent in all_agents if agent in self.sync_locks]
            if locked_agents:
                raise ServiceError(
                    f"Agents are locked for sync: {locked_agents}",
                    "SYNC_AGENT_LOCKED",
                    "agent_persistence_coordinator"
                )
            
            # Lock agents for synchronization
            for agent in all_agents:
                self.sync_locks.add(agent)
            
            try:
                sync_plan = CrossAgentSyncPlan(
                    sync_id=sync_id,
                    primary_agent=primary_agent,
                    dependent_agents=dependent_agents,
                    state_types=state_types,
                    sync_direction=sync_direction,
                    conflict_resolution="primary_wins",
                    created_at=datetime.now()
                )
                
                self.active_syncs[sync_id] = sync_plan
                
                # Execute synchronization
                if sync_direction == "primary_to_dependents":
                    await self._sync_primary_to_dependents(sync_plan)
                elif sync_direction == "dependents_to_primary":
                    await self._sync_dependents_to_primary(sync_plan)
                elif sync_direction == "bidirectional":
                    await self._sync_bidirectional(sync_plan)
                
                self.transaction_metrics['successful_syncs'] += 1
                
                self.logger.info(f"Successfully synchronized agent states", meta={
                    "sync_id": sync_id,
                    "primary_agent": primary_agent,
                    "dependent_agents": dependent_agents,
                    "sync_direction": sync_direction
                })
                
                return sync_id
                
            finally:
                # Release agent locks
                for agent in all_agents:
                    self.sync_locks.discard(agent)
                
                # Remove from active syncs
                self.active_syncs.pop(sync_id, None)
            
        except Exception as e:
            error_msg = f"Failed to synchronize agent states: {str(e)}"
            self.logger.error(error_msg)
            raise ServiceError(error_msg, "SYNC_ERROR", "agent_persistence_coordinator")
    
    async def _sync_primary_to_dependents(self, sync_plan: CrossAgentSyncPlan):
        """Sync states from primary agent to dependent agents"""
        primary_states = {}
        
        # Load states from primary agent
        for state_type in sync_plan.state_types:
            try:
                result = await self.state_manager.load_agent_state(
                    sync_plan.primary_agent, state_type
                )
                if result:
                    state_data, metadata = result
                    primary_states[state_type] = state_data
            except Exception as e:
                self.logger.warning(f"Failed to load {state_type} from primary: {e}")
        
        # Apply states to dependent agents
        for dependent_agent in sync_plan.dependent_agents:
            for state_type, state_data in primary_states.items():
                try:
                    await self.state_manager.save_agent_state(
                        agent_name=dependent_agent,
                        state_type=state_type,
                        state_data=state_data,
                        version="synced",
                        tags=["synced", f"from_{sync_plan.primary_agent}", f"sync_{sync_plan.sync_id}"]
                    )
                except Exception as e:
                    self.logger.error(f"Failed to sync {state_type} to {dependent_agent}: {e}")
    
    async def _sync_dependents_to_primary(self, sync_plan: CrossAgentSyncPlan):
        """Sync states from dependent agents to primary agent"""
        # Collect states from all dependents
        dependent_states = {}
        
        for dependent_agent in sync_plan.dependent_agents:
            for state_type in sync_plan.state_types:
                try:
                    result = await self.state_manager.load_agent_state(
                        dependent_agent, state_type
                    )
                    if result:
                        state_data, metadata = result
                        if state_type not in dependent_states:
                            dependent_states[state_type] = []
                        dependent_states[state_type].append({
                            'agent': dependent_agent,
                            'data': state_data,
                            'timestamp': metadata.timestamp
                        })
                except Exception as e:
                    self.logger.warning(f"Failed to load {state_type} from {dependent_agent}: {e}")
        
        # Merge and apply to primary agent
        for state_type, state_list in dependent_states.items():
            if state_list:
                # Use latest state (could implement more sophisticated merging)
                latest_state = max(state_list, key=lambda x: x['timestamp'])
                
                try:
                    await self.state_manager.save_agent_state(
                        agent_name=sync_plan.primary_agent,
                        state_type=state_type,
                        state_data=latest_state['data'],
                        version="synced",
                        tags=["synced", f"from_dependents", f"sync_{sync_plan.sync_id}"]
                    )
                except Exception as e:
                    self.logger.error(f"Failed to sync {state_type} to primary: {e}")
    
    async def _sync_bidirectional(self, sync_plan: CrossAgentSyncPlan):
        """Bidirectional synchronization between agents"""
        # For bidirectional sync, we implement a conflict resolution strategy
        # This is a simplified implementation - production would need more sophisticated conflict resolution
        
        all_agents = [sync_plan.primary_agent] + sync_plan.dependent_agents
        agent_states = {}
        
        # Collect all states
        for agent in all_agents:
            agent_states[agent] = {}
            for state_type in sync_plan.state_types:
                try:
                    result = await self.state_manager.load_agent_state(agent, state_type)
                    if result:
                        state_data, metadata = result
                        agent_states[agent][state_type] = {
                            'data': state_data,
                            'timestamp': metadata.timestamp
                        }
                except Exception as e:
                    self.logger.warning(f"Failed to load {state_type} from {agent}: {e}")
        
        # Resolve conflicts and sync latest states
        for state_type in sync_plan.state_types:
            # Find latest state across all agents
            latest_state = None
            latest_timestamp = None
            source_agent = None
            
            for agent, states in agent_states.items():
                if state_type in states:
                    timestamp = states[state_type]['timestamp']
                    if latest_timestamp is None or timestamp > latest_timestamp:
                        latest_timestamp = timestamp
                        latest_state = states[state_type]['data']
                        source_agent = agent
            
            # Apply latest state to all agents
            if latest_state is not None:
                for agent in all_agents:
                    if agent != source_agent:  # Don't sync back to source
                        try:
                            await self.state_manager.save_agent_state(
                                agent_name=agent,
                                state_type=state_type,
                                state_data=latest_state,
                                version="synced",
                                tags=["synced", f"from_{source_agent}", f"sync_{sync_plan.sync_id}"]
                            )
                        except Exception as e:
                            self.logger.error(f"Failed to sync {state_type} to {agent}: {e}")
    
    async def create_distributed_snapshot(self, agent_names: List[str]) -> str:
        """Create coordinated snapshots across multiple agents"""
        try:
            transaction_id = await self.begin_cross_agent_transaction(
                transaction_type=TransactionType.DISTRIBUTED_SNAPSHOT,
                participating_agents=agent_names,
                coordinator_agent="ai_model_coordinator"
            )
            
            # Add snapshot operations for each agent
            for agent_name in agent_names:
                await self.add_transaction_operation(
                    transaction_id=transaction_id,
                    agent_name=agent_name,
                    operation_type="create_snapshot",
                    operation_data={}
                )
            
            # Commit the transaction
            success = await self.commit_transaction(transaction_id)
            
            if success:
                self.logger.info(f"Created distributed snapshot for agents: {agent_names}")
                return transaction_id
            else:
                raise ServiceError(
                    "Failed to create distributed snapshot",
                    "SNAPSHOT_FAILED",
                    "agent_persistence_coordinator"
                )
                
        except Exception as e:
            error_msg = f"Failed to create distributed snapshot: {str(e)}"
            self.logger.error(error_msg)
            raise ServiceError(error_msg, "DISTRIBUTED_SNAPSHOT_ERROR", "agent_persistence_coordinator")
    
    async def coordinate_recovery(self, failed_agents: List[str], recovery_strategy: str = "latest_snapshot") -> bool:
        """Coordinate recovery of multiple failed agents"""
        try:
            transaction_id = await self.begin_cross_agent_transaction(
                transaction_type=TransactionType.RECOVERY_COORDINATION,
                participating_agents=failed_agents,
                coordinator_agent="ai_model_coordinator"
            )
            
            recovery_success = True
            
            for agent_name in failed_agents:
                try:
                    if recovery_strategy == "latest_snapshot":
                        # Find latest snapshot for agent
                        snapshots = await self.state_manager.get_agent_snapshots(agent_name, limit=1)
                        if snapshots:
                            latest_snapshot = snapshots[0]
                            await self.add_transaction_operation(
                                transaction_id=transaction_id,
                                agent_name=agent_name,
                                operation_type="restore_from_snapshot",
                                operation_data={'snapshot_id': latest_snapshot['snapshot_id']}
                            )
                        else:
                            self.logger.warning(f"No snapshots found for agent {agent_name}")
                            recovery_success = False
                    
                except Exception as e:
                    self.logger.error(f"Failed to prepare recovery for {agent_name}: {e}")
                    recovery_success = False
            
            if recovery_success:
                # Commit recovery transaction
                success = await self.commit_transaction(transaction_id)
                
                if success:
                    # Synchronize recovered agents with their dependencies
                    await self._post_recovery_sync(failed_agents)
                
                return success
            else:
                return False
                
        except Exception as e:
            error_msg = f"Failed to coordinate recovery: {str(e)}"
            self.logger.error(error_msg)
            return False
    
    async def _post_recovery_sync(self, recovered_agents: List[str]):
        """Synchronize recovered agents with their dependencies"""
        for agent in recovered_agents:
            if agent in self.agent_dependencies:
                dependencies = self.agent_dependencies[agent]
                if dependencies and dependencies != ["all"]:
                    try:
                        await self.synchronize_agent_states(
                            primary_agent=agent,
                            dependent_agents=dependencies,
                            state_types=[StateType.DEPENDENCY_STATE, StateType.COMMUNICATION_STATE],
                            sync_direction="bidirectional"
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed post-recovery sync for {agent}: {e}")
    
    async def get_transaction_status(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a transaction"""
        # Check active transactions
        if transaction_id in self.active_transactions:
            transaction = self.active_transactions[transaction_id]
            return asdict(transaction)
        
        # Check transaction history
        for transaction in self.transaction_history:
            if transaction.transaction_id == transaction_id:
                return asdict(transaction)
        
        return None
    
    async def get_coordination_metrics(self) -> Dict[str, Any]:
        """Get coordination performance metrics"""
        return {
            'transaction_metrics': self.transaction_metrics,
            'active_transactions': len(self.active_transactions),
            'active_syncs': len(self.active_syncs),
            'locked_agents': list(self.sync_locks),
            'agent_dependencies': self.agent_dependencies
        }
    
    async def _transaction_monitor(self):
        """Monitor transactions for timeouts"""
        while True:
            try:
                current_time = datetime.now()
                timed_out_transactions = []
                
                for transaction_id, transaction in self.active_transactions.items():
                    timeout_time = transaction.start_time + timedelta(seconds=transaction.timeout_seconds)
                    if current_time > timeout_time:
                        timed_out_transactions.append(transaction_id)
                
                # Handle timed out transactions
                for transaction_id in timed_out_transactions:
                    try:
                        transaction = self.active_transactions[transaction_id]
                        transaction.status = TransactionStatus.FAILED
                        transaction.error_message = "Transaction timed out"
                        transaction.end_time = current_time
                        
                        # Move to history
                        self.transaction_history.append(transaction)
                        del self.active_transactions[transaction_id]
                        
                        self.transaction_metrics['failed_transactions'] += 1
                        
                        self.logger.warning(f"Transaction {transaction_id} timed out")
                        
                    except Exception as e:
                        self.logger.error(f"Error handling timed out transaction: {e}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Transaction monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _periodic_sync_check(self):
        """Periodic check for automatic synchronization needs"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Check if any dependent agents need synchronization
                # This is a placeholder for more sophisticated sync detection
                self.logger.debug("Performing periodic sync check")
                
            except Exception as e:
                self.logger.error(f"Periodic sync check error: {e}")
                await asyncio.sleep(300)
    
    async def shutdown(self):
        """Gracefully shutdown the persistence coordinator"""
        try:
            self.logger.info("Shutting down Agent Persistence Coordinator")
            
            # Complete or cancel active transactions
            for transaction_id in list(self.active_transactions.keys()):
                transaction = self.active_transactions[transaction_id]
                if transaction.status == TransactionStatus.PENDING:
                    transaction.status = TransactionStatus.FAILED
                    transaction.error_message = "System shutdown"
                    transaction.end_time = datetime.now()
            
            # Clear sync locks
            self.sync_locks.clear()
            self.active_syncs.clear()
            
            self.is_initialized = False
            self.logger.info("Agent Persistence Coordinator shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Factory function for easy instantiation
async def create_agent_persistence_coordinator(state_manager: Optional[AgentStateManager] = None) -> AgentPersistenceCoordinator:
    """Factory function to create and initialize an AgentPersistenceCoordinator"""
    coordinator = AgentPersistenceCoordinator(state_manager)
    
    success = await coordinator.initialize()
    if not success:
        raise RuntimeError("Failed to initialize Agent Persistence Coordinator")
    
    return coordinator


if __name__ == "__main__":
    # Example usage
    async def test_persistence_coordinator():
        try:
            # Create coordinator
            coordinator = await create_agent_persistence_coordinator()
            
            # Test distributed snapshot
            test_agents = ["risk_genius", "pattern_master", "decision_master"]
            snapshot_id = await coordinator.create_distributed_snapshot(test_agents)
            print(f"Created distributed snapshot: {snapshot_id}")
            
            # Test synchronization
            sync_id = await coordinator.synchronize_agent_states(
                primary_agent="risk_genius",
                dependent_agents=["pattern_master"],
                state_types=[StateType.CONFIGURATION],
                sync_direction="primary_to_dependents"
            )
            print(f"Completed synchronization: {sync_id}")
            
            # Get metrics
            metrics = await coordinator.get_coordination_metrics()
            print(f"Coordination metrics: {metrics}")
            
            # Shutdown
            await coordinator.shutdown()
            
        except Exception as e:
            print(f"Test failed: {e}")
            raise
    
    # Run test
    asyncio.run(test_persistence_coordinator())