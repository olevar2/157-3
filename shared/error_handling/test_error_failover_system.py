"""
🧪 ERROR PROPAGATION & FAILOVER TEST SUITE
===========================================

Comprehensive test suite for validating error propagation and failover systems.
Tests all scenarios including cascade prevention, failover strategies, and recovery.

Mission: Ensure bulletproof error handling and failover reliability
"""

import asyncio
import logging
import pytest
import json
from typing import Dict, List, Any
from datetime import datetime, timedelta
import uuid

# Setup logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import systems under test
from .error_propagation_manager import (
    ErrorPropagationManager, ErrorEvent, ErrorSeverity, ErrorCategory,
    PropagationStrategy, AgentErrorState
)
from .failover_coordinator import (
    FailoverCoordinator, FailoverConfig, FailoverTrigger, FailoverStrategy,
    AgentType, AgentStatus
)
from .error_failover_bridge import ErrorFailoverBridge


class ErrorFailoverTestSuite:
    """
    🧪 COMPREHENSIVE TEST SUITE
    
    Tests error propagation, failover coordination, and integration
    across various failure scenarios and recovery procedures.
    """
    
    def __init__(self):
        """Initialize test suite"""
        self.logger = logging.getLogger(__name__)
        
        # Test instances
        self.error_manager = None
        self.failover_coordinator = None
        self.bridge = None
        
        # Test state
        self.test_results = []
        self.test_agents = [
            "test_agent_1", "test_agent_2", "test_agent_3",
            "test_backup_1", "test_backup_2", "test_backup_3"
        ]
        
        self.logger.info("🧪 Error & Failover Test Suite initialized")
    
    async def setup_test_environment(self):
        """Setup test environment with fresh instances"""
        self.logger.info("🧪 Setting up test environment...")
        
        # Create fresh instances for testing
        self.error_manager = ErrorPropagationManager({
            'test_mode': True,
            'propagation_delay_multiplier': 0.1  # Speed up for testing
        })
        
        self.failover_coordinator = FailoverCoordinator({
            'test_mode': True,
            'failover_timeout_multiplier': 0.1  # Speed up for testing
        })
        
        self.bridge = ErrorFailoverBridge()
        
        # Setup test configurations
        await self._setup_test_agents()
        await self._setup_test_callbacks()
        
        self.logger.info("✅ Test environment ready")
    
    async def _setup_test_agents(self):
        """Setup test agents with various configurations"""
        
        # Test agent configurations
        test_configs = [
            {
                'agent_id': 'test_agent_1',
                'type': AgentType.CRITICAL_SERVICE,
                'strategy': FailoverStrategy.HOT_STANDBY,
                'backups': ['test_backup_1']
            },
            {
                'agent_id': 'test_agent_2',
                'type': AgentType.COORDINATION_HUB,
                'strategy': FailoverStrategy.GRACEFUL,
                'backups': ['test_backup_2']
            },
            {
                'agent_id': 'test_agent_3',
                'type': AgentType.STATELESS_SERVICE,
                'strategy': FailoverStrategy.LOAD_BALANCING,
                'backups': ['test_backup_3']
            }
        ]
        
        for config_data in test_configs:
            config = FailoverConfig(
                agent_id=config_data['agent_id'],
                agent_type=config_data['type'],
                strategy=config_data['strategy'],
                backup_agents=config_data['backups'],
                max_failover_time=30,  # Short timeout for testing
                health_check_interval=5,  # Frequent checks for testing
                state_sync_required=True
            )
            
            await self.failover_coordinator.register_agent(config_data['agent_id'], config)
    
    async def _setup_test_callbacks(self):
        """Setup test callbacks for monitoring"""
        
        # Error handler for testing
        async def test_error_handler(error_event: ErrorEvent):
            self.test_results.append({
                'type': 'error_handled',
                'agent': error_event.source_agent,
                'severity': error_event.severity.value,
                'category': error_event.category.value,
                'timestamp': datetime.now().isoformat()
            })
        
        # Failover callback for testing
        async def test_failover_callback(failover_event):
            self.test_results.append({
                'type': 'failover_completed',
                'failed_agent': failover_event.failed_agent,
                'backup_agent': failover_event.backup_agent,
                'success': failover_event.success,
                'strategy': failover_event.strategy.value,
                'timestamp': datetime.now().isoformat()
            })
        
        # Health check callback for testing
        async def test_health_callback(agent_id: str) -> bool:
            # Simulate health check - agents are healthy unless in error state
            error_state = self.error_manager.get_agent_error_state(agent_id)
            if error_state and error_state.consecutive_failures >= 2:
                return False
            return True
        
        # Register callbacks
        self.error_manager.register_error_handler("*", test_error_handler)
        self.failover_coordinator.register_failover_callback(test_failover_callback)
        
        for agent_id in self.test_agents:
            self.failover_coordinator.register_health_check_callback(agent_id, test_health_callback)
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite"""
        self.logger.info("🧪 Starting comprehensive test suite...")
        
        start_time = datetime.now()
        test_summary = {
            'start_time': start_time.isoformat(),
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'test_details': []
        }
        
        # Test categories
        test_categories = [
            ('Error Propagation Tests', self._test_error_propagation),
            ('Failover Coordination Tests', self._test_failover_coordination),
            ('Integration Bridge Tests', self._test_integration_bridge),
            ('Cascade Prevention Tests', self._test_cascade_prevention),
            ('Recovery and Rollback Tests', self._test_recovery_rollback),
            ('Stress and Load Tests', self._test_stress_scenarios)
        ]
        
        for category_name, test_method in test_categories:
            self.logger.info(f"🧪 Running {category_name}...")
            
            try:
                category_results = await test_method()
                test_summary['test_details'].append({
                    'category': category_name,
                    'results': category_results,
                    'passed': category_results.get('passed', 0),
                    'failed': category_results.get('failed', 0)
                })
                
                test_summary['tests_run'] += category_results.get('total', 0)
                test_summary['tests_passed'] += category_results.get('passed', 0)
                test_summary['tests_failed'] += category_results.get('failed', 0)
                
            except Exception as e:
                self.logger.error(f"❌ Test category failed: {category_name} - {e}")
                test_summary['test_details'].append({
                    'category': category_name,
                    'error': str(e),
                    'passed': 0,
                    'failed': 1
                })
                test_summary['tests_failed'] += 1
        
        end_time = datetime.now()
        test_summary['end_time'] = end_time.isoformat()
        test_summary['duration_seconds'] = (end_time - start_time).total_seconds()
        test_summary['success_rate'] = (
            test_summary['tests_passed'] / max(1, test_summary['tests_run']) * 100
        )
        
        self.logger.info(f"🧪 Test suite completed: {test_summary['success_rate']:.1f}% success rate")
        return test_summary
    
    async def _test_error_propagation(self) -> Dict[str, Any]:
        """Test error propagation mechanisms"""
        results = {'total': 0, 'passed': 0, 'failed': 0, 'details': []}
        
        # Test 1: Basic error propagation
        test_name = "Basic Error Propagation"
        try:
            error_event = ErrorEvent(
                source_agent='test_agent_1',
                error_type='test_error',
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.COMPUTATION,
                message='Test error for propagation'
            )
            
            success = await self.error_manager.handle_error_event(error_event)
            
            if success:
                results['passed'] += 1
                results['details'].append(f"✅ {test_name}: Error handled successfully")
            else:
                results['failed'] += 1
                results['details'].append(f"❌ {test_name}: Error handling failed")
            
        except Exception as e:
            results['failed'] += 1
            results['details'].append(f"❌ {test_name}: Exception - {e}")
        
        results['total'] += 1
        
        # Test 2: Severity-based propagation
        test_name = "Severity-based Propagation"
        try:
            # Test different severity levels
            severities = [ErrorSeverity.WARNING, ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]
            
            for severity in severities:
                error_event = ErrorEvent(
                    source_agent='test_agent_2',
                    error_type='severity_test',
                    severity=severity,
                    category=ErrorCategory.NETWORK,
                    message=f'Test {severity.value} error'
                )
                
                await self.error_manager.handle_error_event(error_event)
                await asyncio.sleep(0.1)  # Small delay between errors
            
            results['passed'] += 1
            results['details'].append(f"✅ {test_name}: Multiple severity levels handled")
            
        except Exception as e:
            results['failed'] += 1
            results['details'].append(f"❌ {test_name}: Exception - {e}")
        
        results['total'] += 1
        
        # Test 3: Circuit breaker activation
        test_name = "Circuit Breaker Activation"
        try:
            # Generate multiple errors to trigger circuit breaker
            for i in range(6):  # Exceed threshold
                error_event = ErrorEvent(
                    source_agent='test_agent_3',
                    error_type='circuit_breaker_test',
                    severity=ErrorSeverity.ERROR,
                    category=ErrorCategory.COMPUTATION,
                    message=f'Circuit breaker test error {i+1}'
                )
                await self.error_manager.handle_error_event(error_event)
                await asyncio.sleep(0.05)
            
            # Check if circuit breaker opened
            agent_state = self.error_manager.get_agent_error_state('test_agent_3')
            if agent_state and agent_state.circuit_breaker_open:
                results['passed'] += 1
                results['details'].append(f"✅ {test_name}: Circuit breaker opened correctly")
            else:
                results['failed'] += 1
                results['details'].append(f"❌ {test_name}: Circuit breaker did not open")
            
        except Exception as e:
            results['failed'] += 1
            results['details'].append(f"❌ {test_name}: Exception - {e}")
        
        results['total'] += 1
        
        return results
    
    async def _test_failover_coordination(self) -> Dict[str, Any]:
        """Test failover coordination mechanisms"""
        results = {'total': 0, 'passed': 0, 'failed': 0, 'details': []}
        
        # Test 1: Immediate failover
        test_name = "Immediate Failover"
        try:
            success = await self.failover_coordinator.trigger_failover(
                'test_agent_1', 
                FailoverTrigger.MANUAL_TRIGGER
            )
            
            if success:
                results['passed'] += 1
                results['details'].append(f"✅ {test_name}: Immediate failover succeeded")
            else:
                results['failed'] += 1
                results['details'].append(f"❌ {test_name}: Immediate failover failed")
            
        except Exception as e:
            results['failed'] += 1
            results['details'].append(f"❌ {test_name}: Exception - {e}")
        
        results['total'] += 1
        
        # Test 2: Graceful failover
        test_name = "Graceful Failover"
        try:
            success = await self.failover_coordinator.trigger_failover(
                'test_agent_2', 
                FailoverTrigger.HEALTH_CHECK_FAILED
            )
            
            if success:
                results['passed'] += 1
                results['details'].append(f"✅ {test_name}: Graceful failover succeeded")
            else:
                results['failed'] += 1
                results['details'].append(f"❌ {test_name}: Graceful failover failed")
            
        except Exception as e:
            results['failed'] += 1
            results['details'].append(f"❌ {test_name}: Exception - {e}")
        
        results['total'] += 1
        
        # Test 3: Agent status tracking
        test_name = "Agent Status Tracking"
        try:
            # Check agent statuses after failovers
            status_1 = self.failover_coordinator.get_agent_status('test_agent_1')
            status_2 = self.failover_coordinator.get_agent_status('test_agent_2')
            
            if status_1 and status_2:
                results['passed'] += 1
                results['details'].append(f"✅ {test_name}: Agent statuses tracked correctly")
            else:
                results['failed'] += 1
                results['details'].append(f"❌ {test_name}: Agent status tracking failed")
            
        except Exception as e:
            results['failed'] += 1
            results['details'].append(f"❌ {test_name}: Exception - {e}")
        
        results['total'] += 1
        
        return results
    
    async def _test_integration_bridge(self) -> Dict[str, Any]:
        """Test integration bridge functionality"""
        results = {'total': 0, 'passed': 0, 'failed': 0, 'details': []}
        
        # Test 1: Bridge initialization
        test_name = "Bridge Initialization"
        try:
            await self.bridge.initialize_integrations()
            
            if self.bridge.bridge_active:
                results['passed'] += 1
                results['details'].append(f"✅ {test_name}: Bridge initialized successfully")
            else:
                results['failed'] += 1
                results['details'].append(f"❌ {test_name}: Bridge initialization failed")
            
        except Exception as e:
            results['failed'] += 1
            results['details'].append(f"❌ {test_name}: Exception - {e}")
        
        results['total'] += 1
        
        # Test 2: Unified system status
        test_name = "Unified System Status"
        try:
            status = self.bridge.get_unified_system_status()
            
            if 'error_statistics' in status and 'failover_statistics' in status:
                results['passed'] += 1
                results['details'].append(f"✅ {test_name}: System status retrieved")
            else:
                results['failed'] += 1
                results['details'].append(f"❌ {test_name}: System status incomplete")
            
        except Exception as e:
            results['failed'] += 1
            results['details'].append(f"❌ {test_name}: Exception - {e}")
        
        results['total'] += 1
        
        # Test 3: Emergency failover trigger
        test_name = "Emergency Failover Trigger"
        try:
            success = await self.bridge.trigger_emergency_failover('test_agent_3')
            
            if success:
                results['passed'] += 1
                results['details'].append(f"✅ {test_name}: Emergency failover triggered")
            else:
                results['failed'] += 1
                results['details'].append(f"❌ {test_name}: Emergency failover failed")
            
        except Exception as e:
            results['failed'] += 1
            results['details'].append(f"❌ {test_name}: Exception - {e}")
        
        results['total'] += 1
        
        return results
    
    async def _test_cascade_prevention(self) -> Dict[str, Any]:
        """Test cascade prevention mechanisms"""
        results = {'total': 0, 'passed': 0, 'failed': 0, 'details': []}
        
        # Test 1: Error cascade simulation
        test_name = "Error Cascade Prevention"
        try:
            # Generate rapid errors from multiple agents
            for i in range(3):
                for agent in ['test_agent_1', 'test_agent_2', 'test_agent_3']:
                    error_event = ErrorEvent(
                        source_agent=agent,
                        error_type='cascade_test',
                        severity=ErrorSeverity.ERROR,
                        category=ErrorCategory.COORDINATION,
                        message=f'Cascade test error {i+1}'
                    )
                    await self.error_manager.handle_error_event(error_event)
                    await asyncio.sleep(0.01)  # Very rapid errors
            
            # Check if propagation was suppressed to prevent cascade
            stats = self.error_manager.get_error_statistics()
            
            if stats['agents_failing'] < 3:  # Not all agents should be failing
                results['passed'] += 1
                results['details'].append(f"✅ {test_name}: Cascade prevention worked")
            else:
                results['failed'] += 1
                results['details'].append(f"❌ {test_name}: Cascade not prevented")
            
        except Exception as e:
            results['failed'] += 1
            results['details'].append(f"❌ {test_name}: Exception - {e}")
        
        results['total'] += 1
        
        return results
    
    async def _test_recovery_rollback(self) -> Dict[str, Any]:
        """Test recovery and rollback mechanisms"""
        results = {'total': 0, 'passed': 0, 'failed': 0, 'details': []}
        
        # Test 1: Agent recovery
        test_name = "Agent Recovery"
        try:
            # Force recovery of a failing agent
            success = await self.error_manager.force_agent_recovery('test_agent_1')
            
            if success:
                results['passed'] += 1
                results['details'].append(f"✅ {test_name}: Agent recovery succeeded")
            else:
                results['failed'] += 1
                results['details'].append(f"❌ {test_name}: Agent recovery failed")
            
        except Exception as e:
            results['failed'] += 1
            results['details'].append(f"❌ {test_name}: Exception - {e}")
        
        results['total'] += 1
        
        # Test 2: System-wide recovery
        test_name = "System-wide Recovery"
        try:
            recovery_results = await self.bridge.perform_system_recovery()
            
            if 'agents_recovered' in recovery_results:
                results['passed'] += 1
                results['details'].append(f"✅ {test_name}: System recovery completed")
            else:
                results['failed'] += 1
                results['details'].append(f"❌ {test_name}: System recovery failed")
            
        except Exception as e:
            results['failed'] += 1
            results['details'].append(f"❌ {test_name}: Exception - {e}")
        
        results['total'] += 1
        
        return results
    
    async def _test_stress_scenarios(self) -> Dict[str, Any]:
        """Test stress scenarios and load handling"""
        results = {'total': 0, 'passed': 0, 'failed': 0, 'details': []}
        
        # Test 1: High error rate
        test_name = "High Error Rate Handling"
        try:
            # Generate high volume of errors
            tasks = []
            for i in range(50):  # 50 concurrent errors
                error_event = ErrorEvent(
                    source_agent=f'test_agent_{(i % 3) + 1}',
                    error_type='stress_test',
                    severity=ErrorSeverity.WARNING,
                    category=ErrorCategory.COMPUTATION,
                    message=f'Stress test error {i+1}'
                )
                tasks.append(self.error_manager.handle_error_event(error_event))
            
            # Execute all concurrently
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successes
            successes = sum(1 for r in results_list if r is True)
            
            if successes >= 40:  # At least 80% success rate
                results['passed'] += 1
                results['details'].append(f"✅ {test_name}: High error rate handled ({successes}/50)")
            else:
                results['failed'] += 1
                results['details'].append(f"❌ {test_name}: Poor handling ({successes}/50)")
            
        except Exception as e:
            results['failed'] += 1
            results['details'].append(f"❌ {test_name}: Exception - {e}")
        
        results['total'] += 1
        
        # Test 2: Disaster recovery simulation
        test_name = "Disaster Recovery Simulation"
        try:
            disaster_results = await self.bridge.simulate_disaster_recovery()
            
            if disaster_results.get('success', False):
                results['passed'] += 1
                results['details'].append(f"✅ {test_name}: Disaster recovery succeeded")
            else:
                results['failed'] += 1
                results['details'].append(f"❌ {test_name}: Disaster recovery failed")
            
        except Exception as e:
            results['failed'] += 1
            results['details'].append(f"❌ {test_name}: Exception - {e}")
        
        results['total'] += 1
        
        return results
    
    def generate_test_report(self, test_summary: Dict[str, Any]) -> str:
        """Generate comprehensive test report"""
        report_lines = [
            "🧪 ERROR PROPAGATION & FAILOVER TEST REPORT",
            "=" * 50,
            f"Test Suite Duration: {test_summary['duration_seconds']:.2f} seconds",
            f"Tests Run: {test_summary['tests_run']}",
            f"Tests Passed: {test_summary['tests_passed']}",
            f"Tests Failed: {test_summary['tests_failed']}",
            f"Success Rate: {test_summary['success_rate']:.1f}%",
            "",
            "DETAILED RESULTS:",
            "-" * 20
        ]
        
        for category in test_summary['test_details']:
            report_lines.append(f"\n📋 {category['category']}")
            
            if 'error' in category:
                report_lines.append(f"   ❌ CATEGORY FAILED: {category['error']}")
            else:
                report_lines.append(f"   ✅ Passed: {category['passed']}")
                report_lines.append(f"   ❌ Failed: {category['failed']}")
                
                if 'results' in category and 'details' in category['results']:
                    for detail in category['results']['details']:
                        report_lines.append(f"      {detail}")
        
        # Add system statistics
        if hasattr(self, 'bridge') and self.bridge:
            try:
                system_status = self.bridge.get_unified_system_status()
                report_lines.extend([
                    "\n📊 FINAL SYSTEM STATUS:",
                    "-" * 20,
                    f"Bridge Active: {system_status.get('bridge_active', 'Unknown')}",
                    f"Integrated Agents: {len(system_status.get('integrated_agents', []))}",
                    f"System Health: {system_status.get('system_health', {}).get('overall_healthy', 'Unknown')}",
                    f"Error Rate: {system_status.get('system_health', {}).get('error_rate', 'Unknown')}",
                    f"Healthy Agents: {system_status.get('system_health', {}).get('healthy_agents', 'Unknown')}"
                ])
            except Exception as e:
                report_lines.append(f"\n⚠️ Could not get final system status: {e}")
        
        return "\n".join(report_lines)


# Main test execution
async def run_comprehensive_tests():
    """Run the complete test suite"""
    test_suite = ErrorFailoverTestSuite()
    
    try:
        # Setup test environment
        await test_suite.setup_test_environment()
        
        # Run all tests
        test_summary = await test_suite.run_all_tests()
        
        # Generate and display report
        report = test_suite.generate_test_report(test_summary)
        print(report)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"error_failover_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(test_summary, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        return test_summary
        
    except Exception as e:
        print(f"❌ Test suite execution failed: {e}")
        return {'error': str(e)}


if __name__ == "__main__":
    asyncio.run(run_comprehensive_tests())