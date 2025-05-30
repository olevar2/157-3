#!/usr/bin/env python3
"""
Platform3 AI Platform Integration Test Suite
Tests the complete AI platform implementation including model registry,
coordination, performance monitoring, and MLOps services.
"""

import sys
import os
import time
import asyncio
import traceback
import json
from pathlib import Path
from datetime import datetime

# Add ai-platform to Python path
ai_platform_path = os.path.join(os.path.dirname(__file__), 'ai-platform')
ai_services_path = os.path.join(ai_platform_path, 'ai-services')
sys.path.insert(0, ai_platform_path)
sys.path.insert(0, ai_services_path)

class AIPlatformIntegrationTest:
    """Comprehensive test suite for the AI Platform"""
    
    def __init__(self):
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }
        self.ai_platform_manager = None
        
    def log_test(self, test_name, status, details="", error=None):
        """Log test results"""
        result = {
            'test_name': test_name,
            'status': status,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        if error:
            result['error'] = str(error)
            result['traceback'] = traceback.format_exc()
            
        self.test_results['test_details'].append(result)
        self.test_results['total_tests'] += 1
        
        if status == 'PASSED':
            self.test_results['passed_tests'] += 1
            print(f"✅ {test_name}: PASSED - {details}")
        else:
            self.test_results['failed_tests'] += 1
            print(f"❌ {test_name}: FAILED - {details}")
            if error:
                print(f"   Error: {error}")
    
    def test_imports(self):
        """Test 1: Validate all AI platform imports"""
        try:
            from ai_platform_manager import AIPlatformManager
            from ai_services.model_registry.model_registry import ModelRegistry
            from ai_services.coordination.ai_coordinator import AICoordinator
            from ai_services.performance_monitoring.performance_monitor import PerformanceMonitor
            from ai_services.mlops.mlops_service import MLOpsService
            
            self.log_test("Import Test", "PASSED", "All AI platform modules imported successfully")
            return True
        except Exception as e:
            self.log_test("Import Test", "FAILED", "Failed to import AI platform modules", e)
            return False
    
    def test_model_registry_initialization(self):
        """Test 2: Model Registry initialization and model discovery"""
        try:
            from ai_services.model_registry.model_registry import ModelRegistry
            
            registry = ModelRegistry()
            
            # Test model discovery
            models = registry.discover_models()
            
            if models:
                model_count = len(models)
                categories = set(model['category'] for model in models)
                self.log_test("Model Registry Initialization", "PASSED", 
                            f"Discovered {model_count} models across {len(categories)} categories: {list(categories)}")
                return True
            else:
                self.log_test("Model Registry Initialization", "FAILED", 
                            "No models discovered")
                return False
                
        except Exception as e:
            self.log_test("Model Registry Initialization", "FAILED", 
                        "Model registry initialization failed", e)
            return False
    
    def test_ai_coordinator_initialization(self):
        """Test 3: AI Coordinator initialization and task management"""
        try:
            from ai_services.coordination.ai_coordinator import AICoordinator
            
            coordinator = AICoordinator()
            
            # Test task submission
            test_task = {
                'id': 'test_task_001',
                'type': 'prediction',
                'model': 'test_model',
                'data': {'test': 'data'},
                'priority': 1
            }
            
            coordinator.submit_task(test_task)
            
            # Check task queue
            queue_status = coordinator.get_queue_status()
            
            if queue_status['total_tasks'] > 0:
                self.log_test("AI Coordinator Initialization", "PASSED", 
                            f"Task queue operational, {queue_status['total_tasks']} tasks queued")
                return True
            else:
                self.log_test("AI Coordinator Initialization", "FAILED", 
                            "Task queue empty after task submission")
                return False
                
        except Exception as e:
            self.log_test("AI Coordinator Initialization", "FAILED", 
                        "AI coordinator initialization failed", e)
            return False
    
    def test_performance_monitor_initialization(self):
        """Test 4: Performance Monitor initialization and metric tracking"""
        try:
            from ai_services.performance_monitoring.performance_monitor import PerformanceMonitor
            
            monitor = PerformanceMonitor()
            
            # Test metric recording
            monitor.record_prediction_metric('test_model', 50.0, 0.95, 0.02)
            
            # Get system metrics
            system_metrics = monitor.get_system_metrics()
            
            if system_metrics and 'cpu_percent' in system_metrics:
                self.log_test("Performance Monitor Initialization", "PASSED", 
                            f"Performance monitoring active, CPU: {system_metrics['cpu_percent']:.1f}%")
                return True
            else:
                self.log_test("Performance Monitor Initialization", "FAILED", 
                            "System metrics not available")
                return False
                
        except Exception as e:
            self.log_test("Performance Monitor Initialization", "FAILED", 
                        "Performance monitor initialization failed", e)
            return False
    
    def test_mlops_service_initialization(self):
        """Test 5: MLOps Service initialization and model management"""
        try:
            from ai_services.mlops.mlops_service import MLOpsService
            
            mlops = MLOpsService()
            
            # Test model registration
            test_model_info = {
                'name': 'test_model_v1',
                'version': '1.0.0',
                'type': 'trading',
                'performance_metrics': {'accuracy': 0.95}
            }
            
            result = mlops.register_model_version(test_model_info)
            
            if result.get('success'):
                self.log_test("MLOps Service Initialization", "PASSED", 
                            f"Model version registered: {test_model_info['name']}")
                return True
            else:
                self.log_test("MLOps Service Initialization", "FAILED", 
                            "Model version registration failed")
                return False
                
        except Exception as e:
            self.log_test("MLOps Service Initialization", "FAILED", 
                        "MLOps service initialization failed", e)
            return False
    
    def test_ai_platform_manager_initialization(self):
        """Test 6: AI Platform Manager full initialization"""
        try:
            from ai_platform_manager import AIPlatformManager
            
            self.ai_platform_manager = AIPlatformManager()
            
            # Initialize the platform
            self.ai_platform_manager.initialize()
            
            # Check if all services are running
            health_status = self.ai_platform_manager.get_health_status()
            
            healthy_services = sum(1 for service in health_status['services'].values() 
                                 if service.get('status') == 'healthy')
            total_services = len(health_status['services'])
            
            if healthy_services == total_services:
                self.log_test("AI Platform Manager Initialization", "PASSED", 
                            f"All {total_services} services healthy")
                return True
            else:
                self.log_test("AI Platform Manager Initialization", "FAILED", 
                            f"Only {healthy_services}/{total_services} services healthy")
                return False
                
        except Exception as e:
            self.log_test("AI Platform Manager Initialization", "FAILED", 
                        "AI Platform Manager initialization failed", e)
            return False
    
    def test_unified_prediction_interface(self):
        """Test 7: Unified prediction execution"""
        try:
            if not self.ai_platform_manager:
                self.log_test("Unified Prediction Interface", "SKIPPED", 
                            "AI Platform Manager not initialized")
                return False
            
            # Test prediction request
            test_data = {
                'symbol': 'EURUSD',
                'timeframe': '1H',
                'data': [1.1234, 1.1235, 1.1236, 1.1237, 1.1238]
            }
            
            # This would be an async operation in real implementation
            result = self.ai_platform_manager.execute_prediction('market_analysis', test_data)
            
            if result.get('status') == 'success' or result.get('predictions'):
                self.log_test("Unified Prediction Interface", "PASSED", 
                            "Prediction executed successfully")
                return True
            else:
                self.log_test("Unified Prediction Interface", "PARTIAL", 
                            "Prediction interface available but no models ready")
                return True
                
        except Exception as e:
            self.log_test("Unified Prediction Interface", "FAILED", 
                        "Unified prediction execution failed", e)
            return False
    
    def test_model_discovery_accuracy(self):
        """Test 8: Verify model discovery matches actual file structure"""
        try:
            ai_models_path = Path("ai-platform/ai-models")
            
            if not ai_models_path.exists():
                self.log_test("Model Discovery Accuracy", "FAILED", 
                            "AI models directory not found")
                return False
            
            # Count actual model directories
            actual_models = []
            for category_dir in ai_models_path.iterdir():
                if category_dir.is_dir():
                    for model_dir in category_dir.iterdir():
                        if model_dir.is_dir():
                            actual_models.append({
                                'category': category_dir.name,
                                'name': model_dir.name,
                                'path': str(model_dir)
                            })
            
            # Compare with registry discovery
            from ai_services.model_registry.model_registry import ModelRegistry
            registry = ModelRegistry()
            discovered_models = registry.discover_models()
            
            actual_count = len(actual_models)
            discovered_count = len(discovered_models)
            
            if actual_count > 0 and discovered_count >= actual_count * 0.8:  # 80% discovery rate acceptable
                self.log_test("Model Discovery Accuracy", "PASSED", 
                            f"Discovered {discovered_count}/{actual_count} models ({discovered_count/actual_count*100:.1f}%)")
                return True
            else:
                self.log_test("Model Discovery Accuracy", "FAILED", 
                            f"Low discovery rate: {discovered_count}/{actual_count} models")
                return False
                
        except Exception as e:
            self.log_test("Model Discovery Accuracy", "FAILED", 
                        "Model discovery accuracy test failed", e)
            return False
    
    def test_performance_monitoring_alerts(self):
        """Test 9: Performance monitoring and alert system"""
        try:
            from ai_services.performance_monitoring.performance_monitor import PerformanceMonitor
            
            monitor = PerformanceMonitor()
            
            # Test alert generation with poor performance metrics
            monitor.record_prediction_metric('test_model', 1000.0, 0.3, 5.0)  # High latency, low accuracy
            
            alerts = monitor.get_alerts()
            
            if alerts:
                alert_count = len(alerts)
                self.log_test("Performance Monitoring Alerts", "PASSED", 
                            f"Alert system functional, {alert_count} alerts generated")
                return True
            else:
                self.log_test("Performance Monitoring Alerts", "PARTIAL", 
                            "Alert system available but no alerts triggered")
                return True
                
        except Exception as e:
            self.log_test("Performance Monitoring Alerts", "FAILED", 
                        "Performance monitoring alerts test failed", e)
            return False
    
    def test_mlops_workflow(self):
        """Test 10: Complete MLOps workflow"""
        try:
            from ai_services.mlops.mlops_service import MLOpsService
            
            mlops = MLOpsService()
            
            # Test complete workflow
            model_info = {
                'name': 'integration_test_model',
                'version': '1.0.0',
                'type': 'trading',
                'performance_metrics': {'accuracy': 0.92, 'precision': 0.89}
            }
            
            # Register model
            register_result = mlops.register_model_version(model_info)
            
            # List models
            models = mlops.list_models()
            
            # Test model comparison (if multiple versions exist)
            comparison_result = mlops.compare_models(['integration_test_model'])
            
            workflow_steps = 0
            if register_result.get('success'):
                workflow_steps += 1
            if models:
                workflow_steps += 1
            if comparison_result:
                workflow_steps += 1
            
            if workflow_steps >= 2:
                self.log_test("MLOps Workflow", "PASSED", 
                            f"MLOps workflow operational ({workflow_steps}/3 steps successful)")
                return True
            else:
                self.log_test("MLOps Workflow", "FAILED", 
                            f"MLOps workflow incomplete ({workflow_steps}/3 steps successful)")
                return False
                
        except Exception as e:
            self.log_test("MLOps Workflow", "FAILED", 
                        "MLOps workflow test failed", e)
            return False
    
    async def run_all_tests(self):
        """Execute all tests in sequence"""
        print("🚀 Starting Platform3 AI Platform Integration Tests")
        print("=" * 60)
        
        test_methods = [
            self.test_imports,
            self.test_model_registry_initialization,
            self.test_ai_coordinator_initialization,
            self.test_performance_monitor_initialization,
            self.test_mlops_service_initialization,
            self.test_ai_platform_manager_initialization,
            self.test_unified_prediction_interface,
            self.test_model_discovery_accuracy,
            self.test_performance_monitoring_alerts,
            self.test_mlops_workflow
        ]
        
        for test_method in test_methods:
            try:
                test_method()
                time.sleep(0.5)  # Brief pause between tests
            except Exception as e:
                self.log_test(test_method.__name__, "FAILED", 
                            "Test execution failed", e)
        
        # Generate summary
        self.print_summary()
        self.save_results()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("🎯 TEST SUMMARY")
        print("=" * 60)
        
        total = self.test_results['total_tests']
        passed = self.test_results['passed_tests']
        failed = self.test_results['failed_tests']
        success_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("🎉 AI Platform Integration: SUCCESSFUL")
        elif success_rate >= 60:
            print("⚠️  AI Platform Integration: PARTIAL SUCCESS")
        else:
            print("❌ AI Platform Integration: NEEDS IMPROVEMENT")
        
        print("\nDetailed Results:")
        for test in self.test_results['test_details']:
            status_icon = "✅" if test['status'] == 'PASSED' else "❌" if test['status'] == 'FAILED' else "⚠️"
            print(f"{status_icon} {test['test_name']}: {test['status']} - {test['details']}")
    
    def save_results(self):
        """Save test results to file"""
        try:
            with open('ai_platform_integration_test_results.json', 'w') as f:
                json.dump(self.test_results, f, indent=2)
            print(f"\n📄 Test results saved to: ai_platform_integration_test_results.json")
        except Exception as e:
            print(f"❌ Failed to save test results: {e}")

def main():
    """Main test execution"""
    tester = AIPlatformIntegrationTest()
    
    # Run tests
    try:
        asyncio.run(tester.run_all_tests())
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
