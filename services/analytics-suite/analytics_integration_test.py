"""
Advanced Analytics Framework Integration Test
Comprehensive testing and validation of the analytics framework
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys
import os

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from AdvancedAnalyticsFramework import AdvancedAnalyticsFramework
from AnalyticsWebSocketServer import AnalyticsWebSocketServer
import requests
import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedAnalyticsIntegrationTest:
    """Comprehensive integration test for the Advanced Analytics Framework"""
    
    def __init__(self):
        self.framework = None
        self.websocket_server = None
        self.api_base_url = "http://localhost:8002"
        self.websocket_url = "ws://localhost:8001"
        
        # Test data
        self.test_trades = [
            {
                "symbol": "EURUSD",
                "entry_price": 1.1000,
                "exit_price": 1.1050,
                "quantity": 1000,
                "entry_time": "2025-06-01T10:00:00Z",
                "exit_time": "2025-06-01T10:30:00Z",
                "side": "buy"
            },
            {
                "symbol": "GBPUSD",
                "entry_price": 1.3000,
                "exit_price": 1.2950,
                "quantity": 800,
                "entry_time": "2025-06-01T11:00:00Z",
                "exit_time": "2025-06-01T11:45:00Z",
                "side": "buy"
            },
            {
                "symbol": "USDJPY",
                "entry_price": 150.00,
                "exit_price": 150.75,
                "quantity": 1200,
                "entry_time": "2025-06-01T12:00:00Z",
                "exit_time": "2025-06-01T12:20:00Z",
                "side": "buy"
            }
        ]
        
        # Test results tracking
        self.test_results = {}
        self.passed_tests = 0
        self.failed_tests = 0

    async def run_comprehensive_test(self):
        """Run comprehensive integration test"""
        print("🚀 Starting Advanced Analytics Framework Integration Test")
        print("=" * 70)
        
        try:
            # Phase 1: Framework initialization
            await self._test_framework_initialization()
            
            # Phase 2: Analytics engines testing
            await self._test_analytics_engines()
            
            # Phase 3: Real-time metrics testing
            await self._test_realtime_metrics()
            
            # Phase 4: Report generation testing
            await self._test_report_generation()
            
            # Phase 5: WebSocket integration testing
            await self._test_websocket_integration()
            
            # Phase 6: API integration testing
            await self._test_api_integration()
            
            # Phase 7: Performance testing
            await self._test_performance()
            
            # Generate test report
            await self._generate_test_report()
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            self._record_test_result("integration_test", False, str(e))
        
        finally:
            await self._cleanup()

    async def _test_framework_initialization(self):
        """Test framework initialization"""
        print("\n📋 Phase 1: Framework Initialization")
        
        try:
            # Initialize analytics framework
            self.framework = AdvancedAnalyticsFramework()
            await self.framework.initialize()
            
            # Verify initialization
            assert len(self.framework.engines) > 0, "No analytics engines registered"
            assert self.framework.redis_client is not None, "Redis client not initialized"
            assert self.framework.communication_framework is not None, "Communication framework not initialized"
            
            self._record_test_result("framework_initialization", True, "Successfully initialized")
            print("✅ Framework initialization: PASSED")
            
        except Exception as e:
            self._record_test_result("framework_initialization", False, str(e))
            print(f"❌ Framework initialization: FAILED - {e}")
            raise

    async def _test_analytics_engines(self):
        """Test individual analytics engines"""
        print("\n🔧 Phase 2: Analytics Engines Testing")
        
        for engine_name, engine in self.framework.engines.items():
            try:
                # Test data processing
                test_data = {"trades": self.test_trades}
                result = await engine.process_data(test_data)
                
                assert "performance_score" in result, f"Engine {engine_name} missing performance_score"
                assert isinstance(result["performance_score"], (int, float)), f"Invalid performance_score type in {engine_name}"
                
                # Test report generation
                report = await engine.generate_report("1h")
                assert report.report_id is not None, f"Engine {engine_name} report missing ID"
                assert report.confidence_score >= 0, f"Invalid confidence score in {engine_name}"
                
                self._record_test_result(f"engine_{engine_name}", True, f"Performance: {result['performance_score']}")
                print(f"✅ Engine {engine_name}: PASSED")
                
            except Exception as e:
                self._record_test_result(f"engine_{engine_name}", False, str(e))
                print(f"❌ Engine {engine_name}: FAILED - {e}")

    async def _test_realtime_metrics(self):
        """Test real-time metrics collection"""
        print("\n📊 Phase 3: Real-time Metrics Testing")
        
        try:
            # Stream test data
            test_data = {"trades": self.test_trades}
            results = await self.framework.stream_analytics_data(test_data, "integration_test")
            
            assert isinstance(results, dict), "Invalid results format"
            assert len(results) > 0, "No engine results returned"
            
            # Wait for metrics to be collected
            await asyncio.sleep(2)
            
            # Get real-time metrics
            metrics = self.framework.get_realtime_metrics()
            assert len(metrics) > 0, "No real-time metrics collected"
            
            # Verify metric structure
            for metric_name, metric in metrics.items():
                assert hasattr(metric, 'metric_name'), f"Metric {metric_name} missing name"
                assert hasattr(metric, 'value'), f"Metric {metric_name} missing value"
                assert hasattr(metric, 'timestamp'), f"Metric {metric_name} missing timestamp"
            
            self._record_test_result("realtime_metrics", True, f"Collected {len(metrics)} metrics")
            print(f"✅ Real-time metrics: PASSED ({len(metrics)} metrics)")
            
        except Exception as e:
            self._record_test_result("realtime_metrics", False, str(e))
            print(f"❌ Real-time metrics: FAILED - {e}")

    async def _test_report_generation(self):
        """Test comprehensive report generation"""
        print("\n📄 Phase 4: Report Generation Testing")
        
        try:
            # Generate comprehensive report
            report = await self.framework.generate_comprehensive_report("1h")
            
            assert report.report_id is not None, "Report missing ID"
            assert report.summary is not None, "Report missing summary"
            assert isinstance(report.recommendations, list), "Invalid recommendations format"
            assert 0 <= report.confidence_score <= 100, "Invalid confidence score"
            assert report.data is not None, "Report missing data"
            
            # Verify report data structure
            assert "timeframe" in report.data, "Report missing timeframe"
            assert "engine_reports" in report.data, "Report missing engine reports"
            assert "aggregated_metrics" in report.data, "Report missing aggregated metrics"
            
            self._record_test_result("report_generation", True, f"Generated report {report.report_id}")
            print(f"✅ Report generation: PASSED (Confidence: {report.confidence_score:.1f}%)")
            
        except Exception as e:
            self._record_test_result("report_generation", False, str(e))
            print(f"❌ Report generation: FAILED - {e}")

    async def _test_websocket_integration(self):
        """Test WebSocket server integration"""
        print("\n🔌 Phase 5: WebSocket Integration Testing")
        
        try:
            # Start WebSocket server in background
            self.websocket_server = AnalyticsWebSocketServer(host="localhost", port=8001)
            server_task = asyncio.create_task(self._start_websocket_server())
            
            # Wait for server to start
            await asyncio.sleep(3)
            
            # Test WebSocket connection
            await self._test_websocket_connection()
            
            self._record_test_result("websocket_integration", True, "WebSocket server functional")
            print("✅ WebSocket integration: PASSED")
            
        except Exception as e:
            self._record_test_result("websocket_integration", False, str(e))
            print(f"❌ WebSocket integration: FAILED - {e}")

    async def _start_websocket_server(self):
        """Start WebSocket server for testing"""
        try:
            await self.websocket_server.start_server()
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")

    async def _test_websocket_connection(self):
        """Test WebSocket connection and messaging"""
        try:
            async with websockets.connect(f"{self.websocket_url}/analytics") as websocket:
                # Send test message
                test_message = {"action": "refresh_metrics"}
                await websocket.send(json.dumps(test_message))
                
                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
                
                assert "type" in response_data, "Invalid WebSocket response format"
                print(f"  📡 WebSocket response: {response_data.get('type', 'unknown')}")
                
        except asyncio.TimeoutError:
            raise Exception("WebSocket response timeout")
        except Exception as e:
            raise Exception(f"WebSocket connection failed: {e}")

    async def _test_api_integration(self):
        """Test REST API integration"""
        print("\n🌐 Phase 6: API Integration Testing")
        
        # Note: This assumes the API server is running
        # In a real scenario, you'd start the API server here
        
        try:
            # Test health endpoint
            response = requests.get(f"{self.api_base_url}/analytics/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                assert health_data["status"] == "healthy", "API health check failed"
                print("  🏥 Health check: PASSED")
            else:
                raise Exception(f"Health check failed with status {response.status_code}")
            
            # Test status endpoint
            try:
                response = requests.get(f"{self.api_base_url}/analytics/status", timeout=5)
                if response.status_code == 200:
                    status_data = response.json()
                    print(f"  📊 Status check: PASSED (Engines: {status_data.get('total_engines', 0)})")
                else:
                    print(f"  📊 Status check: API not ready (Status: {response.status_code})")
            except requests.exceptions.RequestException:
                print("  📊 Status check: API server not running (expected for isolated test)")
            
            self._record_test_result("api_integration", True, "API endpoints accessible")
            print("✅ API integration: PASSED")
            
        except requests.exceptions.RequestException as e:
            self._record_test_result("api_integration", False, f"API server not accessible: {e}")
            print("⚠️  API integration: SKIPPED (API server not running)")
        except Exception as e:
            self._record_test_result("api_integration", False, str(e))
            print(f"❌ API integration: FAILED - {e}")

    async def _test_performance(self):
        """Test system performance under load"""
        print("\n⚡ Phase 7: Performance Testing")
        
        try:
            start_time = time.time()
            
            # Process multiple data streams concurrently
            tasks = []
            for i in range(10):
                test_data = {
                    "trades": self.test_trades,
                    "batch_id": i
                }
                task = self.framework.stream_analytics_data(test_data, f"perf_test_{i}")
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Check for errors
            errors = [r for r in results if isinstance(r, Exception)]
            successful = len(results) - len(errors)
            
            assert successful >= 8, f"Too many failed operations: {len(errors)}/10"
            assert processing_time < 30, f"Processing too slow: {processing_time:.2f}s"
            
            self._record_test_result("performance", True, f"{successful}/10 operations in {processing_time:.2f}s")
            print(f"✅ Performance: PASSED ({successful}/10 operations in {processing_time:.2f}s)")
            
        except Exception as e:
            self._record_test_result("performance", False, str(e))
            print(f"❌ Performance: FAILED - {e}")

    async def _generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 70)
        print("📊 INTEGRATION TEST REPORT")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["passed"])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print()
        
        # Detailed results
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result["passed"] else "❌ FAILED"
            print(f"{status} {test_name}: {result['message']}")
        
        # Save report to file
        report_data = {
            "test_execution": {
                "timestamp": datetime.utcnow().isoformat(),
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate
            },
            "test_results": self.test_results,
            "system_info": {
                "framework_version": "1.0.0",
                "test_environment": "integration",
                "engines_tested": len(self.framework.engines) if self.framework else 0
            }
        }
        
        with open("advanced_analytics_integration_test_report.json", "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print("\n📄 Test report saved to: advanced_analytics_integration_test_report.json")
        
        if success_rate >= 80:
            print("\n🎉 INTEGRATION TEST: OVERALL SUCCESS!")
        else:
            print("\n⚠️  INTEGRATION TEST: NEEDS ATTENTION")
        
        print("=" * 70)

    def _record_test_result(self, test_name: str, passed: bool, message: str):
        """Record test result"""
        self.test_results[test_name] = {
            "passed": passed,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1

    async def _cleanup(self):
        """Cleanup test resources"""
        try:
            if self.framework:
                await self.framework.shutdown()
            
            if self.websocket_server:
                await self.websocket_server.shutdown()
            
            print("\n🧹 Cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# Quick validation test
async def quick_validation_test():
    """Quick validation test for basic functionality"""
    print("🔬 Quick Analytics Framework Validation")
    print("-" * 50)
    
    try:
        # Initialize framework
        framework = AdvancedAnalyticsFramework()
        await framework.initialize()
        
        # Test basic functionality
        test_data = {
            "trades": [
                {"symbol": "EURUSD", "entry_price": 1.1000, "exit_price": 1.1050, "quantity": 1000}
            ]
        }
        
        results = await framework.stream_analytics_data(test_data)
        print(f"✅ Data processing: {len(results)} engines responded")
        
        metrics = framework.get_realtime_metrics()
        print(f"✅ Metrics collection: {len(metrics)} metrics available")
        
        report = await framework.generate_comprehensive_report("1h")
        print(f"✅ Report generation: {report.report_id} (Confidence: {report.confidence_score:.1f}%)")
        
        await framework.shutdown()
        print("✅ Framework shutdown: Clean")
        
        print("\n🎉 Quick validation: ALL SYSTEMS OPERATIONAL")
        
    except Exception as e:
        print(f"❌ Quick validation failed: {e}")
        raise

# Main execution
async def main():
    """Main function to run tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Analytics Framework Integration Test")
    parser.add_argument("--quick", action="store_true", help="Run quick validation test only")
    parser.add_argument("--full", action="store_true", help="Run full integration test")
    
    args = parser.parse_args()
    
    if args.quick:
        await quick_validation_test()
    elif args.full:
        test_suite = AdvancedAnalyticsIntegrationTest()
        await test_suite.run_comprehensive_test()
    else:
        print("Please specify --quick or --full test mode")
        print("Use --help for more information")

if __name__ == "__main__":
    asyncio.run(main())
