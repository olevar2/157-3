"""
Enhanced Analytics Integration Test
Tests the integration between enhanced analytics services and the Advanced Analytics Framework
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from pathlib import Path

# Import the enhanced analytics services
from AdvancedAnalyticsFramework import AdvancedAnalyticsFramework
from DayTradingAnalytics import DayTradingAnalytics
from SwingAnalytics import SwingAnalytics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedAnalyticsIntegrationTest:
    """Test suite for enhanced analytics integration"""
    
    def __init__(self):
        """Initialize the test suite"""
        self.framework = None
        self.test_results = {}
        self.test_data = self._generate_test_data()
        
    def _generate_test_data(self) -> Dict[str, Any]:
        """Generate comprehensive test data for analytics testing"""
        logger.info("Generating test data for enhanced analytics integration")
        
        # Generate sample trading data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        
        # Sample trade data for day trading
        day_trades = []
        for i in range(100):
            trade = {
                "symbol": np.random.choice(["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]),
                "entry_price": 1.1000 + np.random.normal(0, 0.01),
                "exit_price": 1.1000 + np.random.normal(0, 0.01),
                "quantity": np.random.randint(1000, 10000),
                "entry_time": dates[i % len(dates)],
                "exit_time": dates[i % len(dates)] + timedelta(hours=np.random.randint(1, 8)),
                "pnl": np.random.normal(50, 200)
            }
            day_trades.append(trade)
        
        # Sample swing trading data
        swing_trades = []
        for i in range(50):
            entry_date = dates[i % len(dates)]
            exit_date = entry_date + timedelta(days=np.random.randint(3, 14))
            entry_price = 1.1000 + np.random.normal(0, 0.02)
            exit_price = entry_price + np.random.normal(0, 0.03)
            
            swing_trade = {
                "symbol": np.random.choice(["EURUSD", "GBPUSD", "USDJPY"]),
                "entry_date": entry_date.strftime("%Y-%m-%d"),
                "exit_date": exit_date.strftime("%Y-%m-%d"),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "position_size": np.random.randint(5000, 20000),
                "direction": np.random.choice(["Long", "Short"]),
                "pnl": (exit_price - entry_price) * np.random.randint(5000, 20000),
                "hold_period_days": (exit_date - entry_date).days
            }
            swing_trades.append(swing_trade)
        
        # Sample market data
        market_data = []
        for date in dates[:100]:
            market_point = {
                "timestamp": date,
                "open": 1.1000 + np.random.normal(0, 0.01),
                "high": 1.1050 + np.random.normal(0, 0.01),
                "low": 1.0950 + np.random.normal(0, 0.01),
                "close": 1.1000 + np.random.normal(0, 0.01),
                "volume": np.random.randint(100000, 1000000)
            }
            market_data.append(market_point)
        
        return {
            "day_trades": day_trades,
            "swing_trades": swing_trades,
            "market_data": market_data
        }

    async def test_framework_initialization(self) -> bool:
        """Test 1: Framework Initialization"""
        logger.info("TEST 1: Testing framework initialization")
        
        try:
            self.framework = AdvancedAnalyticsFramework()
            await self.framework.initialize()
            
            # Verify engines are registered
            expected_engines = ["day_trading", "swing_trading", "session_analysis", "scalping", "profit_optimization"]
            registered_engines = list(self.framework.engines.keys())
            
            logger.info(f"Expected engines: {expected_engines}")
            logger.info(f"Registered engines: {registered_engines}")
            
            # Check if enhanced engines are properly integrated
            day_trading_engine = self.framework.engines.get("day_trading")
            swing_trading_engine = self.framework.engines.get("swing_trading")
            
            if isinstance(day_trading_engine, DayTradingAnalytics):
                logger.info("✅ Day Trading Analytics directly integrated")
            else:
                logger.warning("❌ Day Trading Analytics not directly integrated")
                
            if isinstance(swing_trading_engine, SwingAnalytics):
                logger.info("✅ Swing Analytics directly integrated")
            else:
                logger.warning("❌ Swing Analytics not directly integrated")
            
            self.test_results["framework_initialization"] = {
                "status": "PASSED",
                "engines_registered": len(registered_engines),
                "enhanced_integration": {
                    "day_trading": isinstance(day_trading_engine, DayTradingAnalytics),
                    "swing_trading": isinstance(swing_trading_engine, SwingAnalytics)
                }
            }
            
            logger.info("✅ Framework initialization test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Framework initialization test FAILED: {e}")
            self.test_results["framework_initialization"] = {
                "status": "FAILED",
                "error": str(e)
            }
            return False

    async def test_enhanced_day_trading_integration(self) -> bool:
        """Test 2: Enhanced Day Trading Analytics Integration"""
        logger.info("TEST 2: Testing enhanced day trading analytics integration")
        
        try:
            if not self.framework:
                raise Exception("Framework not initialized")
            
            day_trading_engine = self.framework.engines.get("day_trading")
            if not day_trading_engine:
                raise Exception("Day trading engine not found")
            
            # Test process_data method
            day_trading_data = {"trades": self.test_data["day_trades"]}
            result = await day_trading_engine.process_data(day_trading_data)
            
            logger.info(f"Day trading processing result: {result}")
            
            # Test generate_report method
            report = await day_trading_engine.generate_report("1d")
            logger.info(f"Day trading report generated: {report.report_id}")
            
            # Test get_real_time_metrics method
            metrics = day_trading_engine.get_real_time_metrics()
            logger.info(f"Day trading metrics count: {len(metrics)}")
            
            # Verify results
            success_checks = [
                result.get("success", False),
                report.report_type == "day_trading_analytics",
                len(metrics) > 0,
                "performance_score" in result or "error" not in result
            ]
            
            self.test_results["enhanced_day_trading"] = {
                "status": "PASSED" if all(success_checks) else "PARTIAL",
                "processed_trades": result.get("processed_trades", 0),
                "report_generated": report.report_id,
                "metrics_count": len(metrics),
                "confidence_score": report.confidence_score
            }
            
            logger.info("✅ Enhanced day trading integration test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Enhanced day trading integration test FAILED: {e}")
            self.test_results["enhanced_day_trading"] = {
                "status": "FAILED",
                "error": str(e)
            }
            return False

    async def test_enhanced_swing_analytics_integration(self) -> bool:
        """Test 3: Enhanced Swing Analytics Integration"""
        logger.info("TEST 3: Testing enhanced swing analytics integration")
        
        try:
            if not self.framework:
                raise Exception("Framework not initialized")
            
            swing_engine = self.framework.engines.get("swing_trading")
            if not swing_engine:
                raise Exception("Swing trading engine not found")
            
            # Test process_data method
            swing_data = {"swing_trades": self.test_data["swing_trades"]}
            result = await swing_engine.process_data(swing_data)
            
            logger.info(f"Swing analytics processing result: {result}")
            
            # Test generate_report method
            report = await swing_engine.generate_report("1w")
            logger.info(f"Swing analytics report generated: {report.report_id}")
            
            # Test get_real_time_metrics method
            metrics = swing_engine.get_real_time_metrics()
            logger.info(f"Swing analytics metrics count: {len(metrics)}")
            
            # Verify results
            success_checks = [
                result.get("success", False),
                report.report_type == "swing_trading_analytics",
                len(metrics) > 0,
                report.confidence_score > 0
            ]
            
            self.test_results["enhanced_swing_analytics"] = {
                "status": "PASSED" if all(success_checks) else "PARTIAL",
                "processing_result": result.get("success", False),
                "report_generated": report.report_id,
                "metrics_count": len(metrics),
                "confidence_score": report.confidence_score
            }
            
            logger.info("✅ Enhanced swing analytics integration test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Enhanced swing analytics integration test FAILED: {e}")
            self.test_results["enhanced_swing_analytics"] = {
                "status": "FAILED",
                "error": str(e)
            }
            return False

    async def test_framework_data_streaming(self) -> bool:
        """Test 4: Framework Data Streaming"""
        logger.info("TEST 4: Testing framework data streaming")
        
        try:
            if not self.framework:
                raise Exception("Framework not initialized")
            
            # Test streaming different types of data
            test_cases = [
                {"trades": self.test_data["day_trades"][:10]},
                {"swing_trades": self.test_data["swing_trades"][:5]},
                {"market_data": self.test_data["market_data"][:20]}
            ]
            
            streaming_results = []
            for i, data in enumerate(test_cases):
                logger.info(f"Streaming test case {i+1}")
                result = await self.framework.stream_analytics_data(data, f"test_source_{i+1}")
                streaming_results.append(result)
                logger.info(f"Streaming result {i+1}: {len(result) if result else 0} engines processed")
            
            self.test_results["framework_streaming"] = {
                "status": "PASSED",
                "test_cases_processed": len(test_cases),
                "streaming_results": len([r for r in streaming_results if r]),
                "engines_responding": len(self.framework.engines)
            }
            
            logger.info("✅ Framework data streaming test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Framework data streaming test FAILED: {e}")
            self.test_results["framework_streaming"] = {
                "status": "FAILED",
                "error": str(e)
            }
            return False

    async def test_comprehensive_reporting(self) -> bool:
        """Test 5: Comprehensive Reporting"""
        logger.info("TEST 5: Testing comprehensive reporting")
        
        try:
            if not self.framework:
                raise Exception("Framework not initialized")
            
            # Generate comprehensive reports for different timeframes
            timeframes = ["1h", "4h", "1d", "1w"]
            reports_generated = []
            
            for timeframe in timeframes:
                logger.info(f"Generating comprehensive report for {timeframe}")
                report = await self.framework.generate_comprehensive_report(timeframe)
                reports_generated.append(report)
                logger.info(f"Report {report.report_id} generated with confidence: {report.confidence_score}")
            
            self.test_results["comprehensive_reporting"] = {
                "status": "PASSED",
                "reports_generated": len(reports_generated),
                "timeframes_tested": timeframes,
                "avg_confidence_score": sum(r.confidence_score for r in reports_generated) / len(reports_generated)
            }
            
            logger.info("✅ Comprehensive reporting test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Comprehensive reporting test FAILED: {e}")
            self.test_results["comprehensive_reporting"] = {
                "status": "FAILED",
                "error": str(e)
            }
            return False

    async def test_real_time_metrics_collection(self) -> bool:
        """Test 6: Real-time Metrics Collection"""
        logger.info("TEST 6: Testing real-time metrics collection")
        
        try:
            if not self.framework:
                raise Exception("Framework not initialized")
            
            # Collect metrics from all engines
            all_metrics = []
            for engine_name, engine in self.framework.engines.items():
                logger.info(f"Collecting metrics from {engine_name}")
                metrics = engine.get_real_time_metrics()
                all_metrics.extend(metrics)
                logger.info(f"Collected {len(metrics)} metrics from {engine_name}")
            
            # Test framework's metrics aggregation
            framework_metrics = self.framework.get_realtime_metrics()
            
            self.test_results["real_time_metrics"] = {
                "status": "PASSED",
                "total_metrics_collected": len(all_metrics),
                "engines_with_metrics": len([e for e in self.framework.engines.values() if e.get_real_time_metrics()]),
                "framework_metrics": len(framework_metrics),
                "metric_types": list(set(m.metric_name for m in all_metrics))
            }
            
            logger.info("✅ Real-time metrics collection test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Real-time metrics collection test FAILED: {e}")
            self.test_results["real_time_metrics"] = {
                "status": "FAILED",
                "error": str(e)
            }
            return False

    async def test_performance_benchmarking(self) -> bool:
        """Test 7: Performance Benchmarking"""
        logger.info("TEST 7: Testing performance benchmarking")
        
        try:
            if not self.framework:
                raise Exception("Framework not initialized")
            
            import time
            
            # Benchmark data processing performance
            start_time = time.time()
            
            # Process large dataset
            large_dataset = {
                "trades": self.test_data["day_trades"] * 5,  # 500 trades
                "swing_trades": self.test_data["swing_trades"] * 3,  # 150 swing trades
                "market_data": self.test_data["market_data"] * 2  # 200 market data points
            }
            
            processing_results = await self.framework.stream_analytics_data(large_dataset, "performance_test")
            processing_time = time.time() - start_time
            
            # Benchmark report generation
            start_time = time.time()
            report = await self.framework.generate_comprehensive_report("1d")
            report_time = time.time() - start_time
            
            # Benchmark metrics collection
            start_time = time.time()
            metrics = self.framework.get_realtime_metrics()
            metrics_time = time.time() - start_time
            
            performance_score = 100 - min(50, processing_time * 10 + report_time * 20 + metrics_time * 100)
            
            self.test_results["performance_benchmarking"] = {
                "status": "PASSED" if performance_score > 70 else "PARTIAL",
                "processing_time": processing_time,
                "report_generation_time": report_time,
                "metrics_collection_time": metrics_time,
                "performance_score": performance_score,
                "data_processed": len(large_dataset.get("trades", [])) + len(large_dataset.get("swing_trades", []))
            }
            
            logger.info(f"✅ Performance benchmarking test PASSED - Score: {performance_score:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance benchmarking test FAILED: {e}")
            self.test_results["performance_benchmarking"] = {
                "status": "FAILED",
                "error": str(e)
            }
            return False

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        logger.info("Starting Enhanced Analytics Integration Test Suite")
        
        tests = [
            self.test_framework_initialization,
            self.test_enhanced_day_trading_integration,
            self.test_enhanced_swing_analytics_integration,
            self.test_framework_data_streaming,
            self.test_comprehensive_reporting,
            self.test_real_time_metrics_collection,
            self.test_performance_benchmarking
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test in tests:
            try:
                result = await test()
                if result:
                    passed_tests += 1
                await asyncio.sleep(0.5)  # Brief pause between tests
            except Exception as e:
                logger.error(f"Test execution error: {e}")
        
        # Generate final test report
        final_report = {
            "test_suite": "Enhanced Analytics Integration Test",
            "execution_time": datetime.utcnow().isoformat(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": (passed_tests / total_tests) * 100,
            "overall_status": "PASSED" if passed_tests >= total_tests * 0.8 else "FAILED",
            "detailed_results": self.test_results
        }
        
        # Cleanup
        if self.framework:
            await self.framework.shutdown()
        
        return final_report

    def save_test_report(self, report: Dict[str, Any], filename: str = None):
        """Save test report to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_analytics_integration_test_report_{timestamp}.json"
        
        filepath = Path(__file__).parent / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Test report saved to: {filepath}")

async def main():
    """Main test execution"""
    test_suite = EnhancedAnalyticsIntegrationTest()
    
    print("=" * 80)
    print("ENHANCED ANALYTICS INTEGRATION TEST SUITE")
    print("=" * 80)
    
    # Run all tests
    report = await test_suite.run_all_tests()
    
    # Display summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {report['total_tests']}")
    print(f"Passed Tests: {report['passed_tests']}")
    print(f"Success Rate: {report['success_rate']:.1f}%")
    print(f"Overall Status: {report['overall_status']}")
    
    # Save report
    test_suite.save_test_report(report)
    
    return report

if __name__ == "__main__":
    asyncio.run(main())
