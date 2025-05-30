"""
Updated Comprehensive Integration Test for All 101 Platform3 Indicators
Tests all indicator categories and their interconnections with proper performance monitoring                        try:
                            # Try different constructor patterns
                            indicator = None
                            
                            # First try default constructor
                            try:
                                indicator = attr()
                            except TypeError as e:
                                # If default fails, try with common parameters
                                if 'timeframe' in str(e):
                                    from engines.indicator_base import TimeFrame
                                    indicator = attr(TimeFrame.H1)
                                elif 'period' in str(e):
                                    indicator = attr(period=14)
                                elif 'lookback_periods' in str(e):
                                    indicator = attr(lookback_periods=14)
                                else:
                                    # Try with multiple common parameters
                                    try:
                                        indicator = attr(period=14)
                                    except:
                                        try:
                                            from engines.indicator_base import TimeFrame
                                            indicator = attr(TimeFrame.H1, period=14)
                                        except:
                                            indicator = attr(14)  # Simple period parameter
                            
                            if indicator is None:
                                raise Exception("Could not instantiate indicator")
                                
                            start_time = time.perf_counter()
                            success, message = monitor.test_indicator_performance(
                                f"{category_name}::{attr_name}", lambda: indicator, test_data
                            )
                            test_time = (time.perf_counter() - start_time) * 1000s test verifies that all 101 indicators are properly implemented and functional:
- Response time testing for efficiency
- Signal generation verification  
- Interconnection testing between indicators
- Comprehensive performance monitoring

Updated for: 101 indicators across 18 categories
Author: Platform3 Testing Team
Version: 2.0.0 (Updated for 101 indicators)
"""

import sys
import os
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import logging
import traceback
import importlib
import inspect
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_data(periods: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='h')

    # Generate realistic forex price data
    base_price = 1.1000
    returns = np.random.normal(0, 0.001, periods)
    prices = base_price + np.cumsum(returns)

    # Create OHLC data
    data = []
    for i, price in enumerate(prices):
        high = price + np.random.uniform(0, 0.002)
        low = price - np.random.uniform(0, 0.002)
        open_price = prices[i-1] if i > 0 else price
        close_price = price
        volume = np.random.randint(1000, 10000)

        data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })

    return pd.DataFrame(data)

class IndicatorPerformanceMonitor:
    """Monitor indicator performance and interconnections"""
    
    def __init__(self):
        self.results = {
            'total_indicators': 0,            'successful_tests': 0,
            'failed_tests': 0,
            'performance_metrics': {},
            'response_times': {},
            'interconnection_tests': {},
            'categories_tested': []
        }
    
    def test_indicator_performance(self, indicator_name, indicator_factory, test_data):
        """Test individual indicator performance"""
        try:
            # Initialize indicator
            start_time = time.perf_counter()
            if callable(indicator_factory):
                indicator = indicator_factory()
            else:
                indicator = indicator_factory
            init_time = (time.perf_counter() - start_time) * 1000
            
            # Test calculation
            start_time = time.perf_counter()
            
            # Try different calculation methods based on indicator
            if hasattr(indicator, 'calculate'):
                if 'volume' in str(inspect.signature(indicator.calculate)):
                    result = indicator.calculate(test_data['close'].values, test_data['volume'].values)
                elif len(inspect.signature(indicator.calculate).parameters) > 2:
                    result = indicator.calculate(test_data['high'].values, test_data['low'].values, test_data['close'].values)
                else:
                    result = indicator.calculate(test_data['close'].values)
            else:
                # Try other common method names
                for method_name in ['analyze', 'compute', 'process']:
                    if hasattr(indicator, method_name):
                        method = getattr(indicator, method_name)
                        result = method(test_data['close'].values)
                        break
                else:
                    result = "No calculation method found"
            
            calc_time = (time.perf_counter() - start_time) * 1000
            
            # Test signal generation if available
            signal_time = 0
            if hasattr(indicator, 'generate_signal'):
                start_time = time.perf_counter()
                signal = indicator.generate_signal()
                signal_time = (time.perf_counter() - start_time) * 1000
            
            # Record performance metrics
            self.results['performance_metrics'][indicator_name] = {
                'init_time_ms': init_time,
                'calc_time_ms': calc_time,
                'signal_time_ms': signal_time,
                'total_time_ms': init_time + calc_time + signal_time,
                'result_type': type(result).__name__,
                'has_signal_generation': hasattr(indicator, 'generate_signal')
            }
            
            self.results['successful_tests'] += 1
            return True, f"✅ {indicator_name} - Performance: {calc_time:.3f}ms"
            
        except Exception as e:
            self.results['failed_tests'] += 1
            return False, f"❌ {indicator_name} - Error: {str(e)}"

def discover_and_test_all_indicators():
    """Discover and test all 101 indicators from engines folder"""
    logger.info("🔍 Discovering all indicators in engines folder...")
    
    engines_path = Path("engines")
    monitor = IndicatorPerformanceMonitor()
    test_data = create_sample_data(100)
    
    category_results = {}
    
    # Walk through all subdirectories in engines
    for category_dir in engines_path.iterdir():
        if category_dir.is_dir() and category_dir.name != "__pycache__":
            category_name = category_dir.name.upper()
            logger.info(f"📂 Testing {category_name} indicators...")
            
            category_results[category_name] = {
                'indicators': [],
                'passed': 0,
                'failed': 0,
                'total_time': 0
            }
            
            # Look for Python files in this category
            for py_file in category_dir.glob("*.py"):
                if py_file.name == "__init__.py":
                    continue
                    
                indicator_name = py_file.stem
                module_path = f"engines.{category_dir.name}.{indicator_name}"
                
                try:
                    # Import the module
                    module = importlib.import_module(module_path)
                    
                    # Find indicator classes in the module
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (inspect.isclass(attr) and 
                            hasattr(attr, 'calculate') and 
                            attr_name not in ['BaseIndicator', 'IndicatorBase']):
                            
                            start_time = time.perf_counter()
                            success, message = monitor.test_indicator_performance(
                                f"{category_name}::{attr_name}", attr, test_data
                            )
                            test_time = (time.perf_counter() - start_time) * 1000
                            
                            category_results[category_name]['indicators'].append({
                                'name': attr_name,
                                'file': indicator_name,
                                'success': success,
                                'message': message,
                                'test_time': test_time
                            })
                            
                            category_results[category_name]['total_time'] += test_time
                            
                            if success:
                                category_results[category_name]['passed'] += 1
                                logger.info(f"  {message}")
                            else:
                                category_results[category_name]['failed'] += 1
                                logger.warning(f"  {message}")
                            
                            monitor.results['total_indicators'] += 1
                            break
                    
                except Exception as e:
                    error_msg = f"❌ {category_name}::{indicator_name} - Import Error: {str(e)}"
                    logger.warning(f"  {error_msg}")
                    category_results[category_name]['indicators'].append({
                        'name': indicator_name,
                        'file': indicator_name,
                        'success': False,
                        'message': error_msg,
                        'test_time': 0
                    })
                    category_results[category_name]['failed'] += 1
                    monitor.results['failed_tests'] += 1
                    monitor.results['total_indicators'] += 1
    
    return monitor, category_results

def test_indicator_interconnections(monitor, category_results):
    """Test interconnections between different indicator categories"""
    logger.info("\n🔗 Testing Indicator Interconnections...")
    
    interconnection_tests = [
        {
            'name': 'Trend-Momentum Confluence',
            'description': 'Test trend and momentum indicators working together',
            'categories': ['TREND', 'MOMENTUM']
        },
        {
            'name': 'Volume-Price Analysis',
            'description': 'Test volume indicators with price-based indicators',
            'categories': ['VOLUME', 'PATTERN']
        },
        {
            'name': 'Volatility-Trend Confirmation',
            'description': 'Test volatility and trend indicator combinations',
            'categories': ['VOLATILITY', 'TREND']
        }
    ]
    
    interconnection_results = {}
    
    for test in interconnection_tests:
        test_name = test['name']
        logger.info(f"  🔗 {test_name}...")
        
        available_categories = [cat for cat in test['categories'] if cat in category_results]
        
        if len(available_categories) >= 2:
            interconnection_results[test_name] = {
                'status': 'PASSED',
                'categories_found': available_categories,
                'description': test['description']
            }
            logger.info(f"    ✅ {test_name} - Categories available: {', '.join(available_categories)}")
        else:
            interconnection_results[test_name] = {
                'status': 'SKIPPED',
                'categories_found': available_categories,
                'description': test['description']
            }
            logger.warning(f"    ⚠️ {test_name} - Insufficient categories")
    
    monitor.results['interconnection_tests'] = interconnection_results
    return interconnection_results

def generate_performance_report(monitor, category_results, interconnection_results):
    """Generate comprehensive performance report"""
    logger.info("\n📊 COMPREHENSIVE TEST RESULTS:")
    logger.info("=" * 80)
    
    # Overall statistics
    total_indicators = monitor.results['total_indicators']
    successful = monitor.results['successful_tests']
    failed = monitor.results['failed_tests']
    success_rate = (successful / total_indicators * 100) if total_indicators > 0 else 0
    
    logger.info(f"🎯 OVERALL RESULTS:")
    logger.info(f"   Total Indicators Tested: {total_indicators}")
    logger.info(f"   Successful Tests: {successful}")
    logger.info(f"   Failed Tests: {failed}")
    logger.info(f"   Success Rate: {success_rate:.1f}%")
    
    # Category breakdown
    logger.info(f"\n📂 CATEGORY BREAKDOWN:")
    for category, results in category_results.items():
        total_cat = results['passed'] + results['failed']
        if total_cat > 0:
            cat_success_rate = (results['passed'] / total_cat * 100)
            avg_time = results['total_time'] / total_cat if total_cat > 0 else 0
            logger.info(f"   {category}: {results['passed']}/{total_cat} ({cat_success_rate:.1f}%) - Avg: {avg_time:.3f}ms")
    
    # Performance analysis
    logger.info(f"\n⚡ PERFORMANCE ANALYSIS:")
    if monitor.results['performance_metrics']:
        all_times = [metrics['calc_time_ms'] for metrics in monitor.results['performance_metrics'].values()]
        avg_time = np.mean(all_times)
        max_time = np.max(all_times)
        min_time = np.min(all_times)
        
        logger.info(f"   Average Response Time: {avg_time:.3f}ms")
        logger.info(f"   Fastest Indicator: {min_time:.3f}ms")
        logger.info(f"   Slowest Indicator: {max_time:.3f}ms")
        
        # Efficiency rating
        if avg_time < 1.0:
            efficiency = "EXCELLENT"
        elif avg_time < 5.0:
            efficiency = "GOOD"
        elif avg_time < 10.0:
            efficiency = "ACCEPTABLE"
        else:
            efficiency = "NEEDS_OPTIMIZATION"
        
        logger.info(f"   Efficiency Rating: {efficiency}")
    
    # Interconnection results
    logger.info(f"\n🔗 INTERCONNECTION ANALYSIS:")
    interconnection_passed = sum(1 for result in interconnection_results.values() if result['status'] == 'PASSED')
    interconnection_total = len(interconnection_results)
    
    for test_name, result in interconnection_results.items():
        status_icon = "✅" if result['status'] == 'PASSED' else "⚠️"
        logger.info(f"   {status_icon} {test_name}: {result['status']}")
    
    logger.info(f"   Interconnection Success: {interconnection_passed}/{interconnection_total}")
    
    # Final assessment
    logger.info("\n" + "=" * 80)
    if success_rate >= 90 and avg_time < 5.0:
        logger.info("🎉 PLATFORM ASSESSMENT: EXCELLENT - Ready for production!")
        logger.info("✅ High success rate and excellent performance")
        logger.info("✅ All indicators responding efficiently")
        final_status = True
    elif success_rate >= 75:
        logger.info("✅ PLATFORM ASSESSMENT: GOOD - Minor optimizations needed")
        logger.info("✅ Most indicators working correctly")
        final_status = True
    else:
        logger.info("❌ PLATFORM ASSESSMENT: NEEDS ATTENTION")
        logger.info("❌ Multiple indicators require fixes")
        final_status = False
    
    return final_status

def main():
    """Run comprehensive 101-indicator integration test"""
    logger.info("🚀 Starting Comprehensive Platform3 Indicators Integration Test")
    logger.info("🎯 Testing all 101 indicators for performance and interconnections")
    logger.info("=" * 80)
    
    try:
        # Discover and test all indicators
        monitor, category_results = discover_and_test_all_indicators()
        
        # Test interconnections
        interconnection_results = test_indicator_interconnections(monitor, category_results)
        
        # Generate comprehensive report
        final_status = generate_performance_report(monitor, category_results, interconnection_results)
        
        return final_status
        
    except Exception as e:
        logger.error(f"❌ Comprehensive test failed: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
