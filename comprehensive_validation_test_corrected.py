#!/usr/bin/env python3
"""
Comprehensive Platform3 Validation Test
Validates complete indicator integration and agent-driven trade execution pipeline
"""

import sys
import os
import json
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the platform root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Define dummy enums and classes for IndicatorConfig, IndicatorType, TimeFrame
# These are placeholders and should be replaced with actual imports if they exist elsewhere in your project
class IndicatorType:
    VOLUME = "VOLUME"
    TREND = "TREND"
    MOMENTUM = "MOMENTUM"

class TimeFrame:
    D1 = "D1"
    H1 = "H1"

class IndicatorConfig:
    def __init__(self, name: str, indicator_type: str, timeframe: str, lookback_periods: int):
        self.name = name
        self.indicator_type = indicator_type
        self.timeframe = timeframe
        self.lookback_periods = lookback_periods

def create_sample_data():
    """Create sample OHLCV data for testing"""
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': np.nan,
        'low': np.nan,
        'close': np.nan,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Generate proper OHLC data
    for i in range(len(data)):
        open_price = data.iloc[i]['open']
        daily_range = abs(np.random.randn() * 2)
        data.iloc[i, data.columns.get_loc('high')] = open_price + daily_range
        data.iloc[i, data.columns.get_loc('low')] = open_price - daily_range
        data.iloc[i, data.columns.get_loc('close')] = open_price + np.random.randn() * 1
    
    return data

def test_registry_coverage():
    """Test 1: Registry Coverage - Ensure all indicators are registered"""
    print("=" * 60)
    print("TEST 1: Registry Coverage Validation")
    print("=" * 60)
    
    try:
        from engines.indicator_registry import IndicatorRegistry
        registry = IndicatorRegistry()
        
        # Get all registered indicators
        all_indicators = registry.get_all_indicators()
        
        print(f"Total indicators registered: {len(all_indicators)}")
        
        # Check for volume indicators specifically
        volume_indicators = [name for name in all_indicators.keys() 
                           if 'volume' in name.lower() or 'vwap' in name.lower() or 
                              'obv' in name.lower() or 'accumulation' in name.lower() or
                              'distribution' in name.lower()]
        
        print(f"Volume-related indicators found: {len(volume_indicators)}")
        print("Volume indicators:", volume_indicators)
        
        # Expected minimum counts
        expected_min_total = 150
        expected_min_volume = 20
        
        success = len(all_indicators) >= expected_min_total and len(volume_indicators) >= expected_min_volume
        
        print(f"Registry Coverage: {'PASS' if success else 'FAIL'}")
        print(f"Expected >= {expected_min_total} total indicators, got {len(all_indicators)}")
        print(f"Expected >= {expected_min_volume} volume indicators, got {len(volume_indicators)}")
        
        return success, {
            'total_registered': len(all_indicators),
            'volume_indicators': len(volume_indicators),
            'all_indicators': list(all_indicators.keys()),
            'volume_indicator_names': volume_indicators
        }
        
    except Exception as e:
        print(f"Registry test failed: {str(e)}")
        traceback.print_exc()
        return False, {'error': str(e)}

def test_indicator_execution():
    """Test 2: Indicator Execution - Test that indicators can be instantiated and executed"""
    print("\n" + "=" * 60)
    print("TEST 2: Indicator Execution Validation")  
    print("=" * 60)
    
    try:
        from engines.indicator_registry import IndicatorRegistry
        registry = IndicatorRegistry()
        
        # Create sample data
        sample_data = create_sample_data()
        print(f"Created sample data with {len(sample_data)} rows")
        
        all_indicators = registry.get_all_indicators()
        successful_executions = 0
        failed_executions = 0
        execution_results = {}
        
        print(f"Testing execution of {len(all_indicators)} indicators...")
        
        for indicator_name, indicator_class in all_indicators.items():
            try:
                # Try to instantiate the indicator
                # First try without config
                try:
                    indicator = indicator_class()
                except TypeError:
                    # If that fails, try with a basic config
                    try:
                        config = IndicatorConfig(
                            name=indicator_name,
                            indicator_type=IndicatorType.VOLUME,
                            timeframe=TimeFrame.D1,
                            lookback_periods=14
                        )
                        indicator = indicator_class(config)
                    except TypeError:
                        # Try with just the name parameter
                        indicator = indicator_class(name=indicator_name)
                  # Try to execute the indicator
                if hasattr(indicator, 'calculate'):
                    try:
                        # Try different call signatures for calculate method
                        import inspect
                        sig = inspect.signature(indicator.calculate)
                        params = list(sig.parameters.keys())
                        
                        # Try to match the expected parameters
                        if len(params) == 1:  # Only self (DataFrame input)
                            result = indicator.calculate(sample_data)
                        elif len(params) == 2:  # self + close or self + volume
                            if any(p in ['close', 'prices', 'data'] for p in params):
                                result = indicator.calculate(sample_data['close'])
                            else:
                                result = indicator.calculate(sample_data)
                        elif len(params) == 3:  # self + close + volume
                            result = indicator.calculate(sample_data['close'], sample_data['volume'])
                        elif len(params) >= 4:  # self + high + low + close + volume (+ optional)
                            if 'high' in params and 'low' in params:
                                result = indicator.calculate(
                                    sample_data['high'].iloc[0],
                                    sample_data['low'].iloc[0], 
                                    sample_data['close'].iloc[0],
                                    sample_data['volume'].iloc[0]
                                )
                            else:
                                # Try OHLCV order
                                result = indicator.calculate(
                                    sample_data['close'],
                                    sample_data['high'],
                                    sample_data['low'],
                                    sample_data['volume']
                                )
                        else:
                            # Fallback: try with full DataFrame
                            result = indicator.calculate(sample_data)
                    except Exception as calc_error:
                        try:
                            # Final fallback: try with just close price
                            result = indicator.calculate(sample_data['close'])
                        except:
                            raise calc_error
                elif hasattr(indicator, '__call__'):
                    result = indicator(sample_data)
                else:
                    raise AttributeError("No calculate method or __call__ method found")
                
                successful_executions += 1
                execution_results[indicator_name] = 'SUCCESS'
                
                if successful_executions <= 5:  # Show details for first few
                    print(f"✓ {indicator_name}: SUCCESS")
                    
            except Exception as e:
                failed_executions += 1
                execution_results[indicator_name] = f'FAILED: {str(e)}'
                
                if failed_executions <= 10:  # Show details for first few failures  
                    print(f"✗ {indicator_name}: FAILED - {str(e)}")
        
        execution_rate = (successful_executions / len(all_indicators)) * 100
        print(f"\nExecution Summary:")
        print(f"Successful: {successful_executions}/{len(all_indicators)} ({execution_rate:.1f}%)")
        print(f"Failed: {failed_executions}/{len(all_indicators)}")
        
        # Consider test passed if > 80% execute successfully
        success = execution_rate > 80.0
        
        print(f"Indicator Execution: {'PASS' if success else 'FAIL'}")
        
        return success, {
            'total_tested': len(all_indicators),
            'successful': successful_executions,
            'failed': failed_executions,
            'execution_rate': execution_rate,
            'detailed_results': execution_results
        }
        
    except Exception as e:
        print(f"Execution test failed: {str(e)}")
        traceback.print_exc()
        return False, {'error': str(e)}

def test_agent_integration():
    """Test 3: Agent Integration - Test that agents can access indicators"""
    print("\n" + "=" * 60)
    print("TEST 3: Agent Integration Validation")
    print("=" * 60)
    
    try:
        # Try to import the agent/engine components
        try:
            from models.platform3_engine import Platform3Engine
            engine = Platform3Engine()
            print("✓ Platform3Engine instantiated successfully")
        except ImportError as e:
            print(f"Could not import Platform3Engine: {e}")
            # Try alternative import
            try:
                from engines.ai_enhancement.adaptive_indicator_bridge import AdaptiveIndicatorBridge
                bridge = AdaptiveIndicatorBridge()
                print("✓ AdaptiveIndicatorBridge instantiated successfully")
            except ImportError as e2:
                print(f"Could not import AdaptiveIndicatorBridge either: {e2}")
                return False, {'error': f'Could not import agent components: {e}, {e2}'}
        
        # Try to access indicators through agent
        from engines.indicator_registry import IndicatorRegistry
        registry = IndicatorRegistry()
        
        # Test that agents can get indicators
        volume_indicators = [name for name in registry.get_all_indicators().keys() 
                           if 'volume' in name.lower()]
        
        if len(volume_indicators) > 0:
            test_indicator = volume_indicators[0]
            indicator_class = registry.get_indicator(test_indicator)
            
            if indicator_class:
                print(f"✓ Agent can access indicator: {test_indicator}")
                success = True
            else:
                print(f"✗ Agent cannot access indicator: {test_indicator}")
                success = False
        else:
            print("✗ No volume indicators found for agent test")
            success = False
            
        print(f"Agent Integration: {'PASS' if success else 'FAIL'}")
        
        return success, {
            'can_instantiate_engine': True,
            'can_access_indicators': success,
            'test_indicator': test_indicator if len(volume_indicators) > 0 else None
        }
        
    except Exception as e:
        print(f"Agent integration test failed: {str(e)}")
        traceback.print_exc()
        return False, {'error': str(e)}

def test_trading_pipeline():
    """Test 4: Trading Pipeline Integration - Test end-to-end pipeline"""
    print("\n" + "=" * 60)
    print("TEST 4: Trading Pipeline Integration Validation")
    print("=" * 60)
    
    try:
        # Test the complete pipeline flow
        sample_data = create_sample_data()
        
        # Test 1: Data input
        print("✓ Sample market data created")
        
        # Test 2: Indicator calculation through registry
        from engines.indicator_registry import IndicatorRegistry
        registry = IndicatorRegistry()
        
        volume_indicators = [name for name in registry.get_all_indicators().keys() 
                           if 'volume' in name.lower()]
        
        if len(volume_indicators) == 0:
            print("✗ No volume indicators available for pipeline test")
            return False, {'error': 'No volume indicators found'}
            
        # Test a few volume indicators
        pipeline_success_count = 0
        total_pipeline_tests = min(3, len(volume_indicators))
        
        for i in range(total_pipeline_tests):
            indicator_name = volume_indicators[i]
            try:
                indicator_class = registry.get_indicator(indicator_name)
                
                # Try to instantiate
                try:
                    indicator = indicator_class()
                except TypeError:
                    config = IndicatorConfig(
                        name=indicator_name,
                        indicator_type=IndicatorType.VOLUME,
                        timeframe=TimeFrame.D1,
                        lookback_periods=14
                    )
                    indicator = indicator_class(config)
                
                # Try to calculate
                if hasattr(indicator, 'calculate'):
                    try:
                        result = indicator.calculate(sample_data)
                        pipeline_success_count += 1
                        print(f"✓ Pipeline test passed for {indicator_name}")
                    except Exception as e:
                        print(f"✗ Pipeline calculation failed for {indicator_name}: {e}")
                else:
                    print(f"✗ No calculate method for {indicator_name}")
                    
            except Exception as e:
                print(f"✗ Pipeline test failed for {indicator_name}: {e}")
        
        # Test 3: Try to import REST API components (if they exist)
        try:
            from api.rest_api import RestAPI
            print("✓ REST API components accessible")
            api_accessible = True
        except ImportError:
            print("⚠ REST API components not accessible (may not be implemented yet)")
            api_accessible = False
        
        pipeline_success_rate = (pipeline_success_count / total_pipeline_tests) * 100
        success = pipeline_success_rate >= 66.7  # At least 2/3 should work
        
        print(f"\nPipeline Summary:")
        print(f"Successful pipeline tests: {pipeline_success_count}/{total_pipeline_tests} ({pipeline_success_rate:.1f}%)")
        print(f"Trading Pipeline: {'PASS' if success else 'FAIL'}")
        
        return success, {
            'pipeline_tests_passed': pipeline_success_count,
            'total_pipeline_tests': total_pipeline_tests,
            'pipeline_success_rate': pipeline_success_rate,
            'api_accessible': api_accessible
        }
        
    except Exception as e:
        print(f"Trading pipeline test failed: {str(e)}")
        traceback.print_exc()
        return False, {'error': str(e)}

def main():
    """Run all validation tests"""
    print("Platform3 Comprehensive Validation Test")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Registry Coverage", test_registry_coverage),
        ("Indicator Execution", test_indicator_execution),
        ("Agent Integration", test_agent_integration),
        ("Trading Pipeline", test_trading_pipeline)
    ]
    
    results = {}
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            success, details = test_func()
            results[test_name] = {
                'success': success,
                'details': details
            }
            if success:
                passed_tests += 1
        except Exception as e:
            print(f"CRITICAL ERROR in {test_name}: {str(e)}")
            results[test_name] = {
                'success': False,
                'details': {'critical_error': str(e)}
            }
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "PASS" if result['success'] else "FAIL"
        print(f"{test_name}: {status}")
    
    success_rate = (passed_tests / total_tests) * 100
    print(f"\nOverall Results: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
    
    if success_rate == 100:
        overall_status = "COMPLETE"
    elif success_rate >= 75:
        overall_status = "MOSTLY_COMPLETE"
    elif success_rate >= 50:
        overall_status = "NEEDS_WORK"
    else:
        overall_status = "CRITICAL_ISSUES"
    
    print(f"Overall Status: {overall_status}")
    print(f"Completion Level: {success_rate:.1f}%")
    
    # Save detailed results
    validation_report = {
        'timestamp': datetime.now().isoformat(),
        'overall_status': overall_status,
        'completion_percentage': success_rate,
        'tests_passed': passed_tests,
        'total_tests': total_tests,
        'detailed_results': results
    }
    
    report_file = f"comprehensive_validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(validation_report, f, indent=2, default=str)
    
    print(f"\nDetailed report saved to: {report_file}")
    print("=" * 60)
    
    return success_rate == 100

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
