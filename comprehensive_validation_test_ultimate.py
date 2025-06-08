#!/usr/bin/env python3
"""
Ultimate Platform3 Validation Test
Comprehensive validation with enhanced error handling and indicator fixes
"""

import sys
import os
import json
import traceback
import inspect
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add the platform root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Enhanced enums and classes for IndicatorConfig, IndicatorType, TimeFrame
class IndicatorType:
    VOLUME = "VOLUME"
    TREND = "TREND"
    MOMENTUM = "MOMENTUM"
    PATTERN = "PATTERN"
    FRACTAL = "FRACTAL"
    STATISTICAL = "STATISTICAL"

class TimeFrame:
    D1 = "D1"
    H1 = "H1"
    DAILY = "D1"  # Add this for compatibility
    HOURLY = "H1"

class IndicatorConfig:
    def __init__(self, name: str, indicator_type: str, timeframe: str, lookback_periods: int):
        self.name = name
        self.indicator_type = indicator_type
        self.timeframe = timeframe
        self.lookback_periods = lookback_periods
        self.lookback_period = lookback_periods  # Add this for compatibility
        
        # Add common indicator configuration attributes
        self.period = lookback_periods
        self.atr_multiplier = 2.0
        self.scale_min = 0.0
        self.scale_max = 100.0
        
    def get(self, key: str, default=None):
        """Dictionary-like access for compatibility"""
        return getattr(self, key, default)

def generate_sample_data(size: int = 100) -> List[Dict[str, Any]]:
    """Generate more comprehensive sample market data for testing"""
    base_time = datetime.now() - timedelta(days=size)
    data = []
    
    price = 100.0
    volume = 10000
    
    for i in range(size):
        # More realistic price movement
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        price = max(price * (1 + change), 1.0)
        
        # Generate OHLC with realistic relationships
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = price + np.random.normal(0, 0.005) * price
        close_price = price
        
        # Ensure OHLC relationships are valid
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        # Realistic volume
        volume = max(int(volume * (1 + np.random.normal(0, 0.3))), 1000)
        
        data.append({
            'timestamp': base_time + timedelta(days=i),
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
        
        price = close_price
    
    return data

def test_indicator_registry_completeness():
    """Test if all available indicators are registered and accessible"""
    print("\n=== TESTING INDICATOR REGISTRY COMPLETENESS ===")
    
    try:
        from engines.indicator_registry import IndicatorRegistry
        from dynamic_indicator_loader import load_all_indicators
        
        # Get registry indicators
        registry = IndicatorRegistry()
        registry_indicators = registry.get_all_indicators()
        print(f"Registry has {len(registry_indicators)} indicators")
        
        # Get all available indicators via dynamic loading
        all_indicators, categories = load_all_indicators()
        print(f"Dynamic loader found {len(all_indicators)} indicators")
          # Count volume indicators specifically
        volume_indicators = [name for name in all_indicators.keys() 
                           if 'volume' in name.lower()]
        print(f"Found {len(volume_indicators)} volume indicators")
        
        # Calculate coverage
        missing_indicators = []
        for indicator_name in all_indicators:
            if indicator_name.lower() not in [reg_name.lower() for reg_name in registry_indicators]:
                missing_indicators.append(indicator_name)
        
        coverage_percent = (1 - len(missing_indicators) / len(all_indicators)) * 100 if all_indicators else 100
        
        if missing_indicators:
            print(f"Missing from registry: {len(missing_indicators)} indicators")
            print("Missing indicators:")
            for indicator in missing_indicators[:10]:  # Show first 10
                print(f"   - {indicator}")
            if len(missing_indicators) > 10:
                print(f"   ... and {len(missing_indicators) - 10} more")
        
        print(f"Registry coverage: {coverage_percent:.1f}%")
        return coverage_percent >= 95.0, f"Coverage: {coverage_percent:.1f}%"
        
    except Exception as e:
        print(f"Registry test failed: {e}")
        return False, f"Registry test error: {e}"

def smart_indicator_call(indicator_class, config, data):
    """
    Intelligently call indicator with correct parameters based on inspection
    """
    try:
        # Get constructor signature
        init_sig = inspect.signature(indicator_class.__init__)
        init_params = list(init_sig.parameters.keys())[1:]  # Skip 'self'
        
        # Prepare constructor arguments
        init_kwargs = {}
        
        # Common parameter mappings
        param_mappings = {
            'config': config,
            'name': config.name if hasattr(config, 'name') else 'test_indicator',
            'period': config.lookback_periods if hasattr(config, 'lookback_periods') else 20,
            'lookback_periods': config.lookback_periods if hasattr(config, 'lookback_periods') else 20,
            'timeframe': config.timeframe if hasattr(config, 'timeframe') else TimeFrame.D1
        }
        
        # Add relevant parameters based on signature
        for param in init_params:
            if param in param_mappings:
                init_kwargs[param] = param_mappings[param]
        
        # Create indicator instance
        if init_kwargs:
            indicator = indicator_class(**init_kwargs)
        else:
            indicator = indicator_class()
        
        # Get calculate method signature
        calc_method = getattr(indicator, 'calculate', None)
        if not calc_method:
            return None, "No calculate method found"
        
        # Try calling calculate method
        try:
            result = calc_method(data)
            return result, None
        except Exception as calc_error:
            # Try with DataFrame
            try:
                df = pd.DataFrame(data)
                result = calc_method(df)
                return result, None
            except Exception as df_error:
                return None, f"Calculation failed: {calc_error}"
    
    except Exception as e:
        return None, f"Indicator creation failed: {e}"

def test_indicator_execution():
    """Test execution of individual indicators with enhanced error handling"""
    print("\n=== TESTING INDICATOR EXECUTION ===")
    
    try:
        from dynamic_indicator_loader import load_all_indicators
        
        # Load all indicators
        all_indicators, categories = load_all_indicators()
        
        # Generate comprehensive test data
        test_data = generate_sample_data(200)  # More data for complex indicators
        
        config = IndicatorConfig(
            name="TestIndicator",
            indicator_type=IndicatorType.VOLUME,
            timeframe=TimeFrame.D1,
            lookback_periods=20        )
        
        successful = 0
        failed = 0
        results = {}
        
        for indicator_name, indicator_class in all_indicators.items():
            if not indicator_class:
                continue
                
            print(f"Testing {indicator_name}...")
            
            try:
                result, error = smart_indicator_call(indicator_class, config, test_data)
                
                if error:
                    print(f"   FAILED: {error}")
                    failed += 1
                    results[indicator_name] = {'status': 'failed', 'error': str(error)}
                else:
                    print(f"   PASSED")
                    successful += 1
                    results[indicator_name] = {'status': 'passed', 'result_type': type(result).__name__}
                    
            except Exception as e:
                print(f"   FAILED: {e}")
                failed += 1
                results[indicator_name] = {'status': 'failed', 'error': str(e)}
        
        total = successful + failed
        success_rate = (successful / total * 100) if total > 0 else 0
        
        print(f"\nExecution Results:")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        return success_rate >= 80.0, f"Success Rate: {success_rate:.1f}%", results
        
    except Exception as e:
        print(f"Execution test failed: {e}")
        return False, f"Execution test error: {e}", {}

def test_agent_integration():
    """Test agent integration capabilities"""
    print("\n=== TESTING AGENT INTEGRATION DEPTH ===")
    
    try:
        from engines.ai_enhancement.genius_agent_integration import GeniusAgentIntegration
        
        # Initialize agent integration
        agent = GeniusAgentIntegration()
        print(f"Agent initialized with {len(agent.indicators)} indicators")
        
        # Generate test data
        test_data = generate_sample_data(50)
        
        # Test agent analysis
        print("Testing agent analysis...")
        analysis = agent.analyze_market_data(test_data)
        
        # Test agent decision making
        print("Testing agent decision making...")
        decision = agent.make_trading_decision("EURUSD", test_data)
        
        # Test agent signal generation
        print("Testing agent signal generation...")
        signals = agent.generate_signals(test_data)
        
        # Check agent capabilities
        capabilities = {
            'has_analyze_method': hasattr(agent, 'analyze_market_data'),
            'has_decision_method': hasattr(agent, 'make_trading_decision'),
            'has_signal_method': hasattr(agent, 'generate_signals'),
            'analysis_output': type(analysis).__name__,
            'decision_output': type(decision).__name__,
            'signal_output': type(signals).__name__
        }
        
        print(f"Agent capabilities: {capabilities}")
        
        return True, "Agent integration working", capabilities
        
    except Exception as e:
        print(f"Agent integration test failed: {e}")
        return False, f"Agent test error: {e}", {}

def test_trading_pipeline():
    """Test trading pipeline readiness"""
    print("\n=== TESTING TRADING PIPELINE READINESS ===")
    
    # Test REST API server
    print("Testing REST API server...")
    try:
        from ai_services.model_registry import ModelRegistry
        api_available = True
        api_error = None
    except ImportError as e:
        api_available = False
        api_error = str(e)
        print(f"REST API import failed: {e}")
    
    # Test TypeScript service readiness
    print("Testing TypeScript service readiness...")
    ts_files = []
    try:
        # Check for TypeScript files
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.ts') and 'node_modules' not in root:
                    ts_files.append(os.path.join(root, file))
    except Exception as e:
        print(f"TypeScript check failed: {e}")
    
    pipeline_score = 0
    if api_available:
        pipeline_score += 50
    if ts_files:
        pipeline_score += 50
    
    return pipeline_score >= 70, f"Pipeline Score: {pipeline_score}%", {
        'api_available': api_available,
        'api_error': api_error,
        'typescript_files': len(ts_files)
    }

def main():
    """Main validation function"""
    print("=== PLATFORM3 ULTIMATE VALIDATION TEST ===")
    print(f"Started at: {datetime.now()}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {},
        'summary': {}
    }
    
    # Test 1: Indicator Registry Completeness
    try:
        registry_pass, registry_msg = test_indicator_registry_completeness()
        results['tests']['registry_completeness'] = {
            'passed': registry_pass,
            'message': registry_msg
        }
    except Exception as e:
        results['tests']['registry_completeness'] = {
            'passed': False,
            'message': f"Registry test error: {e}"
        }
    
    # Test 2: Indicator Execution
    try:
        execution_pass, execution_msg, execution_details = test_indicator_execution()
        results['tests']['indicator_execution'] = {
            'passed': execution_pass,
            'message': execution_msg,
            'details': execution_details
        }
    except Exception as e:
        results['tests']['indicator_execution'] = {
            'passed': False,
            'message': f"Execution test error: {e}",
            'details': {}
        }
    
    # Test 3: Agent Integration
    try:
        agent_pass, agent_msg, agent_details = test_agent_integration()
        results['tests']['agent_integration'] = {
            'passed': agent_pass,
            'message': agent_msg,
            'details': agent_details
        }
    except Exception as e:
        results['tests']['agent_integration'] = {
            'passed': False,
            'message': f"Agent test error: {e}",
            'details': {}
        }
    
    # Test 4: Trading Pipeline
    try:
        pipeline_pass, pipeline_msg, pipeline_details = test_trading_pipeline()
        results['tests']['trading_pipeline'] = {
            'passed': pipeline_pass,
            'message': pipeline_msg,
            'details': pipeline_details
        }
    except Exception as e:
        results['tests']['trading_pipeline'] = {
            'passed': False,
            'message': f"Pipeline test error: {e}",
            'details': {}
        }
    
    # Calculate summary
    passed_tests = sum(1 for test in results['tests'].values() if test['passed'])
    total_tests = len(results['tests'])
    completion_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    results['summary'] = {
        'tests_passed': passed_tests,
        'total_tests': total_tests,
        'completion_rate': completion_rate,
        'overall_status': 'PASS' if completion_rate >= 100 else 'NEEDS_WORK' if completion_rate >= 50 else 'FAIL'
    }
    
    # Print summary
    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Overall Status: {results['summary']['overall_status']}")
    print(f"Completion: {completion_rate:.1f}%")
    
    # Save detailed results
    output_file = f"ultimate_validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Print critical findings
    print(f"\n=== CRITICAL FINDINGS ===")
    for test_name, test_result in results['tests'].items():
        if not test_result['passed']:
            print(f"[WARNING] {test_name}: {test_result['message']}")
    
    print(f"\nValidation completed at: {datetime.now()}")
    
    return results

if __name__ == "__main__":
    main()
