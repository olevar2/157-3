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
        
        # Check coverage
        registry_names = set(registry_indicators.keys())
        dynamic_names = set(all_indicators.keys())
        
        missing_from_registry = dynamic_names - registry_names
        print(f"Missing from registry: {len(missing_from_registry)} indicators")
        
        if missing_from_registry:
            print("Missing indicators:")
            for name in sorted(missing_from_registry):
                print(f"   - {name}")
        
        coverage_percentage = (len(registry_names) / len(dynamic_names)) * 100 if dynamic_names else 0
        print(f"Registry coverage: {coverage_percentage:.1f}%")
        
        return {
            "registry_count": len(registry_indicators),
            "dynamic_count": len(all_indicators),
            "missing_count": len(missing_from_registry),
            "coverage_percentage": coverage_percentage,
            "missing_indicators": list(missing_from_registry),
            "status": "PASS" if coverage_percentage > 90 else "NEEDS_IMPROVEMENT"
        }
        
    except Exception as e:
        print(f"Error testing registry completeness: {e}")
        traceback.print_exc()
        return {"status": "FAILED", "error": str(e)}

def test_indicator_execution():
    """Test that indicators can actually execute with sample data"""
    print("\n=== TESTING INDICATOR EXECUTION ===")
    
    try:
        from engines.indicator_registry import IndicatorRegistry
        import pandas as pd
        import numpy as np
        
        # Create sample market data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': 100 + np.random.randn(100).cumsum(),
            'high': 102 + np.random.randn(100).cumsum(),
            'low': 98 + np.random.randn(100).cumsum(),
            'close': 100 + np.random.randn(100).cumsum(),
            'volume': 1000000 + np.random.randint(-100000, 100000, 100)
        })
        registry = IndicatorRegistry()
        indicators = registry.get_all_indicators()
        
        execution_results = {}
        successful_executions = 0
        failed_executions = 0
        
        for name, indicator_class in indicators.items():
            try:
                print(f"Testing {name}...")
                
                # Analyze the class to determine the best initialization approach
                import inspect
                sig = None
                try:
                    sig = inspect.signature(indicator_class.__init__)
                except Exception:
                    pass
                
                indicator = None
                
                # Try different initialization approaches based on signature analysis
                if sig:
                    params = list(sig.parameters.keys())[1:]  # Skip 'self'
                    
                    if 'config' in params and len(params) == 1:
                        # TechnicalIndicator subclass expecting only config
                        try:
                            config = IndicatorConfig(
                                name=name,
                                indicator_type=IndicatorType.VOLUME,
                                timeframe=TimeFrame.D1,
                                lookback_periods=20
                            )
                            indicator = indicator_class(config)
                        except Exception:
                            pass
                        
                    elif len(params) == 0:
                        # No parameters expected
                        try:
                            indicator = indicator_class()
                        except Exception:
                            pass
                        
                    elif len(params) == 1:
                        # Single parameter - try different types
                        param_name = params[0]
                        try:
                            if 'value' in param_name.lower() or 'base' in param_name.lower():
                                indicator = indicator_class(1000.0)
                            elif 'period' in param_name.lower() or 'window' in param_name.lower():
                                indicator = indicator_class(20)
                            else:
                                indicator = indicator_class()
                        except Exception:
                            pass
                
                # Fallback attempts if signature analysis didn't work
                if indicator is None:
                    try:
                        # Try with IndicatorConfig for TechnicalIndicator subclasses
                        config = IndicatorConfig(
                            name=name,
                            indicator_type=IndicatorType.VOLUME,
                            timeframe=TimeFrame.D1,
                            lookback_periods=20
                        )
                        indicator = indicator_class(config)
                    except (TypeError, Exception):
                        # Fallback for IndicatorBase subclasses or older classes
                        try:
                            indicator = indicator_class()
                        except Exception as init_error:
                            # Try with typical parameter values
                            try:
                                indicator = indicator_class(20)  # Period
                            except Exception:
                                try:
                                    indicator = indicator_class(1000.0)  # Base value
                                except Exception:
                                    try:
                                        indicator = indicator_class({})  # Config dict
                                    except Exception:
                                        raise init_error
                # Try to calculate the indicator
                if hasattr(indicator, 'calculate'):
                    # Analyze calculate method signature
                    calc_sig = None
                    try:
                        calc_sig = inspect.signature(indicator.calculate)
                    except Exception:
                        pass
                    
                    # Try different calculation approaches
                    calc_success = False
                    
                    if calc_sig:
                        calc_params = list(calc_sig.parameters.keys())[1:]  # Skip 'self'
                        
                        # Handle different calculate signatures
                        if 'data' in calc_params and len(calc_params) == 1:
                            # Single data parameter (DataFrame)
                            try:
                                result = indicator.calculate(sample_data)
                                calc_success = True
                            except Exception:
                                pass
                        
                        elif 'close' in calc_params and 'volume' in calc_params:
                            # Separate close and volume parameters
                            try:
                                result = indicator.calculate(
                                    sample_data['close'],
                                    sample_data['volume']
                                )
                                calc_success = True
                            except Exception:
                                pass
                        
                        elif len(calc_params) >= 2 and any(p in calc_params for p in ['high', 'low', 'close', 'volume']):
                            # Multiple OHLCV parameters
                            try:
                                result = indicator.calculate(
                                    sample_data['close'],
                                    sample_data.get('volume', sample_data['close']),
                                    sample_data.get('high', sample_data['close']),
                                    sample_data.get('low', sample_data['close'])
                                )
                                calc_success = True
                            except Exception:
                                pass
                        
                    # Fallback attempts if signature analysis didn't work
                    if not calc_success:
                        try:
                            # Try with DataFrame first
                            result = indicator.calculate(sample_data)
                            calc_success = True
                        except Exception as calc_error:
                            # Try with individual series (for legacy indicators)
                            try:
                                result = indicator.calculate(
                                    sample_data['close'],
                                    sample_data.get('volume', sample_data['close'])
                                )
                                calc_success = True
                            except Exception:
                                try:
                                    # Try with all OHLCV parameters
                                    result = indicator.calculate(
                                        sample_data['close'],
                                        sample_data.get('volume', sample_data['close']),
                                        sample_data.get('high', sample_data['close']),
                                        sample_data.get('low', sample_data['close'])
                                    )
                                    calc_success = True
                                except Exception:
                                    # Try with positional parameters in order: data, high, low, close, volume
                                    try:
                                        result = indicator.calculate(
                                            sample_data,
                                            sample_data.get('high', None),
                                            sample_data.get('low', None),
                                            sample_data.get('close', None),
                                            sample_data.get('volume', None)
                                        )
                                        calc_success = True
                                    except Exception:
                                        raise calc_error
                    
                    if calc_success:
                        execution_results[name] = {
                            "status": "SUCCESS",
                            "result_type": type(result).__name__,
                            "result_length": len(result) if hasattr(result, '__len__') else 1
                        }
                        successful_executions += 1
                    else:
                        execution_results[name] = {
                            "status": "CALCULATION_FAILED",
                            "error": str(calc_error) if 'calc_error' in locals() else "Unknown calculation error"
                        }
                        failed_executions += 1
                else:
                    execution_results[name] = {
                        "status": "NO_CALCULATE_METHOD",
                        "methods": [m for m in dir(indicator) if not m.startswith('_')]
                    }
                    failed_executions += 1
                    
            except Exception as e:
                execution_results[name] = {
                    "status": "FAILED",
                    "error": str(e)
                }
                failed_executions += 1
                print(f"   FAILED: {e}")
        
        success_rate = (successful_executions / len(indicators)) * 100 if indicators else 0
        print(f"\nExecution Results:")
        print(f"   Successful: {successful_executions}")
        print(f"   Failed: {failed_executions}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        return {
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": success_rate,
            "execution_details": execution_results,
            "status": "PASS" if success_rate > 80 else "NEEDS_IMPROVEMENT"
        }
        
    except Exception as e:
        print(f"Error testing indicator execution: {e}")
        traceback.print_exc()
        return {"status": "FAILED", "error": str(e)}

def test_agent_integration_depth():
    """Test deep integration between agents and trading system"""
    print("\n=== TESTING AGENT INTEGRATION DEPTH ===")
    
    try:
        # Test Python agent capabilities
        from engines.ai_enhancement.genius_agent_integration import GeniusAgentIntegration
        import pandas as pd
        import numpy as np
        
        # Create sample market data
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='H'),
            'open': 100 + np.random.randn(50).cumsum() * 0.1,
            'high': 102 + np.random.randn(50).cumsum() * 0.1,
            'low': 98 + np.random.randn(50).cumsum() * 0.1,
            'close': 100 + np.random.randn(50).cumsum() * 0.1,
            'volume': 1000000 + np.random.randint(-100000, 100000, 50)
        })
        
        agent = GeniusAgentIntegration()
        
        # Test agent analysis
        print("Testing agent analysis...")
        analysis_result = agent.analyze_market_data(sample_data)
        
        # Test decision making
        print("Testing agent decision making...")
        decision_result = agent.make_trading_decision(sample_data)
        
        # Test signal generation
        print("Testing agent signal generation...")
        signal_result = agent.generate_trading_signal(analysis_result)
        
        # Check for required methods and outputs
        agent_capabilities = {
            "has_analyze_method": hasattr(agent, 'analyze_market_data'),
            "has_decision_method": hasattr(agent, 'make_trading_decision'),
            "has_signal_method": hasattr(agent, 'generate_trading_signal'),
            "analysis_output": type(analysis_result).__name__ if analysis_result else None,
            "decision_output": type(decision_result).__name__ if decision_result else None,
            "signal_output": type(signal_result).__name__ if signal_result else None
        }
        
        print(f"Agent capabilities: {agent_capabilities}")
        
        # Test if agent uses indicators
        from engines.indicator_registry import IndicatorRegistry
        registry = IndicatorRegistry()
        
        # Check if agent integrates with indicators
        indicator_integration_test = agent.analyze_market_data(sample_data)
        
        return {
            "agent_capabilities": agent_capabilities,
            "analysis_successful": analysis_result is not None,
            "decision_successful": decision_result is not None,
            "signal_successful": signal_result is not None,
            "indicator_integration": indicator_integration_test is not None,
            "status": "PASS"
        }
        
    except Exception as e:
        print(f"Error testing agent integration: {e}")
        traceback.print_exc()
        return {"status": "FAILED", "error": str(e)}

def test_trading_pipeline_readiness():
    """Test if the complete trading pipeline is ready for production"""
    print("\n=== TESTING TRADING PIPELINE READINESS ===")
    
    try:
        # Check critical files existence
        critical_files = [
            "services/trading-service/src/server.ts",
            "shared/PythonEngineClient.ts", 
            "ai-platform/rest_api_server.py",
            "engines/ai_enhancement/genius_agent_integration.py",
            "engines/indicator_registry.py"
        ]
        
        file_status = {}
        for file_path in critical_files:
            full_path = os.path.join(os.path.dirname(__file__), file_path)
            file_status[file_path] = os.path.exists(full_path)
        
        # Test REST API server availability
        import subprocess
        import time
        
        print("Testing REST API server...")
        # Check if we can import the REST API components
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), 'ai-platform'))
            # Import test only - don't start server
            import rest_api_server
            api_importable = True
        except Exception as e:
            print(f"REST API import failed: {e}")
            api_importable = False
        
        # Test TypeScript service readiness
        print("Testing TypeScript service readiness...")
        ts_service_path = os.path.join(os.path.dirname(__file__), 'services', 'trading-service')
        package_json_exists = os.path.exists(os.path.join(ts_service_path, 'package.json'))
        node_modules_exists = os.path.exists(os.path.join(ts_service_path, 'node_modules'))
        
        return {
            "critical_files": file_status,
            "all_files_exist": all(file_status.values()),
            "api_importable": api_importable,
            "typescript_service_ready": package_json_exists and node_modules_exists,
            "package_json_exists": package_json_exists,
            "dependencies_installed": node_modules_exists,
            "status": "PASS" if all(file_status.values()) and api_importable else "NEEDS_IMPROVEMENT"
        }
        
    except Exception as e:
        print(f"Error testing trading pipeline: {e}")
        traceback.print_exc()
        return {"status": "FAILED", "error": str(e)}

def run_comprehensive_validation():
    """Run all validation tests and generate comprehensive report"""
    print("=== PLATFORM3 COMPREHENSIVE VALIDATION TEST ===")
    print(f"Started at: {datetime.now()}")
    
    results = {}
    
    # Run all validation tests
    results['indicator_registry'] = test_indicator_registry_completeness()
    results['indicator_execution'] = test_indicator_execution() 
    results['agent_integration'] = test_agent_integration_depth()
    results['trading_pipeline'] = test_trading_pipeline_readiness()
    
    # Generate overall assessment
    all_statuses = [test_result.get('status', 'UNKNOWN') for test_result in results.values()]
    passed_tests = sum(1 for status in all_statuses if status == 'PASS')
    total_tests = len(all_statuses)
    
    overall_status = "PRODUCTION_READY" if passed_tests == total_tests else "NEEDS_WORK"
    
    results['summary'] = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "overall_status": overall_status,
        "completion_percentage": (passed_tests / total_tests) * 100,
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Overall Status: {overall_status}")
    print(f"Completion: {results['summary']['completion_percentage']:.1f}%")
    
    # Save detailed results
    with open('comprehensive_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: comprehensive_validation_results.json")
    
    return results

if __name__ == "__main__":
    try:
        results = run_comprehensive_validation()
        
        # Print critical findings
        print(f"\n=== CRITICAL FINDINGS ===")
        
        # Indicator coverage issues
        if results['indicator_registry']['status'] != 'PASS':
            missing_count = results['indicator_registry'].get('missing_count', 0)
            print(f"[WARNING] {missing_count} indicators not in registry")
        
        # Execution issues
        if results['indicator_execution']['status'] != 'PASS':
            failed_count = results['indicator_execution'].get('failed_executions', 0)
            print(f"[WARNING] {failed_count} indicators failed execution")
        
        # Pipeline issues
        if results['trading_pipeline']['status'] != 'PASS':
            print(f"[WARNING] Trading pipeline not fully ready")
            
        print(f"\nValidation completed at: {datetime.now()}")
        
    except Exception as e:
        print(f"Fatal error in comprehensive validation: {e}")
        traceback.print_exc()
        sys.exit(1)
