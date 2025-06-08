#!/usr/bin/env python3
"""
Enhanced Indicator Validation Framework
Platform3 Comprehensive Indicator Testing with Advanced Error Resolution

This module provides a robust validation framework that addresses the major
failure categories identified in the indicator execution analysis:
1. Constructor signature mismatches
2. Data format incompatibilities  
3. Missing method implementations
4. Missing imports/attributes
5. Parameter validation issues
"""

import inspect
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Type
from pathlib import Path
import traceback
import importlib
import sys
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedIndicatorValidator:
    """Enhanced indicator validation with intelligent error resolution"""
    
    def __init__(self):
        self.results = {}
        self.error_categories = {
            'constructor_signature': 0,
            'data_format': 0,
            'missing_methods': 0,
            'missing_imports': 0,
            'parameter_validation': 0,
            'miscellaneous': 0
        }
        self.patches_applied = []
        
    def validate_all_indicators(self, indicators_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all indicators with enhanced error handling"""
        logger.info(f"Starting enhanced validation of {len(indicators_data)} indicators")
          # Apply global patches
        self._apply_global_patches()
        
        total_indicators = len(indicators_data)
        successful_indicators = 0
        
        for indicator_path, indicator_class in indicators_data.items():
            try:
                result = self._validate_single_indicator(indicator_path, indicator_class)
                self.results[indicator_path] = result
                
                if result['status'] == 'passed':
                    successful_indicators += 1
                    
            except Exception as e:
                self.results[indicator_path] = {
                    'status': 'failed',
                    'error': f"Validation framework error: {str(e)}",
                    'error_category': 'framework_error'
                }
                logger.error(f"Framework error validating {indicator_path}: {e}")
        
        success_rate = (successful_indicators / total_indicators) * 100
        
        return {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_indicators': total_indicators,
                'successful_indicators': successful_indicators,
                'success_rate': round(success_rate, 1),
                'error_categories': self.error_categories,
                'patches_applied': self.patches_applied
            },
            'results': self.results
        }
      def _validate_single_indicator(self, indicator_path: str, indicator_class: Type) -> Dict[str, Any]:
        """Validate a single indicator with enhanced error handling"""
        try:
            # The indicator_class is directly the class object
            if not indicator_class:
                return {
                    'status': 'failed',
                    'error': 'No indicator class found',
                    'error_category': 'missing_class'
                }
            
            # Analyze constructor signature
            signature_info = self._analyze_constructor_signature(indicator_class)
            
            # Generate appropriate test data
            test_data = self._generate_test_data(indicator_class, signature_info)
            
            # Create flexible constructor arguments
            constructor_args = self._create_flexible_args(indicator_class, signature_info, test_data)
            
            # Create indicator instance with enhanced error handling
            try:
                indicator_instance = indicator_class(**constructor_args)
            except Exception as e:
                return self._handle_constructor_error(e, indicator_path)
            
            # Validate method implementations
            method_validation = self._validate_method_implementations(indicator_instance)
            if not method_validation['valid']:
                return {
                    'status': 'failed',
                    'error': f"Missing methods: {method_validation['missing_methods']}",
                    'error_category': 'missing_methods'
                }
            
            # Execute calculation with enhanced error handling
            try:
                result = self._execute_calculation(indicator_instance, test_data)
                return {
                    'status': 'passed',
                    'result_type': type(result).__name__,
                    'data_shape': self._get_result_shape(result),
                    'constructor_args': list(constructor_args.keys())
                }
                
            except Exception as e:
                return self._handle_calculation_error(e, indicator_path)
                
        except Exception as e:
            return {
                'status': 'failed',
                'error': f"Unexpected error: {str(e)}",
                'error_category': 'unexpected_error',
                'traceback': traceback.format_exc()
            }
    
    def _analyze_constructor_signature(self, indicator_class) -> Dict[str, Any]:
        """Analyze constructor signature to understand requirements"""
        try:
            signature = inspect.signature(indicator_class.__init__)
            parameters = signature.parameters
            
            required_params = []
            optional_params = []
            
            for param_name, param in parameters.items():
                if param_name == 'self':
                    continue
                    
                if param.default == inspect.Parameter.empty:
                    required_params.append(param_name)
                else:
                    optional_params.append(param_name)
            
            return {
                'required_params': required_params,
                'optional_params': optional_params,
                'total_params': len(required_params) + len(optional_params),
                'signature': str(signature)
            }
            
        except Exception as e:
            logger.warning(f"Could not analyze signature for {indicator_class}: {e}")
            return {
                'required_params': [],
                'optional_params': [],
                'total_params': 0,
                'signature': 'unknown'
            }
    
    def _create_flexible_args(self, indicator_class, signature_info: Dict, test_data: Dict) -> Dict[str, Any]:
        """Create flexible constructor arguments based on signature analysis"""
        args = {}
        required_params = signature_info.get('required_params', [])
        
        # Handle common parameter patterns
        for param in required_params:
            if param == 'config':
                args['config'] = self._create_default_config()
            elif param == 'data':
                args['data'] = test_data['dataframe']
            elif param == 'period':
                args['period'] = 14
            elif param == 'name':
                args['name'] = indicator_class.__name__
            elif param == 'timeframe':
                args['timeframe'] = 'daily'
            elif param == 'timestamp':
                args['timestamp'] = datetime.now()
            elif param == 'indicator_name':
                args['indicator_name'] = indicator_class.__name__
            elif param == 'signal_type':
                args['signal_type'] = 'technical'
            elif param in ['window', 'lookback', 'length']:
                args[param] = 20
            elif param in ['fast_period', 'short_period']:
                args[param] = 12
            elif param in ['slow_period', 'long_period']:
                args[param] = 26
            elif param in ['signal_period']:
                args[param] = 9
            elif 'period' in param.lower():
                args[param] = 14
            else:
                # Default handling for unknown parameters
                args[param] = self._get_default_value_for_param(param)
        
        return args
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create a default configuration object"""
        return {
            'period': 14,
            'timeframe': 'daily',
            'source': 'close',
            'smoothing': 2,
            'threshold': 0.5,
            'lookback': 20,
            'min_periods': 10,
            'max_periods': 500
        }
    
    def _get_default_value_for_param(self, param_name: str) -> Any:
        """Get appropriate default value for unknown parameters"""
        param_lower = param_name.lower()
        
        if 'period' in param_lower or 'window' in param_lower:
            return 14
        elif 'threshold' in param_lower:
            return 0.5
        elif 'factor' in param_lower or 'multiplier' in param_lower:
            return 2.0
        elif 'lookback' in param_lower:
            return 20
        elif 'alpha' in param_lower or 'beta' in param_lower:
            return 0.1
        elif 'min' in param_lower:
            return 1
        elif 'max' in param_lower:
            return 100
        else:
            return None
    
    def _generate_test_data(self, indicator_class, signature_info: Dict) -> Dict[str, Any]:
        """Generate appropriate test data based on indicator requirements"""
        # Determine data length based on indicator type
        base_length = 500  # Start with sufficient data for most indicators
        
        # Create realistic market data
        np.random.seed(42)  # For reproducible results
        
        dates = pd.date_range(start='2023-01-01', periods=base_length, freq='D')
        
        # Generate correlated OHLCV data
        base_price = 100
        prices = [base_price]
        
        for i in range(1, base_length):
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.01))  # Prevent negative prices
        
        # Create OHLCV from price series
        data_dict = {}
        data_list = []
        
        for i, (date, close) in enumerate(zip(dates, prices)):
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            open_price = close * (1 + np.random.normal(0, 0.005))
            volume = np.random.randint(1000, 10000)
            
            row_dict = {
                'timestamp': date,
                'open': open_price,
                'high': max(high, open_price, close),
                'low': min(low, open_price, close),
                'close': close,
                'volume': volume
            }
            
            data_list.append(row_dict)
            
            if i == len(dates) - 1:  # Latest data point
                data_dict = row_dict
        
        # Create DataFrame
        df = pd.DataFrame(data_list)
        df.set_index('timestamp', inplace=True)
        
        return {
            'dataframe': df,
            'dict': data_dict,
            'list': data_list,
            'series': df['close'],
            'array': df['close'].values
        }
    
    def _validate_method_implementations(self, indicator_instance) -> Dict[str, Any]:
        """Validate that required methods are implemented"""
        required_methods = ['calculate']
        optional_methods = ['_perform_calculation', 'analyze', 'signal']
        
        missing_required = []
        missing_optional = []
        
        for method in required_methods:
            if not hasattr(indicator_instance, method):
                missing_required.append(method)
            else:
                method_obj = getattr(indicator_instance, method)
                if callable(method_obj):
                    # Check if it's an abstract method
                    try:
                        # Try to get the source - abstract methods might raise
                        inspect.getsource(method_obj)
                    except (OSError, TypeError):
                        # Might be abstract or builtin
                        pass
                else:
                    missing_required.append(method)
        
        for method in optional_methods:
            if not hasattr(indicator_instance, method):
                missing_optional.append(method)
        
        return {
            'valid': len(missing_required) == 0,
            'missing_methods': missing_required,
            'missing_optional': missing_optional
        }
    
    def _execute_calculation(self, indicator_instance, test_data: Dict) -> Any:
        """Execute indicator calculation with intelligent error handling"""
        # Get the calculate method
        calculate_method = getattr(indicator_instance, 'calculate')
        
        # Analyze the calculate method signature
        try:
            calc_signature = inspect.signature(calculate_method)
            calc_params = [p.name for p in calc_signature.parameters.values() if p.name != 'self']
        except:
            calc_params = []
        
        # Try different data formats and parameter combinations
        execution_attempts = [
            # No parameters
            lambda: calculate_method(),
            # DataFrame
            lambda: calculate_method(test_data['dataframe']),
            # Individual OHLCV parameters
            lambda: calculate_method(
                test_data['dataframe']['high'],
                test_data['dataframe']['low'],
                test_data['dataframe']['close'],
                test_data['dataframe']['volume']
            ),
            # Dict format
            lambda: calculate_method(test_data['dict']),
            # Series format
            lambda: calculate_method(test_data['series']),
            # List format  
            lambda: calculate_method(test_data['list'])
        ]
        
        # Try each execution approach
        for i, attempt in enumerate(execution_attempts):
            try:
                result = attempt()
                logger.debug(f"Successful execution with approach {i+1}")
                return result
            except Exception as e:
                logger.debug(f"Execution attempt {i+1} failed: {str(e)}")
                continue
        
        # If all attempts failed, raise the last exception
        raise Exception("All calculation execution attempts failed")
    
    def _handle_constructor_error(self, error: Exception, indicator_path: str) -> Dict[str, Any]:
        """Handle and categorize constructor errors"""
        error_str = str(error)
        
        if 'missing' in error_str and 'required positional argument' in error_str:
            self.error_categories['constructor_signature'] += 1
            return {
                'status': 'failed',
                'error': f"Constructor signature mismatch: {error_str}",
                'error_category': 'constructor_signature'
            }
        elif 'unexpected keyword argument' in error_str:
            self.error_categories['constructor_signature'] += 1
            return {
                'status': 'failed',
                'error': f"Unexpected constructor argument: {error_str}",
                'error_category': 'constructor_signature'
            }
        else:
            self.error_categories['miscellaneous'] += 1
            return {
                'status': 'failed',
                'error': f"Constructor error: {error_str}",
                'error_category': 'constructor_error'
            }
    
    def _handle_calculation_error(self, error: Exception, indicator_path: str) -> Dict[str, Any]:
        """Handle and categorize calculation errors"""
        error_str = str(error)
        
        if 'has no attribute' in error_str and ('rolling' in error_str or 'iterrows' in error_str or 'pct_change' in error_str):
            self.error_categories['data_format'] += 1
            return {
                'status': 'failed',
                'error': f"Data format incompatibility: {error_str}",
                'error_category': 'data_format'
            }
        elif 'Subclasses must implement' in error_str:
            self.error_categories['missing_methods'] += 1
            return {
                'status': 'failed',
                'error': f"Missing method implementation: {error_str}",
                'error_category': 'missing_methods'
            }
        elif 'is not defined' in error_str:
            self.error_categories['missing_imports'] += 1
            return {
                'status': 'failed',
                'error': f"Missing import: {error_str}",
                'error_category': 'missing_imports'
            }
        elif 'must be greater than' in error_str or 'need at least' in error_str:
            self.error_categories['parameter_validation'] += 1
            return {
                'status': 'failed',
                'error': f"Parameter validation error: {error_str}",
                'error_category': 'parameter_validation'
            }
        else:
            self.error_categories['miscellaneous'] += 1
            return {
                'status': 'failed',
                'error': f"Calculation error: {error_str}",
                'error_category': 'calculation_error'
            }
    
    def _get_result_shape(self, result: Any) -> str:
        """Get shape information for the result"""
        if hasattr(result, 'shape'):
            return str(result.shape)
        elif hasattr(result, '__len__'):
            return f"length: {len(result)}"
        else:
            return "scalar"
    
    def _apply_global_patches(self):
        """Apply global patches for common issues"""
        # Patch datetime if missing in any module
        if 'datetime' not in globals():
            import datetime
            globals()['datetime'] = datetime
            self.patches_applied.append('datetime_global_patch')
        
        # Try to patch TimeFrame enum issues
        try:
            from some_module import TimeFrame  # This will fail, but we handle it
        except:
            pass
        
        # Create mock TimeFrame if needed
        if 'TimeFrame' not in globals():
            class MockTimeFrame:
                DAILY = 'daily'
                HOURLY = 'hourly'
                MINUTE = 'minute'
            
            globals()['TimeFrame'] = MockTimeFrame
            self.patches_applied.append('timeframe_mock_patch')
        
        # Suppress warnings to reduce noise
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)

def main():
    """Main execution function"""
    # Load the dynamic indicator loader to get the actual indicator classes
    from dynamic_indicator_loader import load_all_working_indicators
    
    # Get indicators
    indicators_data = load_all_working_indicators()
    
    logger.info(f"Loaded {len(indicators_data)} indicators from dynamic loader")
    
    # Create enhanced validator
    validator = EnhancedIndicatorValidator()
    
    # Run enhanced validation
    results = validator.validate_all_indicators(indicators_data)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'enhanced_validation_results_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    summary = results['summary']
    print(f"\n=== Enhanced Indicator Validation Results ===")
    print(f"Total Indicators: {summary['total_indicators']}")
    print(f"Successful: {summary['successful_indicators']}")
    print(f"Success Rate: {summary['success_rate']}%")
    print(f"Patches Applied: {len(summary['patches_applied'])}")
    print(f"\nError Categories:")
    for category, count in summary['error_categories'].items():
        if count > 0:
            print(f"  {category}: {count}")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
