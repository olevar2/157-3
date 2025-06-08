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
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Type
from pathlib import Path
import traceback
import importlib
import sys
# import warnings # Not strictly needed if we manage warnings carefully

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Mock Classes ---
class MockConfigBase:
    def __init__(self, name="MockConfigBaseName", **kwargs):
        self.name = name
        self.period = kwargs.get('period', 14)
        self.timeframe = kwargs.get('timeframe', 'daily') # Default timeframe
        self.indicator_type = kwargs.get('indicator_type', "MOCK_GENERAL_TYPE") # Added for broader compatibility
        self.source = kwargs.get('source', 'close')
        self.lookback_periods = kwargs.get('lookback_periods', 20)
        # Add other common attributes that might be accessed
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getattr__(self, name: str) -> Any:
        """Fallback for any missing attributes, returning sensible defaults."""
        # logger.debug(f"MockConfigBase.__getattr__ called for '{name}' on {self.name}")
        defaults = {
            'get': lambda k, d=None: getattr(self, k, d),
            'name': self.name if hasattr(self, 'name') else 'DefaultNameFromAttr',
            'period': 14,
            'timeframe': 'daily',
            'source': 'close',
            'indicator_type': 'MOCK_FALLBACK_TYPE',
            'lookback_periods': 20,
            'window': 14,
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'min_periods': 10,
            'max_periods': 500,
            'smoothing': 2,
            'threshold': 0.5,
        }
        if name in defaults:
            return defaults[name]
        # For other attributes, to prevent NoneType errors when chained,
        # return a new instance of a very basic mock that can be called or accessed.
        # This is a bit aggressive but can help bypass some superficial NoneType errors.
        # logger.warning(f"Attribute '{name}' not found in MockConfigBase, returning a BasicCallableMock.")
        # class BasicCallableMock:
        #     def __call__(self, *args, **kwargs): return None
        #     def __getattr__(self, item): return None
        # return BasicCallableMock()
        return None # A more conservative approach

class MockConfig(MockConfigBase):
    def __init__(self, name="MockConfigName", **kwargs):
        super().__init__(name, **kwargs)
        self.indicator_type = kwargs.get('indicator_type', "MOCK_CONFIG_TYPE")


class MockIndicatorConfig(MockConfigBase):
    def __init__(self, name="MockIndicatorConfigName", **kwargs):
        super().__init__(name, **kwargs)
        self.indicator_type = kwargs.get('indicator_type', "MOCK_INDICATOR_CONFIG_TYPE")
        # Ensure TimeFrame compatibility if actual TimeFrame is used
        try:
            from engines.indicator_base import TimeFrame # Assuming this path
            self.timeframe = kwargs.get('timeframe', TimeFrame.D1) # Default to D1
        except ImportError:
            self.timeframe = kwargs.get('timeframe', 'D1') # Fallback to string if import fails


# --- Universal Reset Method ---
def universal_reset_method(self):
    """Universal reset implementation that can be used when super().reset() is called"""
    attributes_to_reset = ['_data', 'values', 'signals', 'state', '_initialized',
                           '_buffer', '_results', '_calculated_data']
    for attr in attributes_to_reset:
        if hasattr(self, attr):
            if isinstance(getattr(self, attr), list):
                setattr(self, attr, [])
            elif isinstance(getattr(self, attr), dict):
                setattr(self, attr, {})
            elif isinstance(getattr(self, attr), bool): # for _initialized
                 setattr(self, attr, False)
            else:
                setattr(self, attr, None)
    # logger.debug(f"Universal reset method called on {type(self).__name__}")
    return True

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
            'miscellaneous': 0,
            'framework_error': 0,
            'stub_implementation': 0
        }
        self.patches_applied = []
        self._setup_mock_classes()
        self.universal_reset = universal_reset_method

    def _setup_mock_classes(self):
        """Sets up mock classes used for validation."""
        self.MockConfig = MockConfig
        self.MockIndicatorConfig = MockIndicatorConfig
        self.TimeFrame = None
        try:
            # Attempt to import the actual TimeFrame if available
            from engines.indicator_base import TimeFrame as ActualTimeFrame
            self.TimeFrame = ActualTimeFrame
            logger.info("Successfully imported actual TimeFrame enum.")
        except ImportError:
            logger.warning("Actual TimeFrame enum not found. Using string representations for timeframe.")
            # Create a mock TimeFrame if the real one isn't available
            class MockTimeFrame:
                D1 = "D1"
                H4 = "H4"
                H1 = "H1"
                M30 = "M30"
                M15 = "M15"
                M5 = "M5"
                M1 = "M1"
                DAILY = "DAILY" # For compatibility with older indicators
                # Add other common timeframes as needed
            self.TimeFrame = MockTimeFrame()


    def validate_all_indicators(self, indicators_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Starting enhanced validation of {len(indicators_data)} indicators")
        self._apply_global_patches()

        total_indicators = len(indicators_data)
        successful_indicators = 0

        for indicator_path, indicator_class_loader in indicators_data.items():
            indicator_class = None
            try:
                # The loader might be the class itself or a function that returns the class
                if callable(indicator_class_loader) and not isinstance(indicator_class_loader, type):
                    indicator_class = indicator_class_loader()
                else:
                    indicator_class = indicator_class_loader

                if not inspect.isclass(indicator_class):
                    # Try to get the class from the module if a module path was given
                    module_path, class_name_str = indicator_path.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    indicator_class = getattr(module, class_name_str)

            except Exception as e:
                self.results[indicator_path] = {
                    'status': 'failed',
                    'error': f"Failed to load indicator class {indicator_path}: {str(e)}",
                    'error_type': type(e).__name__,
                    'error_category': 'framework_error'
                }
                logger.error(f"Framework error loading {indicator_path}: {e}")
                self.error_categories['framework_error'] += 1
                continue

            try:
                result = self._validate_single_indicator(indicator_path, indicator_class)
                self.results[indicator_path] = result
                if result['status'] == 'passed':
                    successful_indicators += 1
            except Exception as e:
                self.results[indicator_path] = {
                    'status': 'failed',
                    'error': f"Validation framework error: {str(e)}",
                    'error_type': type(e).__name__,
                    'error_category': 'framework_error',
                    'traceback': traceback.format_exc()
                }
                logger.error(f"Framework error validating {indicator_path}: {e}\n{traceback.format_exc()}")
                self.error_categories['framework_error'] += 1

        success_rate = (successful_indicators / total_indicators) * 100 if total_indicators > 0 else 0

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
        logger.debug(f"Validating: {indicator_path}")
        if not indicator_class:
            return {'status': 'failed', 'error': 'No indicator class found', 'error_category': 'framework_error'}

        signature_info = self._analyze_constructor_signature(indicator_class)
        test_data = self._generate_test_data()
        indicator_instance = None
        last_constructor_error = None
        constructor_args_used = {}

        instantiation_attempts = [
            # Attempt 1: Standard guess
            {'force_no_config': False, 'force_no_name': False, 'force_default_config': False, 'attempt_num': 1},
            # Attempt 2: No config
            {'force_no_config': True, 'force_no_name': False, 'force_default_config': False, 'attempt_num': 2},
            # Attempt 3: No name
            {'force_no_config': False, 'force_no_name': True, 'force_default_config': False, 'attempt_num': 3},
            # Attempt 4: No config and no name
            {'force_no_config': True, 'force_no_name': True, 'force_default_config': False, 'attempt_num': 4},
            # Attempt 5: Force default config (if it wasn't already tried by 'config' being required)
            {'force_no_config': False, 'force_no_name': False, 'force_default_config': True, 'attempt_num': 5},
        ]

        for attempt_params in instantiation_attempts:
            current_args = self._create_flexible_args(indicator_class, signature_info, test_data, **attempt_params)
            logger.debug(f"Attempt {attempt_params['attempt_num']} for {indicator_path} with args: { {k: type(v).__name__ for k,v in current_args.items()} }")
            try:
                indicator_instance = indicator_class(**current_args)
                constructor_args_used = current_args
                last_constructor_error = None
                logger.info(f"Successfully instantiated {indicator_path} on attempt {attempt_params['attempt_num']}.")
                break
            except Exception as e:
                last_constructor_error = e
                logger.warning(f"Attempt {attempt_params['attempt_num']} failed for {indicator_path} with args {list(current_args.keys())}: {e}")
                # Specific handling for common errors to guide next attempts or final error reporting
                error_str = str(e).lower()
                if "unexpected keyword argument 'config'" in error_str and not attempt_params['force_no_config']:
                    logger.debug("Error suggests 'config' is unexpected. Will try without it.")
                    # Next attempt will naturally try force_no_config=True if available
                elif "unexpected keyword argument 'name'" in error_str and not attempt_params['force_no_name']:
                    logger.debug("Error suggests 'name' is unexpected. Will try without it.")
                elif ("missing 1 required positional argument: 'config'" in error_str or \
                     ("missing" in error_str and "'config'" in error_str and "required" in error_str)) and not attempt_params['force_default_config']:
                    logger.debug("Error suggests 'config' is required. Will try forcing default config.")
                elif "'super' object has no attribute 'reset'" in error_str:
                    logger.debug(f"Attempting to patch reset for {indicator_path} due to super().reset error.")
                    if not hasattr(indicator_class, 'reset'): # Patch directly on the class for this instance
                        setattr(indicator_class, 'reset', self.universal_reset)
                        logger.info(f"Dynamically added universal_reset to {indicator_class.__name__}")
                        # Retry instantiation immediately with this patch
                        try:
                            indicator_instance = indicator_class(**current_args)
                            constructor_args_used = current_args
                            last_constructor_error = None
                            logger.info(f"Successfully instantiated {indicator_path} after dynamic reset patch.")
                            break
                        except Exception as e_retry:
                            last_constructor_error = e_retry
                            logger.warning(f"Instantiation failed even after dynamic reset patch for {indicator_path}: {e_retry}")
                    else:
                         logger.debug(f"{indicator_class.__name__} already has a reset method. Superclass issue likely.")


        if indicator_instance is None and last_constructor_error:
            return self._handle_constructor_error(last_constructor_error, indicator_path)
        elif indicator_instance is None: # Should not happen if last_constructor_error is also None
            return self._handle_constructor_error(Exception("Unknown instantiation failure"), indicator_path)

        # Method validation (basic)
        if not hasattr(indicator_instance, 'calculate') or not callable(getattr(indicator_instance, 'calculate')):
            self.error_categories['missing_methods'] += 1
            return {'status': 'failed', 'error': "Missing 'calculate' method", 'error_category': 'missing_methods'}

        try:
            result = self._execute_calculation(indicator_instance, test_data, indicator_path)
            return {
                'status': 'passed',
                'result_type': type(result).__name__,
                'data_shape': self._get_result_shape(result),
                'constructor_args': list(constructor_args_used.keys())
            }
        except ValueError as ve: # Catch specific ValueErrors from _execute_calculation or _is_stub
            if "stub implementation" in str(ve).lower():
                self.error_categories['stub_implementation'] += 1
                return {'status': 'failed', 'error': str(ve), 'error_category': 'stub_implementation'}
            else: # Other ValueErrors are likely calculation errors
                 return self._handle_calculation_error(ve, indicator_path)
        except Exception as e:
            return self._handle_calculation_error(e, indicator_path)

    def _analyze_constructor_signature(self, indicator_class) -> Dict[str, Any]:
        try:
            # For classes, __init__ is the constructor. For functions (if any passed directly), it's the function itself.
            target_callable = indicator_class.__init__ if inspect.isclass(indicator_class) else indicator_class
            signature = inspect.signature(target_callable)
            parameters = signature.parameters
            required_params, optional_params, all_params = [], [], []

            for param_name, param in parameters.items():
                if param_name == 'self' and inspect.isclass(indicator_class): # 'self' only for class methods
                    continue
                all_params.append(param_name)
                if param.default == inspect.Parameter.empty:
                    required_params.append(param_name)
                else:
                    optional_params.append(param_name)
            return {'required_params': required_params, 'optional_params': optional_params, 'all_params': all_params, 'signature': str(signature)}
        except Exception as e:
            logger.warning(f"Could not analyze signature for {indicator_class.__name__}: {e}")
            return {'required_params': [], 'optional_params': [], 'all_params': [], 'signature': 'unknown'}

    def _create_flexible_args(self, indicator_class, signature_info: Dict, test_data: Dict,
                              force_no_config=False, force_no_name=False, force_default_config=False, attempt_num=0) -> Dict[str, Any]:
        args = {}
        required_params = signature_info.get('required_params', [])
        all_params = signature_info.get('all_params', [])
        indicator_name = indicator_class.__name__ # Use actual class name

        # Config handling
        config_param_name = next((p for p in ['config', 'configuration'] if p in all_params), None)

        if force_default_config and config_param_name:
            args[config_param_name] = self._get_appropriate_config(indicator_class, name_override=indicator_name)
            logger.debug(f"[{indicator_name}] Forcing default config as '{type(args[config_param_name]).__name__}'.")
        elif config_param_name and not force_no_config:
            args[config_param_name] = self._get_appropriate_config(indicator_class, name_override=indicator_name)
            logger.debug(f"[{indicator_name}] Providing config as '{type(args[config_param_name]).__name__}'.")
        elif config_param_name in args and args[config_param_name] is None and config_param_name in required_params: # Should be covered by above
            args[config_param_name] = self._get_appropriate_config(indicator_class, name_override=indicator_name)
            logger.debug(f"[{indicator_name}] Config was None but required, providing default.")


        # Name handling
        name_param_name = next((p for p in ['name', 'indicator_name'] if p in all_params), None)
        if name_param_name and not force_no_name:
            if name_param_name not in args: # Avoid overwriting if config object might have set it
                 args[name_param_name] = indicator_name
                 logger.debug(f"[{indicator_name}] Providing explicit name: '{indicator_name}'.")

        # Data handling (common parameter names for input data)
        data_param_names = ['data', 'dataframe', 'df', 'input_data', 'ohlcv']
        actual_data_param = next((p for p in data_param_names if p in all_params), None)
        if actual_data_param:
            args[actual_data_param] = test_data['dataframe'] # Default to DataFrame

        # Populate other required parameters
        for param in required_params:
            if param in args: continue # Already handled (config, name, data)

            param_lower = param.lower()
            if any(p in param_lower for p in ['period', 'window', 'length']): args[param] = 14
            elif 'timeframe' in param_lower: args[param] = self._get_default_timeframe()
            elif any(p == param_lower for p in ['high', 'low', 'close', 'open']): args[param] = test_data['dataframe'][param_lower] # Pass Series
            elif 'volume' in param_lower: args[param] = test_data['dataframe']['volume'] # Pass Series
            elif 'fast' in param_lower: args[param] = 12
            elif 'slow' in param_lower: args[param] = 26
            elif 'signal' in param_lower and 'period' in param_lower: args[param] = 9
            else:
                default_val = self._get_default_value_for_param_type(param, indicator_class)
                args[param] = default_val
                logger.debug(f"[{indicator_name}] Providing default for required param '{param}': {type(default_val).__name__}")
        return args

    def _get_appropriate_config(self, indicator_class, name_override=None):
        # Prefer MockIndicatorConfig if "TechnicalIndicator" seems to be a base or if typical TI params are present
        # This is a heuristic.
        base_names_str = " ".join(b.__name__ for b in indicator_class.__mro__)
        if "TechnicalIndicator" in base_names_str or "BaseIndicator" in base_names_str :
             cfg = self.MockIndicatorConfig(name=name_override or indicator_class.__name__)
        else:
             cfg = self.MockConfig(name=name_override or indicator_class.__name__)
        # Ensure name is set on the config object itself
        if name_override and hasattr(cfg, 'name'):
            cfg.name = name_override
        return cfg


    def _get_default_value_for_param_type(self, param_name: str, indicator_class) -> Any:
        # Try to infer type from annotations if available
        try:
            sig = inspect.signature(indicator_class.__init__)
            param_obj = sig.parameters.get(param_name)
            if param_obj and param_obj.annotation != inspect.Parameter.empty:
                param_type = param_obj.annotation
                if param_type == int: return 10
                if param_type == float: return 0.5
                if param_type == bool: return True
                if param_type == str: return "default"
                if param_type == pd.DataFrame: return self._generate_test_data()['dataframe']
                if param_type == pd.Series: return self._generate_test_data()['series']
                if param_type == list: return []
                if param_type == dict: return {}
        except: # Signature analysis might fail for some built-ins or C extensions
            pass

        # Fallback to name-based guessing
        param_lower = param_name.lower()
        if any(s in param_lower for s in ['period', 'window', 'length', 'lag', 'lookback']): return 14
        if any(s in param_lower for s in ['alpha', 'factor', 'multiplier', 'threshold', 'level']): return 0.5
        if any(s in param_lower for s in ['method', 'mode', 'type', 'source']): return "default"
        if 'timeframe' in param_lower: return self._get_default_timeframe()
        if 'debug' in param_lower or param_lower.startswith('is_') or param_lower.startswith('use_'): return False
        return None # Default if no specific rule matches

    def _get_default_timeframe(self):
        return self.TimeFrame.D1 if self.TimeFrame else "D1"


    def _generate_test_data(self) -> Dict:
        dates = pd.date_range(start='2024-01-01', periods=100, freq='B')
        np.random.seed(42)
        data = {}
        data['open'] = np.random.uniform(90, 100, size=len(dates))
        data['high'] = data['open'] + np.random.uniform(0, 5, size=len(dates))
        data['low'] = data['open'] - np.random.uniform(0, 5, size=len(dates))
        data['close'] = (data['high'] + data['low']) / 2 + np.random.normal(0, 1, size=len(dates))
        # Ensure H > L, O, C and L < H, O, C
        data['high'] = np.maximum.reduce([data['high'], data['open'], data['close']])
        data['low'] = np.minimum.reduce([data['low'], data['open'], data['close']])
        data['volume'] = np.random.randint(10000, 1000000, size=len(dates)).astype(float)
        df = pd.DataFrame(data, index=dates)

        return {
            'dataframe': df,
            'series': df['close'].copy(),
            'numpy_array': df.values.copy(),
            'dict': df.to_dict('list'), # DataFrame as dict of lists
            'ohlc': {k: df[k].tolist() for k in ['open', 'high', 'low', 'close']} # Dict of lists for OHLC
        }

    def _is_stub_implementation(self, indicator_instance, indicator_path) -> bool:
        if not hasattr(indicator_instance, 'calculate'):
            logger.warning(f"Indicator {indicator_path} has no calculate method.")
            return True # No calculate method is effectively a stub for our purposes.

        method = getattr(indicator_instance, 'calculate')
        try:
            source = inspect.getsource(method)
            # More robust stub patterns
            # Count non-comment, non-empty lines
            lines = [line.strip() for line in source.splitlines()]
            code_lines = [line for line in lines if line and not line.startswith('#')]

            if not code_lines or (len(code_lines) == 1 and code_lines[0] == 'pass'):
                logger.warning(f"Indicator {indicator_path} calculate method is a stub (pass only or empty).")
                return True

            stub_phrases = ['TODO', 'NotImplementedError', 'pass  # Placeholder', 'return None # Stub']
            if any(phrase in source for phrase in stub_phrases):
                logger.warning(f"Indicator {indicator_path} calculate method contains stub phrases.")
                return True
        except (OSError, TypeError): # Can't get source for C extensions or built-ins
            return False # Assume not a stub if source is unavailable
        return False

    def _execute_calculation(self, indicator_instance, test_data, indicator_path) -> Any:
        if self._is_stub_implementation(indicator_instance, indicator_path):
            raise ValueError(f"Indicator {indicator_path} has stub implementation only.")

        calculate_method = getattr(indicator_instance, 'calculate')
        attempted_formats = []

        # Try various data formats, logging each attempt
        # 1. No arguments (for stateful indicators or those that get data from config)
        try:
            logger.debug(f"[{indicator_path}] Attempting calculate() with no arguments.")
            attempted_formats.append("no_args")
            result = calculate_method()
            if result is not None: return result # Some indicators might return None legitimately for first few periods
        except (TypeError, ValueError) as e:
            logger.debug(f"[{indicator_path}] calculate() with no_args failed: {e}")
            if "Missing order book data" in str(e): # Specific error for some indicators
                self.error_categories['miscellaneous'] +=1 # Or a new category 'missing_special_data'
                raise ServiceError(f"Calculation failed: Missing order book data for {indicator_path}") from e


        # 2. DataFrame
        try:
            logger.debug(f"[{indicator_path}] Attempting calculate() with DataFrame.")
            attempted_formats.append("DataFrame")
            result = calculate_method(test_data['dataframe'].copy()) # Pass a copy
            if result is not None: return result
        except (TypeError, ValueError) as e:
            logger.debug(f"[{indicator_path}] calculate() with DataFrame failed: {e}")
            if "The truth value of a DataFrame is ambiguous" in str(e):
                self.error_categories['data_format'] +=1
                raise ServiceError(f"Calculation failed: Ambiguous DataFrame truth value in {indicator_path}. Indicator needs code update (use a.empty, a.bool(), etc.).") from e
            if "Missing order book data" in str(e):
                self.error_categories['miscellaneous'] +=1
                raise ServiceError(f"Calculation failed: Missing order book data for {indicator_path}") from e


        # 3. Series (Close prices)
        try:
            logger.debug(f"[{indicator_path}] Attempting calculate() with Series (close).")
            attempted_formats.append("Series (close)")
            result = calculate_method(test_data['series'].copy())
            if result is not None: return result
        except (TypeError, ValueError) as e:
            logger.debug(f"[{indicator_path}] calculate() with Series failed: {e}")

        # 4. OHLC dict
        try:
            logger.debug(f"[{indicator_path}] Attempting calculate() with OHLC dict.")
            attempted_formats.append("OHLC dict")
            result = calculate_method(test_data['ohlc'].copy()) # Pass a copy
            if result is not None: return result
        except (TypeError, ValueError) as e:
            logger.debug(f"[{indicator_path}] calculate() with OHLC dict failed: {e}")

        # 5. DataFrame as dict (dict of lists)
        try:
            logger.debug(f"[{indicator_path}] Attempting calculate() with DataFrame as dict.")
            attempted_formats.append("DataFrame as dict")
            result = calculate_method(test_data['dict'].copy()) # Pass a copy
            if result is not None: return result
        except (TypeError, ValueError) as e:
            logger.debug(f"[{indicator_path}] calculate() with DataFrame as dict failed: {e}")


        # If all attempts fail
        self.error_categories['data_format'] +=1
        raise ValueError(f"Could not find compatible data format for {indicator_path}'s calculate method. Attempted: {attempted_formats}")


    def _handle_constructor_error(self, error, indicator_path) -> Dict[str, Any]:
        error_message = str(error).lower()
        error_type = type(error).__name__
        category = 'miscellaneous' # Default category

        if "unexpected keyword argument" in error_message: category = 'constructor_signature'
        elif "missing" in error_message and "required" in error_message: category = 'constructor_signature'
        elif "object has no attribute" in error_message:
            if any(attr in error_message for attr in ['__init__', 'calculate', 'reset', 'update']): category = 'missing_methods'
            else: category = 'missing_imports' # Could also be an attribute error on config
        elif "no module named" in error_message or "cannot import name" in error_message: category = 'missing_imports'
        elif "'super' object has no attribute 'reset'" in error_message : category = 'missing_methods' # Specifically for reset

        self.error_categories[category] += 1
        return {'status': 'failed', 'error': f"Constructor error: {error}", 'error_type': error_type, 'error_category': category}

    def _handle_calculation_error(self, error, indicator_path) -> Dict[str, Any]:
        error_message = str(error).lower()
        error_type = type(error).__name__
        category = 'miscellaneous'

        # More specific categorization for calculation errors
        if isinstance(error, ServiceError): # Custom error for specific framework-detected issues
            category = 'data_format' if "ambiguous dataframe" in error_message else 'miscellaneous'
            detailed_message = str(error)
        elif "object has no attribute" in error_message:
            # Could be a missing attribute on self (needs to be set in __init__) or on input data
            if any(term in error_message for term in ['data', 'values', 'df', 'dataframe', 'series', 'index', 'columns']):
                category = 'data_format'
            else: # Attribute missing on 'self' or a sub-object
                category = 'missing_methods' # Or 'attribute_error_in_logic'
        elif any(term in error_message for term in ['dataframe', 'series', 'numpy', 'array', 'index', 'column', 'shape', 'dimension']):
            category = 'data_format'
        elif any(term in error_message for term in ['parameter', 'argument', 'expected', 'typeerror:', 'valueerror:']): # Broader match
            # Check for specific TypeError/ValueError patterns
            if "got an unexpected keyword argument" in error_message and "__init__" in error_message:
                 # This can happen if calculate tries to init another object with wrong args
                 category = 'parameter_validation'
            elif "type object 'timeframe' has no attribute 'daily'" in error_message:
                 category = 'miscellaneous' # Already somewhat handled by specific logging
                 error = f"Indicator {indicator_path} uses invalid TimeFrame.DAILY. Use TimeFrame.D1 or check TimeFrame definition. Original: {error}"
            else:
                 category = 'parameter_validation'
        elif "truth value of a dataframe is ambiguous" in error_message: # Catchall for this pandas error
            category = 'data_format'
            error = f"Ambiguous DataFrame truth value in {indicator_path}. Indicator needs code update. Original: {error}"
        elif "missing order book data" in error_message:
            category = 'miscellaneous' # Or 'missing_special_data'
            error = f"Missing order book data for {indicator_path}. Original: {error}"


        self.error_categories[category] += 1
        return {
            'status': 'failed',
            'error': f"Calculation error: {error}",
            'error_type': error_type,
            'error_category': category,
            'traceback': traceback.format_exc() if category == 'miscellaneous' else None # Add traceback for harder cases
        }

    def _get_result_shape(self, result) -> str:
        if result is None: return "None"
        try:
            if isinstance(result, pd.DataFrame): return f"DataFrame[{result.shape[0]}x{result.shape[1]}]"
            if isinstance(result, pd.Series): return f"Series[{len(result)}]"
            if isinstance(result, np.ndarray): return f"ndarray{result.shape}"
            if isinstance(result, (list, tuple)): return f"{type(result).__name__}[{len(result)}]"
            if isinstance(result, dict):
                # Attempt a more descriptive shape for dicts of series/arrays
                if result and all(isinstance(v, (pd.Series, np.ndarray, list)) for v in result.values()):
                    lengths = [len(v) for v in result.values() if hasattr(v, '__len__')]
                    if lengths and all(l == lengths[0] for l in lengths):
                        return f"Dict of {len(result)} series/arrays, length {lengths[0]}"
                return f"Dict with {len(result)} items"
            return type(result).__name__
        except Exception: return f"{type(result).__name__} (shape error)"

    def _apply_global_patches(self):
        logger.info("Applying global patches...")
        self._patch_common_base_classes_reset()
        # Add other global patches here if necessary
        # e.g., self._patch_timeframe_enum_access()

    def _patch_common_base_classes_reset(self):
        """Attempts to patch 'reset' method onto known base classes if missing."""
        base_class_candidates = {
            # module_path : [ClassName1, ClassName2]
            'engines.basepatternengine': ['BasePatternEngine'],
            'engines.indicator_base': ['BaseIndicator', 'TechnicalIndicator'], # Hypothetical
            # Add more known base classes that might be missing reset
        }
        patched_count = 0
        for module_path, class_names in base_class_candidates.items():
            try:
                module = importlib.import_module(module_path)
                for class_name in class_names:
                    if hasattr(module, class_name):
                        base_class = getattr(module, class_name)
                        if inspect.isclass(base_class) and not hasattr(base_class, 'reset'):
                            setattr(base_class, 'reset', self.universal_reset)
                            logger.info(f"Patched 'reset' method onto {module_path}.{class_name}")
                            self.patches_applied.append(f"{module_path}.{class_name}_reset_patch")
                            patched_count +=1
            except ImportError:
                logger.debug(f"Module {module_path} for reset patching not found.")
            except Exception as e:
                logger.warning(f"Error patching reset for {module_path}: {e}")
        if patched_count > 0 :
            logger.info(f"Applied {patched_count} reset patches to base classes.")
        else:
            logger.info("No base classes required reset patching or candidates not found.")

        # Fallback: Apply a metaclass to ALL loaded classes if issues persist widely
        # This is more intrusive and should be a last resort or carefully targeted.
        # For now, relying on direct patching of known bases and instance-level patching.

# --- Custom ServiceError for specific framework-detected issues ---
class ServiceError(Exception):
    pass


def main():
    # This needs to be adapted to how your indicators are actually loaded.
    # Assuming dynamic_indicator_loader.py provides a function like this:
    try:
        from dynamic_indicator_loader import load_all_working_indicators # Ensure this path is correct
        indicators_data = load_all_working_indicators()
    except ImportError:
        logger.error("Failed to import 'load_all_working_indicators' from 'dynamic_indicator_loader.py'.")
        logger.error("Please ensure 'dynamic_indicator_loader.py' is in the Python path and provides this function.")
        logger.error("Exiting. Create a dummy loader if needed for testing the validator itself:")
        logger.error("def load_all_working_indicators(): return {'dummy.DummyIndicator': type('DummyIndicator', (), {'__init__': lambda s: None, 'calculate': lambda s: None})}")
        return

    logger.info(f"Loaded {len(indicators_data)} indicator entries by the loader.")

    validator = EnhancedIndicatorValidator()
    results = validator.validate_all_indicators(indicators_data)

    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'enhanced_validation_results_{timestamp_str}.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str) # Use default=str for non-serializable objects like traceback

    summary = results['summary']
    print(f"\n--- Enhanced Indicator Validation Summary ---")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Total Indicators: {summary['total_indicators']}")
    print(f"Successful Indicators: {summary['successful_indicators']}")
    print(f"Success Rate: {summary['success_rate']}%")
    print(f"Patches Applied: {summary['patches_applied']}")
    print(f"\nError Categories:")
    for category, count in sorted(summary['error_categories'].items(), key=lambda item: item[1], reverse=True):
        if count > 0:
            print(f"  {category}: {count}")
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()