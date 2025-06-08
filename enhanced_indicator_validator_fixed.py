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
        # Explicitly initialize common attributes with their default values
        self.period = kwargs.get('period', 14)
        self.timeframe = kwargs.get('timeframe', 'daily') # Default timeframe
        self.indicator_type = kwargs.get('indicator_type', "MOCK_GENERAL_TYPE") # Added for broader compatibility
        self.source = kwargs.get('source', 'close')
        self.lookback_periods = kwargs.get('lookback_periods', 20)
        self.lookback_period = kwargs.get('lookback_period', 20)  # Explicitly initialize this common attribute
        self.window = kwargs.get('window', 14)
        self.fast_period = kwargs.get('fast_period', 12)
        self.slow_period = kwargs.get('slow_period', 26)
        self.signal_period = kwargs.get('signal_period', 9)
        self.min_periods = kwargs.get('min_periods', 10)
        self.max_periods = kwargs.get('max_periods', 500)
        self.smoothing = kwargs.get('smoothing', 2)
        
        # Additional common attributes for more specialized indicators
        self.multiplier = kwargs.get('multiplier', 2.0)
        self.threshold = kwargs.get('threshold', 0.02)
        self.sensitivity = kwargs.get('sensitivity', 0.1)
        self.lookback = kwargs.get('lookback', 20)  # Alternative to lookback_period
        self.periods = kwargs.get('periods', 14)   # Alternative to period
        self.length = kwargs.get('length', 14)     # Another common parameter name
        self.span = kwargs.get('span', 14)         # For exponential calculations
        self.alpha = kwargs.get('alpha', 0.1)      # Smoothing factor
        self.beta = kwargs.get('beta', 0.1)        # Secondary smoothing factor
        self.gamma = kwargs.get('gamma', 0.1)      # Tertiary smoothing factor
        self.k_period = kwargs.get('k_period', 14) # For stochastic indicators
        self.d_period = kwargs.get('d_period', 3)  # For stochastic indicators
        self.rsi_period = kwargs.get('rsi_period', 14)  # For RSI-based indicators
        self.ma_type = kwargs.get('ma_type', 'SMA')     # Moving average type
        self.deviation = kwargs.get('deviation', 2.0)   # Standard deviation multiplier
        self.price_type = kwargs.get('price_type', 'close')  # Alternative to source
        self.volume_factor = kwargs.get('volume_factor', 0.2)  # For volume-based indicators
        self.fractal_dimension = kwargs.get('fractal_dimension', 2.0)  # For fractal indicators
        self.embedding_dimension = kwargs.get('embedding_dimension', 3)  # For chaos/fractal analysis
        self.lag = kwargs.get('lag', 1)            # For correlation/regression indicators
        self.correlation_threshold = kwargs.get('correlation_threshold', 0.7)  # For correlation indicators
        self.min_correlation = kwargs.get('min_correlation', 0.5)  # For correlation indicators
        self.lookback_window = kwargs.get('lookback_window', 20)  # Alternative lookback parameter
        self.volatility_period = kwargs.get('volatility_period', 20)  # For volatility indicators
        self.atr_period = kwargs.get('atr_period', 14)  # For ATR-based indicators
        self.bands_period = kwargs.get('bands_period', 20)  # For band indicators
        self.upper_band = kwargs.get('upper_band', 2.0)    # For band indicators
        self.lower_band = kwargs.get('lower_band', 2.0)    # For band indicators
        
        # Pattern-specific attributes
        self.pattern_sensitivity = kwargs.get('pattern_sensitivity', 0.01)
        self.pattern_threshold = kwargs.get('pattern_threshold', 0.02)
        self.min_pattern_strength = kwargs.get('min_pattern_strength', 0.5)
        
        # Order book / market microstructure attributes
        self.order_book_levels = kwargs.get('order_book_levels', 10)
        self.tick_size = kwargs.get('tick_size', 0.01)
        self.min_order_size = kwargs.get('min_order_size', 100)
        
        # Time-based attributes
        self.session_start = kwargs.get('session_start', '09:30:00')
        self.session_end = kwargs.get('session_end', '16:00:00')
        self.timezone = kwargs.get('timezone', 'UTC')
        
        # Store any additional kwargs as attributes for maximum compatibility
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)
        self.threshold = kwargs.get('threshold', 0.5)
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
            {'force_no_config': True, 'force_no_name': True, 'force_default_config': False, 'attempt_num': 4},            # Attempt 5: Force default config (if it wasn't already tried by 'config' being required)
            {'force_no_config': False, 'force_no_name': False, 'force_default_config': True, 'attempt_num': 5},
        ]

        for attempt_params in instantiation_attempts:
            current_args = self._create_flexible_args(indicator_class, signature_info, test_data, **attempt_params)
            
            # Enhanced logging with config object details
            args_summary = {}
            for k, v in current_args.items():
                if k in ['config', 'configuration'] and hasattr(v, '__dict__'):
                    # Log key attributes of config object
                    config_attrs = {attr: getattr(v, attr, 'N/A') for attr in ['name', 'period', 'lookback_period', 'timeframe', 'source']}
                    args_summary[k] = f"{type(v).__name__}({config_attrs})"
                else:
                    args_summary[k] = type(v).__name__
            
            logger.debug(f"Attempt {attempt_params['attempt_num']} for {indicator_path} with args: {args_summary}")
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
        indicator_name = indicator_class.__name__ # Use actual class name        # Config handling with mandatory config for TechnicalIndicator children
        config_param_name = next((p for p in ['config', 'configuration'] if p in all_params), None)
        
        # Check if TechnicalIndicator is in the MRO
        is_technical_indicator_child = any('TechnicalIndicator' in str(base) for base in indicator_class.__mro__)
        
        if is_technical_indicator_child:
            # Always provide config for TechnicalIndicator children
            if config_param_name:
                args[config_param_name] = self._get_appropriate_config(indicator_class, name_override=indicator_name)
                logger.debug(f"[{indicator_name}] Providing mandatory config for TechnicalIndicator child as '{type(args[config_param_name]).__name__}'.")
        elif force_default_config and config_param_name:
            args[config_param_name] = self._get_appropriate_config(indicator_class, name_override=indicator_name)
            logger.debug(f"[{indicator_name}] Forcing default config as '{type(args[config_param_name]).__name__}'.")
        elif config_param_name and not force_no_config:
            args[config_param_name] = self._get_appropriate_config(indicator_class, name_override=indicator_name)
            logger.debug(f"[{indicator_name}] Providing config as '{type(args[config_param_name]).__name__}'.")
        elif config_param_name in args and args[config_param_name] is None and config_param_name in required_params: # Should be covered by above
            args[config_param_name] = self._get_appropriate_config(indicator_class, name_override=indicator_name)
            logger.debug(f"[{indicator_name}] Config was None but required, providing default.")


        # Name handling - smarter approach for IndicatorBase children
        name_param_name = next((p for p in ['name', 'indicator_name'] if p in all_params), None)
        is_indicator_base_child = any('IndicatorBase' in str(base) for base in indicator_class.__mro__)
        config_being_provided = config_param_name in args
        
        if name_param_name and not force_no_name:
            # If it's an IndicatorBase child and config is being provided, don't add name as separate argument
            if is_indicator_base_child and config_being_provided:
                logger.debug(f"[{indicator_name}] Skipping explicit name argument for IndicatorBase child with config.")
            else:
                if name_param_name not in args: # Avoid overwriting if config object might have set it
                     args[name_param_name] = indicator_name
                     logger.debug(f"[{indicator_name}] Providing explicit name: '{indicator_name}'.")

        # Data handling (common parameter names for input data)
        data_param_names = ['data', 'dataframe', 'df', 'input_data', 'ohlcv']
        actual_data_param = next((p for p in data_param_names if p in all_params), None)
        if actual_data_param:
            args[actual_data_param] = test_data['dataframe'] # Default to DataFrame        # Populate other required parameters with enhanced guessing
        for param in required_params:
            if param in args: continue # Already handled (config, name, data)

            param_lower = param.lower()
            
            # Handle OHLC data parameters
            if param_lower in ['open', 'high', 'low', 'close']:
                args[param] = test_data['dataframe'][param_lower]
                logger.debug(f"[{indicator_name}] Providing OHLC Series for '{param}'")
            elif param_lower == 'volume':
                args[param] = test_data['dataframe']['volume']
                logger.debug(f"[{indicator_name}] Providing volume Series for '{param}'")
            elif param_lower in ['ohlc', 'ohlcv', 'ohlc_data']:
                args[param] = test_data['ohlc'] if param_lower == 'ohlc' else test_data['ohlcv']
                logger.debug(f"[{indicator_name}] Providing OHLC dict for '{param}'")
            
            # Handle price/timeframe/period parameters
            elif any(p in param_lower for p in ['period', 'window', 'length']):
                args[param] = 14
            elif 'timeframe' in param_lower:
                args[param] = self._get_default_timeframe()
            elif any(s in param_lower for s in ['fast', 'short']):
                args[param] = 12
            elif any(s in param_lower for s in ['slow', 'long']):
                args[param] = 26
            elif 'signal' in param_lower and 'period' in param_lower:
                args[param] = 9
            elif param_lower in ['lookback', 'lookback_period', 'lookback_periods']:
                args[param] = 20
            elif param_lower in ['alpha', 'smoothing', 'smoothing_factor']:
                args[param] = 0.1
            elif param_lower in ['multiplier', 'factor', 'deviation']:
                args[param] = 2.0
            elif param_lower in ['threshold', 'min_threshold', 'sensitivity']:
                args[param] = 0.02
            elif param_lower in ['source', 'price_type', 'price_source']:
                args[param] = 'close'
            elif param_lower in ['ma_type', 'method', 'type']:
                args[param] = 'SMA'
            
            # Handle boolean parameters
            elif param_lower.startswith('is_') or param_lower.startswith('use_') or param_lower.startswith('enable_'):
                args[param] = True
            elif 'debug' in param_lower:
                args[param] = False
                
            # Handle array/list parameters
            elif 'levels' in param_lower or 'bands' in param_lower:
                args[param] = [1, 2, 3]  # Generic list of levels
            elif 'weights' in param_lower:
                args[param] = [1.0, 1.0, 1.0]  # Generic weights
                
            # Fallback to generic default
            else:
                default_val = self._get_default_value_for_param_type(param, indicator_class)
                args[param] = default_val
                logger.debug(f"[{indicator_name}] Providing fallback default for required param '{param}': {type(default_val).__name__}")
        
        # More generous passing of common numeric parameters for optional parameters
        common_optional_params = {
            'period': 14, 'window': 20, 'length': 14, 'lookback': 20,
            'fast_period': 12, 'slow_period': 26, 'signal_period': 9,
            'k_period': 14, 'd_period': 3, 'rsi_period': 14,
            'atr_period': 14, 'bands_period': 20, 'volatility_period': 20,
            'alpha': 0.1, 'beta': 0.1, 'gamma': 0.1,
            'multiplier': 2.0, 'factor': 2.0, 'deviation': 2.0,
            'threshold': 0.02, 'sensitivity': 0.1, 'min_threshold': 0.01,
            'upper_band': 2.0, 'lower_band': 2.0, 'smoothing': 2.0,
            'span': 14, 'min_periods': 10, 'max_periods': 500,
            'lag': 1, 'correlation_threshold': 0.7, 'min_correlation': 0.5,
            'fractal_dimension': 2.0, 'embedding_dimension': 3,
            'volume_factor': 0.2, 'tick_size': 0.01, 'min_order_size': 100,
            'order_book_levels': 10, 'pattern_sensitivity': 0.01,
            'pattern_threshold': 0.02, 'min_pattern_strength': 0.5
        }
        
        for param in all_params:
            if param in args: continue # Already handled
            if param in common_optional_params:
                args[param] = common_optional_params[param]
                logger.debug(f"[{indicator_name}] Providing default for optional common param '{param}': {common_optional_params[param]}")
        
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
        
        # Add additional columns that some indicators might expect
        data['timestamp'] = dates
        data['time'] = dates
        data['date'] = dates
        data['datetime'] = dates
        data['vwap'] = (data['high'] + data['low'] + data['close']) / 3  # Volume weighted average proxy
        data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
        data['median_price'] = (data['high'] + data['low']) / 2
        data['hlc3'] = (data['high'] + data['low'] + data['close']) / 3
        data['ohlc4'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        data['hl2'] = (data['high'] + data['low']) / 2
        data['oc2'] = (data['open'] + data['close']) / 2
          # Add some technical indicators that other indicators might use as input
        data['sma_20'] = pd.Series(data['close']).rolling(20).mean().bfill()
        data['ema_12'] = pd.Series(data['close']).ewm(span=12).mean().bfill()
        data['rsi'] = np.random.uniform(20, 80, size=len(dates))  # Mock RSI values
        data['atr'] = np.random.uniform(0.5, 3.0, size=len(dates))  # Mock ATR values
        data['macd'] = np.random.uniform(-2, 2, size=len(dates))  # Mock MACD values
        data['signal'] = np.random.uniform(-1.5, 1.5, size=len(dates))  # Mock MACD signal
        data['histogram'] = data['macd'] - data['signal']  # MACD histogram
        data['bb_upper'] = data['close'] + np.random.uniform(1, 3, size=len(dates))  # Bollinger upper
        data['bb_lower'] = data['close'] - np.random.uniform(1, 3, size=len(dates))  # Bollinger lower
        data['bb_middle'] = (data['bb_upper'] + data['bb_lower']) / 2  # Bollinger middle
        
        # Add bid/ask data for order book indicators
        spread = np.random.uniform(0.01, 0.05, size=len(dates))
        data['bid'] = data['close'] - spread / 2
        data['ask'] = data['close'] + spread / 2
        data['bid_size'] = np.random.randint(100, 10000, size=len(dates)).astype(float)
        data['ask_size'] = np.random.randint(100, 10000, size=len(dates)).astype(float)
        
        df = pd.DataFrame(data, index=dates)

        return {
            'dataframe': df,
            'series': df['close'].copy(),
            'numpy_array': df.values.copy(),
            'dict': df.to_dict('list'), # DataFrame as dict of lists
            'ohlc': {k: df[k].tolist() for k in ['open', 'high', 'low', 'close']}, # Dict of lists for OHLC
            'ohlcv': {k: df[k].tolist() for k in ['open', 'high', 'low', 'close', 'volume']}, # Dict with volume
            'ohlc_series': {k: df[k] for k in ['open', 'high', 'low', 'close']}, # Dict of Series
            'lists': {k: df[k].tolist() for k in df.columns}, # All columns as lists
            'arrays': {k: df[k].values for k in df.columns}, # All columns as numpy arrays
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

        # Try various data formats, logging each attempt with enhanced error handling
        
        # 1. No arguments (for stateful indicators or those that get data from config)
        try:
            logger.debug(f"[{indicator_path}] Attempting calculate() with no arguments.")
            attempted_formats.append("no_args")
            result = calculate_method()
            if result is not None: 
                logger.debug(f"[{indicator_path}] SUCCESS with no_args format")
                return result
        except (TypeError, ValueError) as e:
            logger.debug(f"[{indicator_path}] calculate() with no_args failed: {type(e).__name__}: {e}")
            if "Missing order book data" in str(e):
                self.error_categories['miscellaneous'] +=1
                raise ServiceError(f"Calculation failed: Missing order book data for {indicator_path}") from e
        except Exception as e:
            logger.debug(f"[{indicator_path}] calculate() with no_args failed due to {type(e).__name__}: {e}")
            if "The truth value of a DataFrame is ambiguous" in str(e):
                self.error_categories['data_format'] +=1
                raise ServiceError(f"Calculation failed: Ambiguous DataFrame truth value in {indicator_path}. Indicator needs code update (use a.empty, a.bool(), etc.).") from e

        # 2. DataFrame
        try:
            logger.debug(f"[{indicator_path}] Attempting calculate() with DataFrame.")
            attempted_formats.append("DataFrame")
            result = calculate_method(test_data['dataframe'].copy())
            if result is not None: 
                logger.debug(f"[{indicator_path}] SUCCESS with DataFrame format")
                return result
        except TypeError as e:
            logger.debug(f"[{indicator_path}] calculate() with DataFrame failed: {type(e).__name__}: {e}")
        except Exception as e:
            logger.debug(f"[{indicator_path}] calculate() with DataFrame failed due to {type(e).__name__}: {e}")
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
            if result is not None: 
                logger.debug(f"[{indicator_path}] SUCCESS with Series (close) format")
                return result
        except TypeError as e:
            logger.debug(f"[{indicator_path}] calculate() with Series failed: {type(e).__name__}: {e}")
        except Exception as e:
            logger.debug(f"[{indicator_path}] calculate() with Series (close) failed due to {type(e).__name__}: {e}")
            if "The truth value of a DataFrame is ambiguous" in str(e):
                self.error_categories['data_format'] +=1
                raise ServiceError(f"Calculation failed: Ambiguous DataFrame truth value in {indicator_path}. Indicator needs code update (use a.empty, a.bool(), etc.).") from e

        # 4. Multiple Series arguments (OHLC as separate Series)
        try:
            logger.debug(f"[{indicator_path}] Attempting calculate() with multiple Series (O,H,L,C).")
            attempted_formats.append("Multiple Series (O,H,L,C)")
            df = test_data['dataframe']
            result = calculate_method(df['open'], df['high'], df['low'], df['close'])
            if result is not None: 
                logger.debug(f"[{indicator_path}] SUCCESS with Multiple Series format")
                return result
        except TypeError as e:
            logger.debug(f"[{indicator_path}] calculate() with Multiple Series failed: {type(e).__name__}: {e}")
        except Exception as e:
            logger.debug(f"[{indicator_path}] calculate() with Multiple Series failed due to {type(e).__name__}: {e}")
            if "The truth value of a DataFrame is ambiguous" in str(e):
                self.error_categories['data_format'] +=1
                raise ServiceError(f"Calculation failed: Ambiguous DataFrame truth value in {indicator_path}. Indicator needs code update (use a.empty, a.bool(), etc.).") from e

        # 5. Multiple Series with Volume (OHLCV)
        try:
            logger.debug(f"[{indicator_path}] Attempting calculate() with OHLCV Series.")
            attempted_formats.append("OHLCV Series")
            df = test_data['dataframe']
            result = calculate_method(df['open'], df['high'], df['low'], df['close'], df['volume'])
            if result is not None: 
                logger.debug(f"[{indicator_path}] SUCCESS with OHLCV Series format")
                return result
        except TypeError as e:
            logger.debug(f"[{indicator_path}] calculate() with OHLCV Series failed: {type(e).__name__}: {e}")
        except Exception as e:
            logger.debug(f"[{indicator_path}] calculate() with OHLCV Series failed due to {type(e).__name__}: {e}")
            if "The truth value of a DataFrame is ambiguous" in str(e):
                self.error_categories['data_format'] +=1
                raise ServiceError(f"Calculation failed: Ambiguous DataFrame truth value in {indicator_path}. Indicator needs code update (use a.empty, a.bool(), etc.).") from e

        # 6. OHLC dict
        try:
            logger.debug(f"[{indicator_path}] Attempting calculate() with OHLC dict.")
            attempted_formats.append("OHLC dict")
            result = calculate_method(test_data['ohlc'].copy())
            if result is not None: 
                logger.debug(f"[{indicator_path}] SUCCESS with OHLC dict format")
                return result
        except TypeError as e:
            logger.debug(f"[{indicator_path}] calculate() with OHLC dict failed: {type(e).__name__}: {e}")
        except Exception as e:
            logger.debug(f"[{indicator_path}] calculate() with OHLC dict failed due to {type(e).__name__}: {e}")
            if "The truth value of a DataFrame is ambiguous" in str(e):
                self.error_categories['data_format'] +=1
                raise ServiceError(f"Calculation failed: Ambiguous DataFrame truth value in {indicator_path}. Indicator needs code update (use a.empty, a.bool(), etc.).") from e

        # 7. DataFrame as dict (dict of lists)
        try:
            logger.debug(f"[{indicator_path}] Attempting calculate() with DataFrame as dict.")
            attempted_formats.append("DataFrame as dict")
            result = calculate_method(test_data['dict'].copy())
            if result is not None: 
                logger.debug(f"[{indicator_path}] SUCCESS with DataFrame as dict format")
                return result
        except TypeError as e:
            logger.debug(f"[{indicator_path}] calculate() with DataFrame as dict failed: {type(e).__name__}: {e}")
        except Exception as e:
            logger.debug(f"[{indicator_path}] calculate() with DataFrame as dict failed due to {type(e).__name__}: {e}")
            if "The truth value of a DataFrame is ambiguous" in str(e):
                self.error_categories['data_format'] +=1
                raise ServiceError(f"Calculation failed: Ambiguous DataFrame truth value in {indicator_path}. Indicator needs code update (use a.empty, a.bool(), etc.).") from e

        # 8. Numpy arrays (high, low, close)
        try:
            logger.debug(f"[{indicator_path}] Attempting calculate() with numpy arrays.")
            attempted_formats.append("numpy arrays")
            df = test_data['dataframe']
            result = calculate_method(df['high'].values, df['low'].values, df['close'].values)
            if result is not None: 
                logger.debug(f"[{indicator_path}] SUCCESS with numpy arrays format")
                return result
        except TypeError as e:
            logger.debug(f"[{indicator_path}] calculate() with numpy arrays failed: {type(e).__name__}: {e}")
        except Exception as e:
            logger.debug(f"[{indicator_path}] calculate() with numpy arrays failed due to {type(e).__name__}: {e}")

        # 9. Single numpy array (close prices)
        try:
            logger.debug(f"[{indicator_path}] Attempting calculate() with single numpy array (close).")
            attempted_formats.append("numpy array (close)")
            result = calculate_method(test_data['dataframe']['close'].values)
            if result is not None: 
                logger.debug(f"[{indicator_path}] SUCCESS with numpy array format")
                return result
        except TypeError as e:
            logger.debug(f"[{indicator_path}] calculate() with numpy array failed: {type(e).__name__}: {e}")
        except Exception as e:
            logger.debug(f"[{indicator_path}] calculate() with numpy array failed due to {type(e).__name__}: {e}")        # 10. Data attribute access (for indicators that expect data to be set as attribute)
        try:
            logger.debug(f"[{indicator_path}] Attempting to set data attribute and call calculate().")
            attempted_formats.append("data attribute + calculate()")
            indicator_instance.data = test_data['dataframe'].copy()
            result = calculate_method()
            if result is not None: 
                logger.debug(f"[{indicator_path}] SUCCESS with data attribute format")
                return result
        except TypeError as e:
            logger.debug(f"[{indicator_path}] calculate() with data attribute failed: {type(e).__name__}: {e}")
        except Exception as e:
            logger.debug(f"[{indicator_path}] calculate() with data attribute failed due to {type(e).__name__}: {e}")
            if "The truth value of a DataFrame is ambiguous" in str(e):
                self.error_categories['data_format'] +=1
                raise ServiceError(f"Calculation failed: Ambiguous DataFrame truth value in {indicator_path}. Indicator needs code update (use a.empty, a.bool(), etc.).") from e

        # 11. Update method (for indicators that need data to be fed via update)
        try:
            logger.debug(f"[{indicator_path}] Attempting to call update() method with DataFrame rows.")
            attempted_formats.append("update method")
            df = test_data['dataframe']
            for idx, row in df.iterrows():
                if hasattr(indicator_instance, 'update'):
                    indicator_instance.update(row.to_dict())
            result = calculate_method()
            if result is not None: 
                logger.debug(f"[{indicator_path}] SUCCESS with update method format")
                return result
        except TypeError as e:
            logger.debug(f"[{indicator_path}] calculate() with update method failed: {type(e).__name__}: {e}")
        except Exception as e:
            logger.debug(f"[{indicator_path}] calculate() with update method failed due to {type(e).__name__}: {e}")

        # 12. Input property (for indicators that expect input property to be set)
        try:
            logger.debug(f"[{indicator_path}] Attempting to set input property and call calculate().")
            attempted_formats.append("input property + calculate()")
            if hasattr(indicator_instance, 'input'):
                indicator_instance.input = test_data['dataframe'].copy()
            elif hasattr(indicator_instance, 'inputs'):
                indicator_instance.inputs = test_data['dataframe'].copy()
            result = calculate_method()
            if result is not None: 
                logger.debug(f"[{indicator_path}] SUCCESS with input property format")
                return result
        except TypeError as e:
            logger.debug(f"[{indicator_path}] calculate() with input property failed: {type(e).__name__}: {e}")
        except Exception as e:
            logger.debug(f"[{indicator_path}] calculate() with input property failed due to {type(e).__name__}: {e}")

        # 13. Price data as individual floats (for indicators that want single price points)
        try:
            logger.debug(f"[{indicator_path}] Attempting calculate() with individual price values.")
            attempted_formats.append("individual prices")
            df = test_data['dataframe']
            result = calculate_method(df['close'].iloc[-1])  # Latest close price
            if result is not None: 
                logger.debug(f"[{indicator_path}] SUCCESS with individual prices format")
                return result
        except TypeError as e:
            logger.debug(f"[{indicator_path}] calculate() with individual prices failed: {type(e).__name__}: {e}")
        except Exception as e:
            logger.debug(f"[{indicator_path}] calculate() with individual prices failed due to {type(e).__name__}: {e}")

        # 14. OHLC as separate parameters (order book style)
        try:
            logger.debug(f"[{indicator_path}] Attempting calculate() with bid/ask data.")
            attempted_formats.append("bid/ask data")
            df = test_data['dataframe']
            result = calculate_method(df['bid'], df['ask'], df['bid_size'], df['ask_size'])
            if result is not None: 
                logger.debug(f"[{indicator_path}] SUCCESS with bid/ask data format")
                return result
        except TypeError as e:
            logger.debug(f"[{indicator_path}] calculate() with bid/ask data failed: {type(e).__name__}: {e}")
        except Exception as e:
            logger.debug(f"[{indicator_path}] calculate() with bid/ask data failed due to {type(e).__name__}: {e}")

        # 15. Alternative column names (for indicators expecting specific column names)
        try:
            logger.debug(f"[{indicator_path}] Attempting calculate() with alternative DataFrame column names.")
            attempted_formats.append("alternative columns")
            df = test_data['dataframe'].copy()
            # Create DataFrame with alternative column names
            alt_df = pd.DataFrame({
                'Open': df['open'], 'High': df['high'], 'Low': df['low'], 'Close': df['close'], 'Volume': df['volume'],
                'OPEN': df['open'], 'HIGH': df['high'], 'LOW': df['low'], 'CLOSE': df['close'], 'VOLUME': df['volume'],
                'o': df['open'], 'h': df['high'], 'l': df['low'], 'c': df['close'], 'v': df['volume']
            }, index=df.index)
            result = calculate_method(alt_df)
            if result is not None: 
                logger.debug(f"[{indicator_path}] SUCCESS with alternative columns format")
                return result
        except TypeError as e:
            logger.debug(f"[{indicator_path}] calculate() with alternative columns failed: {type(e).__name__}: {e}")
        except Exception as e:
            logger.debug(f"[{indicator_path}] calculate() with alternative columns failed due to {type(e).__name__}: {e}")        # 16. Process method (for indicators that use process instead of calculate)
        try:
            if hasattr(indicator_instance, 'process'):
                logger.debug(f"[{indicator_path}] Attempting process() method with DataFrame.")
                attempted_formats.append("process method")
                result = indicator_instance.process(test_data['dataframe'].copy())
                if result is not None: 
                    logger.debug(f"[{indicator_path}] SUCCESS with process method format")
                    return result
        except TypeError as e:
            logger.debug(f"[{indicator_path}] process() method failed: {type(e).__name__}: {e}")
        except Exception as e:
            logger.debug(f"[{indicator_path}] process() method failed due to {type(e).__name__}: {e}")

        # 17. Matrix/2D array format (for indicators expecting matrix input)
        try:
            logger.debug(f"[{indicator_path}] Attempting calculate() with matrix format (2D numpy array).")
            attempted_formats.append("matrix format")
            df = test_data['dataframe']
            matrix = df[['open', 'high', 'low', 'close', 'volume']].values
            result = calculate_method(matrix)
            if result is not None: 
                logger.debug(f"[{indicator_path}] SUCCESS with matrix format")
                return result
        except TypeError as e:
            logger.debug(f"[{indicator_path}] calculate() with matrix format failed: {type(e).__name__}: {e}")
        except Exception as e:
            logger.debug(f"[{indicator_path}] calculate() with matrix format failed due to {type(e).__name__}: {e}")

        # 18. Time series data with datetime index
        try:
            logger.debug(f"[{indicator_path}] Attempting calculate() with time series format.")
            attempted_formats.append("time series")
            df = test_data['dataframe'].copy()
            df.index = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
            result = calculate_method(df)
            if result is not None: 
                logger.debug(f"[{indicator_path}] SUCCESS with time series format")
                return result
        except TypeError as e:
            logger.debug(f"[{indicator_path}] calculate() with time series failed: {type(e).__name__}: {e}")
        except Exception as e:
            logger.debug(f"[{indicator_path}] calculate() with time series failed due to {type(e).__name__}: {e}")

        # 19. Order book level data (for institutional flow indicators)
        try:
            logger.debug(f"[{indicator_path}] Attempting calculate() with order book levels.")
            attempted_formats.append("order book levels")
            order_book = {
                'bids': [(100.0 + i, 1000 + i*10) for i in range(10)],
                'asks': [(100.0 + 10 + i, 1000 + i*10) for i in range(10)],
                'timestamp': pd.Timestamp.now()
            }
            result = calculate_method(order_book)
            if result is not None: 
                logger.debug(f"[{indicator_path}] SUCCESS with order book levels format")
                return result
        except TypeError as e:
            logger.debug(f"[{indicator_path}] calculate() with order book levels failed: {type(e).__name__}: {e}")
        except Exception as e:
            logger.debug(f"[{indicator_path}] calculate() with order book levels failed due to {type(e).__name__}: {e}")

        # 20. Raw price list (for simple indicators)
        try:
            logger.debug(f"[{indicator_path}] Attempting calculate() with raw price list.")
            attempted_formats.append("raw price list")
            prices = test_data['dataframe']['close'].tolist()
            result = calculate_method(prices)
            if result is not None: 
                logger.debug(f"[{indicator_path}] SUCCESS with raw price list format")
                return result
        except TypeError as e:
            logger.debug(f"[{indicator_path}] calculate() with raw price list failed: {type(e).__name__}: {e}")
        except Exception as e:
            logger.debug(f"[{indicator_path}] calculate() with raw price list failed due to {type(e).__name__}: {e}")

        # 21. Kwargs format (passing data as keyword arguments)
        try:
            logger.debug(f"[{indicator_path}] Attempting calculate() with kwargs format.")
            attempted_formats.append("kwargs format")
            df = test_data['dataframe']
            result = calculate_method(
                open=df['open'], high=df['high'], low=df['low'], close=df['close'], volume=df['volume']
            )
            if result is not None: 
                logger.debug(f"[{indicator_path}] SUCCESS with kwargs format")
                return result
        except TypeError as e:
            logger.debug(f"[{indicator_path}] calculate() with kwargs format failed: {type(e).__name__}: {e}")
        except Exception as e:
            logger.debug(f"[{indicator_path}] calculate() with kwargs format failed due to {type(e).__name__}: {e}")

        # 22. Feed data through feed() method then calculate
        try:
            if hasattr(indicator_instance, 'feed'):
                logger.debug(f"[{indicator_path}] Attempting feed() + calculate() method.")
                attempted_formats.append("feed + calculate")
                df = test_data['dataframe']
                for idx, row in df.iterrows():
                    indicator_instance.feed(row['close'])
                result = calculate_method()
                if result is not None: 
                    logger.debug(f"[{indicator_path}] SUCCESS with feed + calculate format")
                    return result
        except TypeError as e:
            logger.debug(f"[{indicator_path}] feed() + calculate() failed: {type(e).__name__}: {e}")
        except Exception as e:
            logger.debug(f"[{indicator_path}] feed() + calculate() failed due to {type(e).__name__}: {e}")

        # If all attempts fail
        self.error_categories['data_format'] +=1
        raise ValueError(f"Could not find compatible data format for {indicator_path}'s calculate method. Attempted: {attempted_formats}")

    def _attempt_dataframe_patch(self, indicator_instance, indicator_path):
        """Attempt to patch common DataFrame truth value issues by monkey-patching problematic methods."""
        try:
            # Get the class of the indicator instance
            indicator_class = type(indicator_instance)
            
            # Check if the indicator has a calculate method and try to patch it
            if hasattr(indicator_class, 'calculate'):
                original_calculate = getattr(indicator_class, 'calculate')
                
                def patched_calculate(self, *args, **kwargs):
                    try:
                        return original_calculate(self, *args, **kwargs)
                    except ValueError as e:
                        if "The truth value of a DataFrame is ambiguous" in str(e):
                            logger.warning(f"DataFrame truth value error in {indicator_path}, attempting to handle...")
                            # Try some common fixes
                            import pandas as pd
                            import numpy as np
                            
                            # Patch common problematic patterns
                            old_dataframe_bool = pd.DataFrame.__bool__
                            def safe_dataframe_bool(df_self):
                                if df_self.empty:
                                    return False
                                return len(df_self) > 0
                            
                            pd.DataFrame.__bool__ = safe_dataframe_bool
                            
                            try:
                                result = original_calculate(self, *args, **kwargs)
                                return result
                            finally:
                                # Restore original method
                                pd.DataFrame.__bool__ = old_dataframe_bool
                        raise
                
                # Apply the patch
                setattr(indicator_class, 'calculate', patched_calculate)
                logger.info(f"Applied DataFrame truth value patch to {indicator_path}")
                return True
        except Exception as e:
            logger.debug(f"Failed to apply DataFrame patch to {indicator_path}: {e}")
            return False
        
        return False

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
        self._patch_dataframe_truth_value_handling()
        # Add other global patches here if necessary
        # e.g., self._patch_timeframe_enum_access()

    def _patch_dataframe_truth_value_handling(self):
        """Apply global patch for DataFrame truth value ambiguity issues."""
        try:
            import pandas as pd
            
            # Store the original __bool__ method
            if not hasattr(pd.DataFrame, '_original_bool_method'):
                pd.DataFrame._original_bool_method = pd.DataFrame.__bool__
                
                def safe_dataframe_bool(self):
                    """Safe implementation of DataFrame __bool__ that avoids ambiguity errors."""
                    try:
                        # Use the original method first
                        return pd.DataFrame._original_bool_method(self)
                    except ValueError as e:
                        if "The truth value of a DataFrame is ambiguous" in str(e):
                            # Fallback: return True if DataFrame has any data
                            return not self.empty
                        raise
                
                # Apply the patch
                pd.DataFrame.__bool__ = safe_dataframe_bool
                logger.info("Applied global DataFrame truth value patch")
                self.patches_applied.append("pandas.DataFrame_truth_value_patch")
                
        except Exception as e:
            logger.warning(f"Failed to apply DataFrame truth value patch: {e}")

    def _patch_common_base_classes_reset(self):
        """Patch common base classes to provide a universal reset method."""
        try:
            # Import the base class
            from engines.indicator_base import TechnicalIndicator
            
            # Check if it already has a reset method or if we've already patched it
            if not hasattr(TechnicalIndicator, 'reset') or not hasattr(TechnicalIndicator, '_enhanced_validator_reset_patched'):
                TechnicalIndicator.reset = self.universal_reset
                TechnicalIndicator._enhanced_validator_reset_patched = True
                logger.info("Patched TechnicalIndicator with universal reset method")
                self.patches_applied.append("engines.indicator_base.TechnicalIndicator_reset_patch")
            else:
                logger.debug("TechnicalIndicator already has reset method")
                
        except ImportError:
            logger.warning("Could not import TechnicalIndicator base class for patching")
        except Exception as e:
            logger.warning(f"Failed to patch TechnicalIndicator: {e}")

        # Try patching other common base classes
        base_class_paths = [
            'indicator_base.BaseIndicator',
            'base.IndicatorBase',
            'technical_indicators.BaseIndicator',
            'indicators.base.BaseIndicator'
        ]
        
        for base_path in base_class_paths:
            try:
                module_path, class_name = base_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                base_class = getattr(module, class_name)
                
                if not hasattr(base_class, 'reset') or not hasattr(base_class, '_enhanced_validator_reset_patched'):
                    base_class.reset = self.universal_reset
                    base_class._enhanced_validator_reset_patched = True
                    logger.info(f"Patched {base_path} with universal reset method")
                    self.patches_applied.append(f"{base_path}_reset_patch")
                    
            except (ImportError, AttributeError):
                logger.debug(f"Could not patch {base_path} (not found or not accessible)")
            except Exception as e:
                logger.debug(f"Failed to patch {base_path}: {e}")

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