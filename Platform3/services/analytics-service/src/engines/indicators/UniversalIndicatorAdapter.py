#!/usr/bin/env python3
"""
Universal Indicator Adapter for Platform3
=========================================

This adapter solves the critical interface inconsistency problem by providing
a standardized way to access all Platform3 indicators regardless of their
individual interfaces.

Features:
- Automatic interface detection
- Standardized data input/output
- Performance optimization
- Error handling and fallbacks
- Comprehensive indicator support

Author: Platform3 Development Team
Version: 1.0.0
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import time
import traceback

# Add the indicators path
sys.path.append('services/analytics-service/src/engines/indicators')
sys.path.append('services/analytics-service/src/engines/volume')
sys.path.append('services/analytics-service/src/engines')
sys.path.append('services/analytics-service/src/engines/gann')
sys.path.append('services/analytics-service/src/engines/daytrading')
sys.path.append('services/analytics-service/src/engines/scalping')
sys.path.append('services/analytics-service/src/engines/swingtrading')
sys.path.append('services/analytics-service/src/engines/ml/scalping')
sys.path.append('services/analytics-service/src/engines/signals')

logger = logging.getLogger(__name__)

class IndicatorCategory(Enum):
    MOMENTUM = "momentum"
    TREND = "trend"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    CYCLE = "cycle"
    ADVANCED = "advanced"
    GANN = "gann"

@dataclass
class MarketData:
    """Standardized market data structure"""
    timestamps: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

@dataclass
class IndicatorResult:
    """Standardized indicator result"""
    indicator_name: str
    category: IndicatorCategory
    values: Union[float, np.ndarray, Dict]
    signals: Optional[Dict]
    metadata: Dict
    calculation_time: float
    success: bool
    error_message: Optional[str] = None

class UniversalIndicatorAdapter:
    """
    Universal adapter that provides standardized access to all Platform3 indicators
    """
    
    def __init__(self):
        self.supported_indicators = {}
        self.interface_cache = {}
        self._initialize_indicator_registry()
    
    def _initialize_indicator_registry(self):
        """Initialize the registry of all supported indicators"""
        self.supported_indicators = {
            # Momentum Indicators
            IndicatorCategory.MOMENTUM: {
                'RSI': ('momentum.RSI', 'RSI'),
                'MACD': ('momentum.MACD', 'MACD'),
                'Stochastic': ('momentum.Stochastic', 'Stochastic'),
                'ScalpingMomentum': ('momentum.ScalpingMomentum', 'ScalpingMomentum'),
                'DayTradingMomentum': ('momentum.DayTradingMomentum', 'DayTradingMomentum'),
                'SwingMomentum': ('momentum.SwingMomentum', 'SwingMomentum'),
            },            # Trend Indicators
            IndicatorCategory.TREND: {
                'SMA_EMA': ('trend.SMA_EMA', 'SMA_EMA'),
                'ADX': ('trend.ADX', 'ADX'),
                'Ichimoku': ('trend.Ichimoku', 'Ichimoku'),
            },            # Volatility Indicators
            IndicatorCategory.VOLATILITY: {
                'ATR': ('volatility.ATR', 'ATR'),
                'BollingerBands': ('volatility.BollingerBands', 'BollingerBands'),
                'Vortex': ('volatility.Vortex', 'Vortex'),
            },
              # Volume Indicators
            IndicatorCategory.VOLUME: {
                'OBV': ('volume.OBV', 'OBV'),
                'VolumeProfiles': ('VolumeProfiles', 'VolumeProfiles'),
                'OrderFlowImbalance': ('OrderFlowImbalance', 'OrderFlowImbalance'),
            },
            
            # Advanced Indicators
            IndicatorCategory.ADVANCED: {
                'AutoencoderFeatures': ('advanced.AutoencoderFeatures', 'AutoencoderFeatures'),
            }
        }
    
    def calculate_indicator(self, 
                          indicator_name: str, 
                          market_data: MarketData,
                          category: Optional[IndicatorCategory] = None,
                          **kwargs) -> IndicatorResult:
        """
        Calculate any indicator with standardized interface
        
        Args:
            indicator_name: Name of the indicator
            market_data: Standardized market data
            category: Indicator category (auto-detected if not provided)
            **kwargs: Additional parameters for indicator initialization
            
        Returns:
            IndicatorResult with standardized output
        """
        start_time = time.time()
        
        try:
            # Find indicator in registry
            indicator_info = self._find_indicator(indicator_name, category)
            if not indicator_info:
                return IndicatorResult(
                    indicator_name=indicator_name,
                    category=category or IndicatorCategory.ADVANCED,
                    values=None,
                    signals=None,
                    metadata={},
                    calculation_time=0,
                    success=False,
                    error_message=f"Indicator '{indicator_name}' not found in registry"
                )
            
            module_path, class_name, found_category = indicator_info
            
            # Import and instantiate indicator
            indicator = self._instantiate_indicator(module_path, class_name, **kwargs)
            if not indicator:
                return IndicatorResult(
                    indicator_name=indicator_name,
                    category=found_category,
                    values=None,
                    signals=None,
                    metadata={},
                    calculation_time=0,
                    success=False,
                    error_message=f"Failed to instantiate indicator '{indicator_name}'"
                )
            
            # Detect and use appropriate interface
            result_values = self._calculate_with_auto_interface(indicator, market_data)
            
            # Extract signals if available
            signals = self._extract_signals(indicator, market_data)
            
            calculation_time = (time.time() - start_time) * 1000  # ms
            
            return IndicatorResult(
                indicator_name=indicator_name,
                category=found_category,
                values=result_values,
                signals=signals,
                metadata={
                    'data_points': len(market_data.close),
                    'interface_used': self.interface_cache.get(class_name, 'unknown')
                },
                calculation_time=calculation_time,
                success=True
            )
            
        except Exception as e:
            calculation_time = (time.time() - start_time) * 1000
            logger.error(f"Error calculating {indicator_name}: {e}")
            
            return IndicatorResult(
                indicator_name=indicator_name,
                category=category or IndicatorCategory.ADVANCED,
                values=None,
                signals=None,
                metadata={},
                calculation_time=calculation_time,
                success=False,
                error_message=str(e)            )
    
    def _find_indicator(self, indicator_name: str, category: Optional[IndicatorCategory] = None) -> Optional[Tuple[str, str, IndicatorCategory]]:
        """Find indicator in registry"""
        if category:
            # Search in specific category
            if category in self.supported_indicators:
                if indicator_name in self.supported_indicators[category]:
                    module_path, class_name = self.supported_indicators[category][indicator_name]
                    return module_path, class_name, category
        else:
            # Search all categories
            for cat, indicators in self.supported_indicators.items():
                if indicator_name in indicators:
                    module_path, class_name = indicators[indicator_name]
                    return module_path, class_name, cat
        
        return None
    
    def _instantiate_indicator(self, module_path: str, class_name: str, **kwargs) -> Optional[Any]:
        """Safely instantiate an indicator"""
        try:
            module = __import__(module_path, fromlist=[class_name])
            indicator_class = getattr(module, class_name)
            
            # Special handling for indicators with required parameters
            if class_name == 'AutoencoderFeatures':
                # Provide default input_dim if not specified
                if 'input_dim' not in kwargs:
                    kwargs['input_dim'] = 10  # Default input dimension
            
            # Try with provided kwargs first
            if kwargs:
                return indicator_class(**kwargs)
            else:
                return indicator_class()
                
        except Exception as e:
            logger.error(f"Error instantiating {class_name}: {e}")
            return None
    
    def _calculate_with_auto_interface(self, indicator: Any, market_data: MarketData) -> Union[float, np.ndarray, Dict]:
        """Automatically detect and use the correct interface for calculation"""
        class_name = indicator.__class__.__name__
        
        # Check cache first
        if class_name in self.interface_cache:
            interface = self.interface_cache[class_name]
            return self._execute_interface(indicator, market_data, interface)
        
        # Try different interface methods in order of preference
        interfaces_to_try = [
            'calculate_high_low_close',      # calculate(high, low, close)
            'calculate_close_only',          # calculate(close)
            'calculate_close_volume',        # calculate(close, volume)
            'analyze_full',                  # analyze(high, low, close, timestamps)
            'analyze_close',                 # analyze(close)
            'calculate_sma',                 # For SMA_EMA type indicators
            'calculate_custom',              # Custom method detection
        ]
        
        for interface in interfaces_to_try:
            try:
                result = self._execute_interface(indicator, market_data, interface)
                if result is not None:
                    # Cache successful interface
                    self.interface_cache[class_name] = interface
                    return result
            except Exception:
                continue
        
        # If all interfaces fail, return None
        return None
    
    def _execute_interface(self, indicator: Any, market_data: MarketData, interface: str) -> Union[float, np.ndarray, Dict]:
        """Execute specific interface method"""
        
        if interface == 'calculate_high_low_close':
            if hasattr(indicator, 'calculate'):
                return indicator.calculate(
                    high=market_data.high,
                    low=market_data.low,
                    close=market_data.close
                )
        
        elif interface == 'calculate_close_only':
            if hasattr(indicator, 'calculate'):
                return indicator.calculate(market_data.close)
        
        elif interface == 'calculate_close_volume':
            if hasattr(indicator, 'calculate'):
                return indicator.calculate(market_data.close, market_data.volume)
        
        elif interface == 'analyze_full':
            if hasattr(indicator, 'analyze'):
                return indicator.analyze(
                    high=market_data.high,
                    low=market_data.low,
                    close=market_data.close,
                    timestamps=market_data.timestamps
                )
        
        elif interface == 'analyze_close':
            if hasattr(indicator, 'analyze'):
                return indicator.analyze(market_data.close)
        
        elif interface == 'calculate_sma':
            # Special handling for SMA_EMA type indicators
            if hasattr(indicator, 'calculate_sma'):
                return {
                    'sma': indicator.calculate_sma(market_data.close, 20),
                    'ema': indicator.calculate_ema(market_data.close, 20) if hasattr(indicator, 'calculate_ema') else None
                }
        
        elif interface == 'calculate_custom':
            # Try to find any calculate-like method
            methods = [method for method in dir(indicator) if 'calculate' in method.lower()]
            for method_name in methods:
                try:
                    method = getattr(indicator, method_name)
                    if callable(method):
                        # Try with close prices
                        result = method(market_data.close)
                        if result is not None:
                            return result
                except:
                    continue
        
        return None
    
    def _extract_signals(self, indicator: Any, market_data: MarketData) -> Optional[Dict]:
        """Extract trading signals from indicator if available"""
        try:
            # Try common signal methods
            signal_methods = ['get_trading_signals', 'get_signals', 'analyze']
            
            for method_name in signal_methods:
                if hasattr(indicator, method_name):
                    method = getattr(indicator, method_name)
                    try:
                        if method_name == 'analyze':
                            result = method(market_data.close)
                        else:
                            result = method(market_data.close)
                        
                        if isinstance(result, dict):
                            return result
                        elif hasattr(result, '__dict__'):
                            return result.__dict__
                    except:
                        continue
            
            return None
            
        except Exception:
            return None
    
    def get_supported_indicators(self, category: Optional[IndicatorCategory] = None) -> Dict:
        """Get list of all supported indicators"""
        if category:
            return self.supported_indicators.get(category, {})
        return self.supported_indicators
    
    def batch_calculate(self, 
                       indicator_names: List[str], 
                       market_data: MarketData,
                       **kwargs) -> Dict[str, IndicatorResult]:
        """Calculate multiple indicators efficiently"""
        results = {}
        
        for indicator_name in indicator_names:
            results[indicator_name] = self.calculate_indicator(
                indicator_name, market_data, **kwargs
            )
        
        return results
    
    def validate_indicator(self, indicator_name: str, category: Optional[IndicatorCategory] = None) -> bool:
        """Check if an indicator is supported and can be instantiated"""
        try:
            # Generate minimal test data
            test_data = MarketData(
                timestamps=np.arange(20),
                open=np.random.uniform(1.1000, 1.1100, 20),
                high=np.random.uniform(1.1050, 1.1150, 20),
                low=np.random.uniform(1.0950, 1.1050, 20),
                close=np.random.uniform(1.1000, 1.1100, 20),
                volume=np.random.uniform(1000, 10000, 20)
            )
            
            result = self.calculate_indicator(indicator_name, test_data, category)
            return result.success
            
        except Exception:
            return False


def create_market_data(timestamps: np.ndarray, 
                      open_prices: np.ndarray,
                      high_prices: np.ndarray, 
                      low_prices: np.ndarray,
                      close_prices: np.ndarray, 
                      volumes: np.ndarray) -> MarketData:
    """Helper function to create MarketData from arrays"""
    return MarketData(
        timestamps=timestamps,
        open=open_prices,
        high=high_prices,
        low=low_prices,
        close=close_prices,
        volume=volumes
    )


# Global adapter instance
adapter = UniversalIndicatorAdapter()
