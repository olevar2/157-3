#!/usr/bin/env python3
"""
Comprehensive Indicator Adapter for Platform3 - EXACTLY 67 INDICATORS
======================================================================

This adapter provides access to EXACTLY 67 Platform3 indicators through a standardized interface.

Author: Platform3 Development Team
Version: 2.1.0 - EXACT 67 COVERAGE
"""

import sys
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import time
import traceback

# Add ALL the indicators paths
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
services_dir = os.path.join(current_dir, 'services', 'analytics-service', 'src', 'engines')

# Add all necessary paths to sys.path
paths_to_add = [
    os.path.join(services_dir, 'indicators'),
    os.path.join(services_dir, 'indicators', 'momentum'),
    os.path.join(services_dir, 'indicators', 'trend'),
    os.path.join(services_dir, 'indicators', 'volatility'),
    os.path.join(services_dir, 'indicators', 'volume'),
    os.path.join(services_dir, 'indicators', 'cycle'),
    os.path.join(services_dir, 'indicators', 'advanced'),
    os.path.join(services_dir, 'volume'),
    os.path.join(services_dir),
    os.path.join(services_dir, 'gann'),
    os.path.join(services_dir, 'daytrading'),
    os.path.join(services_dir, 'scalping'),
    os.path.join(services_dir, 'swingtrading'),
    os.path.join(services_dir, 'ml', 'scalping'),
    os.path.join(services_dir, 'signals'),
    os.path.join(services_dir, 'fibonacci'),
    os.path.join(services_dir, 'fractal_geometry'),
    os.path.join(services_dir, 'indicators', 'pivot'),
    os.path.join(services_dir, 'pivot')
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.append(path)

# Import the swingtrading package to make classes available
try:
    import swingtrading
    # Success: swingtrading package imported
except ImportError as e:
    # Warning: Failed to import swingtrading package
    pass

logger = logging.getLogger(__name__)

class IndicatorCategory(Enum):
    MOMENTUM = "momentum"
    TREND = "trend"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    CYCLE = "cycle"
    ADVANCED = "advanced"
    GANN = "gann"
    SCALPING = "scalping"
    DAYTRADING = "daytrading"
    SWINGTRADING = "swingtrading"
    SIGNALS = "signals"

@dataclass
class MarketData:
    """Standardized market data structure"""
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    timestamp: Optional[np.ndarray] = None

@dataclass
class IndicatorResult:
    """Standardized indicator result structure"""
    indicator_name: str
    category: IndicatorCategory
    values: Union[float, np.ndarray, Dict, None]
    signals: Optional[Dict]
    metadata: Dict
    calculation_time: float
    success: bool
    error_message: Optional[str] = None

class ComprehensiveIndicatorAdapter_67:
    """Comprehensive adapter for EXACTLY 67 Platform3 indicators"""

    def __init__(self):
        self.interface_cache = {}
        self.performance_stats = {}
        self._initialize_complete_registry()

    def _initialize_complete_registry(self):
        """Initialize registry with EXACTLY 67 indicators"""
        self.all_indicators = {
            # MOMENTUM INDICATORS (8 indicators)
            'RSI': ('momentum.RSI', 'RSI', IndicatorCategory.MOMENTUM),
            'MACD': ('momentum.MACD', 'MACD', IndicatorCategory.MOMENTUM),
            'Stochastic': ('momentum.Stochastic', 'Stochastic', IndicatorCategory.MOMENTUM),
            'ScalpingMomentum': ('momentum.ScalpingMomentum', 'ScalpingMomentum', IndicatorCategory.MOMENTUM),
            'DayTradingMomentum': ('momentum.DayTradingMomentum', 'DayTradingMomentum', IndicatorCategory.MOMENTUM),
            'SwingMomentum': ('momentum.SwingMomentum', 'SwingMomentum', IndicatorCategory.MOMENTUM),
            'FastMomentumOscillators': ('FastMomentumOscillators', 'FastMomentumOscillators', IndicatorCategory.MOMENTUM),
            'SessionMomentum': ('SessionMomentum', 'SessionMomentum', IndicatorCategory.MOMENTUM),

            # TREND INDICATORS (4 indicators)
            'SMA_EMA': ('trend.SMA_EMA', 'SMA_EMA', IndicatorCategory.TREND),
            'ADX': ('trend.ADX', 'ADX', IndicatorCategory.TREND),
            'Ichimoku': ('trend.Ichimoku', 'Ichimoku', IndicatorCategory.TREND),
            'IntradayTrendAnalysis': ('IntradayTrendAnalysis', 'IntradayTrendAnalysis', IndicatorCategory.TREND),

            # VOLATILITY INDICATORS (9 indicators)
            'ATR': ('volatility.ATR', 'ATR', IndicatorCategory.VOLATILITY),
            'BollingerBands': ('volatility.BollingerBands', 'BollingerBands', IndicatorCategory.VOLATILITY),
            'Vortex': ('volatility.Vortex', 'Vortex', IndicatorCategory.VOLATILITY),
            'CCI': ('volatility.CCI', 'CCI', IndicatorCategory.VOLATILITY),
            'KeltnerChannels': ('volatility.KeltnerChannels', 'KeltnerChannels', IndicatorCategory.VOLATILITY),
            'ParabolicSAR': ('volatility.ParabolicSAR', 'ParabolicSAR', IndicatorCategory.VOLATILITY),
            'SuperTrend': ('volatility.SuperTrend', 'SuperTrend', IndicatorCategory.VOLATILITY),
            'VolatilitySpikesDetector': ('VolatilitySpikesDetector', 'VolatilitySpikesDetector', IndicatorCategory.VOLATILITY),
            'TimeWeightedVolatility': ('advanced.TimeWeightedVolatility', 'TimeWeightedVolatility', IndicatorCategory.VOLATILITY),

            # VOLUME INDICATORS (9 indicators)
            'OBV': ('OBV', 'OBV', IndicatorCategory.VOLUME),
            'VolumeProfiles': ('VolumeProfiles', 'VolumeProfiles', IndicatorCategory.VOLUME),
            'OrderFlowImbalance': ('OrderFlowImbalance', 'OrderFlowImbalance', IndicatorCategory.VOLUME),
            'MFI': ('MFI', 'MFI', IndicatorCategory.VOLUME),
            'VFI': ('VFI', 'VFI', IndicatorCategory.VOLUME),
            'AdvanceDecline': ('AdvanceDecline', 'AdvanceDecline', IndicatorCategory.VOLUME),
            'SmartMoneyIndicators': ('SmartMoneyIndicators', 'SmartMoneyIndicators', IndicatorCategory.VOLUME),
            'VolumeSpreadAnalysis': ('VolumeSpreadAnalysis', 'VolumeSpreadAnalysis', IndicatorCategory.VOLUME),
            'TickVolumeIndicators': ('TickVolumeIndicators', 'TickVolumeIndicators', IndicatorCategory.VOLUME),

            # CYCLE INDICATORS (3 indicators)
            'HurstExponent': ('cycle.HurstExponent', 'HurstExponent', IndicatorCategory.CYCLE),
            'FisherTransform': ('cycle.FisherTransform', 'FisherTransform', IndicatorCategory.CYCLE),
            'Alligator': ('cycle.Alligator', 'Alligator', IndicatorCategory.CYCLE),

            # ADVANCED INDICATORS (7 indicators)
            'AutoencoderFeatures': ('advanced.AutoencoderFeatures', 'AutoencoderFeatures', IndicatorCategory.ADVANCED),
            'PCAFeatures': ('advanced.PCAFeatures', 'PCAFeatures', IndicatorCategory.ADVANCED),
            'SentimentScores': ('advanced.SentimentScores', 'SentimentScores', IndicatorCategory.ADVANCED),
            'NoiseFilter': ('NoiseFilter', 'NoiseFilter', IndicatorCategory.ADVANCED),
            'ScalpingLSTM': ('ScalpingLSTM', 'ScalpingLSTM', IndicatorCategory.ADVANCED),
            'SpreadPredictor': ('SpreadPredictor', 'SpreadPredictor', IndicatorCategory.ADVANCED),
            'TickClassifier': ('TickClassifier', 'TickClassifier', IndicatorCategory.ADVANCED),

            # GANN INDICATORS (11 indicators)
            'GannAnglesCalculator': ('GannAnglesCalculator', 'GannAnglesCalculator', IndicatorCategory.GANN),
            'GannFanAnalysis': ('GannFanAnalysis', 'GannFanAnalysis', IndicatorCategory.GANN),
            'GannPatternDetector': ('GannPatternDetector', 'GannPatternDetector', IndicatorCategory.GANN),
            'GannSquareOfNine': ('GannSquareOfNine', 'GannSquareOfNine', IndicatorCategory.GANN),
            'GannTimePrice': ('GannTimePrice', 'GannTimePrice', IndicatorCategory.GANN),
            'FractalGeometryIndicator': ('FractalGeometryIndicator', 'FractalGeometryIndicator', IndicatorCategory.GANN),
            'ProjectionArcCalculator': ('ProjectionArcCalculator', 'ProjectionArcCalculator', IndicatorCategory.GANN),
            'TimeZoneAnalysis': ('TimeZoneAnalysis', 'TimeZoneAnalysis', IndicatorCategory.GANN),
            'ConfluenceDetector': ('ConfluenceDetector', 'ConfluenceDetector', IndicatorCategory.GANN),
            'FibonacciExtension': ('FibonacciExtension', 'FibonacciExtension', IndicatorCategory.GANN),
            'FibonacciRetracement': ('FibonacciRetracement', 'FibonacciRetracement', IndicatorCategory.GANN),

            # SCALPING INDICATORS (5 indicators)
            'MicrostructureFilters': ('MicrostructureFilters', 'MicrostructureFilters', IndicatorCategory.SCALPING),
            'OrderBookAnalysis': ('OrderBookAnalysis', 'OrderBookAnalysis', IndicatorCategory.SCALPING),
            'ScalpingPriceAction': ('ScalpingPriceAction', 'ScalpingPriceAction', IndicatorCategory.SCALPING),
            'VWAPScalping': ('VWAPScalping', 'VWAPScalping', IndicatorCategory.SCALPING),
            'PivotPointCalculator': ('pivot.PivotPointCalculator', 'PivotPointCalculator', IndicatorCategory.SCALPING),
              # DAYTRADING INDICATORS (1 indicator)
            'SessionBreakouts': ('SessionBreakouts', 'SessionBreakouts', IndicatorCategory.DAYTRADING),

            # SWINGTRADING INDICATORS (5 indicators)
            'QuickFibonacci': ('QuickFibonacci', 'QuickFibonacci', IndicatorCategory.SWINGTRADING),
            'SessionSupportResistance': ('SessionSupportResistance', 'SessionSupportResistance', IndicatorCategory.SWINGTRADING),
            'ShortTermElliottWaves': ('ShortTermElliottWaves', 'ShortTermElliottWaves', IndicatorCategory.SWINGTRADING),
            'SwingHighLowDetector': ('SwingHighLowDetector', 'SwingHighLowDetector', IndicatorCategory.SWINGTRADING),
            'RapidTrendlines': ('RapidTrendlines', 'RapidTrendlines', IndicatorCategory.SWINGTRADING),

            # SIGNALS INDICATORS (5 indicators)
            'ConfidenceCalculator': ('ConfidenceCalculator', 'ConfidenceCalculator', IndicatorCategory.SIGNALS),
            'ConflictResolver': ('ConflictResolver', 'ConflictResolver', IndicatorCategory.SIGNALS),
            'QuickDecisionMatrix': ('QuickDecisionMatrix', 'QuickDecisionMatrix', IndicatorCategory.SIGNALS),
            'SignalAggregator': ('SignalAggregator', 'SignalAggregator', IndicatorCategory.SIGNALS),
            'TimeframeSynchronizer': ('TimeframeSynchronizer', 'TimeframeSynchronizer', IndicatorCategory.SIGNALS),        }

        # Comprehensive adapter initialized with all indicators

    def calculate_indicator(self, indicator_name: str, market_data: MarketData, **kwargs) -> IndicatorResult:
        """Calculate any of the 67 indicators with standardized interface"""
        start_time = time.time()

        try:
            # Validate input data
            if len(market_data.close) == 0:
                return IndicatorResult(
                    indicator_name=indicator_name,
                    category=IndicatorCategory.ADVANCED,
                    values=None,
                    signals=None,
                    metadata={},
                    calculation_time=0,
                    success=False,
                    error_message="Empty market data provided"
                )

            # Check for data consistency
            data_lengths = [len(market_data.open), len(market_data.high), len(market_data.low),
                          len(market_data.close), len(market_data.volume)]
            if len(set(data_lengths)) > 1:
                return IndicatorResult(
                    indicator_name=indicator_name,
                    category=IndicatorCategory.ADVANCED,
                    values=None,
                    signals=None,
                    metadata={},
                    calculation_time=0,
                    success=False,
                    error_message="Inconsistent data array lengths"
                )

            if indicator_name not in self.all_indicators:
                return IndicatorResult(
                    indicator_name=indicator_name,
                    category=IndicatorCategory.ADVANCED,
                    values=None,
                    signals=None,
                    metadata={},
                    calculation_time=0,
                    success=False,
                    error_message=f"Indicator {indicator_name} not found"
                )

            module_path, class_name, category = self.all_indicators[indicator_name]
            indicator = self._instantiate_indicator(module_path, class_name, **kwargs)

            if indicator is None:
                return IndicatorResult(
                    indicator_name=indicator_name,
                    category=category,
                    values=None,
                    signals=None,
                    metadata={},
                    calculation_time=0,
                    success=False,
                    error_message=f"Failed to instantiate {indicator_name}"
                )

            # Calculate with auto-interface detection
            result_values = self._calculate_with_auto_interface(indicator, market_data)
            calculation_time = (time.time() - start_time) * 1000

            return IndicatorResult(
                indicator_name=indicator_name,
                category=category,
                values=result_values,
                signals=None,
                metadata={'data_points': len(market_data.close)},
                calculation_time=calculation_time,
                success=True
            )

        except Exception as e:
            calculation_time = (time.time() - start_time) * 1000
            return IndicatorResult(
                indicator_name=indicator_name,
                category=IndicatorCategory.ADVANCED,
                values=None,
                signals=None,
                metadata={},
                calculation_time=calculation_time,
                success=False,
                error_message=str(e)
            )

    def _instantiate_indicator(self, module_path: str, class_name: str, **kwargs) -> Optional[Any]:
        """Safely instantiate any indicator with smart parameter handling"""
        try:
            # Special handling for swingtrading classes
            if class_name in ['SwingHighLowDetector', 'RapidTrendlines']:
                import swingtrading
                indicator_class = getattr(swingtrading, class_name)
            else:
                module = __import__(module_path, fromlist=[class_name])
                indicator_class = getattr(module, class_name)

            # Create a default logger for indicators that need it
            import logging
            default_logger = logging.getLogger(f"{class_name}_logger")

            # Smart parameter handling for different indicator types
            if class_name == 'AutoencoderFeatures':
                if 'input_dim' not in kwargs:
                    kwargs['input_dim'] = 10

            # Indicators that require logger parameter
            logger_required_indicators = [
                'FastMomentumOscillators', 'SessionMomentum', 'IntradayTrendAnalysis',
                'VolatilitySpikesDetector', 'MicrostructureFilters', 'OrderBookAnalysis',
                'ScalpingPriceAction', 'VWAPScalping', 'SessionBreakouts'
            ]

            if class_name in logger_required_indicators:
                kwargs['logger'] = default_logger

            # Indicators that require symbol parameter
            symbol_required_indicators = ['ScalpingLSTM', 'TickClassifier']
            if class_name in symbol_required_indicators:
                kwargs['symbol'] = 'EURUSD'  # Default symbol

            # Try different instantiation patterns
            instantiation_attempts = [
                lambda: indicator_class(**kwargs),
                lambda: indicator_class(logger=default_logger, **kwargs),
                lambda: indicator_class(symbol='EURUSD', **kwargs),
                lambda: indicator_class(logger=default_logger, symbol='EURUSD', **kwargs),
                lambda: indicator_class()
            ]

            for attempt in instantiation_attempts:
                try:
                    return attempt()
                except Exception:
                    continue

            # If all attempts fail, log the error
            raise Exception(f"All instantiation attempts failed for {class_name}")

        except Exception as e:
            logger.error(f"Error instantiating {class_name}: {e}")
            return None

    def _calculate_with_auto_interface(self, indicator: Any, market_data: MarketData) -> Union[float, np.ndarray, Dict]:
        """Auto-detect and use correct interface"""
        interfaces_to_try = [
            'calculate_close_only',
            'calculate_high_low_close',
            'calculate_close_volume',
            'analyze_close',
            'calculate_custom',
        ]

        for interface in interfaces_to_try:
            try:
                result = self._execute_interface(indicator, market_data, interface)
                if result is not None:
                    return result
            except:
                continue

        return None

    def _execute_interface(self, indicator: Any, market_data: MarketData, interface: str):
        """Execute specific interface method"""
        if interface == 'calculate_close_only':
            if hasattr(indicator, 'calculate'):
                return indicator.calculate(market_data.close)

        elif interface == 'calculate_high_low_close':
            if hasattr(indicator, 'calculate'):
                return indicator.calculate(market_data.high, market_data.low, market_data.close)

        elif interface == 'calculate_close_volume':
            if hasattr(indicator, 'calculate'):
                return indicator.calculate(market_data.close, market_data.volume)

        elif interface == 'analyze_close':
            if hasattr(indicator, 'analyze'):
                return indicator.analyze(market_data.close)

        elif interface == 'calculate_custom':
            methods = [m for m in dir(indicator) if 'calculate' in m.lower()]
            for method_name in methods:
                try:
                    method = getattr(indicator, method_name)
                    if callable(method):
                        return method(market_data.close)
                except:
                    continue

        return None

    def batch_calculate(self, indicator_names: List[str], market_data: MarketData) -> Dict[str, IndicatorResult]:
        """Calculate multiple indicators efficiently"""
        results = {}
        for name in indicator_names:
            results[name] = self.calculate_indicator(name, market_data)
        return results

    def get_category_indicators(self, category: IndicatorCategory) -> List[str]:
        """Get all indicators in a specific category"""
        return [name for name, (_, _, cat) in self.all_indicators.items() if cat == category]

    def get_all_indicator_names(self) -> List[str]:
        """Get all available indicator names"""
        return list(self.all_indicators.keys())

    def get_performance_stats(self) -> Dict[str, Dict]:
        """Get performance statistics for all indicators"""
        return self.performance_stats.copy()
