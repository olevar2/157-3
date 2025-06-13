"""
DeMarker Indicator

The DeMarker (DeM) indicator is a technical analysis tool developed by Tom Demark that 
identifies potential price exhaustion points. It compares the period's high and low 
to the previous period's corresponding values to measure buying and selling pressure.

Key Features:
- Identifies price exhaustion zones
- Oscillates between 0 and 1
- Overbought/oversold signals
- Leading indicator for reversal points
- Works well in ranging markets

Mathematical Formula:
DeMax(i) = max(High(i) - High(i-1), 0)
DeMin(i) = max(Low(i-1) - Low(i), 0)

DeMarker = SMA(DeMax, period) / (SMA(DeMax, period) + SMA(DeMin, period))

Trading Signals:
- Values above 0.7: Overbought (potential sell signal)
- Values below 0.3: Oversold (potential buy signal)
- Values between 0.3-0.7: Neutral zone
- Crossovers of 0.7 and 0.3 levels provide signals

Author: Platform3.AI Engine
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Temporarily import for testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base_indicator import StandardIndicatorInterface


@dataclass
class DeMarkerConfig:
    """Configuration for DeMarker Indicator calculation."""
    
    # Standard period
    period: int = 14  # Period for SMA calculation
    
    # Signal levels
    overbought_level: float = 0.7  # Overbought threshold
    oversold_level: float = 0.3   # Oversold threshold
    
    # Signal analysis
    enable_signals: bool = True  # Generate trading signals
    enable_divergence: bool = True  # Divergence analysis
    
    # Advanced features
    divergence_periods: int = 10  # Periods to look back for divergence
    
    # Performance settings
    max_history: int = 100  # Maximum historical data to keep
    precision: int = 6  # Decimal precision for calculations


class DeMarkerIndicator(StandardIndicatorInterface):
    """
    DeMarker Indicator
    
    A momentum oscillator that identifies potential price exhaustion points
    by comparing current period highs and lows to previous period values.
    """
    
    def __init__(self, config: Optional[DeMarkerConfig] = None):
        """
        Initialize DeMarker Indicator.
        
        Args:
            config: DeMarker configuration parameters
        """
        self.config = config or DeMarkerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Price histories
        self.high_history: List[float] = []
        self.low_history: List[float] = []
        self.close_history: List[float] = []  # For divergence analysis
        
        # DeMarker components
        self.demax_history: List[float] = []
        self.demin_history: List[float] = []
        
        # DeMarker value
        self.demarker_value: float = 0.5
        self.demarker_history: List[float] = []
        
        # Signal analysis
        self.market_condition: str = "neutral"  # "overbought", "oversold", "neutral"
        self.signal_type: str = "none"  # "buy", "sell", "none"
        
        # Divergence tracking
        self.bullish_divergence: bool = False
        self.bearish_divergence: bool = False
        
        # State management
        self.is_initialized = False
        self._last_calculation_time = None
        
        self.logger.info("DeMarker Indicator initialized successfully")
    
    def calculate(self, data: Union[pd.DataFrame, Dict]) -> Dict:
        """
        Calculate DeMarker Indicator.
        
        Args:
            data: Market data containing OHLC
            
        Returns:
            Dictionary containing DeMarker calculation results
        """
        try:
            # Validate and extract data
            ohlc_data = self._validate_and_extract_data(data)
            if not ohlc_data:
                return self._get_empty_result()
            
            # Add to price histories
            self.high_history.append(ohlc_data['high'])
            self.low_history.append(ohlc_data['low'])
            self.close_history.append(ohlc_data['close'])
            
            # Maintain history size
            if len(self.high_history) > self.config.max_history:
                self.high_history.pop(0)
                self.low_history.pop(0)
                self.close_history.pop(0)
            
            # Calculate DeMax and DeMin
            self._calculate_demax_demin()
            
            # Calculate DeMarker value
            self._calculate_demarker()
            
            # Analyze market condition
            self._analyze_market_condition()
            
            # Generate signals
            signals = {}
            if self.config.enable_signals:
                signals = self._generate_signals()
            
            # Perform divergence analysis
            if self.config.enable_divergence:
                self._analyze_divergence()
            
            # Mark as initialized
            if len(self.high_history) >= self.config.period + 1:  # +1 for comparison
                self.is_initialized = True
            
            self._last_calculation_time = datetime.now()
            
            return self._format_results(ohlc_data, signals)
            
        except Exception as e:
            self.logger.error(f"Error in DeMarker calculation: {str(e)}")
            return self._get_error_result(str(e))
    
    def _validate_and_extract_data(self, data: Union[pd.DataFrame, Dict]) -> Optional[Dict]:
        """Validate input data and extract OHLC values."""
        try:
            if isinstance(data, pd.DataFrame):
                if len(data) == 0:
                    return None
                
                latest = data.iloc[-1]
                return {
                    'high': float(latest.get('high', latest.get('High', 0))),
                    'low': float(latest.get('low', latest.get('Low', 0))),
                    'close': float(latest.get('close', latest.get('Close', 0))),
                    'timestamp': latest.get('timestamp', datetime.now())
                }
            
            elif isinstance(data, dict):
                return {
                    'high': float(data.get('high', data.get('High', 0))),
                    'low': float(data.get('low', data.get('Low', 0))),
                    'close': float(data.get('close', data.get('Close', 0))),
                    'timestamp': data.get('timestamp', datetime.now())
                }
            
            return None
            
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Data validation error: {str(e)}")
            return None
    
    def _calculate_demax_demin(self):
        """Calculate DeMax and DeMin values."""
        if len(self.high_history) < 2 or len(self.low_history) < 2:
            return
        
        # Current period values
        current_high = self.high_history[-1]
        current_low = self.low_history[-1]
        
        # Previous period values
        previous_high = self.high_history[-2]
        previous_low = self.low_history[-2]
        
        # Calculate DeMax: max(High(i) - High(i-1), 0)
        demax = max(current_high - previous_high, 0.0)
        
        # Calculate DeMin: max(Low(i-1) - Low(i), 0)
        demin = max(previous_low - current_low, 0.0)
        
        # Store values
        self.demax_history.append(demax)
        self.demin_history.append(demin)
        
        # Maintain history size
        if len(self.demax_history) > self.config.max_history:
            self.demax_history.pop(0)
            self.demin_history.pop(0)
    
    def _calculate_demarker(self):
        """Calculate DeMarker value."""
        if len(self.demax_history) < self.config.period or len(self.demin_history) < self.config.period:
            return
        
        # Calculate SMA of DeMax and DeMin over the period
        sma_demax = sum(self.demax_history[-self.config.period:]) / self.config.period
        sma_demin = sum(self.demin_history[-self.config.period:]) / self.config.period
        
        # Calculate DeMarker value
        denominator = sma_demax + sma_demin
        if denominator != 0:
            self.demarker_value = sma_demax / denominator
        else:
            self.demarker_value = 0.5  # Neutral value when no movement
        
        # Store DeMarker history
        self.demarker_history.append(self.demarker_value)
        if len(self.demarker_history) > self.config.max_history:
            self.demarker_history.pop(0)
    
    def _analyze_market_condition(self):
        """Analyze current market condition based on DeMarker value."""
        if self.demarker_value >= self.config.overbought_level:
            self.market_condition = "overbought"
        elif self.demarker_value <= self.config.oversold_level:
            self.market_condition = "oversold"
        else:
            self.market_condition = "neutral"
    
    def _generate_signals(self) -> Dict:
        """Generate trading signals based on DeMarker analysis."""
        signals = {
            'market_condition': self.market_condition,
            'signal_type': 'none',
            'signal_strength': 0.0,
            'level_cross': 'none',
            'trend_direction': 'neutral'
        }
        
        if not self.is_initialized or len(self.demarker_history) < 2:
            return signals
        
        current_dem = self.demarker_history[-1]
        previous_dem = self.demarker_history[-2]
        
        # Check for level crossovers
        # Bullish signal: crossing above oversold level
        if previous_dem <= self.config.oversold_level and current_dem > self.config.oversold_level:
            signals['signal_type'] = 'buy'
            signals['level_cross'] = 'oversold_exit'
            signals['signal_strength'] = 0.7
            signals['trend_direction'] = 'bullish'
        
        # Bearish signal: crossing below overbought level
        elif previous_dem >= self.config.overbought_level and current_dem < self.config.overbought_level:
            signals['signal_type'] = 'sell'
            signals['level_cross'] = 'overbought_exit'
            signals['signal_strength'] = -0.7
            signals['trend_direction'] = 'bearish'
        
        # Warning signals: entering extreme zones
        elif previous_dem < self.config.overbought_level and current_dem >= self.config.overbought_level:
            signals['signal_type'] = 'sell_warning'
            signals['level_cross'] = 'overbought_entry'
            signals['signal_strength'] = -0.5
            signals['trend_direction'] = 'bearish_warning'
        
        elif previous_dem > self.config.oversold_level and current_dem <= self.config.oversold_level:
            signals['signal_type'] = 'buy_warning'
            signals['level_cross'] = 'oversold_entry'
            signals['signal_strength'] = 0.5
            signals['trend_direction'] = 'bullish_warning'
        
        # Momentum signals based on DeMarker direction
        if len(self.demarker_history) >= 3:
            momentum = current_dem - self.demarker_history[-3]
            if abs(momentum) > 0.1:  # Significant momentum
                if momentum > 0:
                    signals['trend_direction'] = 'bullish_momentum'
                    if signals['signal_strength'] == 0.0:
                        signals['signal_strength'] = min(momentum * 2, 0.8)
                else:
                    signals['trend_direction'] = 'bearish_momentum'
                    if signals['signal_strength'] == 0.0:
                        signals['signal_strength'] = max(momentum * 2, -0.8)
        
        return signals
    
    def _analyze_divergence(self):
        """Analyze price vs DeMarker divergence."""
        self.bullish_divergence = False
        self.bearish_divergence = False
        
        if (len(self.demarker_history) < self.config.divergence_periods or 
            len(self.close_history) < self.config.divergence_periods):
            return
        
        # Get recent data for analysis
        recent_dem = self.demarker_history[-self.config.divergence_periods:]
        recent_prices = self.close_history[-self.config.divergence_periods:]
        
        # Find price lows and highs
        price_min_idx = recent_prices.index(min(recent_prices))
        price_max_idx = recent_prices.index(max(recent_prices))
        
        # Find DeMarker lows and highs
        dem_min_idx = recent_dem.index(min(recent_dem))
        dem_max_idx = recent_dem.index(max(recent_dem))
        
        # Check for bullish divergence (price makes lower low, DeMarker makes higher low)
        if price_min_idx < len(recent_prices) - 2:
            price_low = recent_prices[price_min_idx]
            current_price = recent_prices[-1]
            
            dem_low = recent_dem[dem_min_idx]
            current_dem = recent_dem[-1]
            
            # Price lower low, DeMarker higher low
            if (current_price <= price_low and current_dem > dem_low and 
                current_dem < self.config.oversold_level):
                self.bullish_divergence = True
        
        # Check for bearish divergence (price makes higher high, DeMarker makes lower high)
        if price_max_idx < len(recent_prices) - 2:
            price_high = recent_prices[price_max_idx]
            current_price = recent_prices[-1]
            
            dem_high = recent_dem[dem_max_idx]
            current_dem = recent_dem[-1]
            
            # Price higher high, DeMarker lower high
            if (current_price >= price_high and current_dem < dem_high and 
                current_dem > self.config.overbought_level):
                self.bearish_divergence = True
    
    def _format_results(self, ohlc_data: Dict, signals: Dict) -> Dict:
        """Format calculation results."""
        result = {
            # Core DeMarker values
            'demarker_value': round(self.demarker_value, self.config.precision),
            'market_condition': self.market_condition,
            
            # Component values
            'demax_sma': round(sum(self.demax_history[-self.config.period:]) / self.config.period 
                             if len(self.demax_history) >= self.config.period else 0, self.config.precision),
            'demin_sma': round(sum(self.demin_history[-self.config.period:]) / self.config.period 
                             if len(self.demin_history) >= self.config.period else 0, self.config.precision),
            
            # Levels
            'overbought_level': self.config.overbought_level,
            'oversold_level': self.config.oversold_level,
            'distance_to_overbought': round(self.config.overbought_level - self.demarker_value, 4),
            'distance_to_oversold': round(self.demarker_value - self.config.oversold_level, 4),
            
            # Analysis
            'bullish_divergence': self.bullish_divergence,
            'bearish_divergence': self.bearish_divergence,
            
            # Metadata
            'is_initialized': self.is_initialized,
            'data_points': len(self.high_history),
            'calculation_time': self._last_calculation_time.isoformat() if self._last_calculation_time else None,
            'status': 'active' if self.is_initialized else 'initializing'
        }
        
        # Add signals if enabled
        if self.config.enable_signals:
            result.update(signals)
        
        return result
    
    def _get_empty_result(self) -> Dict:
        """Return empty result structure."""
        return {
            'demarker_value': 0.5,
            'market_condition': 'neutral',
            'signal_type': 'none',
            'status': 'no_data',
            'error': 'Insufficient data for calculation'
        }
    
    def _get_error_result(self, error_message: str) -> Dict:
        """Return error result structure."""
        return {
            'demarker_value': 0.5,
            'market_condition': 'neutral',
            'signal_type': 'none',
            'status': 'error',
            'error': error_message
        }
    
    def get_historical_values(self, periods: int = 20) -> Dict:
        """Get historical values for DeMarker and components."""
        return {
            'demarker_values': self.demarker_history[-periods:] if self.demarker_history else [],
            'demax_values': self.demax_history[-periods:] if self.demax_history else [],
            'demin_values': self.demin_history[-periods:] if self.demin_history else []
        }
    
    def get_signal_analysis(self) -> Dict:
        """Get detailed signal analysis."""
        return {
            'current_demarker': self.demarker_value,
            'market_condition': self.market_condition,
            'is_overbought': self.demarker_value >= self.config.overbought_level,
            'is_oversold': self.demarker_value <= self.config.oversold_level,
            'is_neutral': self.config.oversold_level < self.demarker_value < self.config.overbought_level,
            'exhaustion_level': 'high' if (self.demarker_value > 0.8 or self.demarker_value < 0.2) else 'normal'
        }
    
    def get_divergence_analysis(self) -> Dict:
        """Get detailed divergence analysis."""
        return {
            'bullish_divergence': self.bullish_divergence,
            'bearish_divergence': self.bearish_divergence,
            'divergence_periods': self.config.divergence_periods,
            'current_demarker': self.demarker_value,
            'reversal_potential': self.bullish_divergence or self.bearish_divergence
        }
    
    def reset(self):
        """Reset indicator state."""
        self.high_history.clear()
        self.low_history.clear()
        self.close_history.clear()
        self.demax_history.clear()
        self.demin_history.clear()
        self.demarker_value = 0.5
        self.demarker_history.clear()
        self.market_condition = "neutral"
        self.signal_type = "none"
        self.bullish_divergence = False
        self.bearish_divergence = False
        self.is_initialized = False
        self._last_calculation_time = None
        
        self.logger.info("DeMarker Indicator reset completed")
    
    def get_metadata(self) -> Dict:
        """Get indicator metadata."""
        return {
            'name': 'DeMarker',
            'full_name': 'DeMarker Indicator',
            'description': 'Momentum oscillator identifying price exhaustion points using high/low comparisons',
            'category': 'Momentum',
            'version': '1.0.0',
            'author': 'Platform3.AI Engine',
            'tags': ['momentum', 'oscillator', 'exhaustion', 'overbought', 'oversold'],
            'inputs': ['high', 'low', 'close'],
            'outputs': ['demarker_value', 'market_condition', 'signals', 'divergence'],
            'parameters': {
                'period': self.config.period,
                'overbought_level': self.config.overbought_level,
                'oversold_level': self.config.oversold_level
            }
        }
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters."""
        try:
            # Validate period
            if self.config.period <= 0:
                self.logger.error("Invalid period: must be positive")
                return False
            
            # Validate levels
            if not (0 <= self.config.oversold_level <= 1):
                self.logger.error("Invalid oversold_level: must be between 0 and 1")
                return False
            
            if not (0 <= self.config.overbought_level <= 1):
                self.logger.error("Invalid overbought_level: must be between 0 and 1")
                return False
            
            if self.config.oversold_level >= self.config.overbought_level:
                self.logger.error("oversold_level must be less than overbought_level")
                return False
            
            # Validate divergence periods
            if self.config.divergence_periods <= 0:
                self.logger.error("Invalid divergence_periods: must be positive")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Parameter validation error: {str(e)}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Test data - oscillating market with exhaustion points
    test_data = []
    base_price = 100.0
    
    # Generate test data with price exhaustion patterns
    for i in range(40):
        # Create oscillating pattern with exhaustion points
        cycle = np.sin(i * 0.3) * 2  # Oscillation
        trend = 0.01 * i  # Slight upward trend
        
        # Add exhaustion spikes
        if i == 15:  # Exhaustion high
            spike = 1.5
        elif i == 30:  # Exhaustion low
            spike = -1.5
        else:
            spike = 0
        
        noise = np.random.normal(0, 0.1)
        
        price_base = base_price + trend + cycle + spike + noise
        high = price_base + abs(np.random.normal(0, 0.2))
        low = price_base - abs(np.random.normal(0, 0.2))
        close = price_base
        
        test_data.append({
            'high': high,
            'low': low,
            'close': close,
            'timestamp': f'2024-01-01 09:{30+i}:00'
        })
    
    # Initialize indicator
    config = DeMarkerConfig(enable_signals=True, enable_divergence=True)
    demarker = DeMarkerIndicator(config)
    
    print("=== DeMarker Indicator Test ===")
    print(f"Config: {config}")
    print()
    
    # Process test data
    for i, data_point in enumerate(test_data, 1):
        result = demarker.calculate(data_point)
        
        # Print every 3rd period after initialization
        if i % 3 == 0 and i >= 18:
            print(f"Period {i}:")
            print(f"  Price: {data_point['close']:.2f}")
            print(f"  DeMarker: {result['demarker_value']}")
            print(f"  Condition: {result['market_condition']}")
            
            if 'signal_type' in result:
                print(f"  Signal: {result['signal_type']}")
                print(f"  Level Cross: {result['level_cross']}")
                print(f"  Signal Strength: {result['signal_strength']:.3f}")
                print(f"  Trend: {result['trend_direction']}")
            
            if result.get('bullish_divergence') or result.get('bearish_divergence'):
                print(f"  Bullish Divergence: {result['bullish_divergence']}")
                print(f"  Bearish Divergence: {result['bearish_divergence']}")
            
            print(f"  Status: {result['status']}")
            print()
    
    # Test signal analysis
    print("=== Signal Analysis ===")
    signal_analysis = demarker.get_signal_analysis()
    for key, value in signal_analysis.items():
        print(f"{key}: {value}")
    
    # Test historical values
    print(f"\n=== Historical Values (last 3 periods) ===")
    historical = demarker.get_historical_values(3)
    for indicator_name, values in historical.items():
        print(f"{indicator_name}: {[round(v, 4) for v in values]}")