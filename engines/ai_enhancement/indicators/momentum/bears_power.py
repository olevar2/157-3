"""
Bears Power Indicator (Elder)

Bears Power is a technical indicator developed by Dr. Alexander Elder that measures the ability 
of bears (sellers) to drive prices below the exponential moving average. It's part of Elder's 
Triple Screen trading system and helps identify the strength of selling pressure.

Key Features:
- Measures selling pressure strength
- Based on relationship between Low and EMA
- Part of Elder's Triple Screen system  
- Used with Bulls Power for complete market analysis
- Identifies bear market strength and potential reversals

Mathematical Formula:
Bears Power = Low - EMA(Close, period)
where:
- Low = Period's lowest price
- EMA = Exponential Moving Average of closing prices
- Standard period = 13

Trading Signals:
- Negative values: Bears are in control (typical)
- Values approaching zero: Weakening bear pressure
- Positive values: Very strong buying pressure (rare)
- Divergence with price: Potential trend reversal
- Use with Bulls Power for complete picture

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
from base_indicator import StandardIndicatorInterface


@dataclass
class BearsPowerConfig:
    """Configuration for Bears Power Indicator calculation."""
    
    # Elder's standard period
    ema_period: int = 13  # EMA period for calculation
    
    # Signal analysis
    enable_signals: bool = True  # Generate trading signals
    enable_divergence: bool = True  # Divergence analysis
    
    # Signal thresholds
    strength_threshold: float = 0.01  # Threshold for strength classification
    divergence_periods: int = 10  # Periods to look back for divergence
    
    # Performance settings
    max_history: int = 100  # Maximum historical data to keep
    precision: int = 6  # Decimal precision for calculations


class BearsPowerIndicator(StandardIndicatorInterface):
    """
    Bears Power Indicator (Elder)
    
    A momentum indicator that measures the power of bears (sellers) to drive
    prices below the exponential moving average, indicating selling pressure strength.
    """
    
    def __init__(self, config: Optional[BearsPowerConfig] = None):
        """
        Initialize Bears Power Indicator.
        
        Args:
            config: Bears Power configuration parameters
        """
        self.config = config or BearsPowerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Price histories
        self.close_history: List[float] = []
        self.low_history: List[float] = []
        
        # EMA calculation
        self.ema_value: float = 0.0
        self.ema_history: List[float] = []
        
        # Bears Power calculation
        self.bears_power: float = 0.0
        self.bears_power_history: List[float] = []
        
        # Analysis variables
        self.bear_strength: str = "neutral"  # "weak", "moderate", "strong", "extreme"
        self.trend_direction: str = "neutral"  # "bullish", "bearish", "neutral"
        
        # Divergence tracking
        self.bullish_divergence: bool = False
        self.bearish_divergence: bool = False
        
        # State management
        self.is_initialized = False
        self._last_calculation_time = None
        
        self.logger.info("Bears Power Indicator initialized successfully")
    
    def calculate(self, data: Union[pd.DataFrame, Dict]) -> Dict:
        """
        Calculate Bears Power Indicator.
        
        Args:
            data: Market data containing OHLC
            
        Returns:
            Dictionary containing Bears Power calculation results
        """
        try:
            # Validate and extract data
            ohlc_data = self._validate_and_extract_data(data)
            if not ohlc_data:
                return self._get_empty_result()
            
            # Add to price histories
            self.close_history.append(ohlc_data['close'])
            self.low_history.append(ohlc_data['low'])
            
            # Maintain history size
            if len(self.close_history) > self.config.max_history:
                self.close_history.pop(0)
                self.low_history.pop(0)
            
            # Calculate EMA
            self._calculate_ema()
            
            # Calculate Bears Power
            self._calculate_bears_power()
            
            # Analyze bear strength
            self._analyze_bear_strength()
            
            # Generate signals
            signals = {}
            if self.config.enable_signals:
                signals = self._generate_signals()
            
            # Perform divergence analysis
            if self.config.enable_divergence:
                self._analyze_divergence()
            
            # Mark as initialized
            if len(self.close_history) >= self.config.ema_period:
                self.is_initialized = True
            
            self._last_calculation_time = datetime.now()
            
            return self._format_results(ohlc_data, signals)
            
        except Exception as e:
            self.logger.error(f"Error in Bears Power calculation: {str(e)}")
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
    
    def _calculate_ema(self):
        """Calculate Exponential Moving Average."""
        if len(self.close_history) == 1:
            # First value - use SMA
            self.ema_value = self.close_history[0]
        elif len(self.close_history) > 1:
            # Calculate EMA using standard formula
            multiplier = 2.0 / (self.config.ema_period + 1)
            self.ema_value = (self.close_history[-1] * multiplier) + (self.ema_value * (1 - multiplier))
        
        # Store EMA history
        self.ema_history.append(self.ema_value)
        if len(self.ema_history) > self.config.max_history:
            self.ema_history.pop(0)
    
    def _calculate_bears_power(self):
        """Calculate Bears Power value."""
        if len(self.low_history) > 0 and self.ema_value > 0:
            self.bears_power = self.low_history[-1] - self.ema_value
        else:
            self.bears_power = 0.0
        
        # Store Bears Power history
        self.bears_power_history.append(self.bears_power)
        if len(self.bears_power_history) > self.config.max_history:
            self.bears_power_history.pop(0)
    
    def _analyze_bear_strength(self):
        """Analyze current bear strength based on Bears Power value."""
        if not self.is_initialized:
            self.bear_strength = "neutral"
            return
        
        # Calculate relative strength based on recent EMA
        if self.ema_value > 0:
            relative_power = abs(self.bears_power) / self.ema_value
            
            if relative_power < 0.005:  # Less than 0.5% of EMA
                self.bear_strength = "weak"
            elif relative_power < 0.015:  # Less than 1.5% of EMA
                self.bear_strength = "moderate"
            elif relative_power < 0.03:  # Less than 3% of EMA
                self.bear_strength = "strong"
            else:
                self.bear_strength = "extreme"
        else:
            self.bear_strength = "neutral"
        
        # Determine overall trend direction
        if self.bears_power > 0:
            self.trend_direction = "bullish"  # Rare - very strong buying
        elif self.bears_power < -self.config.strength_threshold:
            self.trend_direction = "bearish"
        else:
            self.trend_direction = "neutral"
    
    def _generate_signals(self) -> Dict:
        """Generate trading signals based on Bears Power analysis."""
        signals = {
            'bear_strength': self.bear_strength,
            'trend_direction': self.trend_direction,
            'power_level': 'neutral',
            'signal_type': 'none',
            'signal_strength': 0.0
        }
        
        if not self.is_initialized or len(self.bears_power_history) < 2:
            return signals
        
        # Determine power level
        if self.bears_power > 0:
            signals['power_level'] = 'positive'  # Rare bullish signal
        elif self.bears_power < -0.01:
            signals['power_level'] = 'strongly_negative'
        elif self.bears_power < 0:
            signals['power_level'] = 'negative'
        else:
            signals['power_level'] = 'neutral'
        
        # Generate signals based on Bears Power changes
        if len(self.bears_power_history) >= 3:
            current = self.bears_power_history[-1]
            previous = self.bears_power_history[-2]
            prev_prev = self.bears_power_history[-3]
            
            # Check for momentum changes
            if current > previous > prev_prev and current < 0:
                signals['signal_type'] = 'bear_weakening'
                signals['signal_strength'] = 0.6
            elif current < previous < prev_prev and current < 0:
                signals['signal_type'] = 'bear_strengthening'
                signals['signal_strength'] = -0.6
            
            # Check for zero line approaches or crosses
            if previous < 0 and current >= 0:
                signals['signal_type'] = 'bullish_breakout'
                signals['signal_strength'] = 0.8
            elif previous >= 0 and current < 0:
                signals['signal_type'] = 'bearish_return'
                signals['signal_strength'] = -0.4
        
        # Calculate overall signal strength
        if signals['signal_strength'] == 0.0:
            # Base signal strength on Bears Power magnitude and direction
            if self.ema_value > 0:
                normalized_power = self.bears_power / self.ema_value
                signals['signal_strength'] = max(-1.0, min(1.0, normalized_power * 10))
        
        return signals
    
    def _analyze_divergence(self):
        """Analyze price vs Bears Power divergence."""
        self.bullish_divergence = False
        self.bearish_divergence = False
        
        if (len(self.bears_power_history) < self.config.divergence_periods or 
            len(self.close_history) < self.config.divergence_periods):
            return
        
        # Get recent data for analysis
        recent_bears = self.bears_power_history[-self.config.divergence_periods:]
        recent_prices = self.close_history[-self.config.divergence_periods:]
        
        # Find price lows and highs
        price_min_idx = recent_prices.index(min(recent_prices))
        price_max_idx = recent_prices.index(max(recent_prices))
        
        # Find Bears Power lows and highs
        bears_min_idx = recent_bears.index(min(recent_bears))
        bears_max_idx = recent_bears.index(max(recent_bears))
        
        # Check for bullish divergence (price makes lower low, Bears Power makes higher low)
        if price_min_idx < len(recent_prices) - 2:
            price_low = recent_prices[price_min_idx]
            current_price = recent_prices[-1]
            
            bears_low = recent_bears[bears_min_idx]
            current_bears = recent_bears[-1]
            
            # Price lower low, Bears Power higher low (less negative)
            if (current_price <= price_low and current_bears > bears_low and 
                current_bears < 0 and bears_low < 0):
                self.bullish_divergence = True
        
        # Check for bearish divergence (price makes higher high, Bears Power makes lower high)
        if price_max_idx < len(recent_prices) - 2:
            price_high = recent_prices[price_max_idx]
            current_price = recent_prices[-1]
            
            bears_high = recent_bears[bears_max_idx]
            current_bears = recent_bears[-1]
            
            # Price higher high, Bears Power lower high (more negative)
            if (current_price >= price_high and current_bears < bears_high and
                current_bears < 0):
                self.bearish_divergence = True
    
    def _format_results(self, ohlc_data: Dict, signals: Dict) -> Dict:
        """Format calculation results."""
        result = {
            # Core Bears Power values
            'bears_power': round(self.bears_power, self.config.precision),
            'ema_value': round(self.ema_value, self.config.precision),
            'current_low': round(ohlc_data['low'], self.config.precision),
            
            # Analysis
            'bear_strength': self.bear_strength,
            'trend_direction': self.trend_direction,
            'bullish_divergence': self.bullish_divergence,
            'bearish_divergence': self.bearish_divergence,
            
            # Relative metrics
            'power_ratio': round(self.bears_power / self.ema_value if self.ema_value > 0 else 0, 4),
            'distance_from_zero': round(abs(self.bears_power), self.config.precision),
            
            # Metadata
            'is_initialized': self.is_initialized,
            'data_points': len(self.close_history),
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
            'bears_power': 0.0,
            'ema_value': 0.0,
            'bear_strength': 'neutral',
            'trend_direction': 'neutral',
            'status': 'no_data',
            'error': 'Insufficient data for calculation'
        }
    
    def _get_error_result(self, error_message: str) -> Dict:
        """Return error result structure."""
        return {
            'bears_power': 0.0,
            'ema_value': 0.0,
            'bear_strength': 'neutral',
            'trend_direction': 'neutral',
            'status': 'error',
            'error': error_message
        }
    
    def get_historical_values(self, periods: int = 20) -> Dict:
        """
        Get historical values for Bears Power and related indicators.
        
        Args:
            periods: Number of historical periods to return
            
        Returns:
            Dictionary containing historical values
        """
        return {
            'bears_power': self.bears_power_history[-periods:] if self.bears_power_history else [],
            'ema_values': self.ema_history[-periods:] if self.ema_history else [],
            'low_values': self.low_history[-periods:] if self.low_history else []
        }
    
    def get_strength_analysis(self) -> Dict:
        """Get detailed strength analysis."""
        return {
            'current_bears_power': self.bears_power,
            'bear_strength': self.bear_strength,
            'is_bears_in_control': self.bears_power < 0,
            'is_bulls_showing_strength': self.bears_power > -0.001,
            'strength_magnitude': abs(self.bears_power),
            'relative_strength': abs(self.bears_power / self.ema_value) if self.ema_value > 0 else 0
        }
    
    def get_divergence_analysis(self) -> Dict:
        """Get detailed divergence analysis."""
        return {
            'bullish_divergence': self.bullish_divergence,
            'bearish_divergence': self.bearish_divergence,
            'divergence_periods': self.config.divergence_periods,
            'current_bears_power': self.bears_power,
            'trend_confirmation': self.bears_power < 0  # Bears typically in control
        }
    
    def reset(self):
        """Reset indicator state."""
        self.close_history.clear()
        self.low_history.clear()
        self.ema_value = 0.0
        self.ema_history.clear()
        self.bears_power = 0.0
        self.bears_power_history.clear()
        self.bear_strength = "neutral"
        self.trend_direction = "neutral"
        self.bullish_divergence = False
        self.bearish_divergence = False
        self.is_initialized = False
        self._last_calculation_time = None
        
        self.logger.info("Bears Power Indicator reset completed")
    
    def get_metadata(self) -> Dict:
        """Get indicator metadata."""
        return {
            'name': 'Bears Power',
            'full_name': 'Bears Power Indicator (Elder)',
            'description': 'Measures the power of bears (sellers) to drive prices below EMA',
            'category': 'Momentum',
            'version': '1.0.0',
            'author': 'Platform3.AI Engine',
            'tags': ['momentum', 'elder', 'bears', 'selling pressure', 'triple screen'],
            'inputs': ['high', 'low', 'close'],
            'outputs': ['bears_power', 'bear_strength', 'signals', 'divergence'],
            'parameters': {
                'ema_period': self.config.ema_period,
                'strength_threshold': self.config.strength_threshold,
                'divergence_periods': self.config.divergence_periods
            }
        }
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters."""
        try:
            # Validate EMA period
            if self.config.ema_period <= 0:
                self.logger.error("Invalid ema_period: must be positive")
                return False
            
            # Validate strength threshold
            if self.config.strength_threshold < 0:
                self.logger.error("Invalid strength_threshold: must be non-negative")
                return False
            
            # Validate divergence periods
            if self.config.divergence_periods <= 0:
                self.logger.error("Invalid divergence_periods: must be positive")
                return False
            
            # Check Elder's standard configuration
            if self.config.ema_period != 13:
                self.logger.warning("Non-standard EMA period detected. Elder's standard: 13 periods")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Parameter validation error: {str(e)}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Test data - declining market with bear pressure
    test_data = []
    base_price = 100.0
    
    # Generate test data with varying bear pressure
    for i in range(30):
        # Create declining trend with varying selling pressure
        if i < 10:
            trend = -0.01 * i  # Gradual decline
        elif i < 20:
            trend = -0.01 * 10 - 0.03 * (i - 10)  # Steeper decline
        else:
            trend = -0.01 * 10 - 0.03 * 10 - 0.005 * (i - 20)  # Slowing decline
        
        noise = np.random.normal(0, 0.1)
        
        high = base_price + trend + noise + 0.15
        low = base_price + trend + noise - 0.25  # Lower lows showing bear pressure
        close = base_price + trend + noise
        
        test_data.append({
            'high': high,
            'low': low,
            'close': close,
            'timestamp': f'2024-01-01 09:{30+i}:00'
        })
    
    # Initialize indicator
    config = BearsPowerConfig(enable_signals=True, enable_divergence=True)
    bears_indicator = BearsPowerIndicator(config)
    
    print("=== Bears Power Indicator Test ===")
    print(f"Config: {config}")
    print()
    
    # Process test data
    for i, data_point in enumerate(test_data, 1):
        result = bears_indicator.calculate(data_point)
        
        # Print every 3rd period after initialization
        if i % 3 == 0 and i >= 15:
            print(f"Period {i}:")
            print(f"  Price: {data_point['close']:.2f}")
            print(f"  Low: {data_point['low']:.2f}")
            print(f"  EMA: {result['ema_value']}")
            print(f"  Bears Power: {result['bears_power']}")
            print(f"  Bear Strength: {result['bear_strength']}")
            print(f"  Power Ratio: {result['power_ratio']}")
            
            if 'signal_type' in result:
                print(f"  Signal Type: {result['signal_type']}")
                print(f"  Signal Strength: {result['signal_strength']:.3f}")
                print(f"  Power Level: {result['power_level']}")
            
            if result.get('bullish_divergence') or result.get('bearish_divergence'):
                print(f"  Bullish Divergence: {result['bullish_divergence']}")
                print(f"  Bearish Divergence: {result['bearish_divergence']}")
            
            print(f"  Status: {result['status']}")
            print()
    
    # Test strength analysis
    print("=== Strength Analysis ===")
    strength_analysis = bears_indicator.get_strength_analysis()
    for key, value in strength_analysis.items():
        print(f"{key}: {value}")
    
    # Test historical values
    print(f"\n=== Historical Values (last 3 periods) ===")
    historical = bears_indicator.get_historical_values(3)
    for indicator_name, values in historical.items():
        print(f"{indicator_name}: {[round(v, 4) for v in values]}")