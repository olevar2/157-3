"""
Bulls Power Indicator (Elder)

Bulls Power is a technical indicator developed by Dr. Alexander Elder that measures the ability 
of bulls (buyers) to drive prices above the exponential moving average. It's part of Elder's 
Triple Screen trading system and works together with Bears Power for complete market analysis.

Key Features:
- Measures buying pressure strength
- Based on relationship between High and EMA
- Part of Elder's Triple Screen system  
- Used with Bears Power for complete market analysis
- Identifies bull market strength and potential reversals

Mathematical Formula:
Bulls Power = High - EMA(Close, period)
where:
- High = Period's highest price
- EMA = Exponential Moving Average of closing prices
- Standard period = 13

Trading Signals:
- Positive values: Bulls are in control (typical)
- Values approaching zero: Weakening bull pressure
- Negative values: Very strong selling pressure (rare)
- Divergence with price: Potential trend reversal
- Use with Bears Power for complete picture

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
class BullsPowerConfig:
    """Configuration for Bulls Power Indicator calculation."""
    
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


class BullsPowerIndicator(StandardIndicatorInterface):
    """
    Bulls Power Indicator (Elder)
    
    A momentum indicator that measures the power of bulls (buyers) to drive
    prices above the exponential moving average, indicating buying pressure strength.
    """
    
    def __init__(self, config: Optional[BullsPowerConfig] = None):
        """
        Initialize Bulls Power Indicator.
        
        Args:
            config: Bulls Power configuration parameters
        """
        self.config = config or BullsPowerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Price histories
        self.close_history: List[float] = []
        self.high_history: List[float] = []
        
        # EMA calculation
        self.ema_value: float = 0.0
        self.ema_history: List[float] = []
        
        # Bulls Power calculation
        self.bulls_power: float = 0.0
        self.bulls_power_history: List[float] = []
        
        # Analysis variables
        self.bull_strength: str = "neutral"  # "weak", "moderate", "strong", "extreme"
        self.trend_direction: str = "neutral"  # "bullish", "bearish", "neutral"
        
        # Divergence tracking
        self.bullish_divergence: bool = False
        self.bearish_divergence: bool = False
        
        # State management
        self.is_initialized = False
        self._last_calculation_time = None
        
        self.logger.info("Bulls Power Indicator initialized successfully")
    
    def calculate(self, data: Union[pd.DataFrame, Dict]) -> Dict:
        """
        Calculate Bulls Power Indicator.
        
        Args:
            data: Market data containing OHLC
            
        Returns:
            Dictionary containing Bulls Power calculation results
        """
        try:
            # Validate and extract data
            ohlc_data = self._validate_and_extract_data(data)
            if not ohlc_data:
                return self._get_empty_result()
            
            # Add to price histories
            self.close_history.append(ohlc_data['close'])
            self.high_history.append(ohlc_data['high'])
            
            # Maintain history size
            if len(self.close_history) > self.config.max_history:
                self.close_history.pop(0)
                self.high_history.pop(0)
            
            # Calculate EMA
            self._calculate_ema()
            
            # Calculate Bulls Power
            self._calculate_bulls_power()
            
            # Analyze bull strength
            self._analyze_bull_strength()
            
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
            self.logger.error(f"Error in Bulls Power calculation: {str(e)}")
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
    
    def _calculate_bulls_power(self):
        """Calculate Bulls Power value."""
        if len(self.high_history) > 0 and self.ema_value > 0:
            self.bulls_power = self.high_history[-1] - self.ema_value
        else:
            self.bulls_power = 0.0
        
        # Store Bulls Power history
        self.bulls_power_history.append(self.bulls_power)
        if len(self.bulls_power_history) > self.config.max_history:
            self.bulls_power_history.pop(0)
    
    def _analyze_bull_strength(self):
        """Analyze current bull strength based on Bulls Power value."""
        if not self.is_initialized:
            self.bull_strength = "neutral"
            return
        
        # Calculate relative strength based on recent EMA
        if self.ema_value > 0:
            relative_power = abs(self.bulls_power) / self.ema_value
            
            if relative_power < 0.005:  # Less than 0.5% of EMA
                self.bull_strength = "weak"
            elif relative_power < 0.015:  # Less than 1.5% of EMA
                self.bull_strength = "moderate"
            elif relative_power < 0.03:  # Less than 3% of EMA
                self.bull_strength = "strong"
            else:
                self.bull_strength = "extreme"
        else:
            self.bull_strength = "neutral"
        
        # Determine overall trend direction
        if self.bulls_power < 0:
            self.trend_direction = "bearish"  # Rare - very strong selling
        elif self.bulls_power > self.config.strength_threshold:
            self.trend_direction = "bullish"
        else:
            self.trend_direction = "neutral"
    
    def _generate_signals(self) -> Dict:
        """Generate trading signals based on Bulls Power analysis."""
        signals = {
            'bull_strength': self.bull_strength,
            'trend_direction': self.trend_direction,
            'power_level': 'neutral',
            'signal_type': 'none',
            'signal_strength': 0.0
        }
        
        if not self.is_initialized or len(self.bulls_power_history) < 2:
            return signals
        
        # Determine power level
        if self.bulls_power < 0:
            signals['power_level'] = 'negative'  # Rare bearish signal
        elif self.bulls_power > 0.01:
            signals['power_level'] = 'strongly_positive'
        elif self.bulls_power > 0:
            signals['power_level'] = 'positive'
        else:
            signals['power_level'] = 'neutral'
        
        # Generate signals based on Bulls Power changes
        if len(self.bulls_power_history) >= 3:
            current = self.bulls_power_history[-1]
            previous = self.bulls_power_history[-2]
            prev_prev = self.bulls_power_history[-3]
            
            # Check for momentum changes
            if current > previous > prev_prev and current > 0:
                signals['signal_type'] = 'bull_strengthening'
                signals['signal_strength'] = 0.6
            elif current < previous < prev_prev and current > 0:
                signals['signal_type'] = 'bull_weakening'
                signals['signal_strength'] = -0.6
            
            # Check for zero line approaches or crosses
            if previous > 0 and current <= 0:
                signals['signal_type'] = 'bearish_breakout'
                signals['signal_strength'] = -0.8
            elif previous <= 0 and current > 0:
                signals['signal_type'] = 'bullish_return'
                signals['signal_strength'] = 0.4
        
        # Calculate overall signal strength
        if signals['signal_strength'] == 0.0:
            # Base signal strength on Bulls Power magnitude and direction
            if self.ema_value > 0:
                normalized_power = self.bulls_power / self.ema_value
                signals['signal_strength'] = max(-1.0, min(1.0, normalized_power * 10))
        
        return signals
    
    def _analyze_divergence(self):
        """Analyze price vs Bulls Power divergence."""
        self.bullish_divergence = False
        self.bearish_divergence = False
        
        if (len(self.bulls_power_history) < self.config.divergence_periods or 
            len(self.close_history) < self.config.divergence_periods):
            return
        
        # Get recent data for analysis
        recent_bulls = self.bulls_power_history[-self.config.divergence_periods:]
        recent_prices = self.close_history[-self.config.divergence_periods:]
        
        # Find price lows and highs
        price_min_idx = recent_prices.index(min(recent_prices))
        price_max_idx = recent_prices.index(max(recent_prices))
        
        # Find Bulls Power lows and highs
        bulls_min_idx = recent_bulls.index(min(recent_bulls))
        bulls_max_idx = recent_bulls.index(max(recent_bulls))
        
        # Check for bullish divergence (price makes lower low, Bulls Power makes higher low)
        if price_min_idx < len(recent_prices) - 2:
            price_low = recent_prices[price_min_idx]
            current_price = recent_prices[-1]
            
            bulls_low = recent_bulls[bulls_min_idx]
            current_bulls = recent_bulls[-1]
            
            # Price lower low, Bulls Power higher low (less positive or more positive)
            if (current_price <= price_low and current_bulls > bulls_low and 
                current_bulls > 0 and bulls_low > 0):
                self.bullish_divergence = True
        
        # Check for bearish divergence (price makes higher high, Bulls Power makes lower high)
        if price_max_idx < len(recent_prices) - 2:
            price_high = recent_prices[price_max_idx]
            current_price = recent_prices[-1]
            
            bulls_high = recent_bulls[bulls_max_idx]
            current_bulls = recent_bulls[-1]
            
            # Price higher high, Bulls Power lower high (less positive)
            if (current_price >= price_high and current_bulls < bulls_high and
                current_bulls > 0):
                self.bearish_divergence = True
    
    def _format_results(self, ohlc_data: Dict, signals: Dict) -> Dict:
        """Format calculation results."""
        result = {
            # Core Bulls Power values
            'bulls_power': round(self.bulls_power, self.config.precision),
            'ema_value': round(self.ema_value, self.config.precision),
            'current_high': round(ohlc_data['high'], self.config.precision),
            
            # Analysis
            'bull_strength': self.bull_strength,
            'trend_direction': self.trend_direction,
            'bullish_divergence': self.bullish_divergence,
            'bearish_divergence': self.bearish_divergence,
            
            # Relative metrics
            'power_ratio': round(self.bulls_power / self.ema_value if self.ema_value > 0 else 0, 4),
            'distance_from_zero': round(abs(self.bulls_power), self.config.precision),
            
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
            'bulls_power': 0.0,
            'ema_value': 0.0,
            'bull_strength': 'neutral',
            'trend_direction': 'neutral',
            'status': 'no_data',
            'error': 'Insufficient data for calculation'
        }
    
    def _get_error_result(self, error_message: str) -> Dict:
        """Return error result structure."""
        return {
            'bulls_power': 0.0,
            'ema_value': 0.0,
            'bull_strength': 'neutral',
            'trend_direction': 'neutral',
            'status': 'error',
            'error': error_message
        }
    
    def get_historical_values(self, periods: int = 20) -> Dict:
        """Get historical values for Bulls Power and related indicators."""
        return {
            'bulls_power': self.bulls_power_history[-periods:] if self.bulls_power_history else [],
            'ema_values': self.ema_history[-periods:] if self.ema_history else [],
            'high_values': self.high_history[-periods:] if self.high_history else []
        }
    
    def get_strength_analysis(self) -> Dict:
        """Get detailed strength analysis."""
        return {
            'current_bulls_power': self.bulls_power,
            'bull_strength': self.bull_strength,
            'is_bulls_in_control': self.bulls_power > 0,
            'is_bears_showing_strength': self.bulls_power < 0.001,
            'strength_magnitude': abs(self.bulls_power),
            'relative_strength': abs(self.bulls_power / self.ema_value) if self.ema_value > 0 else 0
        }
    
    def get_divergence_analysis(self) -> Dict:
        """Get detailed divergence analysis."""
        return {
            'bullish_divergence': self.bullish_divergence,
            'bearish_divergence': self.bearish_divergence,
            'divergence_periods': self.config.divergence_periods,
            'current_bulls_power': self.bulls_power,
            'trend_confirmation': self.bulls_power > 0  # Bulls typically in control
        }
    
    def reset(self):
        """Reset indicator state."""
        self.close_history.clear()
        self.high_history.clear()
        self.ema_value = 0.0
        self.ema_history.clear()
        self.bulls_power = 0.0
        self.bulls_power_history.clear()
        self.bull_strength = "neutral"
        self.trend_direction = "neutral"
        self.bullish_divergence = False
        self.bearish_divergence = False
        self.is_initialized = False
        self._last_calculation_time = None
        
        self.logger.info("Bulls Power Indicator reset completed")
    
    def get_metadata(self) -> Dict:
        """Get indicator metadata."""
        return {
            'name': 'Bulls Power',
            'full_name': 'Bulls Power Indicator (Elder)',
            'description': 'Measures the power of bulls (buyers) to drive prices above EMA',
            'category': 'Momentum',
            'version': '1.0.0',
            'author': 'Platform3.AI Engine',
            'tags': ['momentum', 'elder', 'bulls', 'buying pressure', 'triple screen'],
            'inputs': ['high', 'low', 'close'],
            'outputs': ['bulls_power', 'bull_strength', 'signals', 'divergence'],
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
    # Test data - rising market with bull pressure
    test_data = []
    base_price = 100.0
    
    # Generate test data with varying bull pressure
    for i in range(30):
        # Create rising trend with varying buying pressure
        if i < 10:
            trend = 0.01 * i  # Gradual rise
        elif i < 20:
            trend = 0.01 * 10 + 0.03 * (i - 10)  # Steeper rise
        else:
            trend = 0.01 * 10 + 0.03 * 10 + 0.005 * (i - 20)  # Slowing rise
        
        noise = np.random.normal(0, 0.1)
        
        high = base_price + trend + noise + 0.25  # Higher highs showing bull pressure
        low = base_price + trend + noise - 0.15
        close = base_price + trend + noise
        
        test_data.append({
            'high': high,
            'low': low,
            'close': close,
            'timestamp': f'2024-01-01 09:{30+i}:00'
        })
    
    # Initialize indicator
    config = BullsPowerConfig(enable_signals=True, enable_divergence=True)
    bulls_indicator = BullsPowerIndicator(config)
    
    print("=== Bulls Power Indicator Test ===")
    print(f"Config: {config}")
    print()
    
    # Process test data
    for i, data_point in enumerate(test_data, 1):
        result = bulls_indicator.calculate(data_point)
        
        # Print every 3rd period after initialization
        if i % 3 == 0 and i >= 15:
            print(f"Period {i}:")
            print(f"  Price: {data_point['close']:.2f}")
            print(f"  High: {data_point['high']:.2f}")
            print(f"  EMA: {result['ema_value']}")
            print(f"  Bulls Power: {result['bulls_power']}")
            print(f"  Bull Strength: {result['bull_strength']}")
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
    strength_analysis = bulls_indicator.get_strength_analysis()
    for key, value in strength_analysis.items():
        print(f"{key}: {value}")
    
    # Test historical values
    print(f"\n=== Historical Values (last 3 periods) ===")
    historical = bulls_indicator.get_historical_values(3)
    for indicator_name, values in historical.items():
        print(f"{indicator_name}: {[round(v, 4) for v in values]}")