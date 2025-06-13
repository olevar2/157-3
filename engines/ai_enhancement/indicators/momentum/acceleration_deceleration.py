"""
Acceleration/Deceleration (AC) Indicator (Bill Williams)

The Acceleration/Deceleration (AC) indicator is a momentum oscillator created by Bill Williams 
that measures the acceleration and deceleration of the current driving force. It shows the 
difference between the Awesome Oscillator and its 5-period simple moving average.

Key Features:
- Measures momentum acceleration/deceleration
- Based on Awesome Oscillator (AO) calculations
- 5-period SMA of AO subtracted from current AO
- Anticipates momentum changes before they appear on AO
- Provides early signals for trend changes

Mathematical Formula:
AC = AO - SMA(AO, 5)
where:
- AO = Awesome Oscillator = SMA(HL2, 5) - SMA(HL2, 34)
- HL2 = (High + Low) / 2
- SMA = Simple Moving Average

Trading Signals:
- Positive AC: Acceleration in current direction
- Negative AC: Deceleration in current direction
- AC crossing zero: Change in momentum acceleration
- AC divergence with price: Early trend change warning

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
class ACConfig:
    """Configuration for Acceleration/Deceleration Indicator calculation."""
    
    # Bill Williams standard periods for AO calculation
    ao_short_period: int = 5   # Short period for AO
    ao_long_period: int = 34   # Long period for AO
    
    # AC calculation period
    ac_period: int = 5  # Period for AC smoothing
    
    # Signal analysis
    enable_signals: bool = True  # Generate trading signals
    enable_divergence: bool = True  # Divergence analysis
    
    # Advanced features
    signal_threshold: float = 0.0001  # Minimum threshold for signal generation
    divergence_periods: int = 10  # Periods to look back for divergence
    
    # Performance settings
    max_history: int = 100  # Maximum historical data to keep
    precision: int = 6  # Decimal precision for calculations


class AccelerationDecelerationIndicator(StandardIndicatorInterface):
    """
    Acceleration/Deceleration (AC) Indicator (Bill Williams)
    
    A momentum oscillator that measures the acceleration and deceleration of
    the current driving force by comparing the Awesome Oscillator to its moving average.
    """
    
    def __init__(self, config: Optional[ACConfig] = None):
        """
        Initialize AC Indicator.
        
        Args:
            config: AC configuration parameters
        """
        self.config = config or ACConfig()
        self.logger = logging.getLogger(__name__)
        
        # Price history for AO calculation
        self.hl2_history: List[float] = []  # (High + Low) / 2 values
        
        # Awesome Oscillator components
        self.ao_short_sma: float = 0.0
        self.ao_long_sma: float = 0.0
        self.awesome_oscillator: float = 0.0
        
        # AC calculation
        self.ao_history: List[float] = []
        self.ao_sma: float = 0.0
        self.ac_value: float = 0.0
        
        # Historical values
        self.ac_history: List[float] = []
        self.price_history: List[float] = []  # For divergence analysis
        
        # Signal analysis
        self.previous_ac: float = 0.0
        self.momentum_direction: str = "neutral"  # "accelerating", "decelerating", "neutral"
        self.zero_cross_signal: str = "none"  # "bullish", "bearish", "none"
        
        # Divergence tracking
        self.bullish_divergence: bool = False
        self.bearish_divergence: bool = False
        
        # State management
        self.is_initialized = False
        self._last_calculation_time = None
        
        self.logger.info("Acceleration/Deceleration Indicator initialized successfully")
    
    def calculate(self, data: Union[pd.DataFrame, Dict]) -> Dict:
        """
        Calculate Acceleration/Deceleration Indicator.
        
        Args:
            data: Market data containing OHLC
            
        Returns:
            Dictionary containing AC calculation results
        """
        try:
            # Validate and extract data
            ohlc_data = self._validate_and_extract_data(data)
            if not ohlc_data:
                return self._get_empty_result()
            
            # Calculate HL2 (median price)
            hl2 = (ohlc_data['high'] + ohlc_data['low']) / 2
            
            # Add to histories
            self.hl2_history.append(hl2)
            self.price_history.append(ohlc_data['close'])
            
            # Maintain history size
            if len(self.hl2_history) > self.config.max_history:
                self.hl2_history.pop(0)
                self.price_history.pop(0)
            
            # Calculate Awesome Oscillator
            self._calculate_awesome_oscillator()
            
            # Calculate AC
            self._calculate_ac()
            
            # Analyze momentum and signals
            signals = {}
            if self.config.enable_signals:
                signals = self._generate_signals()
            
            # Perform divergence analysis
            if self.config.enable_divergence:
                self._analyze_divergence()
            
            # Update state
            self.previous_ac = self.ac_value
            
            # Mark as initialized
            if len(self.hl2_history) >= max(self.config.ao_long_period, self.config.ac_period):
                self.is_initialized = True
            
            self._last_calculation_time = datetime.now()
            
            return self._format_results(ohlc_data, signals)
            
        except Exception as e:
            self.logger.error(f"Error in AC calculation: {str(e)}")
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
    
    def _calculate_awesome_oscillator(self):
        """Calculate the Awesome Oscillator (AO)."""
        # Calculate short period SMA
        if len(self.hl2_history) >= self.config.ao_short_period:
            self.ao_short_sma = sum(self.hl2_history[-self.config.ao_short_period:]) / self.config.ao_short_period
        else:
            self.ao_short_sma = 0.0
        
        # Calculate long period SMA
        if len(self.hl2_history) >= self.config.ao_long_period:
            self.ao_long_sma = sum(self.hl2_history[-self.config.ao_long_period:]) / self.config.ao_long_period
        else:
            self.ao_long_sma = 0.0
        
        # Calculate Awesome Oscillator
        if self.ao_short_sma != 0.0 and self.ao_long_sma != 0.0:
            self.awesome_oscillator = self.ao_short_sma - self.ao_long_sma
        else:
            self.awesome_oscillator = 0.0
        
        # Store AO history
        self.ao_history.append(self.awesome_oscillator)
        if len(self.ao_history) > self.config.max_history:
            self.ao_history.pop(0)
    
    def _calculate_ac(self):
        """Calculate Acceleration/Deceleration value."""
        # Calculate AC period SMA of AO
        if len(self.ao_history) >= self.config.ac_period:
            self.ao_sma = sum(self.ao_history[-self.config.ac_period:]) / self.config.ac_period
        else:
            self.ao_sma = 0.0
        
        # Calculate AC value
        if self.awesome_oscillator != 0.0 and self.ao_sma != 0.0:
            self.ac_value = self.awesome_oscillator - self.ao_sma
        else:
            self.ac_value = 0.0
        
        # Store AC history
        self.ac_history.append(self.ac_value)
        if len(self.ac_history) > self.config.max_history:
            self.ac_history.pop(0)
    
    def _generate_signals(self) -> Dict:
        """Generate trading signals based on AC analysis."""
        signals = {
            'ac_direction': 'neutral',
            'momentum_change': 'none',
            'zero_cross_signal': 'none',
            'signal_strength': 0.0,
            'acceleration_state': 'neutral'
        }
        
        if not self.is_initialized or len(self.ac_history) < 2:
            return signals
        
        # Determine AC direction
        if self.ac_value > self.config.signal_threshold:
            signals['ac_direction'] = 'positive'
            signals['acceleration_state'] = 'accelerating'
        elif self.ac_value < -self.config.signal_threshold:
            signals['ac_direction'] = 'negative'
            signals['acceleration_state'] = 'decelerating'
        else:
            signals['ac_direction'] = 'neutral'
            signals['acceleration_state'] = 'stable'
        
        # Check for momentum changes
        if self.previous_ac != 0:
            if self.ac_value > self.previous_ac:
                signals['momentum_change'] = 'increasing'
            elif self.ac_value < self.previous_ac:
                signals['momentum_change'] = 'decreasing'
        
        # Check for zero line crossings
        if len(self.ac_history) >= 2:
            current_ac = self.ac_history[-1]
            previous_ac = self.ac_history[-2]
            
            if previous_ac <= 0 and current_ac > 0:
                signals['zero_cross_signal'] = 'bullish_cross'
                signals['signal_strength'] = 0.7
            elif previous_ac >= 0 and current_ac < 0:
                signals['zero_cross_signal'] = 'bearish_cross'
                signals['signal_strength'] = -0.7
        
        # Calculate signal strength based on AC magnitude
        if abs(self.ac_value) > 0:
            # Normalize signal strength (adjust scale as needed)
            base_strength = min(abs(self.ac_value) * 1000, 1.0)  # Scale factor
            if self.ac_value > 0:
                signals['signal_strength'] = base_strength
            else:
                signals['signal_strength'] = -base_strength
        
        return signals
    
    def _analyze_divergence(self):
        """Analyze price vs AC divergence."""
        self.bullish_divergence = False
        self.bearish_divergence = False
        
        if len(self.ac_history) < self.config.divergence_periods or len(self.price_history) < self.config.divergence_periods:
            return
        
        # Get recent data for analysis
        recent_ac = self.ac_history[-self.config.divergence_periods:]
        recent_prices = self.price_history[-self.config.divergence_periods:]
        
        # Find price lows and highs
        price_min_idx = recent_prices.index(min(recent_prices))
        price_max_idx = recent_prices.index(max(recent_prices))
        
        # Find AC lows and highs
        ac_min_idx = recent_ac.index(min(recent_ac))
        ac_max_idx = recent_ac.index(max(recent_ac))
        
        # Check for bullish divergence (price makes lower low, AC makes higher low)
        if price_min_idx < len(recent_prices) - 2:  # Not at the very end
            price_low = recent_prices[price_min_idx]
            current_price = recent_prices[-1]
            
            ac_low = recent_ac[ac_min_idx]
            current_ac = recent_ac[-1]
            
            if (current_price < price_low and current_ac > ac_low and 
                current_ac < 0):  # AC is negative but rising
                self.bullish_divergence = True
        
        # Check for bearish divergence (price makes higher high, AC makes lower high)
        if price_max_idx < len(recent_prices) - 2:  # Not at the very end
            price_high = recent_prices[price_max_idx]
            current_price = recent_prices[-1]
            
            ac_high = recent_ac[ac_max_idx]
            current_ac = recent_ac[-1]
            
            if (current_price > price_high and current_ac < ac_high and 
                current_ac > 0):  # AC is positive but falling
                self.bearish_divergence = True
    
    def _format_results(self, ohlc_data: Dict, signals: Dict) -> Dict:
        """Format calculation results."""
        result = {
            # Core AC values
            'ac_value': round(self.ac_value, self.config.precision),
            'awesome_oscillator': round(self.awesome_oscillator, self.config.precision),
            'ao_sma': round(self.ao_sma, self.config.precision),
            
            # Component values
            'ao_short_sma': round(self.ao_short_sma, self.config.precision),
            'ao_long_sma': round(self.ao_long_sma, self.config.precision),
            
            # Analysis
            'momentum_direction': self.momentum_direction,
            'bullish_divergence': self.bullish_divergence,
            'bearish_divergence': self.bearish_divergence,
            
            # Metadata
            'is_initialized': self.is_initialized,
            'data_points': len(self.hl2_history),
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
            'ac_value': 0.0,
            'awesome_oscillator': 0.0,
            'ao_sma': 0.0,
            'momentum_direction': 'neutral',
            'status': 'no_data',
            'error': 'Insufficient data for calculation'
        }
    
    def _get_error_result(self, error_message: str) -> Dict:
        """Return error result structure."""
        return {
            'ac_value': 0.0,
            'awesome_oscillator': 0.0,
            'ao_sma': 0.0,
            'momentum_direction': 'neutral',
            'status': 'error',
            'error': error_message
        }
    
    def get_historical_values(self, periods: int = 20) -> Dict:
        """
        Get historical values for AC and related indicators.
        
        Args:
            periods: Number of historical periods to return
            
        Returns:
            Dictionary containing historical values
        """
        return {
            'ac_values': self.ac_history[-periods:] if self.ac_history else [],
            'ao_values': self.ao_history[-periods:] if self.ao_history else [],
            'hl2_values': self.hl2_history[-periods:] if self.hl2_history else []
        }
    
    def get_divergence_analysis(self) -> Dict:
        """Get detailed divergence analysis."""
        return {
            'bullish_divergence': self.bullish_divergence,
            'bearish_divergence': self.bearish_divergence,
            'divergence_periods': self.config.divergence_periods,
            'current_ac': self.ac_value,
            'current_ao': self.awesome_oscillator
        }
    
    def get_momentum_state(self) -> Dict:
        """Get current momentum state analysis."""
        state = {
            'ac_value': self.ac_value,
            'is_accelerating': self.ac_value > 0,
            'is_decelerating': self.ac_value < 0,
            'momentum_strength': abs(self.ac_value),
            'zero_line_distance': abs(self.ac_value)
        }
        
        # Add momentum classification
        if abs(self.ac_value) < 0.0001:
            state['momentum_class'] = 'neutral'
        elif abs(self.ac_value) < 0.001:
            state['momentum_class'] = 'weak'
        elif abs(self.ac_value) < 0.01:
            state['momentum_class'] = 'moderate'
        else:
            state['momentum_class'] = 'strong'
        
        return state
    
    def reset(self):
        """Reset indicator state."""
        self.hl2_history.clear()
        self.ao_short_sma = 0.0
        self.ao_long_sma = 0.0
        self.awesome_oscillator = 0.0
        self.ao_history.clear()
        self.ao_sma = 0.0
        self.ac_value = 0.0
        self.ac_history.clear()
        self.price_history.clear()
        self.previous_ac = 0.0
        self.momentum_direction = "neutral"
        self.zero_cross_signal = "none"
        self.bullish_divergence = False
        self.bearish_divergence = False
        self.is_initialized = False
        self._last_calculation_time = None
        
        self.logger.info("Acceleration/Deceleration Indicator reset completed")
    
    def get_metadata(self) -> Dict:
        """Get indicator metadata."""
        return {
            'name': 'AC',
            'full_name': 'Acceleration/Deceleration Indicator (Bill Williams)',
            'description': 'Momentum oscillator measuring acceleration/deceleration of driving force',
            'category': 'Momentum',
            'version': '1.0.0',
            'author': 'Platform3.AI Engine',
            'tags': ['momentum', 'oscillator', 'bill williams', 'acceleration', 'awesome oscillator'],
            'inputs': ['high', 'low', 'close'],
            'outputs': ['ac_value', 'awesome_oscillator', 'signals', 'divergence'],
            'parameters': {
                'ao_short_period': self.config.ao_short_period,
                'ao_long_period': self.config.ao_long_period,
                'ac_period': self.config.ac_period,
                'signal_threshold': self.config.signal_threshold
            }
        }
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters."""
        try:
            # Validate AO periods
            if self.config.ao_short_period <= 0:
                self.logger.error("Invalid ao_short_period: must be positive")
                return False
            
            if self.config.ao_long_period <= 0:
                self.logger.error("Invalid ao_long_period: must be positive")
                return False
            
            if self.config.ao_short_period >= self.config.ao_long_period:
                self.logger.error("ao_short_period must be less than ao_long_period")
                return False
            
            # Validate AC period
            if self.config.ac_period <= 0:
                self.logger.error("Invalid ac_period: must be positive")
                return False
            
            # Validate signal threshold
            if self.config.signal_threshold < 0:
                self.logger.error("Invalid signal_threshold: must be non-negative")
                return False
            
            # Validate divergence periods
            if self.config.divergence_periods <= 0:
                self.logger.error("Invalid divergence_periods: must be positive")
                return False
            
            # Check Bill Williams standard configuration
            if (self.config.ao_short_period != 5 or self.config.ao_long_period != 34 or 
                self.config.ac_period != 5):
                self.logger.warning("Non-standard periods detected. Bill Williams standard: AO Short=5, Long=34, AC=5")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Parameter validation error: {str(e)}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Test data - momentum change scenario
    test_data = []
    base_price = 100.0
    
    # Generate test data with momentum changes
    for i in range(50):
        # Create momentum acceleration/deceleration pattern
        if i < 15:
            trend = 0.02 * i  # Accelerating up
        elif i < 30:
            trend = 0.02 * 15 + 0.01 * (i - 15)  # Decelerating up
        elif i < 40:
            trend = 0.02 * 15 + 0.01 * 15 - 0.01 * (i - 30)  # Decelerating down
        else:
            trend = 0.02 * 15 + 0.01 * 15 - 0.01 * 10 - 0.02 * (i - 40)  # Accelerating down
        
        noise = np.random.normal(0, 0.1)
        
        high = base_price + trend + noise + 0.2
        low = base_price + trend + noise - 0.2
        close = base_price + trend + noise
        
        test_data.append({
            'high': high,
            'low': low,
            'close': close,
            'timestamp': f'2024-01-01 09:{30+i}:00'
        })
    
    # Initialize indicator
    config = ACConfig(enable_signals=True, enable_divergence=True)
    ac_indicator = AccelerationDecelerationIndicator(config)
    
    print("=== Acceleration/Deceleration Indicator Test ===")
    print(f"Config: {config}")
    print()
    
    # Process test data
    for i, data_point in enumerate(test_data, 1):
        result = ac_indicator.calculate(data_point)
        
        # Print every 5th period after initialization
        if i % 5 == 0 and i >= 35:
            print(f"Period {i}:")
            print(f"  Price: {data_point['close']:.2f}")
            print(f"  AC Value: {result['ac_value']}")
            print(f"  AO Value: {result['awesome_oscillator']}")
            print(f"  AO SMA: {result['ao_sma']}")
            
            if 'ac_direction' in result:
                print(f"  AC Direction: {result['ac_direction']}")
                print(f"  Acceleration State: {result['acceleration_state']}")
                print(f"  Zero Cross: {result['zero_cross_signal']}")
                print(f"  Signal Strength: {result['signal_strength']:.3f}")
            
            if result.get('bullish_divergence') or result.get('bearish_divergence'):
                print(f"  Bullish Divergence: {result['bullish_divergence']}")
                print(f"  Bearish Divergence: {result['bearish_divergence']}")
            
            print(f"  Status: {result['status']}")
            print()
    
    # Test historical values
    print("=== Historical Values (last 3 periods) ===")
    historical = ac_indicator.get_historical_values(3)
    for indicator_name, values in historical.items():
        print(f"{indicator_name}: {[round(v, 4) for v in values]}")
    
    # Test momentum state
    print(f"\n=== Current Momentum State ===")
    momentum_state = ac_indicator.get_momentum_state()
    for key, value in momentum_state.items():
        print(f"{key}: {value}")