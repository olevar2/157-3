"""
Relative Vigor Index (RVI)

The Relative Vigor Index (RVI) is a momentum oscillator that measures the conviction 
of a trend by comparing closing prices to opening prices. It's based on the principle 
that in bull markets, prices tend to close higher than they open, and in bear markets, 
prices tend to close lower than they open.

Key Features:
- Measures momentum with conviction
- Compares close vs open relationship
- Includes signal line for confirmation
- Oscillates around zero line
- Smoothed calculation reduces noise

Mathematical Formula:
Numerator = SMA((Close - Open), period)
Denominator = SMA((High - Low), period)
RVI = Numerator / Denominator

Signal Line = SMA(RVI, 4)

Trading Signals:
- RVI > 0: Bullish momentum (closes above opens)
- RVI < 0: Bearish momentum (closes below opens)
- RVI crossing signal line: Trend confirmation
- RVI crossing zero: Major trend change

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
class RVIConfig:
    """Configuration for Relative Vigor Index calculation."""
    
    # Standard periods
    period: int = 10  # Period for RVI calculation
    signal_period: int = 4  # Period for signal line
    
    # Signal analysis
    enable_signals: bool = True  # Generate trading signals
    enable_divergence: bool = True  # Divergence analysis
    
    # Signal thresholds
    strong_momentum_threshold: float = 0.5  # Threshold for strong momentum
    divergence_periods: int = 10  # Periods to look back for divergence
    
    # Performance settings
    max_history: int = 100  # Maximum historical data to keep
    precision: int = 6  # Decimal precision for calculations


class RelativeVigorIndexIndicator(StandardIndicatorInterface):
    """
    Relative Vigor Index (RVI) Indicator
    
    A momentum oscillator that measures trend conviction by comparing
    closing prices to opening prices relative to the trading range.
    """
    
    def __init__(self, config: Optional[RVIConfig] = None):
        """
        Initialize RVI Indicator.
        
        Args:
            config: RVI configuration parameters
        """
        self.config = config or RVIConfig()
        self.logger = logging.getLogger(__name__)
        
        # Price histories
        self.open_history: List[float] = []
        self.high_history: List[float] = []
        self.low_history: List[float] = []
        self.close_history: List[float] = []
        
        # RVI components
        self.numerator_history: List[float] = []  # Close - Open
        self.denominator_history: List[float] = []  # High - Low
        
        # RVI values
        self.rvi_value: float = 0.0
        self.rvi_history: List[float] = []
        self.signal_line: float = 0.0
        self.signal_history: List[float] = []
        
        # Analysis variables
        self.momentum_direction: str = "neutral"  # "bullish", "bearish", "neutral"
        self.momentum_strength: str = "weak"  # "weak", "moderate", "strong"
        
        # Divergence tracking
        self.bullish_divergence: bool = False
        self.bearish_divergence: bool = False
        
        # State management
        self.is_initialized = False
        self._last_calculation_time = None
        
        self.logger.info("Relative Vigor Index Indicator initialized successfully")
    
    def calculate(self, data: Union[pd.DataFrame, Dict]) -> Dict:
        """
        Calculate Relative Vigor Index.
        
        Args:
            data: Market data containing OHLCV
            
        Returns:
            Dictionary containing RVI calculation results
        """
        try:
            # Validate and extract data
            ohlc_data = self._validate_and_extract_data(data)
            if not ohlc_data:
                return self._get_empty_result()
            
            # Add to price histories
            self.open_history.append(ohlc_data['open'])
            self.high_history.append(ohlc_data['high'])
            self.low_history.append(ohlc_data['low'])
            self.close_history.append(ohlc_data['close'])
            
            # Maintain history size
            if len(self.open_history) > self.config.max_history:
                self.open_history.pop(0)
                self.high_history.pop(0)
                self.low_history.pop(0)
                self.close_history.pop(0)
            
            # Calculate numerator and denominator
            self._calculate_components()
            
            # Calculate RVI
            self._calculate_rvi()
            
            # Calculate signal line
            self._calculate_signal_line()
            
            # Analyze momentum
            self._analyze_momentum()
            
            # Generate signals
            signals = {}
            if self.config.enable_signals:
                signals = self._generate_signals()
            
            # Perform divergence analysis
            if self.config.enable_divergence:
                self._analyze_divergence()
            
            # Mark as initialized
            if len(self.open_history) >= self.config.period:
                self.is_initialized = True
            
            self._last_calculation_time = datetime.now()
            
            return self._format_results(ohlc_data, signals)
            
        except Exception as e:
            self.logger.error(f"Error in RVI calculation: {str(e)}")
            return self._get_error_result(str(e))
    
    def _validate_and_extract_data(self, data: Union[pd.DataFrame, Dict]) -> Optional[Dict]:
        """Validate input data and extract OHLC values."""
        try:
            if isinstance(data, pd.DataFrame):
                if len(data) == 0:
                    return None
                
                latest = data.iloc[-1]
                return {
                    'open': float(latest.get('open', latest.get('Open', 0))),
                    'high': float(latest.get('high', latest.get('High', 0))),
                    'low': float(latest.get('low', latest.get('Low', 0))),
                    'close': float(latest.get('close', latest.get('Close', 0))),
                    'timestamp': latest.get('timestamp', datetime.now())
                }
            
            elif isinstance(data, dict):
                return {
                    'open': float(data.get('open', data.get('Open', 0))),
                    'high': float(data.get('high', data.get('High', 0))),
                    'low': float(data.get('low', data.get('Low', 0))),
                    'close': float(data.get('close', data.get('Close', 0))),
                    'timestamp': data.get('timestamp', datetime.now())
                }
            
            return None
            
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Data validation error: {str(e)}")
            return None
    
    def _calculate_components(self):
        """Calculate RVI numerator and denominator components."""
        if len(self.open_history) == 0:
            return
        
        # Calculate numerator: Close - Open
        numerator = self.close_history[-1] - self.open_history[-1]
        self.numerator_history.append(numerator)
        
        # Calculate denominator: High - Low
        denominator = self.high_history[-1] - self.low_history[-1]
        # Avoid division by zero
        if denominator == 0:
            denominator = 0.0001  # Small value to prevent division by zero
        self.denominator_history.append(denominator)
        
        # Maintain history size
        if len(self.numerator_history) > self.config.max_history:
            self.numerator_history.pop(0)
            self.denominator_history.pop(0)
    
    def _calculate_rvi(self):
        """Calculate RVI value."""
        if len(self.numerator_history) < self.config.period or len(self.denominator_history) < self.config.period:
            return
        
        # Calculate SMA of numerator and denominator
        sma_numerator = sum(self.numerator_history[-self.config.period:]) / self.config.period
        sma_denominator = sum(self.denominator_history[-self.config.period:]) / self.config.period
        
        # Calculate RVI
        if sma_denominator != 0:
            self.rvi_value = sma_numerator / sma_denominator
        else:
            self.rvi_value = 0.0
        
        # Store RVI history
        self.rvi_history.append(self.rvi_value)
        if len(self.rvi_history) > self.config.max_history:
            self.rvi_history.pop(0)
    
    def _calculate_signal_line(self):
        """Calculate RVI signal line."""
        if len(self.rvi_history) < self.config.signal_period:
            self.signal_line = self.rvi_value
            return
        
        # Calculate SMA of RVI for signal line
        self.signal_line = sum(self.rvi_history[-self.config.signal_period:]) / self.config.signal_period
        
        # Store signal history
        self.signal_history.append(self.signal_line)
        if len(self.signal_history) > self.config.max_history:
            self.signal_history.pop(0)
    
    def _analyze_momentum(self):
        """Analyze momentum direction and strength."""
        # Determine momentum direction
        if self.rvi_value > 0:
            self.momentum_direction = "bullish"
        elif self.rvi_value < 0:
            self.momentum_direction = "bearish"
        else:
            self.momentum_direction = "neutral"
        
        # Determine momentum strength
        abs_rvi = abs(self.rvi_value)
        if abs_rvi >= self.config.strong_momentum_threshold:
            self.momentum_strength = "strong"
        elif abs_rvi >= self.config.strong_momentum_threshold / 2:
            self.momentum_strength = "moderate"
        else:
            self.momentum_strength = "weak"
    
    def _generate_signals(self) -> Dict:
        """Generate trading signals based on RVI analysis."""
        signals = {
            'momentum_direction': self.momentum_direction,
            'momentum_strength': self.momentum_strength,
            'signal_type': 'none',
            'signal_strength': 0.0,
            'rvi_signal_cross': 'none',
            'zero_line_cross': 'none'
        }
        
        if not self.is_initialized or len(self.rvi_history) < 2:
            return signals
        
        current_rvi = self.rvi_history[-1]
        previous_rvi = self.rvi_history[-2]
        current_signal = self.signal_line
        
        # Check for RVI vs Signal line crossovers
        if len(self.signal_history) >= 2:
            previous_signal = self.signal_history[-2]
            
            # Bullish cross: RVI crosses above signal line
            if previous_rvi <= previous_signal and current_rvi > current_signal:
                signals['signal_type'] = 'buy'
                signals['rvi_signal_cross'] = 'bullish'
                signals['signal_strength'] = 0.6
            
            # Bearish cross: RVI crosses below signal line
            elif previous_rvi >= previous_signal and current_rvi < current_signal:
                signals['signal_type'] = 'sell'
                signals['rvi_signal_cross'] = 'bearish'
                signals['signal_strength'] = -0.6
        
        # Check for zero line crossovers (major trend changes)
        if previous_rvi <= 0 and current_rvi > 0:
            signals['zero_line_cross'] = 'bullish'
            if signals['signal_strength'] == 0.0:
                signals['signal_strength'] = 0.7
            signals['signal_type'] = 'strong_buy'
        
        elif previous_rvi >= 0 and current_rvi < 0:
            signals['zero_line_cross'] = 'bearish'
            if signals['signal_strength'] == 0.0:
                signals['signal_strength'] = -0.7
            signals['signal_type'] = 'strong_sell'
        
        # Enhance signal strength based on momentum strength
        if self.momentum_strength == "strong":
            signals['signal_strength'] *= 1.2
        elif self.momentum_strength == "weak":
            signals['signal_strength'] *= 0.7
        
        # Limit signal strength
        signals['signal_strength'] = max(-1.0, min(1.0, signals['signal_strength']))
        
        return signals
    
    def _analyze_divergence(self):
        """Analyze price vs RVI divergence."""
        self.bullish_divergence = False
        self.bearish_divergence = False
        
        if (len(self.rvi_history) < self.config.divergence_periods or 
            len(self.close_history) < self.config.divergence_periods):
            return
        
        # Get recent data for analysis
        recent_rvi = self.rvi_history[-self.config.divergence_periods:]
        recent_prices = self.close_history[-self.config.divergence_periods:]
        
        # Find price lows and highs
        price_min_idx = recent_prices.index(min(recent_prices))
        price_max_idx = recent_prices.index(max(recent_prices))
        
        # Find RVI lows and highs
        rvi_min_idx = recent_rvi.index(min(recent_rvi))
        rvi_max_idx = recent_rvi.index(max(recent_rvi))
        
        # Check for bullish divergence (price makes lower low, RVI makes higher low)
        if price_min_idx < len(recent_prices) - 2:
            price_low = recent_prices[price_min_idx]
            current_price = recent_prices[-1]
            
            rvi_low = recent_rvi[rvi_min_idx]
            current_rvi = recent_rvi[-1]
            
            # Price lower low, RVI higher low
            if (current_price <= price_low and current_rvi > rvi_low and 
                current_rvi < 0):  # RVI in bearish territory but improving
                self.bullish_divergence = True
        
        # Check for bearish divergence (price makes higher high, RVI makes lower high)
        if price_max_idx < len(recent_prices) - 2:
            price_high = recent_prices[price_max_idx]
            current_price = recent_prices[-1]
            
            rvi_high = recent_rvi[rvi_max_idx]
            current_rvi = recent_rvi[-1]
            
            # Price higher high, RVI lower high
            if (current_price >= price_high and current_rvi < rvi_high and 
                current_rvi > 0):  # RVI in bullish territory but weakening
                self.bearish_divergence = True
    
    def _format_results(self, ohlc_data: Dict, signals: Dict) -> Dict:
        """Format calculation results."""
        result = {
            # Core RVI values
            'rvi_value': round(self.rvi_value, self.config.precision),
            'signal_line': round(self.signal_line, self.config.precision),
            
            # Analysis
            'momentum_direction': self.momentum_direction,
            'momentum_strength': self.momentum_strength,
            'bullish_divergence': self.bullish_divergence,
            'bearish_divergence': self.bearish_divergence,
            
            # Component analysis
            'close_open_diff': round(ohlc_data['close'] - ohlc_data['open'], self.config.precision),
            'high_low_range': round(ohlc_data['high'] - ohlc_data['low'], self.config.precision),
            'rvi_signal_diff': round(self.rvi_value - self.signal_line, self.config.precision),
            
            # Metadata
            'is_initialized': self.is_initialized,
            'data_points': len(self.open_history),
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
            'rvi_value': 0.0,
            'signal_line': 0.0,
            'momentum_direction': 'neutral',
            'momentum_strength': 'weak',
            'status': 'no_data',
            'error': 'Insufficient data for calculation'
        }
    
    def _get_error_result(self, error_message: str) -> Dict:
        """Return error result structure."""
        return {
            'rvi_value': 0.0,
            'signal_line': 0.0,
            'momentum_direction': 'neutral',
            'momentum_strength': 'weak',
            'status': 'error',
            'error': error_message
        }
    
    def get_historical_values(self, periods: int = 20) -> Dict:
        """Get historical values for RVI and components."""
        return {
            'rvi_values': self.rvi_history[-periods:] if self.rvi_history else [],
            'signal_values': self.signal_history[-periods:] if self.signal_history else [],
            'numerator_values': self.numerator_history[-periods:] if self.numerator_history else [],
            'denominator_values': self.denominator_history[-periods:] if self.denominator_history else []
        }
    
    def get_momentum_analysis(self) -> Dict:
        """Get detailed momentum analysis."""
        return {
            'current_rvi': self.rvi_value,
            'current_signal': self.signal_line,
            'momentum_direction': self.momentum_direction,
            'momentum_strength': self.momentum_strength,
            'is_bullish': self.rvi_value > 0,
            'is_bearish': self.rvi_value < 0,
            'above_signal': self.rvi_value > self.signal_line,
            'conviction_level': abs(self.rvi_value)  # Distance from zero indicates conviction
        }
    
    def get_divergence_analysis(self) -> Dict:
        """Get detailed divergence analysis."""
        return {
            'bullish_divergence': self.bullish_divergence,
            'bearish_divergence': self.bearish_divergence,
            'divergence_periods': self.config.divergence_periods,
            'current_rvi': self.rvi_value,
            'reversal_potential': self.bullish_divergence or self.bearish_divergence
        }
    
    def reset(self):
        """Reset indicator state."""
        self.open_history.clear()
        self.high_history.clear()
        self.low_history.clear()
        self.close_history.clear()
        self.numerator_history.clear()
        self.denominator_history.clear()
        self.rvi_value = 0.0
        self.rvi_history.clear()
        self.signal_line = 0.0
        self.signal_history.clear()
        self.momentum_direction = "neutral"
        self.momentum_strength = "weak"
        self.bullish_divergence = False
        self.bearish_divergence = False
        self.is_initialized = False
        self._last_calculation_time = None
        
        self.logger.info("Relative Vigor Index Indicator reset completed")
    
    def get_metadata(self) -> Dict:
        """Get indicator metadata."""
        return {
            'name': 'RVI',
            'full_name': 'Relative Vigor Index',
            'description': 'Momentum oscillator measuring trend conviction by comparing close vs open prices',
            'category': 'Momentum',
            'version': '1.0.0',
            'author': 'Platform3.AI Engine',
            'tags': ['momentum', 'oscillator', 'conviction', 'close', 'open'],
            'inputs': ['open', 'high', 'low', 'close'],
            'outputs': ['rvi_value', 'signal_line', 'momentum_analysis', 'signals'],
            'parameters': {
                'period': self.config.period,
                'signal_period': self.config.signal_period,
                'strong_momentum_threshold': self.config.strong_momentum_threshold
            }
        }
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters."""
        try:
            # Validate periods
            if self.config.period <= 0:
                self.logger.error("Invalid period: must be positive")
                return False
            
            if self.config.signal_period <= 0:
                self.logger.error("Invalid signal_period: must be positive")
                return False
            
            # Validate threshold
            if self.config.strong_momentum_threshold <= 0:
                self.logger.error("Invalid strong_momentum_threshold: must be positive")
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
    # Test data - trend with varying conviction levels
    test_data = []
    base_price = 100.0
    
    # Generate test data with varying open/close relationships
    for i in range(30):
        trend = 0.02 * i  # Upward trend
        noise = np.random.normal(0, 0.1)
        
        # Vary the open/close relationship to show conviction changes
        if i < 10:
            # Strong conviction: closes well above opens
            open_price = base_price + trend + noise
            close_price = open_price + abs(np.random.normal(0.2, 0.1))
        elif i < 20:
            # Weakening conviction: smaller close-open differences
            open_price = base_price + trend + noise
            close_price = open_price + np.random.normal(0.05, 0.1)
        else:
            # Bearish conviction: closes below opens despite uptrend
            open_price = base_price + trend + noise
            close_price = open_price - abs(np.random.normal(0.1, 0.05))
        
        high = max(open_price, close_price) + abs(np.random.normal(0.1, 0.05))
        low = min(open_price, close_price) - abs(np.random.normal(0.1, 0.05))
        
        test_data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'timestamp': f'2024-01-01 09:{30+i}:00'
        })
    
    # Initialize indicator
    config = RVIConfig(enable_signals=True, enable_divergence=True)
    rvi = RelativeVigorIndexIndicator(config)
    
    print("=== Relative Vigor Index Indicator Test ===")
    print(f"Config: {config}")
    print()
    
    # Process test data
    for i, data_point in enumerate(test_data, 1):
        result = rvi.calculate(data_point)
        
        # Print every 3rd period after initialization
        if i % 3 == 0 and i >= 12:
            print(f"Period {i}:")
            print(f"  Open: {data_point['open']:.2f}, Close: {data_point['close']:.2f}")
            print(f"  Close-Open: {result['close_open_diff']}")
            print(f"  RVI: {result['rvi_value']}")
            print(f"  Signal: {result['signal_line']}")
            print(f"  Momentum: {result['momentum_direction']} ({result['momentum_strength']})")
            
            if 'signal_type' in result:
                print(f"  Signal Type: {result['signal_type']}")
                print(f"  RVI-Signal Cross: {result['rvi_signal_cross']}")
                print(f"  Zero Cross: {result['zero_line_cross']}")
                print(f"  Signal Strength: {result['signal_strength']:.3f}")
            
            if result.get('bullish_divergence') or result.get('bearish_divergence'):
                print(f"  Bullish Divergence: {result['bullish_divergence']}")
                print(f"  Bearish Divergence: {result['bearish_divergence']}")
            
            print(f"  Status: {result['status']}")
            print()
    
    # Test momentum analysis
    print("=== Momentum Analysis ===")
    momentum_analysis = rvi.get_momentum_analysis()
    for key, value in momentum_analysis.items():
        print(f"{key}: {value}")
    
    # Test historical values
    print(f"\n=== Historical Values (last 3 periods) ===")
    historical = rvi.get_historical_values(3)
    for indicator_name, values in historical.items():
        print(f"{indicator_name}: {[round(v, 4) for v in values]}")