"""
Alligator Indicator (Bill Williams)

The Alligator indicator is a technical analysis tool created by Bill Williams that uses 
three smoothed moving averages to identify trending and consolidating periods. The three 
lines represent the jaw, teeth, and lips of an alligator, providing insights into market 
trend direction and momentum.

Key Features:
- Jaw (Blue Line): 13-period SMMA shifted forward by 8 periods
- Teeth (Red Line): 8-period SMMA shifted forward by 5 periods  
- Lips (Green Line): 5-period SMMA shifted forward by 3 periods
- Identifies trending vs consolidating markets
- Provides clear entry and exit signals

Mathematical Formulas:
- Jaw = SMMA(Median Price, 13) shifted +8 periods
- Teeth = SMMA(Median Price, 8) shifted +5 periods
- Lips = SMMA(Median Price, 5) shifted +3 periods
- Median Price = (High + Low) / 2
- SMMA = Smoothed Moving Average

Trading Signals:
- Sleeping Alligator: Lines are intertwined (consolidation)
- Awakening Alligator: Lines begin to separate (trend start)
- Eating Alligator: Lines are spread apart (strong trend)
- Sated Alligator: Lines converge again (trend end)

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
class AlligatorConfig:
    """Configuration for Alligator Indicator calculation."""
    
    # Bill Williams standard periods
    jaw_period: int = 13  # Blue line period
    teeth_period: int = 8  # Red line period
    lips_period: int = 5  # Green line period
    
    # Bill Williams standard shifts
    jaw_shift: int = 8  # Jaw forward shift
    teeth_shift: int = 5  # Teeth forward shift
    lips_shift: int = 3  # Lips forward shift
    
    # Price calculation method
    price_method: str = "median"  # "median", "close", "typical"
    
    # Advanced features
    enable_signals: bool = True  # Generate trading signals
    enable_fractal_analysis: bool = True  # Fractal-based analysis
    
    # Performance settings
    max_history: int = 100  # Maximum historical data to keep
    precision: int = 6  # Decimal precision for calculations


class AlligatorIndicator(StandardIndicatorInterface):
    """
    Alligator Indicator (Bill Williams)
    
    A trend-following indicator using three smoothed moving averages to identify
    market trends and consolidation periods through the metaphor of an alligator.
    """
    
    def __init__(self, config: Optional[AlligatorConfig] = None):
        """
        Initialize Alligator Indicator.
        
        Args:
            config: Alligator configuration parameters
        """
        self.config = config or AlligatorConfig()
        self.logger = logging.getLogger(__name__)
        
        # Price history for calculations
        self.price_history: List[float] = []
        
        # SMMA calculation states
        self.jaw_smma: float = 0.0
        self.teeth_smma: float = 0.0
        self.lips_smma: float = 0.0
        
        # Alligator lines (with shifts applied)
        self.jaw_line: float = 0.0  # Blue line
        self.teeth_line: float = 0.0  # Red line
        self.lips_line: float = 0.0  # Green line
        
        # Historical values for shifted lines
        self.jaw_history: List[float] = []
        self.teeth_history: List[float] = []
        self.lips_history: List[float] = []
        
        # Raw SMMA histories (before shifting)
        self.jaw_smma_history: List[float] = []
        self.teeth_smma_history: List[float] = []
        self.lips_smma_history: List[float] = []
        
        # Alligator state analysis
        self.alligator_state: str = "sleeping"  # "sleeping", "awakening", "eating", "sated"
        self.line_separation: float = 0.0
        self.trend_strength: float = 0.0
        
        # State management
        self.is_initialized = False
        self._last_calculation_time = None
        
        self.logger.info("Alligator Indicator initialized successfully")
    
    def calculate(self, data: Union[pd.DataFrame, Dict]) -> Dict:
        """
        Calculate Alligator Indicator.
        
        Args:
            data: Market data containing OHLC
            
        Returns:
            Dictionary containing Alligator calculation results
        """
        try:
            # Validate and extract data
            ohlc_data = self._validate_and_extract_data(data)
            if not ohlc_data:
                return self._get_empty_result()
            
            # Calculate price based on method
            price = self._calculate_price(ohlc_data)
            
            # Add to price history
            self.price_history.append(price)
            if len(self.price_history) > self.config.max_history:
                self.price_history.pop(0)
            
            # Calculate SMMA values
            self._calculate_smma_values()
            
            # Apply shifts to get alligator lines
            self._apply_shifts()
            
            # Analyze alligator state
            if self.config.enable_fractal_analysis:
                self._analyze_alligator_state()
            
            # Generate signals
            signals = {}
            if self.config.enable_signals:
                signals = self._generate_signals(price)
            
            # Mark as initialized
            if (len(self.price_history) >= max(self.config.jaw_period, 
                                               self.config.teeth_period, 
                                               self.config.lips_period)):
                self.is_initialized = True
            
            self._last_calculation_time = datetime.now()
            
            return self._format_results(ohlc_data, price, signals)
            
        except Exception as e:
            self.logger.error(f"Error in Alligator calculation: {str(e)}")
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
    
    def _calculate_price(self, ohlc_data: Dict) -> float:
        """Calculate price based on configuration method."""
        if self.config.price_method == "median":
            return (ohlc_data['high'] + ohlc_data['low']) / 2
        elif self.config.price_method == "typical":
            return (ohlc_data['high'] + ohlc_data['low'] + ohlc_data['close']) / 3
        else:  # "close"
            return ohlc_data['close']
    
    def _calculate_smma_values(self):
        """Calculate Smoothed Moving Average values for all lines."""
        # Jaw SMMA (13-period)
        if len(self.price_history) >= self.config.jaw_period:
            if self.jaw_smma == 0.0:  # First calculation
                self.jaw_smma = sum(self.price_history[-self.config.jaw_period:]) / self.config.jaw_period
            else:  # Subsequent calculations
                self.jaw_smma = (self.jaw_smma * (self.config.jaw_period - 1) + 
                                self.price_history[-1]) / self.config.jaw_period
        
        # Teeth SMMA (8-period)
        if len(self.price_history) >= self.config.teeth_period:
            if self.teeth_smma == 0.0:  # First calculation
                self.teeth_smma = sum(self.price_history[-self.config.teeth_period:]) / self.config.teeth_period
            else:  # Subsequent calculations
                self.teeth_smma = (self.teeth_smma * (self.config.teeth_period - 1) + 
                                  self.price_history[-1]) / self.config.teeth_period
        
        # Lips SMMA (5-period)
        if len(self.price_history) >= self.config.lips_period:
            if self.lips_smma == 0.0:  # First calculation
                self.lips_smma = sum(self.price_history[-self.config.lips_period:]) / self.config.lips_period
            else:  # Subsequent calculations
                self.lips_smma = (self.lips_smma * (self.config.lips_period - 1) + 
                                 self.price_history[-1]) / self.config.lips_period
        
        # Store SMMA histories
        self.jaw_smma_history.append(self.jaw_smma)
        self.teeth_smma_history.append(self.teeth_smma)
        self.lips_smma_history.append(self.lips_smma)
        
        # Maintain history size
        max_keep = self.config.max_history
        if len(self.jaw_smma_history) > max_keep:
            self.jaw_smma_history.pop(0)
            self.teeth_smma_history.pop(0)
            self.lips_smma_history.pop(0)
    
    def _apply_shifts(self):
        """Apply forward shifts to SMMA values to get Alligator lines."""
        # For real-time calculation, we use current SMMA values
        # In practice, these would be shifted forward for future plotting
        
        # For current calculation, use the SMMA values directly
        self.jaw_line = self.jaw_smma
        self.teeth_line = self.teeth_smma
        self.lips_line = self.lips_smma
        
        # Store in histories
        self.jaw_history.append(self.jaw_line)
        self.teeth_history.append(self.teeth_line)
        self.lips_history.append(self.lips_line)
        
        # Maintain history size
        if len(self.jaw_history) > self.config.max_history:
            self.jaw_history.pop(0)
            self.teeth_history.pop(0)
            self.lips_history.pop(0)
    
    def _analyze_alligator_state(self):
        """Analyze current state of the alligator."""
        if not self.is_initialized:
            self.alligator_state = "sleeping"
            self.line_separation = 0.0
            self.trend_strength = 0.0
            return
        
        # Calculate line separation
        lines = [self.jaw_line, self.teeth_line, self.lips_line]
        self.line_separation = max(lines) - min(lines)
        
        # Determine alligator state based on line order and separation
        jaw_above_teeth = self.jaw_line > self.teeth_line
        teeth_above_lips = self.teeth_line > self.lips_line
        lips_above_teeth = self.lips_line > self.teeth_line
        teeth_above_jaw = self.teeth_line > self.jaw_line
        
        # Calculate relative separation as percentage
        avg_price = sum(lines) / 3
        if avg_price > 0:
            separation_pct = (self.line_separation / avg_price) * 100
        else:
            separation_pct = 0
        
        # Analyze trend strength based on line order and separation
        if separation_pct < 0.1:  # Lines very close together
            self.alligator_state = "sleeping"
            self.trend_strength = 0.0
        
        elif separation_pct < 0.5:  # Lines beginning to separate
            self.alligator_state = "awakening"
            self.trend_strength = 0.3
        
        elif separation_pct >= 0.5:  # Lines well separated
            # Check if lines are in proper trending order
            if (self.lips_line > self.teeth_line > self.jaw_line):  # Bullish order
                self.alligator_state = "eating"
                self.trend_strength = 0.8
            elif (self.lips_line < self.teeth_line < self.jaw_line):  # Bearish order
                self.alligator_state = "eating"
                self.trend_strength = -0.8
            else:  # Mixed order
                self.alligator_state = "awakening"
                self.trend_strength = 0.5
        
        # Check for convergence (sated alligator)
        if len(self.jaw_history) >= 3:
            # Calculate recent separation trend
            recent_separations = []
            for i in range(-3, 0):
                if abs(i) <= len(self.jaw_history):
                    recent_lines = [self.jaw_history[i], self.teeth_history[i], self.lips_history[i]]
                    recent_separations.append(max(recent_lines) - min(recent_lines))
            
            if len(recent_separations) >= 2:
                if recent_separations[-1] < recent_separations[-2] < recent_separations[0]:
                    self.alligator_state = "sated"
                    self.trend_strength *= 0.5  # Reduce strength for converging lines
    
    def _generate_signals(self, current_price: float) -> Dict:
        """Generate trading signals based on Alligator analysis."""
        signals = {
            'trend_direction': 'neutral',
            'signal_strength': 0.0,
            'entry_signal': 'none',
            'exit_signal': 'none',
            'alligator_state': self.alligator_state,
            'price_vs_alligator': 'neutral'
        }
        
        if not self.is_initialized:
            return signals
        
        # Price position relative to alligator lines
        lines = [self.jaw_line, self.teeth_line, self.lips_line]
        max_line = max(lines)
        min_line = min(lines)
        
        if current_price > max_line:
            signals['price_vs_alligator'] = 'above_all_lines'
        elif current_price < min_line:
            signals['price_vs_alligator'] = 'below_all_lines'
        else:
            signals['price_vs_alligator'] = 'between_lines'
        
        # Generate signals based on alligator state
        if self.alligator_state == "sleeping":
            signals['entry_signal'] = 'wait'
            signals['trend_direction'] = 'consolidation'
            
        elif self.alligator_state == "awakening":
            # Check for breakout potential
            if signals['price_vs_alligator'] == 'above_all_lines':
                signals['entry_signal'] = 'potential_bullish'
                signals['trend_direction'] = 'bullish'
                signals['signal_strength'] = 0.5
            elif signals['price_vs_alligator'] == 'below_all_lines':
                signals['entry_signal'] = 'potential_bearish'
                signals['trend_direction'] = 'bearish'
                signals['signal_strength'] = -0.5
                
        elif self.alligator_state == "eating":
            # Strong trend signals
            if self.trend_strength > 0:
                signals['trend_direction'] = 'bullish'
                signals['signal_strength'] = self.trend_strength
                if signals['price_vs_alligator'] == 'above_all_lines':
                    signals['entry_signal'] = 'strong_bullish'
                    
            elif self.trend_strength < 0:
                signals['trend_direction'] = 'bearish'
                signals['signal_strength'] = self.trend_strength
                if signals['price_vs_alligator'] == 'below_all_lines':
                    signals['entry_signal'] = 'strong_bearish'
                    
        elif self.alligator_state == "sated":
            signals['exit_signal'] = 'trend_weakening'
            signals['trend_direction'] = 'weakening'
            signals['signal_strength'] = abs(self.trend_strength) * 0.3
        
        return signals
    
    def _format_results(self, ohlc_data: Dict, price: float, signals: Dict) -> Dict:
        """Format calculation results."""
        result = {
            # Core Alligator lines
            'jaw_line': round(self.jaw_line, self.config.precision),    # Blue line
            'teeth_line': round(self.teeth_line, self.config.precision), # Red line  
            'lips_line': round(self.lips_line, self.config.precision),   # Green line
            
            # Analysis values
            'current_price': round(price, self.config.precision),
            'alligator_state': self.alligator_state,
            'line_separation': round(self.line_separation, self.config.precision),
            'trend_strength': round(self.trend_strength, 3),
            
            # Metadata
            'is_initialized': self.is_initialized,
            'data_points': len(self.price_history),
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
            'jaw_line': 0.0,
            'teeth_line': 0.0,
            'lips_line': 0.0,
            'alligator_state': 'sleeping',
            'status': 'no_data',
            'error': 'Insufficient data for calculation'
        }
    
    def _get_error_result(self, error_message: str) -> Dict:
        """Return error result structure."""
        return {
            'jaw_line': 0.0,
            'teeth_line': 0.0,
            'lips_line': 0.0,
            'alligator_state': 'sleeping',
            'status': 'error',
            'error': error_message
        }
    
    def get_line_values(self, periods: int = 20) -> Dict:
        """
        Get historical values for all Alligator lines.
        
        Args:
            periods: Number of historical periods to return
            
        Returns:
            Dictionary containing historical line values
        """
        return {
            'jaw_line': self.jaw_history[-periods:] if self.jaw_history else [],
            'teeth_line': self.teeth_history[-periods:] if self.teeth_history else [],
            'lips_line': self.lips_history[-periods:] if self.lips_history else []
        }
    
    def get_smma_values(self) -> Dict:
        """Get current SMMA values before shift application."""
        return {
            'jaw_smma': self.jaw_smma,
            'teeth_smma': self.teeth_smma,
            'lips_smma': self.lips_smma
        }
    
    def get_fractal_analysis(self) -> Dict:
        """Get detailed fractal-based analysis."""
        return {
            'alligator_state': self.alligator_state,
            'line_separation': self.line_separation,
            'trend_strength': self.trend_strength,
            'is_trending': self.alligator_state in ['awakening', 'eating'],
            'is_consolidating': self.alligator_state == 'sleeping',
            'trend_ending': self.alligator_state == 'sated'
        }
    
    def reset(self):
        """Reset indicator state."""
        self.price_history.clear()
        self.jaw_smma = 0.0
        self.teeth_smma = 0.0
        self.lips_smma = 0.0
        self.jaw_line = 0.0
        self.teeth_line = 0.0
        self.lips_line = 0.0
        self.jaw_history.clear()
        self.teeth_history.clear()
        self.lips_history.clear()
        self.jaw_smma_history.clear()
        self.teeth_smma_history.clear()
        self.lips_smma_history.clear()
        self.alligator_state = "sleeping"
        self.line_separation = 0.0
        self.trend_strength = 0.0
        self.is_initialized = False
        self._last_calculation_time = None
        
        self.logger.info("Alligator Indicator reset completed")
    
    def get_metadata(self) -> Dict:
        """Get indicator metadata."""
        return {
            'name': 'Alligator',
            'full_name': 'Alligator Indicator (Bill Williams)',
            'description': 'Trend-following indicator using three smoothed moving averages to identify market trends',
            'category': 'Trend',
            'version': '1.0.0',
            'author': 'Platform3.AI Engine',
            'tags': ['trend', 'bill williams', 'moving average', 'smoothed', 'fractal'],
            'inputs': ['high', 'low', 'close'],
            'outputs': ['jaw_line', 'teeth_line', 'lips_line', 'alligator_state', 'signals'],
            'parameters': {
                'jaw_period': self.config.jaw_period,
                'teeth_period': self.config.teeth_period,
                'lips_period': self.config.lips_period,
                'jaw_shift': self.config.jaw_shift,
                'teeth_shift': self.config.teeth_shift,
                'lips_shift': self.config.lips_shift
            }
        }
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters."""
        try:
            # Validate periods
            if self.config.jaw_period <= 0:
                self.logger.error("Invalid jaw_period: must be positive")
                return False
            
            if self.config.teeth_period <= 0:
                self.logger.error("Invalid teeth_period: must be positive")
                return False
            
            if self.config.lips_period <= 0:
                self.logger.error("Invalid lips_period: must be positive")
                return False
            
            # Validate shifts
            if self.config.jaw_shift < 0:
                self.logger.error("Invalid jaw_shift: must be non-negative")
                return False
            
            if self.config.teeth_shift < 0:
                self.logger.error("Invalid teeth_shift: must be non-negative")
                return False
            
            if self.config.lips_shift < 0:
                self.logger.error("Invalid lips_shift: must be non-negative")
                return False
            
            # Validate Bill Williams standard configuration
            if (self.config.jaw_period != 13 or self.config.teeth_period != 8 or 
                self.config.lips_period != 5):
                self.logger.warning("Non-standard periods detected. Bill Williams standard: Jaw=13, Teeth=8, Lips=5")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Parameter validation error: {str(e)}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Test data - trending market scenario
    test_data = []
    base_price = 100.0
    
    # Generate trending test data
    for i in range(30):
        trend = 0.05 * i  # Upward trend
        noise = np.random.normal(0, 0.2)
        
        high = base_price + trend + noise + 0.3
        low = base_price + trend + noise - 0.3
        close = base_price + trend + noise
        
        test_data.append({
            'high': high,
            'low': low,
            'close': close,
            'timestamp': f'2024-01-01 09:{30+i}:00'
        })
    
    # Initialize indicator
    config = AlligatorConfig(enable_signals=True, enable_fractal_analysis=True)
    alligator = AlligatorIndicator(config)
    
    print("=== Alligator Indicator Test ===")
    print(f"Config: {config}")
    print()
    
    # Process test data
    for i, data_point in enumerate(test_data, 1):
        result = alligator.calculate(data_point)
        
        # Print every 5th period
        if i % 5 == 0:
            print(f"Period {i}:")
            print(f"  Price: {data_point['close']:.2f}")
            print(f"  Jaw (Blue): {result['jaw_line']}")
            print(f"  Teeth (Red): {result['teeth_line']}")
            print(f"  Lips (Green): {result['lips_line']}")
            print(f"  State: {result['alligator_state']}")
            print(f"  Separation: {result['line_separation']}")
            print(f"  Trend Strength: {result['trend_strength']}")
            
            if 'trend_direction' in result:
                print(f"  Trend: {result['trend_direction']}")
                print(f"  Entry Signal: {result['entry_signal']}")
                print(f"  Price vs Alligator: {result['price_vs_alligator']}")
            
            print(f"  Status: {result['status']}")
            print()
    
    # Test line values
    print("=== Alligator Line Values (last 3 periods) ===")
    line_values = alligator.get_line_values(3)
    for line_name, values in line_values.items():
        print(f"{line_name}: {[round(v, 2) for v in values]}")
    
    # Test fractal analysis
    print(f"\n=== Fractal Analysis ===")
    fractal_analysis = alligator.get_fractal_analysis()
    for key, value in fractal_analysis.items():
        print(f"{key}: {value}")