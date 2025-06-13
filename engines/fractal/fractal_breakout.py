"""
Fractal Breakout Indicator for Platform3 AI Trading System

This indicator detects fractal-based breakout patterns by analyzing price movements
relative to fractal support and resistance levels. It combines traditional breakout
detection with fractal geometry principles to identify high-probability breakout scenarios.

Key Features:
- Multi-timeframe fractal analysis
- Dynamic support/resistance level detection
- Breakout strength measurement using fractal dimension
- Signal generation with confidence scoring
- AI/ML ready outputs for trading agents

Author: Platform3 AI Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent

from shared.logging.platform3_logger import Platform3Logger

# Simple IndicatorBase for compatibility
class IndicatorBase:
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = Platform3Logger(self.__class__.__name__)


class FractalBreakoutIndicator(IndicatorBase):
    """
    Fractal Breakout Indicator
    
    Identifies breakout patterns using fractal analysis principles.
    Combines traditional breakout detection with fractal dimension calculations
    to provide enhanced signal quality and reduced false breakouts.
    """
    
    def __init__(self, 
                 fractal_period: int = 5,
                 strength_period: int = 14,
                 confirmation_period: int = 3,
                 min_fractal_dimension: float = 1.3,
                 max_fractal_dimension: float = 1.7):
        """
        Initialize Fractal Breakout Indicator
        
        Args:
            fractal_period: Period for fractal pattern detection
            strength_period: Period for breakout strength calculation
            confirmation_period: Periods needed for breakout confirmation
            min_fractal_dimension: Minimum fractal dimension for valid patterns
            max_fractal_dimension: Maximum fractal dimension for valid patterns
        """
        super().__init__()
        
        self.fractal_period = fractal_period
        self.strength_period = strength_period
        self.confirmation_period = confirmation_period
        self.min_fractal_dimension = min_fractal_dimension
        self.max_fractal_dimension = max_fractal_dimension
        
        # Initialize logger
        self.logger = Platform3Logger(self.__class__.__name__)
        
        # Initialize calculation variables
        self.prices = []
        self.highs = []
        self.lows = []
        self.volumes = []
        
        # Fractal levels storage
        self.resistance_levels = []
        self.support_levels = []
        self.fractal_dimensions = []
        
        # Breakout tracking
        self.breakout_signals = []
        self.breakout_strength = []
        self.confirmation_count = 0
        
        self.logger.info(f"Initialized FractalBreakoutIndicator with fractal_period={fractal_period}")
    
    def _identify_fractal_highs(self, data: np.ndarray, period: int) -> List[int]:
        """Identify fractal high points"""
        fractal_highs = []
        
        for i in range(period, len(data) - period):
            is_fractal = True
            center_value = data[i]
            
            # Check if center is highest among surrounding values
            for j in range(i - period, i + period + 1):
                if j != i and data[j] >= center_value:
                    is_fractal = False
                    break
            
            if is_fractal:
                fractal_highs.append(i)
        
        return fractal_highs
    
    def _identify_fractal_lows(self, data: np.ndarray, period: int) -> List[int]:
        """Identify fractal low points"""
        fractal_lows = []
        
        for i in range(period, len(data) - period):
            is_fractal = True
            center_value = data[i]
            
            # Check if center is lowest among surrounding values
            for j in range(i - period, i + period + 1):
                if j != i and data[j] <= center_value:
                    is_fractal = False
                    break
            
            if is_fractal:
                fractal_lows.append(i)
        
        return fractal_lows
    
    def _calculate_fractal_dimension(self, data: np.ndarray, method: str = 'higuchi') -> float:
        """
        Calculate fractal dimension using specified method
        
        Args:
            data: Price data array
            method: Method to use ('higuchi', 'box_counting')
        """
        if len(data) < 10:
            return 1.5  # Default neutral value
        
        if method == 'higuchi':
            return self._higuchi_fractal_dimension(data)
        elif method == 'box_counting':
            return self._box_counting_dimension(data)
        else:
            return self._higuchi_fractal_dimension(data)
    
    def _higuchi_fractal_dimension(self, data: np.ndarray, k_max: int = 10) -> float:
        """Calculate fractal dimension using Higuchi's method"""
        try:
            N = len(data)
            if N < k_max * 2:
                k_max = max(2, N // 2)
            
            lk_values = []
            k_values = []
            
            for k in range(1, k_max + 1):
                lk = 0
                for m in range(k):
                    lm = 0
                    max_i = int((N - m - 1) / k)
                    if max_i > 0:
                        for i in range(1, max_i + 1):
                            lm += abs(data[m + i * k] - data[m + (i - 1) * k])
                        lm = lm * (N - 1) / (max_i * k * k)
                        lk += lm
                
                if k > 0:
                    lk = lk / k
                    if lk > 0:
                        lk_values.append(np.log(lk))
                        k_values.append(np.log(1.0 / k))
            
            if len(lk_values) > 1:
                # Linear regression to find slope
                coeffs = np.polyfit(k_values, lk_values, 1)
                return coeffs[0]
            else:
                return 1.5
                
        except Exception as e:
            self.logger.warning(f"Error in Higuchi calculation: {e}")
            return 1.5
    
    def _box_counting_dimension(self, data: np.ndarray) -> float:
        """Calculate fractal dimension using box counting method"""
        try:
            # Normalize data to [0, 1]
            data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
            
            # Define box sizes
            scales = np.logspace(0.01, 1, num=20)
            counts = []
            
            for scale in scales:
                # Count boxes needed to cover the data
                grid_size = int(1 / scale)
                if grid_size < 2:
                    continue
                    
                boxes = set()
                for i, value in enumerate(data_normalized):
                    x_box = int(i * grid_size / len(data))
                    y_box = int(value * grid_size)
                    boxes.add((x_box, y_box))
                
                if len(boxes) > 0:
                    counts.append(len(boxes))
            
            if len(counts) > 1:
                # Calculate dimension as slope
                log_scales = np.log(scales[:len(counts)])
                log_counts = np.log(counts)
                coeffs = np.polyfit(log_scales, log_counts, 1)
                return -coeffs[0]  # Negative because we want positive dimension
            else:
                return 1.5
                
        except Exception as e:
            self.logger.warning(f"Error in box counting calculation: {e}")
            return 1.5
    
    def _calculate_breakout_strength(self, price: float, level: float, volume: float) -> float:
        """Calculate breakout strength based on price penetration and volume"""
        if level == 0:
            return 0.0
        
        # Price penetration strength
        penetration = abs(price - level) / level
        
        # Volume factor (normalized)
        avg_volume = np.mean(self.volumes[-self.strength_period:]) if len(self.volumes) >= self.strength_period else volume
        volume_factor = volume / avg_volume if avg_volume > 0 else 1.0
        
        # Combined strength
        strength = penetration * volume_factor
        
        return min(strength, 10.0)  # Cap at 10
    
    def _detect_breakout(self, current_price: float, current_volume: float) -> Dict[str, Any]:
        """Detect if current price represents a breakout"""
        breakout_signal = {
            'signal': 'NONE',
            'type': 'none',
            'strength': 0.0,
            'confidence': 0.0,
            'level': None,
            'fractal_dimension': None
        }
        
        if not self.resistance_levels and not self.support_levels:
            return breakout_signal
        
        # Check resistance breakout (bullish)
        if self.resistance_levels:
            nearest_resistance = min(self.resistance_levels, key=lambda x: abs(x - current_price))
            if current_price > nearest_resistance:
                strength = self._calculate_breakout_strength(current_price, nearest_resistance, current_volume)
                if strength > 0.02:  # Minimum 2% breakout
                    breakout_signal = {
                        'signal': 'BUY',
                        'type': 'resistance_breakout',
                        'strength': strength,
                        'confidence': min(strength * 10, 1.0),
                        'level': nearest_resistance,
                        'fractal_dimension': self.fractal_dimensions[-1] if self.fractal_dimensions else 1.5
                    }
        
        # Check support breakdown (bearish)
        if self.support_levels:
            nearest_support = min(self.support_levels, key=lambda x: abs(x - current_price))
            if current_price < nearest_support:
                strength = self._calculate_breakout_strength(current_price, nearest_support, current_volume)
                if strength > 0.02:  # Minimum 2% breakdown
                    if breakout_signal['signal'] == 'NONE' or strength > breakout_signal['strength']:
                        breakout_signal = {
                            'signal': 'SELL',
                            'type': 'support_breakdown',
                            'strength': strength,
                            'confidence': min(strength * 10, 1.0),
                            'level': nearest_support,
                            'fractal_dimension': self.fractal_dimensions[-1] if self.fractal_dimensions else 1.5
                        }
        
        return breakout_signal
    
    def calculate(self, data: Dict[str, Union[float, int]], **kwargs) -> Dict[str, Any]:
        """
        Calculate fractal breakout signals
        
        Args:
            data: Dictionary containing 'close', 'high', 'low', 'volume'
            
        Returns:
            Dictionary containing breakout analysis results
        """
        try:
            # Extract price data
            close = float(data.get('close', 0))
            high = float(data.get('high', close))
            low = float(data.get('low', close))
            volume = float(data.get('volume', 0))
            
            # Store data
            self.prices.append(close)
            self.highs.append(high)
            self.lows.append(low)
            self.volumes.append(volume)
            
            # Keep only required data length
            max_length = max(100, self.strength_period * 3)
            if len(self.prices) > max_length:
                self.prices = self.prices[-max_length:]
                self.highs = self.highs[-max_length:]
                self.lows = self.lows[-max_length:]
                self.volumes = self.volumes[-max_length:]
            
            # Calculate results
            result = {
                'signal': 'NONE',
                'fractal_breakout_strength': 0.0,
                'fractal_breakout_confidence': 0.0,
                'fractal_dimension': 1.5,
                'resistance_level': None,
                'support_level': None,
                'breakout_type': 'none',
                'confirmation_periods': 0
            }
            
            # Need sufficient data for analysis
            if len(self.prices) < self.fractal_period * 2 + 1:
                return result
            
            # Update fractal levels
            highs_array = np.array(self.highs)
            lows_array = np.array(self.lows)
            
            # Identify fractal points
            fractal_highs = self._identify_fractal_highs(highs_array, self.fractal_period)
            fractal_lows = self._identify_fractal_lows(lows_array, self.fractal_period)
            
            # Update resistance and support levels
            if fractal_highs:
                recent_highs = [highs_array[i] for i in fractal_highs[-3:]]  # Last 3 fractal highs
                self.resistance_levels = recent_highs
            
            if fractal_lows:
                recent_lows = [lows_array[i] for i in fractal_lows[-3:]]  # Last 3 fractal lows
                self.support_levels = recent_lows
            
            # Calculate fractal dimension
            recent_prices = np.array(self.prices[-self.strength_period:])
            fractal_dimension = self._calculate_fractal_dimension(recent_prices)
            self.fractal_dimensions.append(fractal_dimension)
            
            # Keep only recent fractal dimensions
            if len(self.fractal_dimensions) > 50:
                self.fractal_dimensions = self.fractal_dimensions[-50:]
            
            # Detect breakout
            breakout = self._detect_breakout(close, volume)
            
            # Update confirmation count
            if breakout['signal'] != 'NONE':
                if len(self.breakout_signals) > 0 and self.breakout_signals[-1]['signal'] == breakout['signal']:
                    self.confirmation_count += 1
                else:
                    self.confirmation_count = 1
            else:
                self.confirmation_count = 0
            
            # Store breakout signal
            self.breakout_signals.append(breakout)
            if len(self.breakout_signals) > 20:
                self.breakout_signals = self.breakout_signals[-20:]
            
            # Update result
            result.update({
                'signal': breakout['signal'],
                'fractal_breakout_strength': breakout['strength'],
                'fractal_breakout_confidence': breakout['confidence'],
                'fractal_dimension': fractal_dimension,
                'resistance_level': self.resistance_levels[-1] if self.resistance_levels else None,
                'support_level': self.support_levels[-1] if self.support_levels else None,
                'breakout_type': breakout['type'],
                'confirmation_periods': self.confirmation_count
            })
            
            # Adjust confidence based on fractal dimension validity
            if (self.min_fractal_dimension <= fractal_dimension <= self.max_fractal_dimension and
                self.confirmation_count >= self.confirmation_period):
                result['fractal_breakout_confidence'] = min(result['fractal_breakout_confidence'] * 1.5, 1.0)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in FractalBreakoutIndicator calculation: {e}")
            return {
                'signal': 'NONE',
                'fractal_breakout_strength': 0.0,
                'fractal_breakout_confidence': 0.0,
                'fractal_dimension': 1.5,
                'resistance_level': None,
                'support_level': None,
                'breakout_type': 'none',
                'confirmation_periods': 0
            }
    
    def get_signals(self) -> Dict[str, Any]:
        """Get current fractal breakout signals"""
        if not self.breakout_signals:
            return {'signal': 'NONE', 'strength': 0.0, 'confidence': 0.0}
        
        latest_signal = self.breakout_signals[-1]
        return {
            'signal': latest_signal['signal'],
            'strength': latest_signal['strength'],
            'confidence': latest_signal['confidence'],
            'type': latest_signal['type'],
            'confirmation_periods': self.confirmation_count
        }
    
    def get_fractal_levels(self) -> Dict[str, List[float]]:
        """Get current fractal support and resistance levels"""
        return {
            'resistance_levels': self.resistance_levels.copy(),
            'support_levels': self.support_levels.copy()
        }
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current fractal breakout metrics"""
        if not self.fractal_dimensions:
            return {
                'current_fractal_dimension': 1.5,
                'avg_fractal_dimension': 1.5,
                'breakout_frequency': 0.0,
                'signal_strength': 0.0
            }
        
        recent_breakouts = [s for s in self.breakout_signals[-20:] if s['signal'] != 'NONE']
        
        return {
            'current_fractal_dimension': self.fractal_dimensions[-1],
            'avg_fractal_dimension': np.mean(self.fractal_dimensions[-10:]),
            'breakout_frequency': len(recent_breakouts) / 20,
            'signal_strength': np.mean([s['strength'] for s in recent_breakouts]) if recent_breakouts else 0.0
        }
    
    def interpret_signal(self, signal_data: Dict[str, Any]) -> str:
        """
        Interpret fractal breakout signal for trading decisions
        
        Args:
            signal_data: Signal data from calculate method
            
        Returns:
            Human-readable interpretation
        """
        signal = signal_data.get('signal', 'NONE')
        strength = signal_data.get('fractal_breakout_strength', 0)
        confidence = signal_data.get('fractal_breakout_confidence', 0)
        breakout_type = signal_data.get('breakout_type', 'none')
        fractal_dim = signal_data.get('fractal_dimension', 1.5)
        confirmation = signal_data.get('confirmation_periods', 0)
        
        if signal == 'NONE':
            return f"No breakout detected. Fractal dimension: {fractal_dim:.3f} (Market complexity: {'High' if fractal_dim > 1.6 else 'Low' if fractal_dim < 1.4 else 'Medium'})"
        
        strength_desc = "Strong" if strength > 0.1 else "Moderate" if strength > 0.05 else "Weak"
        confidence_desc = "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
        
        interpretation = f"{signal} signal detected - {breakout_type.replace('_', ' ').title()}\n"
        interpretation += f"Strength: {strength_desc} ({strength:.3f}), Confidence: {confidence_desc} ({confidence:.3f})\n"
        interpretation += f"Fractal Dimension: {fractal_dim:.3f} - "
        
        if fractal_dim > 1.6:
            interpretation += "High market complexity, trend may be unstable"
        elif fractal_dim < 1.4:
            interpretation += "Low market complexity, strong trend likely"
        else:
            interpretation += "Medium market complexity, balanced conditions"
        
        if confirmation >= 3:
            interpretation += f"\nConfirmed signal ({confirmation} periods) - Higher reliability"
        elif confirmation > 0:
            interpretation += f"\nPartial confirmation ({confirmation} periods) - Monitor for strengthening"
        else:
            interpretation += "\nNew signal - Wait for confirmation"
        
        return interpretation
