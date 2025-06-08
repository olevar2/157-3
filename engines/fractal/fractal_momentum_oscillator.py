"""
Fractal Momentum Oscillator for Platform3 AI Trading System

This indicator combines traditional momentum analysis with fractal geometry principles
to create a sophisticated momentum oscillator. It uses fractal dimension calculations
to weight momentum signals and identify turning points in price action.

Key Features:
- Multi-timeframe fractal momentum analysis
- Adaptive period adjustment based on market complexity
- Momentum strength measurement using fractal dimension
- Overbought/oversold detection with fractal validation
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
sys.path.append(str(project_root))
sys.path.append(str(project_root / "shared"))
sys.path.append(str(project_root / "engines"))

from shared.logging.platform3_logger import Platform3Logger

# Simple IndicatorBase for compatibility
class IndicatorBase:
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = Platform3Logger(self.__class__.__name__)


class FractalMomentumOscillator(IndicatorBase):
    """
    Fractal Momentum Oscillator
    
    Analyzes momentum using fractal geometry principles to provide
    enhanced momentum signals with reduced noise and improved timing.
    """
    
    def __init__(self, 
                 momentum_period: int = 14,
                 fractal_period: int = 5,
                 smoothing_period: int = 3,
                 overbought_level: float = 80.0,
                 oversold_level: float = 20.0,
                 adaptive_periods: bool = True):
        """
        Initialize Fractal Momentum Oscillator
        
        Args:
            momentum_period: Period for momentum calculation
            fractal_period: Period for fractal dimension calculation
            smoothing_period: Period for signal smoothing
            overbought_level: Overbought threshold (0-100)
            oversold_level: Oversold threshold (0-100)
            adaptive_periods: Whether to adapt periods based on fractal dimension
        """
        super().__init__()
        
        self.momentum_period = momentum_period
        self.fractal_period = fractal_period
        self.smoothing_period = smoothing_period
        self.overbought_level = overbought_level
        self.oversold_level = oversold_level
        self.adaptive_periods = adaptive_periods
        
        # Initialize logger
        self.logger = Platform3Logger(self.__class__.__name__)
        
        # Initialize calculation variables
        self.prices = []
        self.momentum_values = []
        self.fractal_dimensions = []
        self.oscillator_values = []
        self.smoothed_values = []
        
        # Signal tracking
        self.signals = []
        self.signal_strength = []
        
        self.logger.info(f"Initialized FractalMomentumOscillator with momentum_period={momentum_period}")
    
    def _calculate_momentum(self, prices: np.ndarray, period: int) -> float:
        """Calculate traditional momentum"""
        if len(prices) < period + 1:
            return 0.0
        
        current_price = prices[-1]
        past_price = prices[-(period + 1)]
        
        if past_price == 0:
            return 0.0
        
        momentum = ((current_price - past_price) / past_price) * 100
        return momentum
    
    def _calculate_rate_of_change(self, prices: np.ndarray, period: int) -> float:
        """Calculate rate of change"""
        if len(prices) < period + 1:
            return 0.0
        
        current_price = prices[-1]
        past_price = prices[-(period + 1)]
        
        if past_price == 0:
            return 0.0
        
        roc = ((current_price / past_price) - 1) * 100
        return roc
    
    def _calculate_fractal_dimension(self, data: np.ndarray, method: str = 'higuchi') -> float:
        """
        Calculate fractal dimension using specified method
        
        Args:
            data: Price data array
            method: Method to use ('higuchi', 'variance')
        """
        if len(data) < 10:
            return 1.5  # Default neutral value
        
        if method == 'higuchi':
            return self._higuchi_fractal_dimension(data)
        elif method == 'variance':
            return self._variance_fractal_dimension(data)
        else:
            return self._higuchi_fractal_dimension(data)
    
    def _higuchi_fractal_dimension(self, data: np.ndarray, k_max: int = 8) -> float:
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
                return max(1.0, min(2.0, coeffs[0]))
            else:
                return 1.5
                
        except Exception as e:
            self.logger.warning(f"Error in Higuchi calculation: {e}")
            return 1.5
    
    def _variance_fractal_dimension(self, data: np.ndarray) -> float:
        """Calculate fractal dimension using variance scaling method"""
        try:
            if len(data) < 4:
                return 1.5
            
            # Calculate variances at different scales
            scales = []
            variances = []
            
            for scale in range(2, min(len(data) // 2, 10)):
                # Downsample data at this scale
                downsampled = data[::scale]
                if len(downsampled) > 2:
                    variance = np.var(np.diff(downsampled))
                    if variance > 0:
                        scales.append(np.log(scale))
                        variances.append(np.log(variance))
            
            if len(scales) > 2:
                # Linear regression to find scaling exponent
                coeffs = np.polyfit(scales, variances, 1)
                # Convert to fractal dimension (D = 2 - H/2, where H is Hurst exponent)
                hurst = coeffs[0] / 2
                dimension = 2 - hurst
                return max(1.0, min(2.0, dimension))
            else:
                return 1.5
                
        except Exception as e:
            self.logger.warning(f"Error in variance fractal calculation: {e}")
            return 1.5
    
    def _adaptive_period_adjustment(self, fractal_dimension: float, base_period: int) -> int:
        """Adjust calculation period based on fractal dimension"""
        if not self.adaptive_periods:
            return base_period
        
        # Higher fractal dimension (more complex) -> shorter period
        # Lower fractal dimension (more trending) -> longer period
        if fractal_dimension > 1.7:
            adjustment_factor = 0.7  # Reduce period by 30%
        elif fractal_dimension > 1.5:
            adjustment_factor = 0.85  # Reduce period by 15%
        elif fractal_dimension < 1.3:
            adjustment_factor = 1.3  # Increase period by 30%
        elif fractal_dimension < 1.4:
            adjustment_factor = 1.15  # Increase period by 15%
        else:
            adjustment_factor = 1.0  # No adjustment
        
        adjusted_period = int(base_period * adjustment_factor)
        return max(3, min(50, adjusted_period))
    
    def _calculate_fractal_weighted_momentum(self, momentum: float, fractal_dimension: float) -> float:
        """Weight momentum by fractal dimension"""
        # Normalize fractal dimension to 0-1 range (assuming typical range 1.0-2.0)
        normalized_fd = (fractal_dimension - 1.0) / 1.0
        normalized_fd = max(0.0, min(1.0, normalized_fd))
        
        # Weight factor: higher fractal dimension = more noise = lower weight
        weight_factor = 1.0 - (normalized_fd * 0.5)  # Weight between 0.5 and 1.0
        
        weighted_momentum = momentum * weight_factor
        return weighted_momentum
    
    def _normalize_to_oscillator(self, values: List[float], period: int = 20) -> float:
        """Normalize momentum values to 0-100 oscillator range"""
        if len(values) < period:
            return 50.0  # Neutral value
        
        recent_values = values[-period:]
        current_value = values[-1]
        
        min_val = min(recent_values)
        max_val = max(recent_values)
        
        if max_val == min_val:
            return 50.0
        
        # Normalize to 0-100 range
        normalized = ((current_value - min_val) / (max_val - min_val)) * 100
        return normalized
    
    def _generate_signals(self, oscillator_value: float, fractal_dimension: float) -> Dict[str, Any]:
        """Generate trading signals based on oscillator and fractal analysis"""
        signal_data = {
            'signal': 'NONE',
            'strength': 0.0,
            'confidence': 0.0,
            'condition': 'neutral'
        }
        
        # Adjust thresholds based on fractal dimension
        # Higher fractal dimension = more noise = stricter thresholds
        fd_adjustment = (fractal_dimension - 1.5) * 10  # -5 to +5 adjustment
        
        adjusted_overbought = self.overbought_level + fd_adjustment
        adjusted_oversold = self.oversold_level - fd_adjustment
        
        # Ensure thresholds are within reasonable bounds
        adjusted_overbought = max(75, min(95, adjusted_overbought))
        adjusted_oversold = max(5, min(25, adjusted_oversold))
        
        # Generate signals
        if oscillator_value >= adjusted_overbought:
            signal_data = {
                'signal': 'SELL',
                'strength': min((oscillator_value - adjusted_overbought) / 10, 1.0),
                'confidence': 0.7 if fractal_dimension < 1.6 else 0.5,
                'condition': 'overbought'
            }
        elif oscillator_value <= adjusted_oversold:
            signal_data = {
                'signal': 'BUY',
                'strength': min((adjusted_oversold - oscillator_value) / 10, 1.0),
                'confidence': 0.7 if fractal_dimension < 1.6 else 0.5,
                'condition': 'oversold'
            }
        elif oscillator_value > 50 + fd_adjustment:
            signal_data['condition'] = 'bullish'
        elif oscillator_value < 50 - fd_adjustment:
            signal_data['condition'] = 'bearish'
        
        return signal_data
    
    def calculate(self, data: Dict[str, Union[float, int]], **kwargs) -> Dict[str, Any]:
        """
        Calculate fractal momentum oscillator
        
        Args:
            data: Dictionary containing 'close' price
            
        Returns:
            Dictionary containing oscillator analysis results
        """
        try:
            # Extract price data
            close = float(data.get('close', 0))
            
            # Store price
            self.prices.append(close)
            
            # Keep only required data length
            max_length = max(200, self.momentum_period * 5)
            if len(self.prices) > max_length:
                self.prices = self.prices[-max_length:]
            
            # Initialize result
            result = {
                'fractal_momentum_oscillator': 50.0,
                'fractal_momentum_smoothed': 50.0,
                'fractal_dimension': 1.5,
                'momentum_raw': 0.0,
                'signal': 'NONE',
                'signal_strength': 0.0,
                'signal_confidence': 0.0,
                'market_condition': 'neutral',
                'adaptive_period': self.momentum_period
            }
            
            # Need sufficient data for calculation
            if len(self.prices) < max(self.momentum_period + 1, self.fractal_period * 2):
                return result
            
            prices_array = np.array(self.prices)
            
            # Calculate fractal dimension
            recent_prices = prices_array[-self.fractal_period * 2:]
            fractal_dimension = self._calculate_fractal_dimension(recent_prices)
            self.fractal_dimensions.append(fractal_dimension)
            
            # Keep only recent fractal dimensions
            if len(self.fractal_dimensions) > 100:
                self.fractal_dimensions = self.fractal_dimensions[-100:]
            
            # Adjust period based on fractal dimension
            adaptive_period = self._adaptive_period_adjustment(fractal_dimension, self.momentum_period)
            
            # Calculate raw momentum
            raw_momentum = self._calculate_momentum(prices_array, adaptive_period)
            
            # Calculate fractal-weighted momentum
            weighted_momentum = self._calculate_fractal_weighted_momentum(raw_momentum, fractal_dimension)
            self.momentum_values.append(weighted_momentum)
            
            # Keep only recent momentum values
            if len(self.momentum_values) > 100:
                self.momentum_values = self.momentum_values[-100:]
            
            # Normalize to oscillator range (0-100)
            oscillator_value = self._normalize_to_oscillator(self.momentum_values)
            self.oscillator_values.append(oscillator_value)
            
            # Calculate smoothed oscillator
            if len(self.oscillator_values) >= self.smoothing_period:
                smoothed_value = np.mean(self.oscillator_values[-self.smoothing_period:])
            else:
                smoothed_value = oscillator_value
            
            self.smoothed_values.append(smoothed_value)
            
            # Keep only recent values
            if len(self.oscillator_values) > 100:
                self.oscillator_values = self.oscillator_values[-100:]
            if len(self.smoothed_values) > 100:
                self.smoothed_values = self.smoothed_values[-100:]
            
            # Generate signals
            signal_data = self._generate_signals(smoothed_value, fractal_dimension)
            self.signals.append(signal_data)
            
            # Keep only recent signals
            if len(self.signals) > 50:
                self.signals = self.signals[-50:]
            
            # Update result
            result.update({
                'fractal_momentum_oscillator': oscillator_value,
                'fractal_momentum_smoothed': smoothed_value,
                'fractal_dimension': fractal_dimension,
                'momentum_raw': raw_momentum,
                'signal': signal_data['signal'],
                'signal_strength': signal_data['strength'],
                'signal_confidence': signal_data['confidence'],
                'market_condition': signal_data['condition'],
                'adaptive_period': adaptive_period
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in FractalMomentumOscillator calculation: {e}")
            return {
                'fractal_momentum_oscillator': 50.0,
                'fractal_momentum_smoothed': 50.0,
                'fractal_dimension': 1.5,
                'momentum_raw': 0.0,
                'signal': 'NONE',
                'signal_strength': 0.0,
                'signal_confidence': 0.0,
                'market_condition': 'neutral',
                'adaptive_period': self.momentum_period
            }
    
    def get_signals(self) -> Dict[str, Any]:
        """Get current momentum signals"""
        if not self.signals:
            return {'signal': 'NONE', 'strength': 0.0, 'confidence': 0.0, 'condition': 'neutral'}
        
        latest_signal = self.signals[-1]
        return {
            'signal': latest_signal['signal'],
            'strength': latest_signal['strength'],
            'confidence': latest_signal['confidence'],
            'condition': latest_signal['condition']
        }
    
    def get_oscillator_levels(self) -> Dict[str, float]:
        """Get current oscillator levels and thresholds"""
        current_fd = self.fractal_dimensions[-1] if self.fractal_dimensions else 1.5
        fd_adjustment = (current_fd - 1.5) * 10
        
        return {
            'current_value': self.smoothed_values[-1] if self.smoothed_values else 50.0,
            'overbought_threshold': max(75, min(95, self.overbought_level + fd_adjustment)),
            'oversold_threshold': max(5, min(25, self.oversold_level - fd_adjustment)),
            'neutral_level': 50.0
        }
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current fractal momentum metrics"""
        if not self.momentum_values or not self.fractal_dimensions:
            return {
                'avg_momentum': 0.0,
                'momentum_volatility': 0.0,
                'avg_fractal_dimension': 1.5,
                'signal_frequency': 0.0
            }
        
        recent_signals = [s for s in self.signals[-20:] if s['signal'] != 'NONE']
        
        return {
            'avg_momentum': np.mean(self.momentum_values[-20:]),
            'momentum_volatility': np.std(self.momentum_values[-20:]),
            'avg_fractal_dimension': np.mean(self.fractal_dimensions[-20:]),
            'signal_frequency': len(recent_signals) / 20
        }
    
    def interpret_signal(self, signal_data: Dict[str, Any]) -> str:
        """
        Interpret fractal momentum oscillator signal for trading decisions
        
        Args:
            signal_data: Signal data from calculate method
            
        Returns:
            Human-readable interpretation
        """
        signal = signal_data.get('signal', 'NONE')
        oscillator = signal_data.get('fractal_momentum_oscillator', 50)
        smoothed = signal_data.get('fractal_momentum_smoothed', 50)
        fractal_dim = signal_data.get('fractal_dimension', 1.5)
        condition = signal_data.get('market_condition', 'neutral')
        strength = signal_data.get('signal_strength', 0)
        confidence = signal_data.get('signal_confidence', 0)
        
        interpretation = f"Fractal Momentum Oscillator: {oscillator:.1f} (Smoothed: {smoothed:.1f})\n"
        interpretation += f"Market Condition: {condition.capitalize()}\n"
        interpretation += f"Fractal Dimension: {fractal_dim:.3f} - "
        
        if fractal_dim > 1.7:
            interpretation += "Very noisy market, signals less reliable"
        elif fractal_dim > 1.5:
            interpretation += "Moderate market noise"
        elif fractal_dim < 1.3:
            interpretation += "Strong trending market, signals more reliable"
        else:
            interpretation += "Balanced market conditions"
        
        if signal != 'NONE':
            strength_desc = "Strong" if strength > 0.7 else "Moderate" if strength > 0.4 else "Weak"
            confidence_desc = "High" if confidence > 0.6 else "Medium" if confidence > 0.4 else "Low"
            
            interpretation += f"\n\n{signal} Signal: {strength_desc} strength ({strength:.2f}), {confidence_desc} confidence ({confidence:.2f})"
            
            if signal == 'BUY':
                interpretation += "\nOversold conditions detected - potential bounce opportunity"
            else:
                interpretation += "\nOverbought conditions detected - potential pullback expected"
        else:
            interpretation += f"\n\nNo clear signals. Monitor for momentum shifts."
        
        return interpretation
