"""
Fractal Volume Analysis for Platform3 AI Trading System

This indicator applies fractal analysis principles to volume data to identify
volume patterns, accumulation/distribution phases, and volume-based signals.
It combines traditional volume analysis with fractal geometry to provide
enhanced insights into market participation and strength.

Key Features:
- Multi-timeframe fractal volume analysis
- Volume pattern recognition using fractal dimension
- Accumulation/distribution detection with fractal validation
- Volume strength measurement with complexity analysis
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


class FractalVolumeAnalysis(IndicatorBase):
    """
    Fractal Volume Analysis Indicator
    
    Analyzes volume using fractal geometry principles to identify
    volume patterns, market phases, and strength of price movements.
    """
    
    def __init__(self, 
                 volume_period: int = 20,
                 fractal_period: int = 14,
                 trend_period: int = 50,
                 volume_threshold: float = 1.5,
                 complexity_threshold: float = 1.6):
        """
        Initialize Fractal Volume Analysis
        
        Args:
            volume_period: Period for volume calculations
            fractal_period: Period for fractal dimension calculation
            trend_period: Period for trend analysis
            volume_threshold: Multiplier for high volume detection
            complexity_threshold: Fractal dimension threshold for complexity
        """
        super().__init__()
        
        self.volume_period = volume_period
        self.fractal_period = fractal_period
        self.trend_period = trend_period
        self.volume_threshold = volume_threshold
        self.complexity_threshold = complexity_threshold
        
        # Initialize logger
        self.logger = Platform3Logger(self.__class__.__name__)
        
        # Initialize calculation variables
        self.prices = []
        self.volumes = []
        self.price_changes = []
        self.volume_fractal_dimensions = []
        self.price_volume_correlation = []
        
        # Analysis results
        self.volume_patterns = []
        self.accumulation_distribution = []
        self.volume_strength_signals = []
        
        # Volume metrics
        self.avg_volume = 0.0
        self.volume_volatility = 0.0
        self.volume_trend = 'neutral'
        
        self.logger.info(f"Initialized FractalVolumeAnalysis with volume_period={volume_period}")
    
    def _calculate_volume_fractal_dimension(self, volumes: np.ndarray, method: str = 'higuchi') -> float:
        """
        Calculate fractal dimension of volume data
        
        Args:
            volumes: Volume data array
            method: Method to use ('higuchi', 'box_counting')
        """
        if len(volumes) < 10:
            return 1.5  # Default neutral value
        
        # Normalize volumes to reduce scale effects
        volumes_normalized = volumes / np.mean(volumes)
        
        if method == 'higuchi':
            return self._higuchi_fractal_dimension(volumes_normalized)
        elif method == 'box_counting':
            return self._box_counting_dimension(volumes_normalized)
        else:
            return self._higuchi_fractal_dimension(volumes_normalized)
    
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
                return max(1.0, min(2.0, coeffs[0]))
            else:
                return 1.5
                
        except Exception as e:
            self.logger.warning(f"Error in Higuchi calculation: {e}")
            return 1.5
    
    def _box_counting_dimension(self, data: np.ndarray) -> float:
        """Calculate fractal dimension using box counting method"""
        try:
            # Normalize data to [0, 1]
            data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
            
            # Define box sizes
            scales = np.logspace(-2, 0, num=15)
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
            
            if len(counts) > 2:
                # Calculate dimension as slope
                log_scales = np.log(scales[:len(counts)])
                log_counts = np.log(counts)
                coeffs = np.polyfit(log_scales, log_counts, 1)
                return max(1.0, min(2.0, -coeffs[0]))  # Negative because we want positive dimension
            else:
                return 1.5
                
        except Exception as e:
            self.logger.warning(f"Error in box counting calculation: {e}")
            return 1.5
    
    def _calculate_volume_pattern(self, volumes: np.ndarray, prices: np.ndarray) -> Dict[str, Any]:
        """Identify volume patterns using fractal analysis"""
        pattern_data = {
            'pattern_type': 'normal',
            'strength': 0.5,
            'reliability': 0.5,
            'description': 'Normal volume activity'
        }
        
        if len(volumes) < self.volume_period:
            return pattern_data
        
        # Calculate volume statistics
        recent_volume = volumes[-1]
        avg_volume = np.mean(volumes[-self.volume_period:])
        volume_std = np.std(volumes[-self.volume_period:])
        
        # Calculate price change
        if len(prices) >= 2:
            price_change = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] != 0 else 0
        else:
            price_change = 0
        
        # Calculate volume fractal dimension
        volume_fd = self._calculate_volume_fractal_dimension(volumes[-self.fractal_period:])
        
        # Volume surge detection
        if recent_volume > avg_volume * self.volume_threshold:
            if abs(price_change) > 0.01:  # Significant price movement
                if volume_fd < self.complexity_threshold:  # Low complexity = strong pattern
                    pattern_data = {
                        'pattern_type': 'breakout' if price_change > 0 else 'breakdown',
                        'strength': min((recent_volume / avg_volume) / self.volume_threshold, 1.0),
                        'reliability': max(0.6, 1.0 - (volume_fd - 1.0)),
                        'description': f"High volume {'breakout' if price_change > 0 else 'breakdown'} with low complexity"
                    }
                else:
                    pattern_data = {
                        'pattern_type': 'noise',
                        'strength': 0.3,
                        'reliability': 0.3,
                        'description': 'High volume but high complexity suggests noise'
                    }
            else:
                # High volume, low price movement
                pattern_data = {
                    'pattern_type': 'accumulation' if price_change >= 0 else 'distribution',
                    'strength': min((recent_volume / avg_volume) / self.volume_threshold, 1.0),
                    'reliability': max(0.5, 1.0 - (volume_fd - 1.0)),
                    'description': f"High volume {'accumulation' if price_change >= 0 else 'distribution'} phase"
                }
        
        # Low volume patterns
        elif recent_volume < avg_volume * 0.7:
            if volume_fd > self.complexity_threshold:
                pattern_data = {
                    'pattern_type': 'indecision',
                    'strength': 0.3,
                    'reliability': 0.4,
                    'description': 'Low volume with high complexity indicates indecision'
                }
            else:
                pattern_data = {
                    'pattern_type': 'consolidation',
                    'strength': 0.4,
                    'reliability': 0.6,
                    'description': 'Low volume consolidation pattern'
                }
        
        return pattern_data
    
    def _calculate_accumulation_distribution(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate accumulation/distribution using fractal-weighted approach"""
        if len(prices) < 2 or len(volumes) < 2:
            return 0.0
        
        # Traditional A/D calculation
        high = max(prices[-2:])
        low = min(prices[-2:])
        close = prices[-1]
        volume = volumes[-1]
        
        if high == low:
            money_flow_multiplier = 0
        else:
            money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
        
        money_flow_volume = money_flow_multiplier * volume
        
        # Weight by volume fractal dimension (lower complexity = higher weight)
        if len(volumes) >= self.fractal_period:
            volume_fd = self._calculate_volume_fractal_dimension(volumes[-self.fractal_period:])
            weight = max(0.3, 2.0 - volume_fd)  # Weight between 0.3 and 1.0
            weighted_money_flow = money_flow_volume * weight
        else:
            weighted_money_flow = money_flow_volume
        
        return weighted_money_flow
    
    def _calculate_price_volume_correlation(self, prices: np.ndarray, volumes: np.ndarray, period: int) -> float:
        """Calculate correlation between price changes and volume"""
        if len(prices) < period + 1 or len(volumes) < period:
            return 0.0
        
        # Calculate price changes
        price_changes = np.diff(prices[-period-1:])
        volume_data = volumes[-period:]
        
        if len(price_changes) != len(volume_data):
            min_len = min(len(price_changes), len(volume_data))
            price_changes = price_changes[-min_len:]
            volume_data = volume_data[-min_len:]
        
        try:
            correlation = np.corrcoef(price_changes, volume_data)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _generate_volume_signals(self, pattern: Dict[str, Any], ad_line: float, correlation: float) -> Dict[str, Any]:
        """Generate trading signals based on volume analysis"""
        signal_data = {
            'signal': 'NONE',
            'strength': 0.0,
            'confidence': 0.0,
            'signal_type': 'none'
        }
        
        pattern_type = pattern['pattern_type']
        pattern_strength = pattern['strength']
        pattern_reliability = pattern['reliability']
        
        # Strong volume signals
        if pattern_type in ['breakout', 'breakdown'] and pattern_reliability > 0.6:
            signal_data = {
                'signal': 'BUY' if pattern_type == 'breakout' else 'SELL',
                'strength': pattern_strength,
                'confidence': pattern_reliability,
                'signal_type': f'volume_{pattern_type}'
            }
        
        # Accumulation/Distribution signals
        elif pattern_type in ['accumulation', 'distribution'] and pattern_reliability > 0.5:
            if abs(ad_line) > 0:  # Significant A/D movement
                signal_data = {
                    'signal': 'BUY' if pattern_type == 'accumulation' else 'SELL',
                    'strength': min(pattern_strength * 0.8, 1.0),  # Slightly lower strength
                    'confidence': pattern_reliability * 0.9,  # Slightly lower confidence
                    'signal_type': f'volume_{pattern_type}'
                }
        
        # Correlation-based signals
        if abs(correlation) > 0.7:  # Strong correlation
            if signal_data['signal'] == 'NONE':
                # Use correlation as confirmation
                signal_data = {
                    'signal': 'BUY' if correlation > 0 else 'SELL',
                    'strength': abs(correlation) * 0.6,
                    'confidence': 0.5,
                    'signal_type': 'volume_correlation'
                }
            else:
                # Enhance existing signal if correlation confirms
                if ((signal_data['signal'] == 'BUY' and correlation > 0) or
                    (signal_data['signal'] == 'SELL' and correlation < 0)):
                    signal_data['confidence'] = min(signal_data['confidence'] * 1.2, 1.0)
        
        return signal_data
    
    def calculate(self, data: Dict[str, Union[float, int]], **kwargs) -> Dict[str, Any]:
        """
        Calculate fractal volume analysis
        
        Args:
            data: Dictionary containing 'close', 'volume'
            
        Returns:
            Dictionary containing volume analysis results
        """
        try:
            # Extract data
            close = float(data.get('close', 0))
            volume = float(data.get('volume', 0))
            
            # Store data
            self.prices.append(close)
            self.volumes.append(volume)
            
            # Keep only required data length
            max_length = max(200, self.trend_period * 2)
            if len(self.prices) > max_length:
                self.prices = self.prices[-max_length:]
                self.volumes = self.volumes[-max_length:]
            
            # Initialize result
            result = {
                'volume_fractal_dimension': 1.5,
                'volume_pattern_type': 'normal',
                'volume_pattern_strength': 0.5,
                'volume_pattern_reliability': 0.5,
                'accumulation_distribution': 0.0,
                'price_volume_correlation': 0.0,
                'volume_signal': 'NONE',
                'volume_signal_strength': 0.0,
                'volume_signal_confidence': 0.0,
                'volume_relative_strength': 1.0,
                'volume_trend': 'neutral'
            }
            
            # Need sufficient data for analysis
            if len(self.volumes) < max(self.volume_period, self.fractal_period):
                return result
            
            volumes_array = np.array(self.volumes)
            prices_array = np.array(self.prices)
            
            # Calculate volume fractal dimension
            volume_fd = self._calculate_volume_fractal_dimension(volumes_array[-self.fractal_period:])
            self.volume_fractal_dimensions.append(volume_fd)
            
            # Keep only recent fractal dimensions
            if len(self.volume_fractal_dimensions) > 100:
                self.volume_fractal_dimensions = self.volume_fractal_dimensions[-100:]
            
            # Calculate volume pattern
            pattern = self._calculate_volume_pattern(volumes_array, prices_array)
            self.volume_patterns.append(pattern)
            
            # Calculate accumulation/distribution
            ad_value = self._calculate_accumulation_distribution(prices_array, volumes_array)
            self.accumulation_distribution.append(ad_value)
            
            # Calculate price-volume correlation
            correlation = self._calculate_price_volume_correlation(prices_array, volumes_array, self.volume_period)
            self.price_volume_correlation.append(correlation)
            
            # Keep only recent values
            for attr in ['volume_patterns', 'accumulation_distribution', 'price_volume_correlation']:
                values = getattr(self, attr)
                if len(values) > 100:
                    setattr(self, attr, values[-100:])
            
            # Calculate volume metrics
            if len(volumes_array) >= self.volume_period:
                self.avg_volume = np.mean(volumes_array[-self.volume_period:])
                self.volume_volatility = np.std(volumes_array[-self.volume_period:])
                
                # Volume trend
                if len(volumes_array) >= self.trend_period:
                    recent_avg = np.mean(volumes_array[-self.volume_period:])
                    older_avg = np.mean(volumes_array[-self.trend_period:-self.volume_period])
                    if recent_avg > older_avg * 1.1:
                        self.volume_trend = 'increasing'
                    elif recent_avg < older_avg * 0.9:
                        self.volume_trend = 'decreasing'
                    else:
                        self.volume_trend = 'stable'
            
            # Generate signals
            signal_data = self._generate_volume_signals(pattern, ad_value, correlation)
            self.volume_strength_signals.append(signal_data)
            
            # Keep only recent signals
            if len(self.volume_strength_signals) > 50:
                self.volume_strength_signals = self.volume_strength_signals[-50:]
            
            # Calculate relative volume strength
            current_volume = volumes_array[-1]
            relative_strength = current_volume / self.avg_volume if self.avg_volume > 0 else 1.0
            
            # Update result
            result.update({
                'volume_fractal_dimension': volume_fd,
                'volume_pattern_type': pattern['pattern_type'],
                'volume_pattern_strength': pattern['strength'],
                'volume_pattern_reliability': pattern['reliability'],
                'accumulation_distribution': ad_value,
                'price_volume_correlation': correlation,
                'volume_signal': signal_data['signal'],
                'volume_signal_strength': signal_data['strength'],
                'volume_signal_confidence': signal_data['confidence'],
                'volume_relative_strength': relative_strength,
                'volume_trend': self.volume_trend
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in FractalVolumeAnalysis calculation: {e}")
            return {
                'volume_fractal_dimension': 1.5,
                'volume_pattern_type': 'normal',
                'volume_pattern_strength': 0.5,
                'volume_pattern_reliability': 0.5,
                'accumulation_distribution': 0.0,
                'price_volume_correlation': 0.0,
                'volume_signal': 'NONE',
                'volume_signal_strength': 0.0,
                'volume_signal_confidence': 0.0,
                'volume_relative_strength': 1.0,
                'volume_trend': 'neutral'
            }
    
    def get_signals(self) -> Dict[str, Any]:
        """Get current volume signals"""
        if not self.volume_strength_signals:
            return {'signal': 'NONE', 'strength': 0.0, 'confidence': 0.0, 'type': 'none'}
        
        latest_signal = self.volume_strength_signals[-1]
        return {
            'signal': latest_signal['signal'],
            'strength': latest_signal['strength'],
            'confidence': latest_signal['confidence'],
            'type': latest_signal['signal_type']
        }
    
    def get_volume_metrics(self) -> Dict[str, float]:
        """Get current volume metrics"""
        return {
            'avg_volume': self.avg_volume,
            'volume_volatility': self.volume_volatility,
            'current_volume_fd': self.volume_fractal_dimensions[-1] if self.volume_fractal_dimensions else 1.5,
            'avg_volume_fd': np.mean(self.volume_fractal_dimensions[-20:]) if len(self.volume_fractal_dimensions) >= 20 else 1.5,
            'avg_correlation': np.mean(self.price_volume_correlation[-20:]) if len(self.price_volume_correlation) >= 20 else 0.0
        }
    
    def get_pattern_analysis(self) -> Dict[str, Any]:
        """Get current pattern analysis"""
        if not self.volume_patterns:
            return {'type': 'normal', 'strength': 0.5, 'reliability': 0.5, 'description': 'Insufficient data'}
        
        latest_pattern = self.volume_patterns[-1]
        return {
            'type': latest_pattern['pattern_type'],
            'strength': latest_pattern['strength'],
            'reliability': latest_pattern['reliability'],
            'description': latest_pattern['description']
        }
    
    def interpret_signal(self, signal_data: Dict[str, Any]) -> str:
        """
        Interpret fractal volume analysis signal for trading decisions
        
        Args:
            signal_data: Signal data from calculate method
            
        Returns:
            Human-readable interpretation
        """
        volume_fd = signal_data.get('volume_fractal_dimension', 1.5)
        pattern_type = signal_data.get('volume_pattern_type', 'normal')
        pattern_strength = signal_data.get('volume_pattern_strength', 0.5)
        pattern_reliability = signal_data.get('volume_pattern_reliability', 0.5)
        signal = signal_data.get('volume_signal', 'NONE')
        correlation = signal_data.get('price_volume_correlation', 0.0)
        relative_strength = signal_data.get('volume_relative_strength', 1.0)
        volume_trend = signal_data.get('volume_trend', 'neutral')
        
        interpretation = f"Volume Fractal Dimension: {volume_fd:.3f} - "
        
        if volume_fd > 1.7:
            interpretation += "Very complex volume patterns (high noise)\n"
        elif volume_fd > 1.5:
            interpretation += "Moderate volume complexity\n"
        elif volume_fd < 1.3:
            interpretation += "Simple volume patterns (strong trends)\n"
        else:
            interpretation += "Balanced volume complexity\n"
        
        interpretation += f"Volume Pattern: {pattern_type.upper()} "
        interpretation += f"(Strength: {pattern_strength:.2f}, Reliability: {pattern_reliability:.2f})\n"
        
        interpretation += f"Volume Trend: {volume_trend.upper()}, "
        interpretation += f"Relative Strength: {relative_strength:.2f}x average\n"
        
        interpretation += f"Price-Volume Correlation: {correlation:.3f} - "
        if abs(correlation) > 0.7:
            interpretation += "Strong correlation (reliable signals)"
        elif abs(correlation) > 0.3:
            interpretation += "Moderate correlation"
        else:
            interpretation += "Weak correlation (mixed signals)"
        
        if signal != 'NONE':
            strength = signal_data.get('volume_signal_strength', 0)
            confidence = signal_data.get('volume_signal_confidence', 0)
            
            strength_desc = "Strong" if strength > 0.7 else "Moderate" if strength > 0.4 else "Weak"
            confidence_desc = "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
            
            interpretation += f"\n\n{signal} SIGNAL: {strength_desc} strength, {confidence_desc} confidence"
            interpretation += f"\nPattern suggests {pattern_type} phase with {'high' if pattern_reliability > 0.6 else 'moderate'} reliability"
        else:
            interpretation += f"\n\nNo clear volume signals. Current pattern: {pattern_type}"
        
        return interpretation
