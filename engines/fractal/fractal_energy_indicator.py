"""
Fractal Energy Indicator
========================

Advanced energy measurement indicator that analyzes market energy levels
using fractal geometry principles. Combines price movement efficiency,
volume dynamics, and fractal dimension to quantify market energy states.

The indicator identifies:
- Energy accumulation and release phases
- Market exhaustion points
- Trend continuation probability
- Optimal entry/exit based on energy levels

Author: Platform3 AI System
Created: December 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Fix import - use absolute import with fallback
try:
    from engines.indicator_base import IndicatorBase
except ImportError:
    import sys
    import os
        from indicator_base import IndicatorBase


class FractalEnergyIndicator(IndicatorBase):
    """
    Fractal Energy Indicator for market energy analysis.
    
    Measures market energy using:
    - Fractal dimension analysis
    - Price efficiency ratios
    - Volume energy metrics
    - Momentum fractals
    """
    
    def __init__(self,
                 period: int = 14,
                 energy_threshold_high: float = 0.618,
                 energy_threshold_low: float = 0.382,
                 fractal_period: int = 5,
                 smooth_factor: int = 3):
        """
        Initialize Fractal Energy Indicator.
        
        Args:
            period: Main calculation period
            energy_threshold_high: High energy threshold (0.618 = golden ratio)
            energy_threshold_low: Low energy threshold
            fractal_period: Period for fractal calculations
            smooth_factor: Smoothing factor for energy values
        """
        super().__init__()
        
        self.period = period
        self.energy_threshold_high = energy_threshold_high
        self.energy_threshold_low = energy_threshold_low
        self.fractal_period = fractal_period
        self.smooth_factor = smooth_factor
        
        # Validation
        if period < 5:
            raise ValueError("Period must be at least 5")
        if not 0 < energy_threshold_low < energy_threshold_high < 1:
            raise ValueError("Invalid energy thresholds")
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Calculate Fractal Energy Indicator values.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing energy analysis
        """
        try:
            # Validate input data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            self._validate_data(data, required_columns)
            
            if len(data) < self.period + self.fractal_period:
                raise ValueError(f"Insufficient data: need at least {self.period + self.fractal_period} periods")
            
            # Extract data arrays
            opens = data['open'].values
            highs = data['high'].values
            lows = data['low'].values
            closes = data['close'].values
            volumes = data['volume'].values
            
            # Calculate fractal energy components
            price_energy = self._calculate_price_energy(opens, highs, lows, closes)
            volume_energy = self._calculate_volume_energy(volumes, closes)
            fractal_energy = self._calculate_fractal_energy(closes)
            momentum_energy = self._calculate_momentum_energy(closes)
            
            # Combine energy components
            total_energy = self._combine_energy_components(
                price_energy, volume_energy, fractal_energy, momentum_energy
            )
            
            # Smooth energy values
            smoothed_energy = self._smooth_energy(total_energy)
            
            # Calculate energy states and zones
            energy_states = self._calculate_energy_states(smoothed_energy)
            energy_zones = self._identify_energy_zones(smoothed_energy)
            
            # Generate trading signals
            signals = self._generate_energy_signals(smoothed_energy, energy_states)
            
            # Calculate additional metrics
            metrics = self._calculate_energy_metrics(smoothed_energy, volumes, closes)
            
            return {
                'fractal_energy': total_energy,
                'smoothed_energy': smoothed_energy,
                'energy_states': energy_states,
                'energy_zones': energy_zones,
                'signals': signals,
                'metrics': metrics,
                'interpretation': self._interpret_energy_state(
                    smoothed_energy[-1] if len(smoothed_energy) > 0 else 0.5,
                    energy_states[-1] if len(energy_states) > 0 else 'neutral',
                    signals[-1] if len(signals) > 0 else 0
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Fractal Energy Indicator: {e}")
            raise
    
    def _calculate_price_energy(self, opens: np.ndarray, highs: np.ndarray, 
                               lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
        """Calculate price-based energy component."""
        energy = np.full(len(closes), np.nan)
        
        for i in range(self.period - 1, len(closes)):
            # Price range energy
            period_highs = highs[i - self.period + 1:i + 1]
            period_lows = lows[i - self.period + 1:i + 1]
            period_closes = closes[i - self.period + 1:i + 1]
            
            # Calculate true range
            true_ranges = []
            for j in range(1, len(period_closes)):
                tr = max(
                    period_highs[j] - period_lows[j],
                    abs(period_highs[j] - period_closes[j-1]),
                    abs(period_lows[j] - period_closes[j-1])
                )
                true_ranges.append(tr)
            
            if true_ranges:
                avg_true_range = np.mean(true_ranges)
                price_movement = abs(closes[i] - closes[i - self.period + 1])
                
                # Energy as ratio of directional movement to volatility
                if avg_true_range > 0:
                    energy[i] = min(1.0, price_movement / (avg_true_range * self.period))
                else:
                    energy[i] = 0.5
            else:
                energy[i] = 0.5
        
        return energy
    
    def _calculate_volume_energy(self, volumes: np.ndarray, closes: np.ndarray) -> np.ndarray:
        """Calculate volume-based energy component."""
        energy = np.full(len(volumes), np.nan)
        
        for i in range(self.period - 1, len(volumes)):
            period_volumes = volumes[i - self.period + 1:i + 1]
            period_closes = closes[i - self.period + 1:i + 1]
            
            # Volume momentum
            avg_volume = np.mean(period_volumes)
            current_volume = volumes[i]
            
            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                
                # Price-volume correlation
                price_changes = np.diff(period_closes)
                volume_changes = np.diff(period_volumes)
                
                if len(price_changes) > 0 and np.std(price_changes) > 0 and np.std(volume_changes) > 0:
                    correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
                    
                    # Energy combines volume ratio and correlation
                    energy[i] = (1 + correlation) / 2 * min(2.0, volume_ratio) / 2
                else:
                    energy[i] = min(1.0, volume_ratio)
            else:
                energy[i] = 0.5
        
        return energy
    
    def _calculate_fractal_energy(self, closes: np.ndarray) -> np.ndarray:
        """Calculate fractal dimension-based energy."""
        energy = np.full(len(closes), np.nan)
        
        for i in range(self.period - 1, len(closes)):
            period_data = closes[i - self.period + 1:i + 1]
            
            # Calculate fractal dimension
            fractal_dim = self._calculate_fractal_dimension(period_data)
            
            # Convert fractal dimension to energy
            # Lower dimension (trending) = higher energy
            # Higher dimension (choppy) = lower energy
            if fractal_dim <= 1.0:
                energy[i] = 1.0
            elif fractal_dim >= 2.0:
                energy[i] = 0.0
            else:
                energy[i] = 2.0 - fractal_dim  # Linear mapping from [1,2] to [1,0]
        
        return energy
    
    def _calculate_momentum_energy(self, closes: np.ndarray) -> np.ndarray:
        """Calculate momentum-based energy."""
        energy = np.full(len(closes), np.nan)
        
        for i in range(self.period - 1, len(closes)):
            period_closes = closes[i - self.period + 1:i + 1]
            
            # Calculate momentum components
            short_momentum = (closes[i] - closes[i - self.period // 2]) / closes[i - self.period // 2]
            long_momentum = (closes[i] - closes[i - self.period + 1]) / closes[i - self.period + 1]
            
            # Rate of change acceleration
            if i >= self.period * 2:
                prev_momentum = (closes[i - self.period] - closes[i - self.period * 2]) / closes[i - self.period * 2]
                acceleration = long_momentum - prev_momentum
            else:
                acceleration = 0
            
            # Combine momentum factors
            momentum_strength = abs(short_momentum) + abs(long_momentum) + abs(acceleration)
            
            # Normalize to [0, 1]
            energy[i] = 1 - np.exp(-momentum_strength * 10)  # Exponential scaling
        
        return energy
    
    def _calculate_fractal_dimension(self, data: np.ndarray) -> float:
        """Calculate fractal dimension using Higuchi method."""
        if len(data) < 4:
            return 1.5
        
        # Higuchi fractal dimension
        k_max = min(8, len(data) // 2)
        lk = []
        
        for k in range(1, k_max + 1):
            l_mk = 0
            for m in range(k):
                ll = 0
                n = (len(data) - m - 1) // k
                if n > 0:
                    for i in range(1, n + 1):
                        ll += abs(data[m + i * k] - data[m + (i - 1) * k])
                    ll = ll * (len(data) - 1) / (n * k)
                    l_mk += ll
            
            if k > 0:
                lk.append(l_mk / k)
        
        # Calculate dimension
        if len(lk) > 1:
            k_values = np.arange(1, len(lk) + 1)
            lk = np.array(lk)
            valid_mask = lk > 0
            
            if np.sum(valid_mask) > 1:
                log_k = np.log(k_values[valid_mask])
                log_lk = np.log(lk[valid_mask])
                
                # Linear regression
                slope = np.polyfit(log_k, log_lk, 1)[0]
                return max(1.0, min(2.0, 2 - slope))
        
        return 1.5
    
    def _combine_energy_components(self, price_energy: np.ndarray, volume_energy: np.ndarray,
                                  fractal_energy: np.ndarray, momentum_energy: np.ndarray) -> np.ndarray:
        """Combine different energy components into total energy."""
        # Weights for different components
        weights = {
            'price': 0.3,
            'volume': 0.2,
            'fractal': 0.3,
            'momentum': 0.2
        }
        
        combined_energy = np.full(len(price_energy), np.nan)
        
        for i in range(len(price_energy)):
            if not any(np.isnan([price_energy[i], volume_energy[i], fractal_energy[i], momentum_energy[i]])):
                combined_energy[i] = (
                    weights['price'] * price_energy[i] +
                    weights['volume'] * volume_energy[i] +
                    weights['fractal'] * fractal_energy[i] +
                    weights['momentum'] * momentum_energy[i]
                )
        
        return combined_energy
    
    def _smooth_energy(self, energy: np.ndarray) -> np.ndarray:
        """Apply smoothing to energy values."""
        smoothed = np.full(len(energy), np.nan)
        
        for i in range(self.smooth_factor - 1, len(energy)):
            window = energy[i - self.smooth_factor + 1:i + 1]
            valid_values = window[~np.isnan(window)]
            
            if len(valid_values) > 0:
                smoothed[i] = np.mean(valid_values)
        
        return smoothed
    
    def _calculate_energy_states(self, energy: np.ndarray) -> np.ndarray:
        """Calculate energy states (high/medium/low)."""
        states = np.full(len(energy), 'neutral', dtype=object)
        
        for i in range(len(energy)):
            if np.isnan(energy[i]):
                continue
            
            if energy[i] >= self.energy_threshold_high:
                states[i] = 'high'
            elif energy[i] <= self.energy_threshold_low:
                states[i] = 'low'
            else:
                states[i] = 'medium'
        
        return states
    
    def _identify_energy_zones(self, energy: np.ndarray) -> np.ndarray:
        """Identify energy zones and transitions."""
        zones = np.full(len(energy), 'neutral', dtype=object)
        
        for i in range(1, len(energy)):
            if np.isnan(energy[i]) or np.isnan(energy[i-1]):
                continue
            
            # Accumulation zone (low energy building up)
            if energy[i] < self.energy_threshold_low and energy[i] > energy[i-1]:
                zones[i] = 'accumulation'
            
            # Distribution zone (high energy dissipating)
            elif energy[i] > self.energy_threshold_high and energy[i] < energy[i-1]:
                zones[i] = 'distribution'
            
            # Breakout zone (energy surge)
            elif energy[i] > self.energy_threshold_high and energy[i] > energy[i-1]:
                zones[i] = 'breakout'
            
            # Exhaustion zone (energy depletion)
            elif energy[i] < self.energy_threshold_low and energy[i] < energy[i-1]:
                zones[i] = 'exhaustion'
            
            else:
                zones[i] = 'neutral'
        
        return zones
    
    def _generate_energy_signals(self, energy: np.ndarray, states: np.ndarray) -> np.ndarray:
        """Generate trading signals based on energy analysis."""
        signals = np.zeros(len(energy))
        
        for i in range(2, len(energy)):
            if np.isnan(energy[i]):
                continue
            
            # Energy crossover signals
            if (energy[i] > self.energy_threshold_high and 
                energy[i-1] <= self.energy_threshold_high):
                signals[i] = 1  # High energy breakout
            
            elif (energy[i] < self.energy_threshold_low and 
                  energy[i-1] >= self.energy_threshold_low):
                signals[i] = -1  # Low energy breakdown
            
            # Energy reversal signals
            elif i >= 3:
                # Energy bottoming out
                if (energy[i] > energy[i-1] and energy[i-1] < energy[i-2] and
                    energy[i-1] < self.energy_threshold_low):
                    signals[i] = 0.5  # Weak buy
                
                # Energy topping out
                elif (energy[i] < energy[i-1] and energy[i-1] > energy[i-2] and
                      energy[i-1] > self.energy_threshold_high):
                    signals[i] = -0.5  # Weak sell
        
        return signals
    
    def _calculate_energy_metrics(self, energy: np.ndarray, volumes: np.ndarray, 
                                 closes: np.ndarray) -> Dict:
        """Calculate additional energy metrics."""
        valid_energy = energy[~np.isnan(energy)]
        
        if len(valid_energy) == 0:
            return {
                'avg_energy': 0.5,
                'energy_volatility': 0.0,
                'energy_trend': 0.0,
                'energy_efficiency': 0.0
            }
        
        metrics = {
            'avg_energy': np.mean(valid_energy),
            'energy_volatility': np.std(valid_energy),
            'current_energy': valid_energy[-1] if len(valid_energy) > 0 else 0.5
        }
        
        # Energy trend
        if len(valid_energy) >= 10:
            recent_energy = valid_energy[-10:]
            x = np.arange(len(recent_energy))
            slope = np.polyfit(x, recent_energy, 1)[0]
            metrics['energy_trend'] = slope
        else:
            metrics['energy_trend'] = 0.0
        
        # Energy efficiency (how well energy translates to price movement)
        if len(valid_energy) >= self.period:
            energy_sum = np.sum(valid_energy[-self.period:])
            price_change = abs(closes[-1] - closes[-self.period]) / closes[-self.period]
            
            if energy_sum > 0:
                metrics['energy_efficiency'] = price_change / energy_sum
            else:
                metrics['energy_efficiency'] = 0.0
        else:
            metrics['energy_efficiency'] = 0.0
        
        # High/Low energy periods
        metrics['high_energy_pct'] = np.sum(valid_energy >= self.energy_threshold_high) / len(valid_energy)
        metrics['low_energy_pct'] = np.sum(valid_energy <= self.energy_threshold_low) / len(valid_energy)
        
        return metrics
    
    def _interpret_energy_state(self, current_energy: float, current_state: str, 
                               current_signal: float) -> Dict:
        """Provide interpretation of current energy state."""
        interpretation = {
            'energy_level': current_energy,
            'energy_state': current_state,
            'market_condition': '',
            'trading_bias': '',
            'risk_level': '',
            'recommendations': []
        }
        
        # Determine market condition
        if current_state == 'high':
            interpretation['market_condition'] = 'High energy - Strong trending conditions'
            interpretation['risk_level'] = 'Medium-High'
            
            if current_signal > 0:
                interpretation['trading_bias'] = 'Bullish'
                interpretation['recommendations'].append('Follow trend with tight stops')
            elif current_signal < 0:
                interpretation['trading_bias'] = 'Bearish'
                interpretation['recommendations'].append('Consider profit taking')
            else:
                interpretation['trading_bias'] = 'Neutral'
                interpretation['recommendations'].append('Monitor for energy exhaustion')
        
        elif current_state == 'low':
            interpretation['market_condition'] = 'Low energy - Consolidation or accumulation'
            interpretation['risk_level'] = 'Low-Medium'
            
            if current_signal > 0:
                interpretation['trading_bias'] = 'Accumulation phase'
                interpretation['recommendations'].append('Build positions gradually')
            else:
                interpretation['trading_bias'] = 'Wait for energy buildup'
                interpretation['recommendations'].append('Avoid aggressive trades')
        
        else:  # medium
            interpretation['market_condition'] = 'Medium energy - Transitional phase'
            interpretation['risk_level'] = 'Medium'
            interpretation['trading_bias'] = 'Neutral'
            interpretation['recommendations'].append('Wait for clear energy direction')
        
        # Add energy-specific insights
        interpretation['recommendations'].append(
            f"Current energy: {current_energy:.3f} ({current_state})"
        )
        
        return interpretation


def create_fractal_energy_indicator(period: int = 14, **kwargs) -> FractalEnergyIndicator:
    """Factory function to create Fractal Energy Indicator."""
    return FractalEnergyIndicator(period=period, **kwargs)
