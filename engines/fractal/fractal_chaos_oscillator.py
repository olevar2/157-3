"""
Fractal Chaos Oscillator
========================

Advanced oscillator that measures market chaos levels using fractal geometry
and chaos theory principles. Oscillates between order and chaos states to
identify optimal trading opportunities.

The oscillator detects:
- Chaos/order transitions
- Market regime changes
- Volatility expansions/contractions
- Trend emergence from chaos

Author: Platform3 AI System
Created: December 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# Fix import - use absolute import with fallback
try:
    from engines.indicator_base import IndicatorBase
except ImportError:
    import sys
    import os
        from indicator_base import IndicatorBase


class FractalChaosOscillator(IndicatorBase):
    """
    Fractal Chaos Oscillator for chaos-based market analysis.
    
    Measures market chaos using:
    - Lyapunov exponents
    - Phase space analysis
    - Fractal dimension variations
    - Entropy measurements
    """
    
    def __init__(self,
                 chaos_period: int = 21,
                 oscillator_period: int = 14,
                 overbought_level: float = 70,
                 oversold_level: float = 30,
                 smoothing: int = 3):
        """
        Initialize Fractal Chaos Oscillator.
        
        Args:
            chaos_period: Period for chaos calculations
            oscillator_period: Period for oscillator calculations
            overbought_level: Overbought threshold (0-100)
            oversold_level: Oversold threshold (0-100)
            smoothing: Smoothing period for oscillator
        """
        super().__init__()
        
        self.chaos_period = chaos_period
        self.oscillator_period = oscillator_period
        self.overbought_level = overbought_level
        self.oversold_level = oversold_level
        self.smoothing = smoothing
        
        # Chaos theory parameters
        self.embedding_dimension = 3
        self.time_delay = 1
        
        # Validation
        if chaos_period < 10:
            raise ValueError("Chaos period must be at least 10")
        if not 0 <= oversold_level < overbought_level <= 100:
            raise ValueError("Invalid overbought/oversold levels")
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Calculate Fractal Chaos Oscillator values.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing chaos oscillator analysis
        """
        try:
            # Validate input data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            self._validate_data(data, required_columns)
            
            if len(data) < self.chaos_period:
                raise ValueError(f"Insufficient data: need at least {self.chaos_period} periods")
            
            # Extract data
            closes = data['close'].values
            highs = data['high'].values
            lows = data['low'].values
            volumes = data['volume'].values
            
            # Calculate chaos components
            lyapunov_values = self._calculate_lyapunov_exponents(closes)
            fractal_dims = self._calculate_fractal_dimensions(closes)
            entropy_values = self._calculate_entropy_series(closes)
            phase_metrics = self._calculate_phase_space_metrics(closes)
            
            # Combine into chaos index
            chaos_index = self._calculate_chaos_index(
                lyapunov_values, fractal_dims, entropy_values, phase_metrics
            )
            
            # Convert to oscillator
            oscillator_values = self._convert_to_oscillator(chaos_index)
            
            # Smooth oscillator
            smoothed_oscillator = self._smooth_oscillator(oscillator_values)
            
            # Calculate oscillator components
            momentum = self._calculate_chaos_momentum(smoothed_oscillator)
            divergence = self._calculate_chaos_divergence(smoothed_oscillator, closes)
            
            # Generate signals
            signals = self._generate_chaos_signals(
                smoothed_oscillator, momentum, divergence
            )
            
            # Calculate zones
            zones = self._identify_chaos_zones(smoothed_oscillator)
            
            # Additional metrics
            metrics = self._calculate_chaos_metrics(
                smoothed_oscillator, chaos_index, closes
            )
            
            return {
                'chaos_oscillator': oscillator_values,
                'smoothed_oscillator': smoothed_oscillator,
                'chaos_momentum': momentum,
                'chaos_divergence': divergence,
                'chaos_zones': zones,
                'signals': signals,
                'metrics': metrics,
                'interpretation': self._interpret_chaos_state(
                    smoothed_oscillator[-1] if len(smoothed_oscillator) > 0 else 50,
                    zones[-1] if len(zones) > 0 else 'neutral',
                    signals[-1] if len(signals) > 0 else 0
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Fractal Chaos Oscillator: {e}")
            raise
    
    def _calculate_lyapunov_exponents(self, closes: np.ndarray) -> np.ndarray:
        """Calculate Lyapunov exponents for chaos detection."""
        lyapunov = np.full(len(closes), np.nan)
        
        for i in range(self.chaos_period - 1, len(closes)):
            period_data = closes[i - self.chaos_period + 1:i + 1]
            
            # Phase space reconstruction
            embedded = self._phase_space_reconstruction(period_data)
            
            if len(embedded) < 5:
                continue
            
            # Calculate largest Lyapunov exponent
            lyapunov[i] = self._calculate_largest_lyapunov(embedded)
        
        return lyapunov
    
    def _calculate_fractal_dimensions(self, closes: np.ndarray) -> np.ndarray:
        """Calculate rolling fractal dimensions."""
        fractal_dims = np.full(len(closes), np.nan)
        
        for i in range(self.oscillator_period - 1, len(closes)):
            period_data = closes[i - self.oscillator_period + 1:i + 1]
            fractal_dims[i] = self._higuchi_dimension(period_data)
        
        return fractal_dims
    
    def _calculate_entropy_series(self, closes: np.ndarray) -> np.ndarray:
        """Calculate rolling entropy values."""
        entropy = np.full(len(closes), np.nan)
        
        for i in range(self.oscillator_period - 1, len(closes)):
            period_data = closes[i - self.oscillator_period + 1:i + 1]
            
            # Calculate returns
            returns = np.diff(np.log(period_data))
            
            if len(returns) > 0:
                entropy[i] = self._calculate_shannon_entropy(returns)
        
        return entropy
    
    def _calculate_phase_space_metrics(self, closes: np.ndarray) -> np.ndarray:
        """Calculate phase space complexity metrics."""
        metrics = np.full(len(closes), np.nan)
        
        for i in range(self.chaos_period - 1, len(closes)):
            period_data = closes[i - self.chaos_period + 1:i + 1]
            
            # Phase space reconstruction
            embedded = self._phase_space_reconstruction(period_data)
            
            if len(embedded) >= 3:
                # Calculate phase space volume
                volume = self._calculate_phase_space_volume(embedded)
                metrics[i] = volume
        
        return metrics
    
    def _phase_space_reconstruction(self, data: np.ndarray) -> np.ndarray:
        """Reconstruct phase space using time delay embedding."""
        if len(data) < self.embedding_dimension * self.time_delay:
            return np.array([])
        
        embedded = []
        for i in range(len(data) - (self.embedding_dimension - 1) * self.time_delay):
            vector = []
            for j in range(self.embedding_dimension):
                vector.append(data[i + j * self.time_delay])
            embedded.append(vector)
        
        return np.array(embedded)
    
    def _calculate_largest_lyapunov(self, embedded: np.ndarray) -> float:
        """Calculate largest Lyapunov exponent."""
        if len(embedded) < 2:
            return 0.0
        
        # Simplified Wolf algorithm
        lyapunov_sum = 0.0
        count = 0
        
        for i in range(len(embedded) - 1):
            # Find nearest neighbor
            distances = np.array([np.linalg.norm(embedded[j] - embedded[i]) 
                                for j in range(len(embedded)) if j != i])
            
            if len(distances) == 0:
                continue
            
            min_idx = np.argmin(distances)
            min_distance = distances[min_idx]
            
            if min_distance > 0 and i + 1 < len(embedded) and min_idx + 1 < len(embedded):
                # Evolution after one step
                evolved_distance = np.linalg.norm(embedded[i + 1] - embedded[min_idx + 1])
                
                if evolved_distance > 0:
                    lyapunov_sum += np.log(evolved_distance / min_distance)
                    count += 1
        
        return lyapunov_sum / count if count > 0 else 0.0
    
    def _higuchi_dimension(self, data: np.ndarray) -> float:
        """Calculate Higuchi fractal dimension."""
        if len(data) < 4:
            return 1.5
        
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
        
        if len(lk) > 1:
            k_values = np.arange(1, len(lk) + 1)
            lk_array = np.array(lk)
            valid_mask = lk_array > 0
            
            if np.sum(valid_mask) > 1:
                log_k = np.log(k_values[valid_mask])
                log_lk = np.log(lk_array[valid_mask])
                slope = np.polyfit(log_k, log_lk, 1)[0]
                return max(1.0, min(2.0, 2 - slope))
        
        return 1.5
    
    def _calculate_shannon_entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy."""
        if len(data) == 0:
            return 0.0
        
        # Discretize data
        bins = 10
        hist, _ = np.histogram(data, bins=bins)
        hist = hist / np.sum(hist)  # Normalize
        
        # Remove zero bins
        hist = hist[hist > 0]
        
        if len(hist) == 0:
            return 0.0
        
        # Calculate entropy
        return -np.sum(hist * np.log2(hist))
    
    def _calculate_phase_space_volume(self, embedded: np.ndarray) -> float:
        """Calculate phase space volume."""
        if len(embedded) < 2:
            return 0.0
        
        # Calculate bounding box volume
        min_coords = np.min(embedded, axis=0)
        max_coords = np.max(embedded, axis=0)
        ranges = max_coords - min_coords
        
        # Avoid zero volume
        ranges = np.maximum(ranges, 1e-10)
        
        # Return log volume to handle large values
        return np.log(np.prod(ranges))
    
    def _calculate_chaos_index(self, lyapunov: np.ndarray, fractal: np.ndarray,
                              entropy: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """Combine chaos metrics into single index."""
        chaos_index = np.full(len(lyapunov), np.nan)
        
        for i in range(len(lyapunov)):
            components = []
            
            # Normalize Lyapunov (positive = chaotic)
            if not np.isnan(lyapunov[i]):
                lyap_normalized = (1 + np.tanh(lyapunov[i])) / 2
                components.append(lyap_normalized)
            
            # Normalize fractal dimension (higher = more chaotic)
            if not np.isnan(fractal[i]):
                fractal_normalized = (fractal[i] - 1.0) / 1.0  # Map [1,2] to [0,1]
                components.append(fractal_normalized)
            
            # Normalize entropy (higher = more chaotic)
            if not np.isnan(entropy[i]):
                entropy_normalized = min(1.0, entropy[i] / 3.0)  # Assume max entropy ~3
                components.append(entropy_normalized)
            
            # Normalize phase space volume
            if not np.isnan(phase[i]):
                phase_normalized = (1 + np.tanh(phase[i] / 10)) / 2
                components.append(phase_normalized)
            
            if components:
                chaos_index[i] = np.mean(components)
        
        return chaos_index
    
    def _convert_to_oscillator(self, chaos_index: np.ndarray) -> np.ndarray:
        """Convert chaos index to oscillator scale (0-100)."""
        oscillator = np.full(len(chaos_index), np.nan)
        
        for i in range(self.oscillator_period - 1, len(chaos_index)):
            if np.isnan(chaos_index[i]):
                continue
            
            # Get period values
            period_values = chaos_index[i - self.oscillator_period + 1:i + 1]
            valid_values = period_values[~np.isnan(period_values)]
            
            if len(valid_values) >= 3:
                # Normalize current value relative to period range
                min_val = np.min(valid_values)
                max_val = np.max(valid_values)
                
                if max_val > min_val:
                    normalized = (chaos_index[i] - min_val) / (max_val - min_val)
                    oscillator[i] = normalized * 100
                else:
                    oscillator[i] = 50
        
        return oscillator
    
    def _smooth_oscillator(self, oscillator: np.ndarray) -> np.ndarray:
        """Apply smoothing to oscillator values."""
        smoothed = np.full(len(oscillator), np.nan)
        
        for i in range(self.smoothing - 1, len(oscillator)):
            window = oscillator[i - self.smoothing + 1:i + 1]
            valid_values = window[~np.isnan(window)]
            
            if len(valid_values) > 0:
                smoothed[i] = np.mean(valid_values)
        
        return smoothed
    
    def _calculate_chaos_momentum(self, oscillator: np.ndarray) -> np.ndarray:
        """Calculate momentum of chaos oscillator."""
        momentum = np.full(len(oscillator), np.nan)
        
        lookback = 5
        for i in range(lookback, len(oscillator)):
            if not np.isnan(oscillator[i]) and not np.isnan(oscillator[i - lookback]):
                momentum[i] = oscillator[i] - oscillator[i - lookback]
        
        return momentum
    
    def _calculate_chaos_divergence(self, oscillator: np.ndarray, closes: np.ndarray) -> np.ndarray:
        """Calculate divergence between oscillator and price."""
        divergence = np.zeros(len(oscillator))
        
        lookback = 10
        for i in range(lookback, len(oscillator)):
            if np.isnan(oscillator[i]) or np.isnan(oscillator[i - lookback]):
                continue
            
            # Price trend
            price_change = (closes[i] - closes[i - lookback]) / closes[i - lookback]
            
            # Oscillator trend
            osc_change = (oscillator[i] - oscillator[i - lookback]) / 50  # Normalize
            
            # Divergence
            if price_change > 0 and osc_change < 0:
                divergence[i] = -1  # Bearish divergence
            elif price_change < 0 and osc_change > 0:
                divergence[i] = 1   # Bullish divergence
        
        return divergence
    
    def _generate_chaos_signals(self, oscillator: np.ndarray, momentum: np.ndarray,
                               divergence: np.ndarray) -> np.ndarray:
        """Generate trading signals from chaos oscillator."""
        signals = np.zeros(len(oscillator))
        
        for i in range(1, len(oscillator)):
            if np.isnan(oscillator[i]):
                continue
            
            # Overbought/oversold signals
            if oscillator[i] > self.overbought_level and oscillator[i-1] <= self.overbought_level:
                signals[i] = -1  # Sell signal
            elif oscillator[i] < self.oversold_level and oscillator[i-1] >= self.oversold_level:
                signals[i] = 1   # Buy signal
            
            # Momentum signals
            if not np.isnan(momentum[i]):
                if momentum[i] > 10 and oscillator[i] > 50:
                    signals[i] = max(signals[i], 0.5)  # Bullish momentum
                elif momentum[i] < -10 and oscillator[i] < 50:
                    signals[i] = min(signals[i], -0.5)  # Bearish momentum
            
            # Divergence signals
            if divergence[i] != 0:
                signals[i] = divergence[i] * 0.75  # Weight divergence signals
        
        return signals
    
    def _identify_chaos_zones(self, oscillator: np.ndarray) -> np.ndarray:
        """Identify chaos zones in the market."""
        zones = np.full(len(oscillator), 'neutral', dtype=object)
        
        for i in range(len(oscillator)):
            if np.isnan(oscillator[i]):
                continue
            
            if oscillator[i] >= 80:
                zones[i] = 'extreme_chaos'
            elif oscillator[i] >= self.overbought_level:
                zones[i] = 'high_chaos'
            elif oscillator[i] >= 50:
                zones[i] = 'moderate_chaos'
            elif oscillator[i] >= self.oversold_level:
                zones[i] = 'low_chaos'
            else:
                zones[i] = 'ordered'
        
        return zones
    
    def _calculate_chaos_metrics(self, oscillator: np.ndarray, chaos_index: np.ndarray,
                                closes: np.ndarray) -> Dict:
        """Calculate chaos oscillator metrics."""
        valid_osc = oscillator[~np.isnan(oscillator)]
        valid_chaos = chaos_index[~np.isnan(chaos_index)]
        
        if len(valid_osc) == 0:
            return {
                'avg_chaos_level': 50,
                'chaos_volatility': 0,
                'time_in_chaos': 0,
                'time_in_order': 0
            }
        
        metrics = {
            'avg_chaos_level': np.mean(valid_osc),
            'chaos_volatility': np.std(valid_osc),
            'current_chaos': valid_osc[-1] if len(valid_osc) > 0 else 50
        }
        
        # Time spent in different zones
        metrics['time_in_chaos'] = np.sum(valid_osc > self.overbought_level) / len(valid_osc)
        metrics['time_in_order'] = np.sum(valid_osc < self.oversold_level) / len(valid_osc)
        
        # Chaos trend
        if len(valid_osc) >= 10:
            recent = valid_osc[-10:]
            x = np.arange(len(recent))
            metrics['chaos_trend'] = np.polyfit(x, recent, 1)[0]
        else:
            metrics['chaos_trend'] = 0
        
        # Chaos efficiency (how chaos translates to volatility)
        if len(valid_chaos) >= 20:
            returns = np.diff(np.log(closes[-len(valid_chaos):]))
            volatility = np.std(returns)
            avg_chaos = np.mean(valid_chaos)
            
            if avg_chaos > 0:
                metrics['chaos_efficiency'] = volatility / avg_chaos
            else:
                metrics['chaos_efficiency'] = 0
        else:
            metrics['chaos_efficiency'] = 0
        
        return metrics
    
    def _interpret_chaos_state(self, current_chaos: float, current_zone: str,
                              current_signal: float) -> Dict:
        """Interpret current chaos state."""
        interpretation = {
            'chaos_level': current_chaos,
            'chaos_zone': current_zone,
            'market_state': '',
            'trading_implications': '',
            'risk_assessment': '',
            'recommendations': []
        }
        
        # Determine market state
        if current_zone == 'extreme_chaos':
            interpretation['market_state'] = 'Extreme chaos - Highly unpredictable'
            interpretation['risk_assessment'] = 'Very High Risk'
            interpretation['trading_implications'] = 'Avoid new positions'
            interpretation['recommendations'].append('Wait for chaos to subside')
            interpretation['recommendations'].append('Tighten stops on existing positions')
        
        elif current_zone == 'high_chaos':
            interpretation['market_state'] = 'High chaos - Volatile conditions'
            interpretation['risk_assessment'] = 'High Risk'
            interpretation['trading_implications'] = 'Trade with caution'
            interpretation['recommendations'].append('Reduce position sizes')
            interpretation['recommendations'].append('Use wider stops')
        
        elif current_zone == 'moderate_chaos':
            interpretation['market_state'] = 'Moderate chaos - Normal volatility'
            interpretation['risk_assessment'] = 'Medium Risk'
            interpretation['trading_implications'] = 'Standard trading conditions'
            interpretation['recommendations'].append('Follow normal trading rules')
        
        elif current_zone == 'low_chaos':
            interpretation['market_state'] = 'Low chaos - Calm conditions'
            interpretation['risk_assessment'] = 'Low Risk'
            interpretation['trading_implications'] = 'Potential for breakout'
            interpretation['recommendations'].append('Watch for volatility expansion')
            interpretation['recommendations'].append('Consider breakout strategies')
        
        else:  # ordered
            interpretation['market_state'] = 'Ordered state - Very predictable'
            interpretation['risk_assessment'] = 'Very Low Risk'
            interpretation['trading_implications'] = 'Trend following favorable'
            interpretation['recommendations'].append('Increase position sizes')
            interpretation['recommendations'].append('Use tighter stops')
        
        # Add signal-based recommendations
        if current_signal > 0:
            interpretation['recommendations'].append(f"Buy signal (strength: {abs(current_signal):.2f})")
        elif current_signal < 0:
            interpretation['recommendations'].append(f"Sell signal (strength: {abs(current_signal):.2f})")
        
        return interpretation


def create_fractal_chaos_oscillator(chaos_period: int = 21, **kwargs) -> FractalChaosOscillator:
    """Factory function to create Fractal Chaos Oscillator."""
    return FractalChaosOscillator(chaos_period=chaos_period, **kwargs)
