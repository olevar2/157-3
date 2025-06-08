"""
Fractal Correlation Dimension Indicator
=======================================

Advanced fractal analysis indicator that calculates the correlation dimension
of price time series to measure the complexity and fractal nature of market behavior.
Uses sophisticated Grassberger-Procaccia algorithm for precise dimension estimation.

The correlation dimension reveals:
- Market complexity and chaos levels
- Fractal structure in price movements  
- Regime changes and phase transitions
- Optimal embedding dimensions for prediction

Author: Platform3 AI System
Created: June 3, 2025
Purpose: Help genius agents make profitable trades for charitable causes
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, Tuple, List
from datetime import datetime
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Fix import - use absolute import with fallback
try:
    from engines.indicator_base import IndicatorBase
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicator_base import IndicatorBase


class FractalCorrelationDimension(IndicatorBase):
    """
    Advanced Fractal Correlation Dimension Indicator.
    
    Implements the Grassberger-Procaccia algorithm to calculate the correlation
    dimension D2, which quantifies the fractal complexity of price time series.
    
    Mathematical Foundation:
    C(r) = lim(N→∞) (1/N²) Σᵢ Σⱼ Θ(r - |xᵢ - xⱼ|)
    D₂ = lim(r→0) log(C(r)) / log(r)
    """
    
    def __init__(self, 
                 window_size: int = 500,
                 embedding_dimension: int = 5,
                 time_delay: int = 1,
                 min_radius_ratio: float = 0.001,
                 max_radius_ratio: float = 0.1,
                 num_radius_points: int = 50,
                 min_points_threshold: int = 10):
        """
        Initialize Fractal Correlation Dimension indicator.
        
        Args:
            window_size: Size of the sliding window for analysis
            embedding_dimension: Dimension for phase space reconstruction
            time_delay: Time delay for embedding (tau)
            min_radius_ratio: Minimum radius as ratio of data range
            max_radius_ratio: Maximum radius as ratio of data range  
            num_radius_points: Number of radius values to test
            min_points_threshold: Minimum points needed for valid calculation
        """
        super().__init__()
        
        self.window_size = window_size
        self.embedding_dimension = embedding_dimension
        self.time_delay = time_delay
        self.min_radius_ratio = min_radius_ratio
        self.max_radius_ratio = max_radius_ratio
        self.num_radius_points = num_radius_points
        self.min_points_threshold = min_points_threshold
        
        # Validation
        if window_size < 100:
            raise ValueError("Window size must be at least 100 for reliable fractal analysis")
        if embedding_dimension < 2 or embedding_dimension > 10:
            raise ValueError("Embedding dimension must be between 2 and 10")
        if time_delay < 1:
            raise ValueError("Time delay must be positive")
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Calculate Fractal Correlation Dimension with advanced mathematical precision.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Dictionary containing correlation dimension and analysis
        """
        try:
            # Validate input data
            required_columns = ['close']
            self._validate_data(data, required_columns)
            
            if len(data) < self.window_size:
                raise ValueError(f"Insufficient data: need at least {self.window_size} periods")
            
            prices = data['close'].values
            
            # Calculate correlation dimensions using sliding window
            correlation_dims = self._calculate_correlation_dimensions(prices)
            
            # Advanced fractal analysis
            fractal_metrics = self._calculate_fractal_metrics(correlation_dims, prices)
            
            # Generate sophisticated trading signals
            signals = self._generate_advanced_signals(correlation_dims, fractal_metrics)
            
            return {
                'correlation_dimension': correlation_dims,
                'fractal_metrics': fractal_metrics,
                'signals': signals,
                'interpretation': self._interpret_fractal_state(
                    correlation_dims[-1] if len(correlation_dims) > 0 else np.nan,
                    fractal_metrics,
                    signals[-1] if len(signals) > 0 else 0
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Fractal Correlation Dimension: {e}")
            raise
    
    def _calculate_correlation_dimensions(self, prices: np.ndarray) -> np.ndarray:
        """Calculate correlation dimensions using advanced Grassberger-Procaccia algorithm."""
        correlation_dims = np.full(len(prices), np.nan)
        
        for i in range(self.window_size - 1, len(prices)):
            window_data = prices[i - self.window_size + 1:i + 1]
            
            # Phase space reconstruction with optimal embedding
            embedded_data = self._reconstruct_phase_space(window_data)
            
            if embedded_data.shape[0] < self.min_points_threshold:
                continue
            
            # Calculate correlation dimension using Grassberger-Procaccia
            correlation_dim = self._grassberger_procaccia_algorithm(embedded_data)
            correlation_dims[i] = correlation_dim
        
        return correlation_dims
    
    def _reconstruct_phase_space(self, data: np.ndarray) -> np.ndarray:
        """
        Reconstruct phase space using time delay embedding.
        Creates m-dimensional vectors from 1D time series.
        """
        n = len(data)
        m = self.embedding_dimension
        tau = self.time_delay
        
        # Calculate number of embedded vectors
        num_vectors = n - (m - 1) * tau
        
        if num_vectors <= 0:
            raise ValueError("Insufficient data for phase space reconstruction")
        
        # Create embedded matrix
        embedded = np.zeros((num_vectors, m))
        
        for i in range(m):
            start_idx = i * tau
            end_idx = start_idx + num_vectors
            embedded[:, i] = data[start_idx:end_idx]
        
        return embedded
    
    def _grassberger_procaccia_algorithm(self, embedded_data: np.ndarray) -> float:
        """
        Implement Grassberger-Procaccia algorithm for correlation dimension.
        
        Args:
            embedded_data: Phase space reconstructed data
            
        Returns:
            Correlation dimension D2
        """
        try:
            # Calculate pairwise distances
            distances = pdist(embedded_data, metric='euclidean')
            
            if len(distances) == 0:
                return np.nan
            
            # Determine radius range based on distance distribution
            min_dist = np.min(distances[distances > 0])
            max_dist = np.max(distances)
            
            if min_dist >= max_dist:
                return np.nan
            
            # Create logarithmic radius scale
            radius_min = max(min_dist, self.min_radius_ratio * max_dist)
            radius_max = min(max_dist, self.max_radius_ratio * max_dist)
            
            radii = np.logspace(
                np.log10(radius_min),
                np.log10(radius_max),
                self.num_radius_points
            )
            
            # Calculate correlation sums
            correlation_sums = []
            valid_radii = []
            
            for r in radii:
                # Count pairs within radius r
                pairs_within_r = np.sum(distances <= r)
                total_pairs = len(distances)
                
                if total_pairs > 0:
                    correlation_sum = pairs_within_r / total_pairs
                    if correlation_sum > 0:
                        correlation_sums.append(correlation_sum)
                        valid_radii.append(r)
            
            if len(correlation_sums) < 5:
                return np.nan
            
            # Fit linear regression in log-log space
            log_radii = np.log10(valid_radii)
            log_corr_sums = np.log10(correlation_sums)
            
            # Use robust linear fitting
            correlation_dimension = self._fit_correlation_dimension(log_radii, log_corr_sums)
            
            # Validate result
            if correlation_dimension < 0 or correlation_dimension > self.embedding_dimension:
                return np.nan
            
            return correlation_dimension
            
        except Exception as e:
            self.logger.warning(f"Error in Grassberger-Procaccia algorithm: {e}")
            return np.nan
    
    def _fit_correlation_dimension(self, log_radii: np.ndarray, log_corr_sums: np.ndarray) -> float:
        """
        Fit correlation dimension using robust linear regression in log-log space.
        """
        try:
            # Find linear scaling region (middle portion)
            n_points = len(log_radii)
            start_idx = max(1, n_points // 4)
            end_idx = min(n_points - 1, 3 * n_points // 4)
            
            if end_idx <= start_idx:
                # Fallback to full range
                x = log_radii
                y = log_corr_sums
            else:
                x = log_radii[start_idx:end_idx]
                y = log_corr_sums[start_idx:end_idx]
            
            if len(x) < 3:
                return np.nan
            
            # Linear regression: log(C(r)) = D2 * log(r) + constant
            coeffs = np.polyfit(x, y, 1)
            correlation_dimension = coeffs[0]  # Slope is the correlation dimension
            
            return correlation_dimension
            
        except Exception:
            return np.nan
    
    def _calculate_fractal_metrics(self, correlation_dims: np.ndarray, prices: np.ndarray) -> Dict:
        """
        Calculate advanced fractal metrics and complexity measures.
        """
        valid_dims = correlation_dims[~np.isnan(correlation_dims)]
        
        if len(valid_dims) == 0:
            return {
                'mean_dimension': np.nan,
                'dimension_volatility': np.nan,
                'fractal_efficiency': np.nan,
                'complexity_index': np.nan,
                'dimension_trend': np.nan,
                'regime_stability': np.nan
            }
        
        # Basic statistics
        mean_dim = np.mean(valid_dims)
        dim_volatility = np.std(valid_dims)
        
        # Fractal efficiency (how close to random walk)
        random_walk_dim = 1.5  # Theoretical dimension for random walk
        fractal_efficiency = 1.0 - abs(mean_dim - random_walk_dim) / random_walk_dim
        
        # Complexity index (normalized dimension variance)
        complexity_index = dim_volatility / (mean_dim + 1e-6)
        
        # Dimension trend analysis
        if len(valid_dims) >= 10:
            recent_dims = valid_dims[-10:]
            earlier_dims = valid_dims[-20:-10] if len(valid_dims) >= 20 else valid_dims[:-10]
            
            if len(earlier_dims) > 0:
                dimension_trend = np.mean(recent_dims) - np.mean(earlier_dims)
            else:
                dimension_trend = 0.0
        else:
            dimension_trend = 0.0
        
        # Regime stability (consistency of dimension)
        if len(valid_dims) >= 5:
            regime_stability = 1.0 / (1.0 + dim_volatility)
        else:
            regime_stability = 0.5
        
        return {
            'mean_dimension': mean_dim,
            'dimension_volatility': dim_volatility,
            'fractal_efficiency': fractal_efficiency,
            'complexity_index': complexity_index,
            'dimension_trend': dimension_trend,
            'regime_stability': regime_stability
        }
    
    def _generate_advanced_signals(self, correlation_dims: np.ndarray, fractal_metrics: Dict) -> np.ndarray:
        """
        Generate sophisticated trading signals based on fractal analysis.
        """
        signals = np.zeros(len(correlation_dims))
        
        if len(correlation_dims) == 0:
            return signals
        
        # Extract metrics
        mean_dim = fractal_metrics.get('mean_dimension', np.nan)
        complexity_index = fractal_metrics.get('complexity_index', 0)
        dimension_trend = fractal_metrics.get('dimension_trend', 0)
        regime_stability = fractal_metrics.get('regime_stability', 0.5)
        
        for i in range(len(correlation_dims)):
            if np.isnan(correlation_dims[i]):
                continue
            
            current_dim = correlation_dims[i]
            signal_strength = 0.0
            
            # Signal 1: Dimension level analysis
            if current_dim < 1.2:
                # Low dimension = trending market
                signal_strength += 0.3
            elif current_dim > 1.8:
                # High dimension = chaotic/random market
                signal_strength -= 0.3
            
            # Signal 2: Dimension trend analysis
            if dimension_trend > 0.1:
                # Increasing complexity = potential reversal
                signal_strength -= 0.2
            elif dimension_trend < -0.1:
                # Decreasing complexity = trend continuation
                signal_strength += 0.2
            
            # Signal 3: Regime stability
            if regime_stability > 0.7:
                # Stable regime = reliable signals
                signal_strength *= 1.2
            elif regime_stability < 0.3:
                # Unstable regime = reduce signal confidence
                signal_strength *= 0.5
            
            # Signal 4: Complexity-based adjustments
            if complexity_index > 0.5:
                # High complexity = reduce position size
                signal_strength *= 0.7
            
            # Normalize signal strength
            signals[i] = np.clip(signal_strength, -1.0, 1.0)
        
        return signals
    
    def _interpret_fractal_state(self, current_dim: float, fractal_metrics: Dict, current_signal: float) -> Dict:
        """
        Provide comprehensive interpretation of current fractal state.
        """
        if np.isnan(current_dim):
            return {
                'market_regime': 'Unknown',
                'complexity_level': 'Unknown',
                'trading_recommendation': 'Wait for valid data',
                'confidence': 0.0,
                'key_insights': ['Insufficient data for fractal analysis']
            }
        
        # Determine market regime
        if current_dim < 1.2:
            market_regime = 'Trending'
            regime_desc = 'Strong directional movement with low noise'
        elif current_dim < 1.5:
            market_regime = 'Weakly Trending'
            regime_desc = 'Moderate directional bias with some volatility'
        elif current_dim < 1.8:
            market_regime = 'Random Walk'
            regime_desc = 'Efficient market with balanced buyer/seller forces'
        else:
            market_regime = 'Chaotic'
            regime_desc = 'High volatility with complex, unpredictable patterns'
        
        # Determine complexity level
        complexity_index = fractal_metrics.get('complexity_index', 0)
        if complexity_index < 0.2:
            complexity_level = 'Low'
        elif complexity_index < 0.5:
            complexity_level = 'Moderate'
        else:
            complexity_level = 'High'
        
        # Generate trading recommendation
        regime_stability = fractal_metrics.get('regime_stability', 0.5)
        
        if abs(current_signal) > 0.5 and regime_stability > 0.6:
            if current_signal > 0:
                trading_recommendation = 'Strong Buy Signal'
                confidence = min(0.9, abs(current_signal) * regime_stability)
            else:
                trading_recommendation = 'Strong Sell Signal'
                confidence = min(0.9, abs(current_signal) * regime_stability)
        elif abs(current_signal) > 0.2:
            if current_signal > 0:
                trading_recommendation = 'Weak Buy Signal'
            else:
                trading_recommendation = 'Weak Sell Signal'
            confidence = abs(current_signal) * regime_stability * 0.7
        else:
            trading_recommendation = 'Hold/Wait'
            confidence = 0.3
        
        # Generate key insights
        insights = [
            f"Market regime: {regime_desc}",
            f"Correlation dimension: {current_dim:.3f}",
            f"Complexity level: {complexity_level}"
        ]
        
        dimension_trend = fractal_metrics.get('dimension_trend', 0)
        if abs(dimension_trend) > 0.05:
            trend_direction = "increasing" if dimension_trend > 0 else "decreasing"
            insights.append(f"Dimension trend: {trend_direction} complexity")
        
        if regime_stability < 0.4:
            insights.append("Warning: Unstable fractal regime detected")
        
        return {
            'market_regime': market_regime,
            'complexity_level': complexity_level,
            'trading_recommendation': trading_recommendation,
            'confidence': confidence,
            'key_insights': insights
        }


def create_fractal_correlation_dimension(window_size: int = 500, **kwargs) -> FractalCorrelationDimension:
    """Factory function to create Fractal Correlation Dimension indicator."""
    return FractalCorrelationDimension(window_size=window_size, **kwargs)
