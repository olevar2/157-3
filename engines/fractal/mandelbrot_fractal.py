"""
Mandelbrot Fractal Indicator
===========================

Market self-similarity detection using Mandelbrot set principles.
Detects fractal patterns in price movements that repeat across different time scales.

Author: Platform3 AI System
Created: June 3, 2025
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import sys
import os

# Add the engines directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from indicator_base import IndicatorBase
except ImportError:
    from engines.indicator_base import IndicatorBase


class MandelbrotFractalIndicator(IndicatorBase):
    """
    Mandelbrot Fractal Indicator for detecting market self-similarity.
    
    This indicator applies Mandelbrot set principles to financial markets,
    identifying fractal patterns that exhibit self-similarity across time scales.
    """
    
    def __init__(self, 
                 max_iterations: int = 100,
                 escape_radius: float = 2.0,
                 complexity_threshold: float = 0.6,
                 window_size: int = 20):
        """
        Initialize Mandelbrot Fractal Indicator.
        
        Args:
            max_iterations: Maximum iterations for Mandelbrot calculation
            escape_radius: Escape radius for convergence test
            complexity_threshold: Threshold for fractal complexity detection
            window_size: Window size for price analysis
        """
        super().__init__()
        
        self.max_iterations = max_iterations
        self.escape_radius = escape_radius
        self.complexity_threshold = complexity_threshold
        self.window_size = window_size
        
        # Validation
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if escape_radius <= 0:
            raise ValueError("escape_radius must be positive") 
        if window_size <= 0:
            raise ValueError("window_size must be positive")
    
    def _mandelbrot_iterations(self, c: complex) -> int:
        """
        Calculate number of iterations for Mandelbrot set convergence.
        
        Args:
            c: Complex number representing price coordinates
            
        Returns:
            Number of iterations until escape or max_iterations
        """
        z = 0
        for i in range(self.max_iterations):
            if abs(z) > self.escape_radius:
                return i
            z = z*z + c
        return self.max_iterations
    
    def _price_to_complex(self, 
                         prices: np.ndarray, 
                         volumes: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convert price data to complex number coordinates.
        
        Args:
            prices: Price array
            volumes: Optional volume array
            
        Returns:
            Array of complex numbers
        """
        # Normalize prices to suitable range for Mandelbrot calculation
        price_norm = (prices - np.mean(prices)) / np.std(prices)
        
        if volumes is not None:
            # Use volume as imaginary component
            volume_norm = (volumes - np.mean(volumes)) / np.std(volumes)
            return price_norm + 1j * volume_norm
        else:
            # Use price differences as imaginary component
            price_diff = np.diff(prices, prepend=prices[0])
            price_diff_norm = (price_diff - np.mean(price_diff)) / np.std(price_diff)
            return price_norm + 1j * price_diff_norm
    
    def _calculate_fractal_dimension(self, iterations: np.ndarray) -> float:
        """
        Calculate fractal dimension from Mandelbrot iterations.
        
        Args:
            iterations: Array of iteration counts
            
        Returns:
            Estimated fractal dimension
        """
        # Use box counting method on iteration data
        unique_vals, counts = np.unique(iterations, return_counts=True)
        
        if len(unique_vals) <= 1:
            return 1.0
            
        # Calculate fractal dimension using log-log relationship
        log_counts = np.log(counts + 1e-10)
        log_scales = np.log(unique_vals + 1e-10)
        
        # Linear regression to find dimension
        try:
            coeffs = np.polyfit(log_scales, log_counts, 1)
            dimension = abs(coeffs[0])
            return min(max(dimension, 1.0), 2.0)  # Clamp between 1 and 2
        except:
            return 1.5  # Default value
    
    def _detect_self_similarity(self, data: np.ndarray) -> Dict[str, float]:
        """
        Detect self-similarity patterns in the data.
        
        Args:
            data: Input data array
            
        Returns:
            Dictionary with similarity metrics
        """
        if len(data) < self.window_size:
            return {
                'similarity_index': 0.0,
                'pattern_strength': 0.0,
                'complexity_score': 0.0
            }
        
        # Calculate correlations at different scales
        scales = [2, 4, 8, 16]
        similarities = []
        
        for scale in scales:
            if len(data) >= scale * 2:
                # Downsample data
                downsampled = data[::scale]
                
                # Compare with original (appropriately sized)
                min_len = min(len(data), len(downsampled))
                correlation = np.corrcoef(
                    data[:min_len], 
                    downsampled[:min_len]
                )[0, 1] if min_len > 1 else 0.0
                
                if not np.isnan(correlation):
                    similarities.append(abs(correlation))
        
        similarity_index = np.mean(similarities) if similarities else 0.0
        
        # Calculate pattern strength
        pattern_strength = np.std(data) / (np.mean(np.abs(data)) + 1e-10)
        
        # Calculate complexity score
        complexity_score = len(np.unique(data)) / len(data)
        
        return {
            'similarity_index': similarity_index,
            'pattern_strength': pattern_strength,
            'complexity_score': complexity_score
        }
    
    def calculate(self, 
                 data: pd.DataFrame,
                 price_column: str = 'close',
                 volume_column: Optional[str] = 'volume') -> pd.DataFrame:
        """
        Calculate Mandelbrot Fractal Indicator.
        
        Args:
            data: DataFrame with OHLCV data
            price_column: Column name for price data
            volume_column: Column name for volume data (optional)
            
        Returns:
            DataFrame with fractal analysis results
        """
        if len(data) < self.window_size:
            raise ValueError(f"Insufficient data. Need at least {self.window_size} rows")
        
        prices = data[price_column].values
        volumes = data[volume_column].values if volume_column and volume_column in data.columns else None
        
        results = []
        
        for i in range(self.window_size - 1, len(data)):
            # Get window of data
            start_idx = max(0, i - self.window_size + 1)
            window_prices = prices[start_idx:i + 1]
            window_volumes = volumes[start_idx:i + 1] if volumes is not None else None
            
            # Convert to complex coordinates
            complex_coords = self._price_to_complex(window_prices, window_volumes)
            
            # Calculate Mandelbrot iterations
            iterations = np.array([
                self._mandelbrot_iterations(c) for c in complex_coords
            ])
            
            # Calculate fractal dimension
            fractal_dim = self._calculate_fractal_dimension(iterations)
            
            # Detect self-similarity
            similarity_metrics = self._detect_self_similarity(window_prices)
            
            # Calculate fractal signal strength
            avg_iterations = np.mean(iterations)
            signal_strength = (avg_iterations / self.max_iterations) * \
                            similarity_metrics['similarity_index']
            
            # Determine fractal state
            is_fractal = (
                similarity_metrics['similarity_index'] > self.complexity_threshold and
                fractal_dim > 1.2 and
                signal_strength > 0.3
            )
            
            results.append({
                'fractal_dimension': fractal_dim,
                'similarity_index': similarity_metrics['similarity_index'],
                'pattern_strength': similarity_metrics['pattern_strength'],
                'complexity_score': similarity_metrics['complexity_score'],
                'signal_strength': signal_strength,
                'avg_iterations': avg_iterations,
                'is_fractal': is_fractal,
                'fractal_signal': 1.0 if is_fractal else 0.0
            })
        
        # Create result DataFrame
        result_df = pd.DataFrame(results)
        
        # Pad with NaN for the initial window
        pad_rows = self.window_size - 1
        for col in result_df.columns:
            result_df = pd.concat([
                pd.DataFrame({col: [np.nan] * pad_rows}),
                result_df
            ], ignore_index=True)
        
        return result_df
    
    def get_signals(self, 
                   indicator_data: pd.DataFrame,
                   signal_threshold: float = 0.7) -> pd.DataFrame:
        """
        Generate trading signals based on fractal analysis.
        
        Args:
            indicator_data: DataFrame from calculate() method
            signal_threshold: Threshold for signal generation
            
        Returns:
            DataFrame with trading signals
        """
        signals = pd.DataFrame(index=indicator_data.index)
        
        # Fractal emergence signal
        signals['fractal_long'] = (
            (indicator_data['similarity_index'] > signal_threshold) &
            (indicator_data['signal_strength'] > signal_threshold) &
            (indicator_data['is_fractal'] == True)
        ).astype(int)
        
        # Fractal breakdown signal  
        signals['fractal_short'] = (
            (indicator_data['similarity_index'] < (1 - signal_threshold)) &
            (indicator_data['pattern_strength'] < 0.3) &
            (indicator_data['is_fractal'] == False)
        ).astype(int)
        
        # Fractal strength signal
        signals['fractal_strength'] = indicator_data['signal_strength']
        
        return signals
    
    def get_interpretation(self, latest_values: Dict) -> str:
        """
        Provide interpretation of current fractal state.
        
        Args:
            latest_values: Dictionary with latest indicator values
            
        Returns:
            String interpretation
        """
        fractal_dim = latest_values.get('fractal_dimension', 1.0)
        similarity = latest_values.get('similarity_index', 0.0)
        is_fractal = latest_values.get('is_fractal', False)
        signal_strength = latest_values.get('signal_strength', 0.0)
        
        if is_fractal and signal_strength > 0.7:
            return f"Strong fractal pattern detected (D={fractal_dim:.2f}). Market showing high self-similarity ({similarity:.2f}). Expect pattern continuation."
        elif is_fractal and signal_strength > 0.5:
            return f"Moderate fractal pattern (D={fractal_dim:.2f}). Some self-similarity present ({similarity:.2f}). Pattern may persist."
        elif fractal_dim > 1.8:
            return f"High complexity market (D={fractal_dim:.2f}). Low predictability. Exercise caution."
        elif fractal_dim < 1.2:
            return f"Low complexity market (D={fractal_dim:.2f}). Trending behavior likely."
        else:
            return f"Normal market complexity (D={fractal_dim:.2f}). Standard trading conditions."


def create_mandelbrot_fractal_indicator(max_iterations: int = 100,
                                       escape_radius: float = 2.0,
                                       complexity_threshold: float = 0.6,
                                       window_size: int = 20) -> MandelbrotFractalIndicator:
    """
    Factory function to create Mandelbrot Fractal Indicator.
    
    Args:
        max_iterations: Maximum iterations for Mandelbrot calculation
        escape_radius: Escape radius for convergence test  
        complexity_threshold: Threshold for fractal complexity detection
        window_size: Window size for price analysis
        
    Returns:
        Configured MandelbrotFractalIndicator instance
    """
    return MandelbrotFractalIndicator(
        max_iterations=max_iterations,
        escape_radius=escape_radius,
        complexity_threshold=complexity_threshold,
        window_size=window_size
    )
