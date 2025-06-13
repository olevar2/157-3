"""
Self Similarity Signal Indicator

This indicator measures the self-similarity characteristics in price data using fractal analysis.
Self-similarity is a key property of fractal structures where patterns repeat at different scales.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings
from scipy import stats
from dataclasses import dataclass

# For direct script testing
try:
    from engines.ai_enhancement.indicators.base_indicator import BaseIndicator, IndicatorConfig
except ImportError:
    import sys
    import os
    # Add the project root to the path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    sys.path.insert(0, project_root)
    try:
        from engines.ai_enhancement.indicators.base_indicator import BaseIndicator, IndicatorConfig
    except ImportError:
        # Fallback for direct testing
        class IndicatorConfig:
            def __init__(self):
                pass
            def __post_init__(self):
                pass
        
        class BaseIndicator:
            def __init__(self, config):
                self.config = config
            def _handle_error(self, msg):
                print(f"Error: {msg}")
            def reset(self):
                pass


@dataclass
class SelfSimilarityConfig(IndicatorConfig):
    """Configuration for Self Similarity Signal indicator."""
    
    window_size: int = 50
    fractal_dimension_method: str = 'box_counting'  # 'box_counting', 'variance'
    similarity_threshold: float = 0.8
    scale_range: Tuple[int, int] = (2, 20)
    signal_smoothing: int = 5
    
    def __post_init__(self):
        super().__post_init__()
        if self.window_size < 20:
            raise ValueError("Window size must be at least 20")
        if not 0.1 <= self.similarity_threshold <= 1.0:
            raise ValueError("Similarity threshold must be between 0.1 and 1.0")
        if self.scale_range[0] >= self.scale_range[1]:
            raise ValueError("Scale range must be (min, max) with min < max")


class SelfSimilaritySignal(BaseIndicator):
    """
    Self Similarity Signal Indicator
    
    This indicator analyzes the self-similarity characteristics of price data
    using fractal analysis techniques. It measures how similar patterns are
    across different time scales, which can indicate market regime changes
    and potential reversal points.
    
    The indicator uses:
    1. Fractal dimension calculation
    2. Scale-invariant pattern analysis
    3. Self-similarity coefficient measurement
    4. Regime change detection
    
    Formula:
    - Fractal Dimension (FD) = log(N) / log(1/r)
    - Self-Similarity Index = correlation(pattern_scale1, pattern_scale2)
    - Signal = smoothed(Self-Similarity Index)
    
    Interpretation:
    - High values (>0.8): Strong self-similarity, trending behavior
    - Medium values (0.4-0.8): Moderate self-similarity, transitional periods
    - Low values (<0.4): Weak self-similarity, random/chaotic behavior
    - Signal crossovers: Potential regime changes
    """
    
    def __init__(self, config: Optional[SelfSimilarityConfig] = None):
        """Initialize Self Similarity Signal indicator."""
        if config is None:
            config = SelfSimilarityConfig()
        
        super().__init__(config)
        self.config: SelfSimilarityConfig = config
        
        # Internal state
        self._price_buffer = []
        self._similarity_buffer = []
        self._fractal_dimensions = []
        
        # Results storage
        self.similarity_index = []
        self.fractal_dimension = []
        self.signal = []
        self.regime_state = []
        
    def _calculate_fractal_dimension(self, data: np.ndarray) -> float:
        """Calculate fractal dimension using specified method."""
        try:
            if self.config.fractal_dimension_method == 'box_counting':
                return self._box_counting_dimension(data)
            elif self.config.fractal_dimension_method == 'variance':
                return self._variance_dimension(data)
            else:
                return 1.5  # Default fallback
        except Exception:
            return 1.5
    
    def _box_counting_dimension(self, data: np.ndarray) -> float:
        """Calculate fractal dimension using box counting method."""
        if len(data) < 10:
            return 1.5
        
        # Normalize data
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
        
        scales = []
        counts = []
        
        min_scale, max_scale = self.config.scale_range
        
        for scale in range(min_scale, min(max_scale, len(data) // 4)):
            # Count boxes at this scale
            boxes = set()
            for i in range(len(data_norm) - scale + 1):
                segment = data_norm[i:i+scale]
                # Discretize segment
                box_coord = tuple(np.round(segment * 10).astype(int))
                boxes.add(box_coord)
            
            if len(boxes) > 0:
                scales.append(1.0 / scale)
                counts.append(len(boxes))
        
        if len(scales) < 3:
            return 1.5
        
        # Linear regression to find slope
        log_scales = np.log(scales)
        log_counts = np.log(counts)
        
        slope, _, r_value, _, _ = stats.linregress(log_scales, log_counts)
        
        # Fractal dimension is the slope
        dimension = abs(slope)
        
        # Bound the result
        return max(1.0, min(2.0, dimension))
    
    def _variance_dimension(self, data: np.ndarray) -> float:
        """Calculate fractal dimension using variance method."""
        if len(data) < 10:
            return 1.5
        
        scales = []
        variances = []
        
        min_scale, max_scale = self.config.scale_range
        
        for scale in range(min_scale, min(max_scale, len(data) // 4)):
            # Calculate variance at this scale
            scaled_data = []
            for i in range(0, len(data) - scale + 1, scale):
                segment = data[i:i+scale]
                if len(segment) == scale:
                    scaled_data.append(np.mean(segment))
            
            if len(scaled_data) > 1:
                variance = np.var(scaled_data)
                if variance > 0:
                    scales.append(scale)
                    variances.append(variance)
        
        if len(scales) < 3:
            return 1.5
        
        # Linear regression
        log_scales = np.log(scales)
        log_variances = np.log(variances)
        
        slope, _, r_value, _, _ = stats.linregress(log_scales, log_variances)
        
        # Fractal dimension from variance scaling
        dimension = 1.5 + slope / 2.0
        
        # Bound the result
        return max(1.0, min(2.0, dimension))
    
    def _calculate_self_similarity(self, data: np.ndarray) -> float:
        """Calculate self-similarity coefficient."""
        if len(data) < 20:
            return 0.5
        
        # Compare patterns at different scales
        correlations = []
        
        # Use multiple scale comparisons
        for scale1 in [5, 10, 15]:
            for scale2 in [scale1 * 2, scale1 * 3]:
                if scale2 < len(data) // 2:
                    pattern1 = self._extract_pattern(data, scale1)
                    pattern2 = self._extract_pattern(data, scale2)
                    
                    if len(pattern1) > 0 and len(pattern2) > 0:
                        # Align patterns and calculate correlation
                        min_len = min(len(pattern1), len(pattern2))
                        p1 = pattern1[:min_len]
                        p2 = pattern2[:min_len]
                        
                        if np.std(p1) > 0 and np.std(p2) > 0:
                            corr = np.corrcoef(p1, p2)[0, 1]
                            if not np.isnan(corr):
                                correlations.append(abs(corr))
        
        if len(correlations) == 0:
            return 0.5
        
        # Return average correlation
        return np.mean(correlations)
    
    def _extract_pattern(self, data: np.ndarray, scale: int) -> np.ndarray:
        """Extract pattern at specific scale."""
        if scale >= len(data):
            return np.array([])
        
        # Downsample data at given scale
        pattern = []
        for i in range(0, len(data) - scale + 1, scale):
            segment = data[i:i+scale]
            if len(segment) == scale:
                # Use relative changes
                pattern.append(np.mean(segment))
        
        if len(pattern) < 2:
            return np.array([])
        
        # Convert to relative changes
        pattern = np.array(pattern)
        changes = np.diff(pattern) / (pattern[:-1] + 1e-8)
        
        return changes
    
    def _classify_regime(self, similarity: float, fractal_dim: float) -> str:
        """Classify market regime based on similarity and fractal dimension."""
        if similarity > 0.8 and fractal_dim < 1.3:
            return "Strong Trend"
        elif similarity > 0.6 and fractal_dim < 1.5:
            return "Weak Trend"
        elif similarity < 0.4 and fractal_dim > 1.7:
            return "Random/Chaotic"
        elif 0.4 <= similarity <= 0.6:
            return "Transitional"
        else:
            return "Mixed"
    
    def update(self, price: float, volume: Optional[float] = None, 
               timestamp: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """Update indicator with new price data."""
        try:
            # Store price
            self._price_buffer.append(price)
            
            # Maintain buffer size
            if len(self._price_buffer) > self.config.window_size * 2:
                self._price_buffer = self._price_buffer[-self.config.window_size * 2:]
            
            # Initialize results
            similarity = 0.5
            fractal_dim = 1.5
            signal_value = 0.5
            regime = "Insufficient Data"
            
            # Calculate when we have enough data
            if len(self._price_buffer) >= self.config.window_size:
                data = np.array(self._price_buffer[-self.config.window_size:])
                
                # Calculate fractal dimension
                fractal_dim = self._calculate_fractal_dimension(data)
                
                # Calculate self-similarity
                similarity = self._calculate_self_similarity(data)
                
                # Store similarity for smoothing
                self._similarity_buffer.append(similarity)
                if len(self._similarity_buffer) > self.config.signal_smoothing:
                    self._similarity_buffer = self._similarity_buffer[-self.config.signal_smoothing:]
                
                # Calculate smoothed signal
                signal_value = np.mean(self._similarity_buffer)
                
                # Classify regime
                regime = self._classify_regime(similarity, fractal_dim)
            
            # Store results
            self.similarity_index.append(similarity)
            self.fractal_dimension.append(fractal_dim)
            self.signal.append(signal_value)
            self.regime_state.append(regime)
            
            # Maintain result buffer sizes
            max_history = 1000
            if len(self.similarity_index) > max_history:
                self.similarity_index = self.similarity_index[-max_history:]
                self.fractal_dimension = self.fractal_dimension[-max_history:]
                self.signal = self.signal[-max_history:]
                self.regime_state = self.regime_state[-max_history:]
            
            return {
                'similarity_index': similarity,
                'fractal_dimension': fractal_dim,
                'signal': signal_value,
                'regime_state': regime,
                'self_similarity_strength': self._interpret_similarity(similarity),
                'trend_persistence': self._interpret_fractal_dim(fractal_dim)
            }
            
        except Exception as e:
            self._handle_error(f"Error in Self Similarity Signal update: {e}")
            return {
                'similarity_index': 0.5,
                'fractal_dimension': 1.5,
                'signal': 0.5,
                'regime_state': "Error",
                'self_similarity_strength': "Unknown",
                'trend_persistence': "Unknown"
            }
    
    def _interpret_similarity(self, similarity: float) -> str:
        """Interpret similarity index value."""
        if similarity > 0.8:
            return "Very Strong"
        elif similarity > 0.6:
            return "Strong"
        elif similarity > 0.4:
            return "Moderate"
        elif similarity > 0.2:
            return "Weak"
        else:
            return "Very Weak"
    
    def _interpret_fractal_dim(self, fractal_dim: float) -> str:
        """Interpret fractal dimension value."""
        if fractal_dim < 1.3:
            return "Highly Persistent"
        elif fractal_dim < 1.5:
            return "Persistent"
        elif fractal_dim < 1.7:
            return "Random Walk"
        else:
            return "Anti-Persistent"
    
    def get_signal_line(self) -> List[float]:
        """Get the main signal line."""
        return self.signal.copy()
    
    def get_similarity_data(self) -> Dict[str, List]:
        """Get all similarity-related data."""
        return {
            'similarity_index': self.similarity_index.copy(),
            'fractal_dimension': self.fractal_dimension.copy(),
            'signal': self.signal.copy(),
            'regime_state': self.regime_state.copy()
        }
    
    def reset(self):
        """Reset indicator state."""
        super().reset()
        self._price_buffer.clear()
        self._similarity_buffer.clear()
        self._fractal_dimensions.clear()
        self.similarity_index.clear()
        self.fractal_dimension.clear()
        self.signal.clear()
        self.regime_state.clear()


# Example usage and testing
if __name__ == "__main__":
    print("Testing Self Similarity Signal Indicator")
    print("=" * 50)
    
    # Create indicator
    config = SelfSimilarityConfig(
        window_size=30,
        fractal_dimension_method='box_counting',
        similarity_threshold=0.7,
        scale_range=(2, 15),
        signal_smoothing=3
    )
    
    indicator = SelfSimilaritySignal(config)
    
    # Generate test data with different regimes
    np.random.seed(42)
    
    # Trending data
    trend_data = np.cumsum(np.random.randn(100) * 0.5 + 0.1) + 100
    
    # Random data
    random_data = np.random.randn(50) * 2 + 105
    
    # Fractal data (self-similar)
    fractal_data = []
    base_pattern = [1, -0.5, 0.8, -0.3, 0.6]
    for i in range(50):
        scale = 1 + i * 0.01
        fractal_data.extend([x * scale + 110 + i * 0.1 for x in base_pattern])
    
    # Combine all data
    test_data = np.concatenate([trend_data, random_data, fractal_data[:100]])
    
    print(f"Processing {len(test_data)} data points...")
    
    # Process data
    results = []
    for i, price in enumerate(test_data):
        result = indicator.update(price)
        results.append(result)
        
        # Print periodic updates
        if i > 0 and (i + 1) % 50 == 0:
            print(f"\nData point {i + 1}:")
            print(f"  Price: {price:.2f}")
            print(f"  Similarity Index: {result['similarity_index']:.3f}")
            print(f"  Fractal Dimension: {result['fractal_dimension']:.3f}")
            print(f"  Signal: {result['signal']:.3f}")
            print(f"  Regime: {result['regime_state']}")
            print(f"  Similarity Strength: {result['self_similarity_strength']}")
            print(f"  Trend Persistence: {result['trend_persistence']}")
    
    # Get final signal data
    signal_data = indicator.get_similarity_data()
    
    print("\nFinal Analysis:")
    print("-" * 30)
    
    if len(signal_data['signal']) > 0:
        latest_signal = signal_data['signal'][-1]
        latest_similarity = signal_data['similarity_index'][-1]
        latest_fractal = signal_data['fractal_dimension'][-1]
        latest_regime = signal_data['regime_state'][-1]
        
        print(f"Latest Signal: {latest_signal:.3f}")
        print(f"Latest Similarity Index: {latest_similarity:.3f}")
        print(f"Latest Fractal Dimension: {latest_fractal:.3f}")
        print(f"Latest Regime: {latest_regime}")
        
        # Signal statistics
        signals = signal_data['signal']
        similarities = signal_data['similarity_index']
        fractals = signal_data['fractal_dimension']
        
        if len(signals) > 10:
            print(f"\nSignal Statistics:")
            print(f"  Average Signal: {np.mean(signals):.3f}")
            print(f"  Signal Range: {np.min(signals):.3f} - {np.max(signals):.3f}")
            print(f"  Average Similarity: {np.mean(similarities):.3f}")
            print(f"  Average Fractal Dim: {np.mean(fractals):.3f}")
            
            # Regime distribution
            regimes = signal_data['regime_state']
            unique_regimes = list(set(regimes))
            print(f"\nRegime Distribution:")
            for regime in unique_regimes:
                count = regimes.count(regime)
                percentage = (count / len(regimes)) * 100
                print(f"  {regime}: {count} ({percentage:.1f}%)")
    
    print("\nSelf Similarity Signal Indicator test completed successfully!")