"""
Fractal Wave Counter
====================

Advanced wave counting algorithm using fractal analysis to identify and count
market waves across multiple timeframes. Combines fractal geometry with
wave theory to detect nested wave structures and their relationships.

The indicator:
- Identifies fractal turning points in price
- Counts waves using fractal decomposition
- Detects nested wave structures across scales
- Calculates wave relationships and ratios
- Provides multi-timeframe wave analysis

Mathematical Foundation:
- Fractal dimension analysis for wave identification
- Self-similarity detection for wave relationships
- Multi-scale decomposition for nested structures
- Statistical validation of wave counts

Author: Platform3 AI System
Created: December 2024
Purpose: Enhance trading decisions through fractal wave analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Fix import - use absolute import with fallback
try:
    from engines.indicator_base import IndicatorBase
except ImportError:
    import sys
    import os
        from indicator_base import IndicatorBase


class WaveType(Enum):
    """Types of waves identified by fractal analysis"""
    IMPULSE = "impulse"
    CORRECTIVE = "corrective"
    COMPLEX = "complex"
    FRACTAL = "fractal"
    UNDEFINED = "undefined"


class WaveDegree(Enum):
    """Degrees of waves from largest to smallest"""
    SUPERCYCLE = 9
    CYCLE = 8
    PRIMARY = 7
    INTERMEDIATE = 6
    MINOR = 5
    MINUTE = 4
    MINUETTE = 3
    SUBMINUETTE = 2
    MICRO = 1


@dataclass
class FractalWave:
    """Represents a single fractal wave"""
    start_index: int
    end_index: int
    start_price: float
    end_price: float
    wave_type: WaveType
    degree: WaveDegree
    fractal_dimension: float
    confidence: float
    sub_waves: List['FractalWave'] = None
    
    def __post_init__(self):
        if self.sub_waves is None:
            self.sub_waves = []
    
    @property
    def length(self) -> int:
        return self.end_index - self.start_index
    
    @property
    def price_change(self) -> float:
        return self.end_price - self.start_price
    
    @property
    def price_change_pct(self) -> float:
        if self.start_price == 0:
            return 0.0
        return (self.price_change / self.start_price) * 100


class FractalWaveCounter(IndicatorBase):
    """
    Advanced Fractal Wave Counter using multi-scale fractal analysis.
    
    Identifies and counts market waves using fractal geometry principles,
    detecting self-similar patterns across multiple timeframes.
    """
    
    def __init__(self,
                 min_wave_size: float = 0.02,
                 max_wave_size: float = 0.5,
                 fractal_threshold: float = 1.4,
                 min_sub_waves: int = 3,
                 max_sub_waves: int = 9,
                 confidence_threshold: float = 0.6):
        """
        Initialize Fractal Wave Counter.
        
        Args:
            min_wave_size: Minimum wave size as fraction of price (2%)
            max_wave_size: Maximum wave size as fraction of price (50%)
            fractal_threshold: Fractal dimension threshold for wave detection
            min_sub_waves: Minimum number of sub-waves
            max_sub_waves: Maximum number of sub-waves
            confidence_threshold: Minimum confidence for wave validation
        """
        super().__init__()
        
        self.min_wave_size = min_wave_size
        self.max_wave_size = max_wave_size
        self.fractal_threshold = fractal_threshold
        self.min_sub_waves = min_sub_waves
        self.max_sub_waves = max_sub_waves
        self.confidence_threshold = confidence_threshold
        
        # Wave storage
        self.waves: List[FractalWave] = []
        self.wave_counts: Dict[WaveDegree, int] = {}
        
        # Analysis cache
        self._fractal_points_cache = {}
        self._dimension_cache = {}
        
        # Validation
        if min_wave_size <= 0 or min_wave_size >= max_wave_size:
            raise ValueError("Invalid wave size parameters")
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Perform fractal wave counting analysis.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing wave analysis results
        """
        try:
            # Validate input data
            required_columns = ['high', 'low', 'close']
            self._validate_data(data, required_columns)
            
            if len(data) < 10:
                raise ValueError("Insufficient data for wave analysis")
            
            # Clear previous analysis
            self.waves.clear()
            self.wave_counts.clear()
            self._fractal_points_cache.clear()
            self._dimension_cache.clear()
            
            # Extract price data
            highs = data['high'].values
            lows = data['low'].values
            closes = data['close'].values
            
            # Identify fractal points at multiple scales
            fractal_points = self._identify_multiscale_fractals(highs, lows, closes)
            
            # Count waves at each degree
            for degree in WaveDegree:
                degree_waves = self._count_waves_at_degree(
                    fractal_points, closes, degree
                )
                self.waves.extend(degree_waves)
                self.wave_counts[degree] = len(degree_waves)
            
            # Build wave hierarchy
            wave_hierarchy = self._build_wave_hierarchy(self.waves)
            
            # Calculate wave statistics
            wave_stats = self._calculate_wave_statistics(self.waves)
            
            # Generate wave analysis
            analysis = self._analyze_wave_patterns(wave_hierarchy, closes)
            
            # Create signals based on wave patterns
            signals = self._generate_wave_signals(wave_hierarchy, closes)
            
            return {
                'waves': self._waves_to_dict(wave_hierarchy),
                'wave_counts': {deg.name: count for deg, count in self.wave_counts.items()},
                'statistics': wave_stats,
                'analysis': analysis,
                'signals': signals,
                'total_waves': len(self.waves),
                'current_wave': self._identify_current_wave(wave_hierarchy, len(closes) - 1),
                'interpretation': self._interpret_wave_structure(
                    wave_hierarchy, wave_stats, analysis
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error in fractal wave counting: {e}")
            raise
    
    def _identify_multiscale_fractals(self, highs: np.ndarray, 
                                    lows: np.ndarray, 
                                    closes: np.ndarray) -> Dict[int, List[Tuple[int, float, str]]]:
        """Identify fractal points at multiple scales."""
        fractal_points = {}
        
        # Define scales based on wave degrees
        scales = {
            WaveDegree.MICRO: 5,
            WaveDegree.SUBMINUETTE: 13,
            WaveDegree.MINUETTE: 21,
            WaveDegree.MINUTE: 34,
            WaveDegree.MINOR: 55,
            WaveDegree.INTERMEDIATE: 89,
            WaveDegree.PRIMARY: 144,
            WaveDegree.CYCLE: 233,
            WaveDegree.SUPERCYCLE: 377
        }
        
        for degree, scale in scales.items():
            if len(closes) < scale * 2:
                continue
            
            points = self._find_fractal_points(highs, lows, closes, scale)
            if points:
                fractal_points[degree.value] = points
        
        return fractal_points
    
    def _find_fractal_points(self, highs: np.ndarray, 
                           lows: np.ndarray, 
                           closes: np.ndarray, 
                           scale: int) -> List[Tuple[int, float, str]]:
        """Find fractal turning points at given scale."""
        points = []
        half_scale = scale // 2
        
        # Find fractal highs
        for i in range(half_scale, len(highs) - half_scale):
            # Check if it's a fractal high
            is_fractal_high = True
            for j in range(1, half_scale + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_fractal_high = False
                    break
            
            if is_fractal_high:
                points.append((i, highs[i], 'high'))
        
        # Find fractal lows
        for i in range(half_scale, len(lows) - half_scale):
            # Check if it's a fractal low
            is_fractal_low = True
            for j in range(1, half_scale + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_fractal_low = False
                    break
            
            if is_fractal_low:
                points.append((i, lows[i], 'low'))
        
        # Sort by index
        points.sort(key=lambda x: x[0])
        
        # Filter alternating highs and lows
        filtered_points = []
        if points:
            filtered_points.append(points[0])
            
            for i in range(1, len(points)):
                # Ensure alternating pattern
                if points[i][2] != filtered_points[-1][2]:
                    # Check minimum price change
                    price_change = abs(points[i][1] - filtered_points[-1][1])
                    avg_price = (points[i][1] + filtered_points[-1][1]) / 2
                    
                    if price_change / avg_price >= self.min_wave_size:
                        filtered_points.append(points[i])
        
        return filtered_points
    
    def _count_waves_at_degree(self, fractal_points: Dict, 
                             closes: np.ndarray, 
                             degree: WaveDegree) -> List[FractalWave]:
        """Count waves at specific degree."""
        waves = []
        
        if degree.value not in fractal_points:
            return waves
        
        points = fractal_points[degree.value]
        
        # Create waves from consecutive fractal points
        for i in range(len(points) - 1):
            start_idx, start_price, start_type = points[i]
            end_idx, end_price, end_type = points[i + 1]
            
            # Calculate fractal dimension for this segment
            segment_closes = closes[start_idx:end_idx + 1]
            fractal_dim = self._calculate_fractal_dimension(segment_closes)
            
            # Determine wave type based on fractal dimension
            wave_type = self._classify_wave_type(fractal_dim, start_type, end_type)
            
            # Calculate confidence
            confidence = self._calculate_wave_confidence(
                segment_closes, fractal_dim, wave_type
            )
            
            if confidence >= self.confidence_threshold:
                wave = FractalWave(
                    start_index=start_idx,
                    end_index=end_idx,
                    start_price=start_price,
                    end_price=end_price,
                    wave_type=wave_type,
                    degree=degree,
                    fractal_dimension=fractal_dim,
                    confidence=confidence
                )
                waves.append(wave)
        
        return waves
    
    def _calculate_fractal_dimension(self, prices: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method."""
        if len(prices) < 3:
            return 1.0
        
        # Normalize prices
        min_price = np.min(prices)
        max_price = np.max(prices)
        
        if max_price == min_price:
            return 1.0
        
        normalized = (prices - min_price) / (max_price - min_price)
        
        # Box-counting algorithm
        scales = []
        counts = []
        
        for scale in range(2, min(len(prices) // 2, 10)):
            # Count boxes needed to cover the curve
            boxes = set()
            
            for i in range(len(normalized)):
                x_box = i // scale
                y_box = int(normalized[i] * scale)
                boxes.add((x_box, y_box))
            
            scales.append(scale)
            counts.append(len(boxes))
        
        if len(scales) < 2:
            return 1.5  # Default dimension
        
        # Calculate dimension from log-log plot
        log_scales = np.log(scales)
        log_counts = np.log(counts)
        
        # Linear regression
        coeffs = np.polyfit(log_scales, log_counts, 1)
        dimension = -coeffs[0]
        
        # Clamp to reasonable range
        return np.clip(dimension, 1.0, 2.0)
    
    def _classify_wave_type(self, fractal_dim: float, 
                          start_type: str, 
                          end_type: str) -> WaveType:
        """Classify wave type based on fractal dimension."""
        if fractal_dim < 1.3:
            # Low dimension - trending/impulse
            return WaveType.IMPULSE
        elif fractal_dim < 1.6:
            # Medium dimension - corrective
            return WaveType.CORRECTIVE
        elif fractal_dim < 1.8:
            # High dimension - complex
            return WaveType.COMPLEX
        else:
            # Very high dimension - fractal/chaotic
            return WaveType.FRACTAL
    
    def _calculate_wave_confidence(self, prices: np.ndarray, 
                                 fractal_dim: float, 
                                 wave_type: WaveType) -> float:
        """Calculate confidence in wave identification."""
        confidence = 1.0
        
        # Factor 1: Price trend consistency
        if len(prices) > 2:
            returns = np.diff(prices) / prices[:-1]
            trend_consistency = 1.0 - np.std(returns) / (np.mean(np.abs(returns)) + 1e-6)
            confidence *= np.clip(trend_consistency, 0.5, 1.0)
        
        # Factor 2: Fractal dimension reliability
        if wave_type == WaveType.IMPULSE:
            dim_reliability = 1.0 - abs(fractal_dim - 1.2) / 0.8
        elif wave_type == WaveType.CORRECTIVE:
            dim_reliability = 1.0 - abs(fractal_dim - 1.5) / 0.5
        else:
            dim_reliability = 0.7
        
        confidence *= np.clip(dim_reliability, 0.3, 1.0)
        
        # Factor 3: Wave size appropriateness
        price_change = abs(prices[-1] - prices[0]) / prices[0]
        if self.min_wave_size <= price_change <= self.max_wave_size:
            size_factor = 1.0
        else:
            size_factor = 0.7
        
        confidence *= size_factor
        
        return confidence
    
    def _build_wave_hierarchy(self, waves: List[FractalWave]) -> List[FractalWave]:
        """Build hierarchical structure of waves."""
        # Sort waves by degree (largest first) and start index
        sorted_waves = sorted(waves, key=lambda w: (-w.degree.value, w.start_index))
        
        # Build hierarchy
        root_waves = []
        
        for wave in sorted_waves:
            # Find parent wave
            parent_found = False
            
            for potential_parent in sorted_waves:
                if (potential_parent.degree.value > wave.degree.value and
                    potential_parent.start_index <= wave.start_index and
                    potential_parent.end_index >= wave.end_index):
                    
                    # Add as sub-wave
                    if wave not in potential_parent.sub_waves:
                        potential_parent.sub_waves.append(wave)
                    parent_found = True
                    break
            
            # If no parent found, it's a root wave
            if not parent_found and wave.degree.value >= WaveDegree.MINOR.value:
                root_waves.append(wave)
        
        return root_waves
    
    def _calculate_wave_statistics(self, waves: List[FractalWave]) -> Dict:
        """Calculate comprehensive wave statistics."""
        if not waves:
            return {}
        
        stats = {}
        
        # Group waves by type
        by_type = {}
        for wave in waves:
            if wave.wave_type not in by_type:
                by_type[wave.wave_type] = []
            by_type[wave.wave_type].append(wave)
        
        # Statistics by wave type
        for wave_type, type_waves in by_type.items():
            type_stats = {
                'count': len(type_waves),
                'avg_length': np.mean([w.length for w in type_waves]),
                'avg_price_change': np.mean([abs(w.price_change_pct) for w in type_waves]),
                'avg_fractal_dimension': np.mean([w.fractal_dimension for w in type_waves]),
                'avg_confidence': np.mean([w.confidence for w in type_waves])
            }
            stats[wave_type.value] = type_stats
        
        # Overall statistics
        stats['total_waves'] = len(waves)
        stats['avg_waves_per_degree'] = len(waves) / len(WaveDegree)
        stats['dominant_wave_type'] = max(by_type.items(), key=lambda x: len(x[1]))[0].value
        
        # Fractal statistics
        all_dimensions = [w.fractal_dimension for w in waves]
        stats['fractal_stats'] = {
            'mean': np.mean(all_dimensions),
            'std': np.std(all_dimensions),
            'min': np.min(all_dimensions),
            'max': np.max(all_dimensions)
        }
        
        return stats
    
    def _analyze_wave_patterns(self, wave_hierarchy: List[FractalWave], 
                             closes: np.ndarray) -> Dict:
        """Analyze wave patterns for trading insights."""
        analysis = {
            'current_structure': 'unknown',
            'completion_status': 0.0,
            'next_wave_type': 'unknown',
            'key_levels': {},
            'pattern_confidence': 0.0
        }
        
        if not wave_hierarchy:
            return analysis
        
        # Find most recent major wave
        recent_waves = sorted(wave_hierarchy, key=lambda w: w.end_index)
        current_wave = recent_waves[-1] if recent_waves else None
        
        if not current_wave:
            return analysis
        
        # Analyze current structure
        if current_wave.wave_type == WaveType.IMPULSE:
            analysis['current_structure'] = 'impulse'
            
            # Count sub-waves
            impulse_count = sum(1 for w in current_wave.sub_waves 
                              if w.wave_type == WaveType.IMPULSE)
            
            if impulse_count >= 5:
                analysis['completion_status'] = 1.0
                analysis['next_wave_type'] = 'corrective'
            else:
                analysis['completion_status'] = impulse_count / 5.0
                analysis['next_wave_type'] = 'impulse_continuation'
        
        elif current_wave.wave_type == WaveType.CORRECTIVE:
            analysis['current_structure'] = 'corrective'
            
            # Count corrective sub-waves
            corrective_count = len(current_wave.sub_waves)
            
            if corrective_count >= 3:
                analysis['completion_status'] = 1.0
                analysis['next_wave_type'] = 'impulse'
            else:
                analysis['completion_status'] = corrective_count / 3.0
                analysis['next_wave_type'] = 'corrective_continuation'
        
        # Identify key levels
        all_waves = self._flatten_wave_hierarchy(wave_hierarchy)
        price_levels = []
        
        for wave in all_waves:
            price_levels.extend([wave.start_price, wave.end_price])
        
        if price_levels:
            analysis['key_levels'] = {
                'resistance': np.percentile(price_levels, 75),
                'support': np.percentile(price_levels, 25),
                'pivot': np.median(price_levels)
            }
        
        # Calculate pattern confidence
        analysis['pattern_confidence'] = current_wave.confidence
        
        return analysis
    
    def _generate_wave_signals(self, wave_hierarchy: List[FractalWave], 
                             closes: np.ndarray) -> List[Dict]:
        """Generate trading signals based on wave patterns."""
        signals = []
        
        if not wave_hierarchy or len(closes) == 0:
            return signals
        
        current_price = closes[-1]
        current_time = len(closes) - 1
        
        # Find active waves at current time
        active_waves = []
        for wave in self._flatten_wave_hierarchy(wave_hierarchy):
            if wave.start_index <= current_time <= wave.end_index:
                active_waves.append(wave)
        
        # Generate signals based on wave positions
        for wave in active_waves:
            # Signal 1: Wave completion
            position_in_wave = (current_time - wave.start_index) / wave.length
            
            if position_in_wave > 0.8:
                if wave.wave_type == WaveType.IMPULSE:
                    signals.append({
                        'type': 'wave_completion',
                        'direction': 'reverse',
                        'strength': wave.confidence * 0.8,
                        'reason': f'{wave.wave_type.value} wave near completion',
                        'target': wave.start_price
                    })
                elif wave.wave_type == WaveType.CORRECTIVE:
                    # Corrective completion suggests trend resumption
                    trend_direction = 'up' if wave.price_change < 0 else 'down'
                    signals.append({
                        'type': 'trend_resumption',
                        'direction': trend_direction,
                        'strength': wave.confidence * 0.7,
                        'reason': 'Corrective wave completion',
                        'target': None
                    })
        
        # Signal 2: Fractal dimension extremes
        recent_waves = [w for w in active_waves if w.confidence > 0.7]
        if recent_waves:
            avg_fractal_dim = np.mean([w.fractal_dimension for w in recent_waves])
            
            if avg_fractal_dim < 1.3:
                signals.append({
                    'type': 'strong_trend',
                    'direction': 'follow',
                    'strength': 0.8,
                    'reason': 'Low fractal dimension indicates strong trend',
                    'target': None
                })
            elif avg_fractal_dim > 1.7:
                signals.append({
                    'type': 'high_volatility',
                    'direction': 'neutral',
                    'strength': 0.6,
                    'reason': 'High fractal dimension indicates chaotic market',
                    'target': None
                })
        
        return signals
    
    def _identify_current_wave(self, wave_hierarchy: List[FractalWave], 
                             current_index: int) -> Optional[Dict]:
        """Identify the current active wave."""
        for wave in self._flatten_wave_hierarchy(wave_hierarchy):
            if wave.start_index <= current_index <= wave.end_index:
                return {
                    'degree': wave.degree.name,
                    'type': wave.wave_type.value,
                    'progress': (current_index - wave.start_index) / wave.length,
                    'fractal_dimension': wave.fractal_dimension,
                    'confidence': wave.confidence
                }
        return None
    
    def _interpret_wave_structure(self, wave_hierarchy: List[FractalWave],
                                wave_stats: Dict, analysis: Dict) -> Dict:
        """Provide comprehensive interpretation of wave structure."""
        interpretation = {
            'market_state': 'unknown',
            'trend_strength': 0.0,
            'volatility': 'normal',
            'trading_bias': 'neutral',
            'key_observations': []
        }
        
        if not wave_hierarchy:
            return interpretation
        
        # Determine market state
        if 'fractal_stats' in wave_stats:
            avg_dimension = wave_stats['fractal_stats']['mean']
            
            if avg_dimension < 1.4:
                interpretation['market_state'] = 'trending'
                interpretation['trend_strength'] = 0.8
                interpretation['trading_bias'] = 'follow_trend'
                interpretation['key_observations'].append('Strong trending behavior detected')
            elif avg_dimension < 1.6:
                interpretation['market_state'] = 'normal'
                interpretation['trend_strength'] = 0.5
                interpretation['trading_bias'] = 'balanced'
                interpretation['key_observations'].append('Normal market conditions')
            else:
                interpretation['market_state'] = 'volatile'
                interpretation['trend_strength'] = 0.2
                interpretation['trading_bias'] = 'range_trading'
                interpretation['key_observations'].append('High volatility/chaos detected')
        
        # Analyze wave type distribution
        if 'impulse' in wave_stats:
            impulse_ratio = wave_stats['impulse']['count'] / wave_stats['total_waves']
            if impulse_ratio > 0.6:
                interpretation['key_observations'].append('Market dominated by impulse waves')
        
        # Add completion status
        if analysis['completion_status'] > 0.8:
            interpretation['key_observations'].append(
                f"Current {analysis['current_structure']} structure near completion"
            )
        
        return interpretation
    
    def _flatten_wave_hierarchy(self, wave_hierarchy: List[FractalWave]) -> List[FractalWave]:
        """Flatten hierarchical wave structure into list."""
        flat_list = []
        
        def add_waves(waves):
            for wave in waves:
                flat_list.append(wave)
                if wave.sub_waves:
                    add_waves(wave.sub_waves)
        
        add_waves(wave_hierarchy)
        return flat_list
    
    def _waves_to_dict(self, waves: List[FractalWave]) -> List[Dict]:
        """Convert waves to dictionary format."""
        result = []
        
        for wave in waves:
            wave_dict = {
                'start_index': wave.start_index,
                'end_index': wave.end_index,
                'start_price': wave.start_price,
                'end_price': wave.end_price,
                'wave_type': wave.wave_type.value,
                'degree': wave.degree.name,
                'fractal_dimension': wave.fractal_dimension,
                'confidence': wave.confidence,
                'length': wave.length,
                'price_change': wave.price_change,
                'price_change_pct': wave.price_change_pct
            }
            
            if wave.sub_waves:
                wave_dict['sub_waves'] = self._waves_to_dict(wave.sub_waves)
            
            result.append(wave_dict)
        
        return result


def create_fractal_wave_counter(min_wave_size: float = 0.02, **kwargs) -> FractalWaveCounter:
    """Factory function to create Fractal Wave Counter."""
    return FractalWaveCounter(min_wave_size=min_wave_size, **kwargs)
