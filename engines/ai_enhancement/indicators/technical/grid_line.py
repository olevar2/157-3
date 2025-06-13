"""
Grid Line Indicator

Creates a grid-based support and resistance system by identifying significant
price levels and creating horizontal grid lines for trading reference points.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from ..base_indicator import StandardIndicatorInterface


@dataclass
class GridLineResult:
    major_levels: List[float]      # Primary support/resistance levels
    minor_levels: List[float]      # Secondary grid levels
    current_zone: str             # "support", "resistance", "between"
    nearest_level: float          # Closest grid level to current price
    level_strength: Dict[float, float]  # Strength score for each level
    grid_spacing: float           # Average spacing between levels
    timestamp: Optional[str] = None


class GridLine(StandardIndicatorInterface):
    """
    Grid Line Support/Resistance Indicator
    
    Automatically identifies key price levels and creates a systematic grid
    of support and resistance lines based on historical price action,
    volume concentration, and fractal analysis.
    """
    
    CATEGORY = "technical"
    
    def __init__(self,
                 lookback: int = 200,
                 min_touches: int = 3,
                 level_tolerance: float = 0.002,
                 grid_density: int = 10,
                 strength_threshold: float = 0.6,
                 **kwargs):
        """
        Initialize Grid Line indicator.
        
        Args:
            lookback: Historical periods to analyze for level identification
            min_touches: Minimum touches required for a valid level
            level_tolerance: Price tolerance for level clustering (as fraction)
            grid_density: Target number of grid levels to maintain
            strength_threshold: Minimum strength score for level inclusion
        """
        super().__init__(**kwargs)
        self.lookback = lookback
        self.min_touches = min_touches
        self.level_tolerance = level_tolerance
        self.grid_density = grid_density
        self.strength_threshold = strength_threshold
    
    def calculate(self, data: pd.DataFrame) -> GridLineResult:
        """
        Calculate grid lines and support/resistance levels.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            GridLineResult with grid line analysis
        """
        try:
            if len(data) < self.lookback:
                current_price = float(data['close'].iloc[-1]) if len(data) > 0 else 0.0
                return GridLineResult(
                    major_levels=[current_price],
                    minor_levels=[],
                    current_zone="insufficient_data",
                    nearest_level=current_price,
                    level_strength={current_price: 0.0},
                    grid_spacing=0.0
                )
            
            # Get recent data
            recent_data = data.tail(self.lookback).copy()
            current_price = float(recent_data['close'].iloc[-1])
            
            # Identify potential levels from highs and lows
            potential_levels = self._find_potential_levels(recent_data)
            
            # Cluster and validate levels
            validated_levels = self._validate_levels(recent_data, potential_levels)
            
            # Calculate level strengths
            level_strengths = self._calculate_level_strengths(recent_data, validated_levels)
            
            # Filter by strength and create grid
            major_levels, minor_levels = self._create_grid_structure(
                validated_levels, level_strengths, current_price
            )
            
            # Determine current zone
            current_zone = self._determine_current_zone(current_price, major_levels)
            
            # Find nearest level
            all_levels = major_levels + minor_levels
            nearest_level = min(all_levels, key=lambda x: abs(x - current_price)) if all_levels else current_price
            
            # Calculate average grid spacing
            if len(major_levels) > 1:
                grid_spacing = np.mean(np.diff(sorted(major_levels)))
            else:
                grid_spacing = current_price * 0.01  # 1% default
            
            return GridLineResult(
                major_levels=sorted(major_levels),
                minor_levels=sorted(minor_levels),
                current_zone=current_zone,
                nearest_level=nearest_level,
                level_strength=level_strengths,
                grid_spacing=grid_spacing,
                timestamp=recent_data.index[-1].isoformat() if hasattr(recent_data.index[-1], 'isoformat') else None
            )
            
        except Exception as e:
            current_price = float(data['close'].iloc[-1]) if len(data) > 0 else 0.0
            return GridLineResult(
                major_levels=[current_price],
                minor_levels=[],
                current_zone="error",
                nearest_level=current_price,
                level_strength={},
                grid_spacing=0.0
            )
    
    def _find_potential_levels(self, data: pd.DataFrame) -> List[float]:
        """Find potential support/resistance levels from price action."""
        potential_levels = []
        
        # Find swing highs and lows
        highs = data['high'].values
        lows = data['low'].values
        
        # Identify local extrema
        for i in range(2, len(data) - 2):
            # Swing high
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                potential_levels.append(highs[i])
            
            # Swing low
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                potential_levels.append(lows[i])
        
        # Add psychological levels (round numbers)
        price_range = data['high'].max() - data['low'].min()
        if price_range > 0:
            base_price = data['low'].min()
            step = self._calculate_psychological_step(price_range)
            
            level = base_price
            while level <= data['high'].max():
                potential_levels.append(level)
                level += step
        
        return potential_levels
    
    def _calculate_psychological_step(self, price_range: float) -> float:
        """Calculate appropriate step size for psychological levels."""
        magnitude = 10 ** int(np.log10(price_range))
        
        if price_range / magnitude >= 5:
            return magnitude
        elif price_range / magnitude >= 2:
            return magnitude * 0.5
        else:
            return magnitude * 0.2
    
    def _validate_levels(self, data: pd.DataFrame, potential_levels: List[float]) -> List[float]:
        """Validate levels by clustering and counting touches."""
        if not potential_levels:
            return []
        
        # Cluster nearby levels
        clustered_levels = self._cluster_levels(potential_levels)
        
        # Count touches for each level
        validated_levels = []
        for level in clustered_levels:
            touches = self._count_level_touches(data, level)
            if touches >= self.min_touches:
                validated_levels.append(level)
        
        return validated_levels
    
    def _cluster_levels(self, levels: List[float]) -> List[float]:
        """Cluster nearby price levels together."""
        if not levels:
            return []
        
        sorted_levels = sorted(levels)
        clustered = []
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] <= self.level_tolerance:
                current_cluster.append(level)
            else:
                # Save average of current cluster
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        
        # Don't forget the last cluster
        clustered.append(np.mean(current_cluster))
        
        return clustered
    
    def _count_level_touches(self, data: pd.DataFrame, level: float) -> int:
        """Count how many times price touched a level."""
        tolerance = level * self.level_tolerance
        
        touches = 0
        for _, row in data.iterrows():
            # Check if high or low touched the level
            if (abs(row['high'] - level) <= tolerance or 
                abs(row['low'] - level) <= tolerance):
                touches += 1
        
        return touches
    
    def _calculate_level_strengths(self, data: pd.DataFrame, levels: List[float]) -> Dict[float, float]:
        """Calculate strength scores for each level."""
        strengths = {}
        
        for level in levels:
            # Factors contributing to level strength:
            touches = self._count_level_touches(data, level)
            volume_at_level = self._calculate_volume_at_level(data, level)
            time_since_touch = self._time_since_last_touch(data, level)
            bounce_quality = self._calculate_bounce_quality(data, level)
            
            # Normalize and combine factors
            touch_score = min(touches / 10.0, 1.0)  # Normalize to [0, 1]
            volume_score = min(volume_at_level, 1.0)
            time_score = max(0, 1.0 - time_since_touch / len(data))  # Fresher = better
            bounce_score = bounce_quality
            
            # Weighted combination
            strength = (touch_score * 0.3 + volume_score * 0.2 + 
                       time_score * 0.2 + bounce_score * 0.3)
            
            strengths[level] = strength
        
        return strengths
    
    def _calculate_volume_at_level(self, data: pd.DataFrame, level: float) -> float:
        """Calculate relative volume concentration at a price level."""
        tolerance = level * self.level_tolerance
        
        volume_at_level = 0
        total_volume = data['volume'].sum()
        
        for _, row in data.iterrows():
            if (abs(row['high'] - level) <= tolerance or 
                abs(row['low'] - level) <= tolerance):
                volume_at_level += row['volume']
        
        return volume_at_level / total_volume if total_volume > 0 else 0
    
    def _time_since_last_touch(self, data: pd.DataFrame, level: float) -> int:
        """Calculate periods since level was last touched."""
        tolerance = level * self.level_tolerance
        
        for i in range(len(data) - 1, -1, -1):
            row = data.iloc[i]
            if (abs(row['high'] - level) <= tolerance or 
                abs(row['low'] - level) <= tolerance):
                return len(data) - 1 - i
        
        return len(data)  # Never touched
    
    def _calculate_bounce_quality(self, data: pd.DataFrame, level: float) -> float:
        """Calculate quality of bounces from the level."""
        tolerance = level * self.level_tolerance
        bounce_scores = []
        
        for i in range(1, len(data) - 1):
            row = data.iloc[i]
            prev_row = data.iloc[i - 1]
            next_row = data.iloc[i + 1]
            
            # Check for bounce at level
            if abs(row['low'] - level) <= tolerance:
                # Measure strength of subsequent bounce
                bounce_strength = (next_row['high'] - row['low']) / row['low']
                bounce_scores.append(min(bounce_strength * 100, 1.0))
            elif abs(row['high'] - level) <= tolerance:
                # Measure strength of subsequent rejection
                rejection_strength = (row['high'] - next_row['low']) / row['high']
                bounce_scores.append(min(rejection_strength * 100, 1.0))
        
        return np.mean(bounce_scores) if bounce_scores else 0.0
    
    def _create_grid_structure(self, levels: List[float], strengths: Dict[float, float], 
                              current_price: float) -> Tuple[List[float], List[float]]:
        """Create major and minor grid levels."""
        # Filter by strength threshold
        strong_levels = [level for level in levels if strengths.get(level, 0) >= self.strength_threshold]
        
        # Sort by strength and take top levels
        sorted_by_strength = sorted(strong_levels, key=lambda x: strengths.get(x, 0), reverse=True)
        
        # Select major levels (top strength levels)
        max_major = min(self.grid_density // 2, len(sorted_by_strength))
        major_levels = sorted_by_strength[:max_major]
        
        # Create minor levels between major levels
        minor_levels = []
        if len(major_levels) >= 2:
            sorted_major = sorted(major_levels)
            for i in range(len(sorted_major) - 1):
                mid_level = (sorted_major[i] + sorted_major[i + 1]) / 2
                minor_levels.append(mid_level)
        
        # Ensure current price area is covered
        if not any(abs(level - current_price) / current_price < 0.01 for level in major_levels + minor_levels):
            # Add a level near current price
            nearest_major = min(major_levels, key=lambda x: abs(x - current_price)) if major_levels else current_price
            if abs(nearest_major - current_price) / current_price > 0.02:
                minor_levels.append(current_price)
        
        return major_levels, minor_levels
    
    def _determine_current_zone(self, current_price: float, major_levels: List[float]) -> str:
        """Determine if current price is at support, resistance, or between levels."""
        if not major_levels:
            return "no_levels"
        
        sorted_levels = sorted(major_levels)
        tolerance = current_price * 0.005  # 0.5% tolerance
        
        # Check if at a level
        for level in sorted_levels:
            if abs(current_price - level) <= tolerance:
                # Determine if support or resistance based on recent price action
                levels_above = [l for l in sorted_levels if l > current_price + tolerance]
                levels_below = [l for l in sorted_levels if l < current_price - tolerance]
                
                if levels_above and not levels_below:
                    return "support"
                elif levels_below and not levels_above:
                    return "resistance"
                else:
                    return "at_level"
        
        return "between"
    
    def get_display_name(self) -> str:
        return "Grid Line Support/Resistance"
    
    def get_parameters(self) -> Dict:
        return {
            "lookback": self.lookback,
            "min_touches": self.min_touches,
            "level_tolerance": self.level_tolerance,
            "grid_density": self.grid_density,
            "strength_threshold": self.strength_threshold
        }