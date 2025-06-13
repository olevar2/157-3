"""
Three Outside Signal Pattern Detector

This module implements detection for the Three Outside Up/Down candlestick patterns,
which are three-candle reversal patterns combining engulfing and confirmation candles.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..base_indicator import StandardIndicatorInterface, IndicatorMetadata


@dataclass
class ThreeOutsideResult:
    """Result container for Three Outside pattern detection."""
    timestamp: pd.Timestamp
    pattern_type: str  # 'three_outside_up' or 'three_outside_down'
    strength: float
    engulfing_ratio: float
    confirmation_size: float
    reliability_score: float
    volume_confirmation: bool


class ThreeOutsideSignal(StandardIndicatorInterface):
    """
    Three Outside Up/Down Pattern Detector
    
    The Three Outside patterns are three-candle reversal formations:
    
    Three Outside Up (Bullish):
    1. Small bearish candle
    2. Large bullish candle that engulfs the first (bullish engulfing)
    3. Bullish confirmation candle that closes above the second candle
    
    Three Outside Down (Bearish):
    1. Small bullish candle
    2. Large bearish candle that engulfs the first (bearish engulfing)
    3. Bearish confirmation candle that closes below the second candle
    """    
    def __init__(self,
                 min_engulf_ratio: float = 1.2,
                 max_first_body_ratio: float = 0.4,
                 min_confirmation_ratio: float = 0.3,
                 volume_threshold: float = 1.1):
        """
        Initialize Three Outside detector.
        
        Args:
            min_engulf_ratio: Minimum engulfing ratio for second candle
            max_first_body_ratio: Maximum body ratio for first small candle
            min_confirmation_ratio: Minimum confirmation candle body ratio
            volume_threshold: Volume multiplier for confirmation
        """
        self.min_engulf_ratio = min_engulf_ratio
        self.max_first_body_ratio = max_first_body_ratio
        self.min_confirmation_ratio = min_confirmation_ratio
        self.volume_threshold = volume_threshold
        
        self.metadata = IndicatorMetadata(
            name="Three Outside Signal",
            description="Detects Three Outside Up/Down reversal patterns",
            category="Pattern Recognition",
            subcategory="Reversal Patterns",
            timeframe_compatibility=["1m", "5m", "15m", "1h", "4h", "1d"],
            data_requirements=["open", "high", "low", "close", "volume"]
        )
    
    def calculate(self, data: pd.DataFrame) -> List[ThreeOutsideResult]:
        """
        Calculate Three Outside patterns.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            List of ThreeOutsideResult objects
        """
        if len(data) < 3:
            return []
        
        results = []
        
        for i in range(2, len(data)):
            # Get three candles for pattern analysis
            candles = data.iloc[i-2:i+1].copy()
            
            # Check for Three Outside Up pattern
            outside_up = self._detect_three_outside_up(candles)
            if outside_up:
                results.append(outside_up)
            
            # Check for Three Outside Down pattern
            outside_down = self._detect_three_outside_down(candles)
            if outside_down:
                results.append(outside_down)
        
        return results
    
    def _detect_three_outside_up(self, candles: pd.DataFrame) -> Optional[ThreeOutsideResult]:
        """Detect Three Outside Up (bullish) pattern."""
        opens = candles['open'].values
        highs = candles['high'].values
        lows = candles['low'].values
        closes = candles['close'].values
        volumes = candles['volume'].values if 'volume' in candles.columns else np.ones(3)
        
        # First candle: Small bearish
        if closes[0] >= opens[0]:  # Not bearish
            return None
        
        first_body = opens[0] - closes[0]
        first_range = highs[0] - lows[0]
        if first_range == 0 or first_body / first_range > self.max_first_body_ratio:
            return None
        
        # Second candle: Bullish engulfing
        if closes[1] <= opens[1]:  # Not bullish
            return None
        
        # Check engulfing criteria
        if not (opens[1] <= closes[0] and closes[1] >= opens[0]):
            return None
        
        second_body = closes[1] - opens[1]
        engulf_ratio = second_body / first_body if first_body > 0 else 0
        if engulf_ratio < self.min_engulf_ratio:
            return None
        
        # Third candle: Confirmation bullish
        if closes[2] <= opens[2]:  # Not bullish
            return None
        
        # Must close above second candle's close
        if closes[2] <= closes[1]:
            return None
        
        third_body = closes[2] - opens[2]
        third_range = highs[2] - lows[2]
        confirmation_ratio = third_body / third_range if third_range > 0 else 0
        if confirmation_ratio < self.min_confirmation_ratio:
            return None
        
        # Calculate pattern metrics
        strength = self._calculate_strength(candles, 'bullish')
        reliability_score = self._calculate_reliability(candles, 'bullish')
        
        # Volume confirmation
        avg_volume = np.mean(volumes[:2])
        volume_confirmation = volumes[2] > self.volume_threshold * avg_volume
        
        return ThreeOutsideResult(
            timestamp=candles.index[-1],
            pattern_type='three_outside_up',
            strength=strength,
            engulfing_ratio=engulf_ratio,
            confirmation_size=third_body,
            reliability_score=reliability_score,
            volume_confirmation=volume_confirmation
        )
    
    def _detect_three_outside_down(self, candles: pd.DataFrame) -> Optional[ThreeOutsideResult]:
        """Detect Three Outside Down (bearish) pattern."""
        opens = candles['open'].values
        highs = candles['high'].values
        lows = candles['low'].values
        closes = candles['close'].values
        volumes = candles['volume'].values if 'volume' in candles.columns else np.ones(3)
        
        # First candle: Small bullish
        if closes[0] <= opens[0]:  # Not bullish
            return None
        
        first_body = closes[0] - opens[0]
        first_range = highs[0] - lows[0]
        if first_range == 0 or first_body / first_range > self.max_first_body_ratio:
            return None
        
        # Second candle: Bearish engulfing
        if closes[1] >= opens[1]:  # Not bearish
            return None
        
        # Check engulfing criteria
        if not (opens[1] >= closes[0] and closes[1] <= opens[0]):
            return None
        
        second_body = opens[1] - closes[1]
        engulf_ratio = second_body / first_body if first_body > 0 else 0
        if engulf_ratio < self.min_engulf_ratio:
            return None
        
        # Third candle: Confirmation bearish
        if closes[2] >= opens[2]:  # Not bearish
            return None
        
        # Must close below second candle's close
        if closes[2] >= closes[1]:
            return None
        
        third_body = opens[2] - closes[2]
        third_range = highs[2] - lows[2]
        confirmation_ratio = third_body / third_range if third_range > 0 else 0
        if confirmation_ratio < self.min_confirmation_ratio:
            return None
        
        # Calculate pattern metrics
        strength = self._calculate_strength(candles, 'bearish')
        reliability_score = self._calculate_reliability(candles, 'bearish')
        
        # Volume confirmation
        avg_volume = np.mean(volumes[:2])
        volume_confirmation = volumes[2] > self.volume_threshold * avg_volume
        
        return ThreeOutsideResult(
            timestamp=candles.index[-1],
            pattern_type='three_outside_down',
            strength=strength,
            engulfing_ratio=engulf_ratio,
            confirmation_size=third_body,
            reliability_score=reliability_score,
            volume_confirmation=volume_confirmation
        )
    
    def _calculate_strength(self, candles: pd.DataFrame, pattern_type: str) -> float:
        """Calculate pattern strength."""
        opens = candles['open'].values
        highs = candles['high'].values
        lows = candles['low'].values
        closes = candles['close'].values
        
        if pattern_type == 'bullish':
            # Measure total bullish move
            total_move = closes[2] - lows[0]
            total_range = highs[2] - lows[0]
        else:
            # Measure total bearish move
            total_move = highs[0] - closes[2]
            total_range = highs[0] - lows[2]
        
        return min(total_move / total_range, 1.0) if total_range > 0 else 0.5
    
    def _calculate_reliability(self, candles: pd.DataFrame, pattern_type: str) -> float:
        """Calculate pattern reliability score."""
        opens = candles['open'].values
        highs = candles['high'].values
        lows = candles['low'].values
        closes = candles['close'].values
        
        # Base reliability
        reliability = 0.6
        
        # Engulfing completeness
        if pattern_type == 'bullish':
            engulf_completeness = min((closes[1] - opens[0]) / (opens[0] - closes[0]), 2.0) / 2.0
        else:
            engulf_completeness = min((opens[0] - closes[1]) / (closes[0] - opens[0]), 2.0) / 2.0
        
        reliability += 0.2 * engulf_completeness
        
        # Confirmation strength
        confirmation_body = abs(closes[2] - opens[2])
        confirmation_range = highs[2] - lows[2]
        if confirmation_range > 0:
            confirmation_strength = confirmation_body / confirmation_range
            reliability += 0.2 * confirmation_strength
        
        return min(reliability, 1.0)
    
    def get_signal_strength(self, results: List[ThreeOutsideResult]) -> Dict[str, float]:
        """Get aggregated signal strength."""
        if not results:
            return {"bullish": 0.0, "bearish": 0.0}
        
        recent_results = results[-3:]  # Last 3 patterns
        
        bullish_strength = sum(r.strength for r in recent_results if r.pattern_type == 'three_outside_up')
        bearish_strength = sum(r.strength for r in recent_results if r.pattern_type == 'three_outside_down')
        
        return {
            "bullish": min(bullish_strength, 1.0),
            "bearish": min(bearish_strength, 1.0)
        }