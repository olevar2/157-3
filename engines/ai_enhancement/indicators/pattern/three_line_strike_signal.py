"""
Three Line Strike Signal Pattern Detector

This module implements detection for the Three Line Strike candlestick pattern,
which is a reversal pattern consisting of three consecutive candles followed by a striking candle.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..base_indicator import StandardIndicatorInterface, IndicatorMetadata


@dataclass
class ThreeLineStrikeResult:
    """Result container for Three Line Strike pattern detection."""
    timestamp: pd.Timestamp
    pattern_type: str  # 'bullish_strike' or 'bearish_strike'
    strength: float
    strike_candle_open: float
    strike_candle_close: float
    first_three_range: float
    reliability_score: float
    volume_confirmation: bool


class ThreeLineStrikeSignal(StandardIndicatorInterface):
    """
    Three Line Strike Pattern Detector
    
    The Three Line Strike is a four-candle reversal pattern:
    - First three candles move in one direction (bullish or bearish)
    - Fourth candle (strike candle) opens within the third candle and closes beyond the first candle
    
    Pattern Types:
    - Bullish Strike: Three declining candles followed by a strong bullish candle
    - Bearish Strike: Three advancing candles followed by a strong bearish candle
    """
    
    def __init__(self,
                 min_body_ratio: float = 0.6,
                 min_strike_ratio: float = 1.2,
                 volume_threshold: float = 1.1):
        """
        Initialize Three Line Strike detector.
        
        Args:
            min_body_ratio: Minimum body-to-range ratio for pattern candles
            min_strike_ratio: Minimum strike candle size relative to average of first three
            volume_threshold: Volume multiplier for confirmation
        """
        self.min_body_ratio = min_body_ratio
        self.min_strike_ratio = min_strike_ratio
        self.volume_threshold = volume_threshold
        
        self.metadata = IndicatorMetadata(
            name="Three Line Strike Signal",
            description="Detects Three Line Strike reversal patterns",
            category="Pattern Recognition",
            subcategory="Reversal Patterns",
            timeframe_compatibility=["1m", "5m", "15m", "1h", "4h", "1d"],
            data_requirements=["open", "high", "low", "close", "volume"]
        )
    
    def calculate(self, data: pd.DataFrame) -> List[ThreeLineStrikeResult]:
        """
        Calculate Three Line Strike patterns.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            List of ThreeLineStrikeResult objects
        """
        if len(data) < 4:
            return []
        
        results = []
        
        for i in range(3, len(data)):
            # Get four candles for pattern analysis
            candles = data.iloc[i-3:i+1].copy()
            
            # Check for bullish strike pattern
            bullish_strike = self._detect_bullish_strike(candles)
            if bullish_strike:
                results.append(bullish_strike)
            
            # Check for bearish strike pattern
            bearish_strike = self._detect_bearish_strike(candles)
            if bearish_strike:
                results.append(bearish_strike)
        
        return results
    
    def _detect_bullish_strike(self, candles: pd.DataFrame) -> Optional[ThreeLineStrikeResult]:
        """Detect bullish three line strike pattern."""
        opens = candles['open'].values
        highs = candles['high'].values
        lows = candles['low'].values
        closes = candles['close'].values
        volumes = candles['volume'].values if 'volume' in candles.columns else np.ones(4)
        
        # First three candles should be bearish (declining)
        for i in range(3):
            if closes[i] >= opens[i]:  # Not bearish
                return None
            
            # Check body ratio
            body_size = abs(closes[i] - opens[i])
            candle_range = highs[i] - lows[i]
            if candle_range == 0 or body_size / candle_range < self.min_body_ratio:
                return None
        
        # Check that closes are declining
        if not (closes[0] > closes[1] > closes[2]):
            return None
        
        # Fourth candle (strike) should be bullish
        if closes[3] <= opens[3]:
            return None
        
        # Strike candle should open within third candle's range
        if not (lows[2] <= opens[3] <= highs[2]):
            return None
        
        # Strike candle should close above first candle's open
        if closes[3] <= opens[0]:
            return None
        
        # Calculate pattern metrics
        first_three_range = opens[0] - closes[2]
        strike_size = closes[3] - opens[3]
        avg_first_three = np.mean([abs(closes[i] - opens[i]) for i in range(3)])
        
        # Check strike size ratio
        if strike_size < self.min_strike_ratio * avg_first_three:
            return None
        
        # Calculate strength and reliability
        strength = min(strike_size / first_three_range, 2.0)
        reliability_score = self._calculate_reliability(candles, 'bullish')
        
        # Volume confirmation
        avg_volume = np.mean(volumes[:3])
        volume_confirmation = volumes[3] > self.volume_threshold * avg_volume
        
        return ThreeLineStrikeResult(
            timestamp=candles.index[-1],
            pattern_type='bullish_strike',
            strength=strength,
            strike_candle_open=opens[3],
            strike_candle_close=closes[3],
            first_three_range=first_three_range,
            reliability_score=reliability_score,
            volume_confirmation=volume_confirmation
        )
    
    def _detect_bearish_strike(self, candles: pd.DataFrame) -> Optional[ThreeLineStrikeResult]:
        """Detect bearish three line strike pattern."""
        opens = candles['open'].values
        highs = candles['high'].values
        lows = candles['low'].values
        closes = candles['close'].values
        volumes = candles['volume'].values if 'volume' in candles.columns else np.ones(4)
        
        # First three candles should be bullish (advancing)
        for i in range(3):
            if closes[i] <= opens[i]:  # Not bullish
                return None
            
            # Check body ratio
            body_size = abs(closes[i] - opens[i])
            candle_range = highs[i] - lows[i]
            if candle_range == 0 or body_size / candle_range < self.min_body_ratio:
                return None
        
        # Check that closes are advancing
        if not (closes[0] < closes[1] < closes[2]):
            return None
        
        # Fourth candle (strike) should be bearish
        if closes[3] >= opens[3]:
            return None
        
        # Strike candle should open within third candle's range
        if not (lows[2] <= opens[3] <= highs[2]):
            return None
        
        # Strike candle should close below first candle's open
        if closes[3] >= opens[0]:
            return None
        
        # Calculate pattern metrics
        first_three_range = closes[2] - opens[0]
        strike_size = opens[3] - closes[3]
        avg_first_three = np.mean([abs(closes[i] - opens[i]) for i in range(3)])
        
        # Check strike size ratio
        if strike_size < self.min_strike_ratio * avg_first_three:
            return None
        
        # Calculate strength and reliability
        strength = min(strike_size / first_three_range, 2.0)
        reliability_score = self._calculate_reliability(candles, 'bearish')
        
        # Volume confirmation
        avg_volume = np.mean(volumes[:3])
        volume_confirmation = volumes[3] > self.volume_threshold * avg_volume
        
        return ThreeLineStrikeResult(
            timestamp=candles.index[-1],
            pattern_type='bearish_strike',
            strength=strength,
            strike_candle_open=opens[3],
            strike_candle_close=closes[3],
            first_three_range=first_three_range,
            reliability_score=reliability_score,
            volume_confirmation=volume_confirmation
        )
    
    def _calculate_reliability(self, candles: pd.DataFrame, pattern_type: str) -> float:
        """Calculate pattern reliability score."""
        opens = candles['open'].values
        highs = candles['high'].values
        lows = candles['low'].values
        closes = candles['close'].values
        
        # Base reliability factors
        reliability = 0.5
        
        # Body size consistency in first three candles
        body_sizes = [abs(closes[i] - opens[i]) for i in range(3)]
        body_consistency = 1.0 - (np.std(body_sizes) / np.mean(body_sizes)) if np.mean(body_sizes) > 0 else 0
        reliability += 0.2 * body_consistency
        
        # Strike candle dominance
        strike_body = abs(closes[3] - opens[3])
        avg_first_three_body = np.mean(body_sizes)
        if avg_first_three_body > 0:
            dominance = min(strike_body / avg_first_three_body, 3.0) / 3.0
            reliability += 0.2 * dominance
        
        # Gap presence (increases reliability)
        if pattern_type == 'bullish':
            if opens[3] > closes[2]:  # Gap up
                reliability += 0.1
        else:
            if opens[3] < closes[2]:  # Gap down
                reliability += 0.1
        
        return min(reliability, 1.0)
    
    def get_signal_strength(self, results: List[ThreeLineStrikeResult]) -> Dict[str, float]:
        """Get aggregated signal strength."""
        if not results:
            return {"bullish": 0.0, "bearish": 0.0}
        
        recent_results = results[-5:]  # Last 5 patterns
        
        bullish_strength = sum(r.strength for r in recent_results if r.pattern_type == 'bullish_strike')
        bearish_strength = sum(r.strength for r in recent_results if r.pattern_type == 'bearish_strike')
        
        return {
            "bullish": min(bullish_strength, 1.0),
            "bearish": min(bearish_strength, 1.0)
        }