import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.market_data import OHLCV
from engines.base_pattern import BasePatternEngine

"""
Morning Star and Evening Star Pattern Detection Engine

Star patterns are three-candle reversal patterns:

Morning Star (Bullish reversal):
- First candle: Long bearish candle
- Second candle: Small-bodied candle (star) that gaps down
- Third candle: Long bullish candle that closes well into first candle's body

Evening Star (Bearish reversal):
- First candle: Long bullish candle
- Second candle: Small-bodied candle (star) that gaps up
- Third candle: Long bearish candle that closes well into first candle's body
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
# from ..base_pattern import BasePatternEngine  # Fixed import
# from ...models.market_data import OHLCV  # Fixed import
# from ...utils.pattern_validation import PatternValidator  # Fixed import


@dataclass
class StarSignal:
    """Signal data for Star patterns"""
    pattern_type: str  # 'morning_star' or 'evening_star'
    confidence: float
    entry_price: float
    stop_loss: float
    target_price: float
    timestamp: str
    candles_involved: List[OHLCV]
    star_gap_size: float
    penetration_percentage: float
    strength: str  # 'weak', 'moderate', 'strong'


class StarPatternEngine(BasePatternEngine):
    """
    Morning Star and Evening Star Pattern Detection and Signal Generation Engine
    
    Detects both morning and evening star patterns with proper gap and penetration validation
    """
    
    def __init__(self, min_body_ratio: float = 0.6, min_star_gap: float = 0.1, min_penetration: float = 0.5):
        super().__init__()
        self.min_body_ratio = min_body_ratio  # Minimum body ratio for first and third candles
        self.min_star_gap = min_star_gap  # Minimum gap for star candle (as % of first candle)
        self.min_penetration = min_penetration  # Minimum penetration into first candle's body

        
    def detect_star_pattern(self, candles: List[OHLCV]) -> Optional[Dict[str, Any]]:
        """
        Detect Star pattern in the given candles
        
        Args:
            candles: List of OHLCV data (minimum 3 candles needed)
            
        Returns:
            Pattern detection result or None if no pattern found
        """
        if len(candles) < 3:
            return None
            
        # Get the last three candles
        first_candle = candles[-3]
        star_candle = candles[-2]
        third_candle = candles[-1]
        
        # Check for morning star pattern
        morning_result = self._detect_morning_star(first_candle, star_candle, third_candle)
        if morning_result:
            return morning_result
            
        # Check for evening star pattern
        evening_result = self._detect_evening_star(first_candle, star_candle, third_candle)
        if evening_result:
            return evening_result
            
        return None
        
    def _detect_morning_star(self, first: OHLCV, star: OHLCV, third: OHLCV) -> Optional[Dict[str, Any]]:
        """Detect morning star pattern"""
        # First candle should be long bearish
        if not self._is_long_bearish(first):
            return None
            
        # Star candle should be small-bodied
        if not self._is_small_bodied(star):
            return None
            
        # Third candle should be long bullish
        if not self._is_long_bullish(third):
            return None
            
        # Check gap down for star candle
        star_gap = first.low - star.high
        if star_gap <= 0:  # No gap or gap up
            return None
            
        # Check gap size relative to first candle
        first_range = first.high - first.low
        gap_percentage = (star_gap / first_range) * 100
        
        if gap_percentage < self.min_star_gap:
            return None
            
        # Check penetration of third candle into first candle's body
        first_body_mid = (first.open + first.close) / 2
        penetration = third.close - first.close
        first_body_size = first.open - first.close
        penetration_percentage = (penetration / first_body_size) * 100
        
        if penetration_percentage < self.min_penetration * 100:
            return None
            
        # Calculate pattern strength
        strength = self._calculate_pattern_strength(gap_percentage, penetration_percentage)
        
        return {
            'pattern_type': 'morning_star',
            'confidence': self._calculate_confidence(first, star, third, gap_percentage, penetration_percentage),
            'star_gap_size': star_gap,
            'gap_percentage': gap_percentage,
            'penetration_percentage': penetration_percentage,
            'strength': strength,
            'candles': [first, star, third]
        }
        
    def _detect_evening_star(self, first: OHLCV, star: OHLCV, third: OHLCV) -> Optional[Dict[str, Any]]:
        """Detect evening star pattern"""
        # First candle should be long bullish
        if not self._is_long_bullish(first):
            return None
            
        # Star candle should be small-bodied
        if not self._is_small_bodied(star):
            return None
            
        # Third candle should be long bearish
        if not self._is_long_bearish(third):
            return None
            
        # Check gap up for star candle
        star_gap = star.low - first.high
        if star_gap <= 0:  # No gap or gap down
            return None
            
        # Check gap size relative to first candle
        first_range = first.high - first.low
        gap_percentage = (star_gap / first_range) * 100
        
        if gap_percentage < self.min_star_gap:
            return None
            
        # Check penetration of third candle into first candle's body
        first_body_mid = (first.open + first.close) / 2
        penetration = first.close - third.close
        first_body_size = first.close - first.open
        penetration_percentage = (penetration / first_body_size) * 100
        
        if penetration_percentage < self.min_penetration * 100:
            return None
            
        # Calculate pattern strength
        strength = self._calculate_pattern_strength(gap_percentage, penetration_percentage)
        
        return {
            'pattern_type': 'evening_star',
            'confidence': self._calculate_confidence(first, star, third, gap_percentage, penetration_percentage),
            'star_gap_size': star_gap,
            'gap_percentage': gap_percentage,
            'penetration_percentage': penetration_percentage,
            'strength': strength,
            'candles': [first, star, third]
        }
        
    def _is_long_bullish(self, candle: OHLCV) -> bool:
        """Check if candle is long bullish"""
        if candle.close <= candle.open:
            return False
            
        body_size = candle.close - candle.open
        total_range = candle.high - candle.low
        
        if total_range == 0:
            return False
            
        body_ratio = body_size / total_range
        return body_ratio >= self.min_body_ratio
        
    def _is_long_bearish(self, candle: OHLCV) -> bool:
        """Check if candle is long bearish"""
        if candle.close >= candle.open:
            return False
            
        body_size = candle.open - candle.close
        total_range = candle.high - candle.low
        
        if total_range == 0:
            return False
            
        body_ratio = body_size / total_range
        return body_ratio >= self.min_body_ratio
        
    def _is_small_bodied(self, candle: OHLCV) -> bool:
        """Check if candle has small body (star characteristics)"""
        body_size = abs(candle.close - candle.open)
        total_range = candle.high - candle.low
        
        if total_range == 0:
            return True  # Doji-like
            
        body_ratio = body_size / total_range
        return body_ratio <= 0.3  # Small body relative to range
        
    def _calculate_pattern_strength(self, gap_percentage: float, penetration_percentage: float) -> str:
        """Calculate pattern strength based on gap size and penetration"""
        if gap_percentage >= 1.0 and penetration_percentage >= 70:
            return 'strong'
        elif gap_percentage >= 0.5 and penetration_percentage >= 60:
            return 'moderate'
        else:
            return 'weak'
            
    def _calculate_confidence(self, first: OHLCV, star: OHLCV, third: OHLCV, 
                            gap_percentage: float, penetration_percentage: float) -> float:
        """Calculate pattern confidence score"""
        confidence = 0.5  # Base confidence
        
        # Add confidence based on gap size
        if gap_percentage >= 1.0:
            confidence += 0.2
        elif gap_percentage >= 0.5:
            confidence += 0.15
        else:
            confidence += 0.1
            
        # Add confidence based on penetration
        if penetration_percentage >= 70:
            confidence += 0.2
        elif penetration_percentage >= 60:
            confidence += 0.15
        else:
            confidence += 0.1
            
        # Add confidence based on candle body strength
        first_body_ratio = abs(first.close - first.open) / (first.high - first.low)
        third_body_ratio = abs(third.close - third.open) / (third.high - third.low)
        
        avg_body_ratio = (first_body_ratio + third_body_ratio) / 2
        if avg_body_ratio >= 0.8:
            confidence += 0.1
        elif avg_body_ratio >= 0.7:
            confidence += 0.05
            
        return min(confidence, 0.95)  # Cap at 95%
        
    def generate_signal(self, pattern_data: Dict[str, Any]) -> StarSignal:
        """Generate trading signal from pattern detection"""
        candles = pattern_data['candles']
        third_candle = candles[-1]
        
        if pattern_data['pattern_type'] == 'morning_star':
            entry_price = third_candle.close
            stop_loss = min(candle.low for candle in candles) * 0.995  # 0.5% buffer
            target_price = entry_price + (entry_price - stop_loss) * 2  # 2:1 R/R ratio
        else:  # evening_star
            entry_price = third_candle.close
            stop_loss = max(candle.high for candle in candles) * 1.005  # 0.5% buffer
            target_price = entry_price - (stop_loss - entry_price) * 2  # 2:1 R/R ratio
            
        return StarSignal(
            pattern_type=pattern_data['pattern_type'],
            confidence=pattern_data['confidence'],
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            timestamp=third_candle.timestamp,
            candles_involved=candles,
            star_gap_size=pattern_data['star_gap_size'],
            penetration_percentage=pattern_data['penetration_percentage'],
            strength=pattern_data['strength']
        )
        
    def validate_pattern(self, pattern_data: Dict[str, Any]) -> bool:
        """Validate the detected pattern"""
        return self.validator.validate_star_pattern(pattern_data)
