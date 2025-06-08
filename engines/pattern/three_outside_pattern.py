import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

"""
Three Outside Up/Down Pattern Detection Engine

Three Outside Up (Bullish reversal):
- First candle: Bearish candle
- Second candle: Bullish engulfing candle (completely engulfs first candle)
- Third candle: Bullish candle that closes higher than second candle

Three Outside Down (Bearish reversal):
- First candle: Bullish candle
- Second candle: Bearish engulfing candle (completely engulfs first candle)
- Third candle: Bearish candle that closes lower than second candle
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from engines.base_pattern import BasePatternEngine
from models.market_data import OHLCV


@dataclass
class ThreeOutsideSignal:
    """Signal data for Three Outside Up/Down patterns"""
    pattern_type: str  # 'three_outside_up' or 'three_outside_down'
    confidence: float
    entry_price: float
    stop_loss: float
    target_price: float
    timestamp: str
    candles_involved: List[OHLCV]
    engulfing_ratio: float
    continuation_strength: float
    strength: str  # 'weak', 'moderate', 'strong'


class ThreeOutsidePatternEngine(BasePatternEngine):
    """
    Three Outside Up/Down Pattern Detection and Signal Generation Engine
    
    Detects both bullish and bearish three-candle reversal patterns with engulfing confirmation
    """
    
    def __init__(self, min_engulfing_ratio: float = 1.2, min_continuation: float = 0.1):
        super().__init__()
        self.min_engulfing_ratio = min_engulfing_ratio  # Minimum size ratio for engulfing candle
        self.min_continuation = min_continuation  # Minimum continuation distance as % of second candle
        # self.validator = PatternValidator() # Temporarily disabled
        
    def detect_three_outside_pattern(self, candles: List[OHLCV]) -> Optional[Dict[str, Any]]:
        """
        Detect Three Outside Up/Down pattern in the given candles
        
        Args:
            candles: List of OHLCV data (minimum 3 candles needed)
            
        Returns:
            Pattern detection result or None if no pattern found
        """
        if len(candles) < 3:
            return None
            
        # Get the last three candles
        first_candle = candles[-3]
        second_candle = candles[-2]
        third_candle = candles[-1]
        
        # Check for three outside up pattern
        outside_up_result = self._detect_three_outside_up(first_candle, second_candle, third_candle)
        if outside_up_result:
            return outside_up_result
            
        # Check for three outside down pattern
        outside_down_result = self._detect_three_outside_down(first_candle, second_candle, third_candle)
        if outside_down_result:
            return outside_down_result
            
        return None
        
    def _detect_three_outside_up(self, first: OHLCV, second: OHLCV, third: OHLCV) -> Optional[Dict[str, Any]]:
        """Detect three outside up pattern"""
        # First candle should be bearish
        if first.close >= first.open:
            return None
            
        # Second candle should be bullish engulfing
        if not self._is_bullish_engulfing(first, second):
            return None
            
        # Third candle should be bullish and close higher than second candle
        if third.close <= third.open:  # Must be bullish
            return None
            
        if third.close <= second.close:  # Must continue higher
            return None
            
        # Calculate engulfing ratio
        first_body_size = first.open - first.close
        second_body_size = second.close - second.open
        engulfing_ratio = second_body_size / first_body_size if first_body_size > 0 else 0
        
        if engulfing_ratio < self.min_engulfing_ratio:
            return None
            
        # Calculate continuation strength
        continuation_distance = third.close - second.close
        second_range = second.high - second.low
        continuation_strength = (continuation_distance / second_range) * 100 if second_range > 0 else 0
        
        if continuation_strength < self.min_continuation:
            return None
            
        # Calculate pattern strength
        strength = self._calculate_pattern_strength(engulfing_ratio, continuation_strength)
        
        return {
            'pattern_type': 'three_outside_up',
            'confidence': self._calculate_confidence(first, second, third, engulfing_ratio, continuation_strength),
            'engulfing_ratio': engulfing_ratio,
            'continuation_strength': continuation_strength,
            'strength': strength,
            'candles': [first, second, third]
        }
        
    def _detect_three_outside_down(self, first: OHLCV, second: OHLCV, third: OHLCV) -> Optional[Dict[str, Any]]:
        """Detect three outside down pattern"""
        # First candle should be bullish
        if first.close <= first.open:
            return None
            
        # Second candle should be bearish engulfing
        if not self._is_bearish_engulfing(first, second):
            return None
            
        # Third candle should be bearish and close lower than second candle
        if third.close >= third.open:  # Must be bearish
            return None
            
        if third.close >= second.close:  # Must continue lower
            return None
            
        # Calculate engulfing ratio
        first_body_size = first.close - first.open
        second_body_size = second.open - second.close
        engulfing_ratio = second_body_size / first_body_size if first_body_size > 0 else 0
        
        if engulfing_ratio < self.min_engulfing_ratio:
            return None
            
        # Calculate continuation strength
        continuation_distance = second.close - third.close
        second_range = second.high - second.low
        continuation_strength = (continuation_distance / second_range) * 100 if second_range > 0 else 0
        
        if continuation_strength < self.min_continuation:
            return None
            
        # Calculate pattern strength
        strength = self._calculate_pattern_strength(engulfing_ratio, continuation_strength)
        
        return {
            'pattern_type': 'three_outside_down',
            'confidence': self._calculate_confidence(first, second, third, engulfing_ratio, continuation_strength),
            'engulfing_ratio': engulfing_ratio,
            'continuation_strength': continuation_strength,
            'strength': strength,
            'candles': [first, second, third]
        }
        
    def _is_bullish_engulfing(self, first: OHLCV, second: OHLCV) -> bool:
        """Check if second candle is bullish engulfing first candle"""
        # Second candle must be bullish
        if second.close <= second.open:
            return False
            
        # Second candle must completely engulf first candle
        if second.open >= first.close or second.close <= first.open:
            return False
            
        # Second candle should also engulf the shadows (complete engulfing)
        if second.low > first.low or second.high < first.high:
            return False
            
        return True
        
    def _is_bearish_engulfing(self, first: OHLCV, second: OHLCV) -> bool:
        """Check if second candle is bearish engulfing first candle"""
        # Second candle must be bearish
        if second.close >= second.open:
            return False
            
        # Second candle must completely engulf first candle
        if second.open <= first.close or second.close >= first.open:
            return False
            
        # Second candle should also engulf the shadows (complete engulfing)
        if second.low > first.low or second.high < first.high:
            return False
            
        return True
        
    def _calculate_pattern_strength(self, engulfing_ratio: float, continuation_strength: float) -> str:
        """Calculate pattern strength based on engulfing characteristics and continuation"""
        if engulfing_ratio >= 2.0 and continuation_strength >= 1.0:
            return 'strong'
        elif engulfing_ratio >= 1.5 and continuation_strength >= 0.5:
            return 'moderate'
        else:
            return 'weak'
            
    def _calculate_confidence(self, first: OHLCV, second: OHLCV, third: OHLCV, 
                            engulfing_ratio: float, continuation_strength: float) -> float:
        """Calculate pattern confidence score"""
        confidence = 0.6  # Base confidence
        
        # Add confidence based on engulfing strength
        if engulfing_ratio >= 2.0:
            confidence += 0.2
        elif engulfing_ratio >= 1.5:
            confidence += 0.15
        else:
            confidence += 0.1
            
        # Add confidence based on continuation strength
        if continuation_strength >= 1.0:
            confidence += 0.15
        elif continuation_strength >= 0.5:
            confidence += 0.1
        else:
            confidence += 0.05
            
        # Add confidence based on second candle body strength
        second_body_size = abs(second.close - second.open)
        second_range = second.high - second.low
        second_body_ratio = second_body_size / second_range if second_range > 0 else 0
        
        if second_body_ratio >= 0.8:
            confidence += 0.1
        elif second_body_ratio >= 0.6:
            confidence += 0.05
            
        # Add confidence based on third candle strength
        third_body_size = abs(third.close - third.open)
        third_range = third.high - third.low
        third_body_ratio = third_body_size / third_range if third_range > 0 else 0
        
        if third_body_ratio >= 0.6:
            confidence += 0.05
            
        return min(confidence, 0.95)  # Cap at 95%
        
    def generate_signal(self, pattern_data: Dict[str, Any]) -> ThreeOutsideSignal:
        """Generate trading signal from pattern detection"""
        candles = pattern_data['candles']
        third_candle = candles[-1]
        
        if pattern_data['pattern_type'] == 'three_outside_up':
            entry_price = third_candle.close
            stop_loss = min(candle.low for candle in candles) * 0.995  # 0.5% buffer
            target_price = entry_price + (entry_price - stop_loss) * 2  # 2:1 R/R ratio
        else:  # three_outside_down
            entry_price = third_candle.close
            stop_loss = max(candle.high for candle in candles) * 1.005  # 0.5% buffer
            target_price = entry_price - (stop_loss - entry_price) * 2  # 2:1 R/R ratio
            
        return ThreeOutsideSignal(
            pattern_type=pattern_data['pattern_type'],
            confidence=pattern_data['confidence'],
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            timestamp=third_candle.timestamp,
            candles_involved=candles,
            engulfing_ratio=pattern_data['engulfing_ratio'],
            continuation_strength=pattern_data['continuation_strength'],
            strength=pattern_data['strength']
        )
        
    def validate_pattern(self, pattern_data: Dict[str, Any]) -> bool:
        """Validate the detected pattern"""
        return self.validator.validate_three_outside_pattern(pattern_data)
