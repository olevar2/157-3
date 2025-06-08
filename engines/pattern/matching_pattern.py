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
Matching Low and Matching High Pattern Detection Engine

Matching Low (Bullish reversal):
- Two or more candles with the same or very similar low prices
- Indicates strong support level
- Often occurs at the end of a downtrend

Matching High (Bearish reversal):
- Two or more candles with the same or very similar high prices
- Indicates strong resistance level
- Often occurs at the end of an uptrend
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
# from ..base_pattern import BasePatternEngine  # Fixed import
# from ...models.market_data import OHLCV  # Fixed import
# from ...utils.pattern_validation import PatternValidator  # Fixed import


@dataclass
class MatchingSignal:
    """Signal data for Matching Low/High patterns"""
    pattern_type: str  # 'matching_low' or 'matching_high'
    confidence: float
    entry_price: float
    stop_loss: float
    target_price: float
    timestamp: str
    candles_involved: List[OHLCV]
    matching_level: float
    price_deviation: float
    support_resistance_strength: float
    strength: str  # 'weak', 'moderate', 'strong'


class MatchingPatternEngine(BasePatternEngine):
    """
    Matching Low and Matching High Pattern Detection and Signal Generation Engine
    
    Detects both bullish and bearish matching patterns with support/resistance validation
    """
    
    def __init__(self, max_deviation_percentage: float = 0.1, min_candles: int = 2, max_candles: int = 5):
        super().__init__()
        self.max_deviation_percentage = max_deviation_percentage  # Maximum allowed price deviation
        self.min_candles = min_candles  # Minimum number of candles for pattern
        self.max_candles = max_candles  # Maximum number of candles to look back

        
    def detect_matching_pattern(self, candles: List[OHLCV]) -> Optional[Dict[str, Any]]:
        """
        Detect Matching Low/High pattern in the given candles
        
        Args:
            candles: List of OHLCV data (minimum 2 candles needed)
            
        Returns:
            Pattern detection result or None if no pattern found
        """
        if len(candles) < self.min_candles:
            return None
            
        # Check for matching low pattern
        matching_low_result = self._detect_matching_low(candles)
        if matching_low_result:
            return matching_low_result
            
        # Check for matching high pattern
        matching_high_result = self._detect_matching_high(candles)
        if matching_high_result:
            return matching_high_result
            
        return None
        
    def _detect_matching_low(self, candles: List[OHLCV]) -> Optional[Dict[str, Any]]:
        """Detect matching low pattern"""
        # Look back at most max_candles
        lookback_candles = candles[-self.max_candles:] if len(candles) >= self.max_candles else candles
        
        if len(lookback_candles) < self.min_candles:
            return None
            
        # Find potential matching lows
        matching_groups = self._find_matching_lows(lookback_candles)
        
        if not matching_groups:
            return None
            
        # Use the largest matching group
        best_group = max(matching_groups, key=len)
        
        if len(best_group['candles']) < self.min_candles:
            return None
            
        # Calculate pattern metrics
        matching_level = best_group['level']
        price_deviation = best_group['deviation']
        
        # Calculate support strength
        support_strength = self._calculate_support_strength(best_group['candles'], matching_level)
        
        # Validate that recent candles show potential reversal
        latest_candle = candles[-1]
        if not self._is_potential_bullish_reversal(latest_candle, matching_level):
            return None
            
        # Calculate pattern strength
        strength = self._calculate_pattern_strength(len(best_group['candles']), price_deviation, support_strength)
        
        return {
            'pattern_type': 'matching_low',
            'confidence': self._calculate_confidence(best_group['candles'], price_deviation, support_strength),
            'matching_level': matching_level,
            'price_deviation': price_deviation,
            'support_resistance_strength': support_strength,
            'strength': strength,
            'candles': best_group['candles']
        }
        
    def _detect_matching_high(self, candles: List[OHLCV]) -> Optional[Dict[str, Any]]:
        """Detect matching high pattern"""
        # Look back at most max_candles
        lookback_candles = candles[-self.max_candles:] if len(candles) >= self.max_candles else candles
        
        if len(lookback_candles) < self.min_candles:
            return None
            
        # Find potential matching highs
        matching_groups = self._find_matching_highs(lookback_candles)
        
        if not matching_groups:
            return None
            
        # Use the largest matching group
        best_group = max(matching_groups, key=len)
        
        if len(best_group['candles']) < self.min_candles:
            return None
            
        # Calculate pattern metrics
        matching_level = best_group['level']
        price_deviation = best_group['deviation']
        
        # Calculate resistance strength
        resistance_strength = self._calculate_resistance_strength(best_group['candles'], matching_level)
        
        # Validate that recent candles show potential reversal
        latest_candle = candles[-1]
        if not self._is_potential_bearish_reversal(latest_candle, matching_level):
            return None
            
        # Calculate pattern strength
        strength = self._calculate_pattern_strength(len(best_group['candles']), price_deviation, resistance_strength)
        
        return {
            'pattern_type': 'matching_high',
            'confidence': self._calculate_confidence(best_group['candles'], price_deviation, resistance_strength),
            'matching_level': matching_level,
            'price_deviation': price_deviation,
            'support_resistance_strength': resistance_strength,
            'strength': strength,
            'candles': best_group['candles']
        }
        
    def _find_matching_lows(self, candles: List[OHLCV]) -> List[Dict[str, Any]]:
        """Find groups of candles with matching lows"""
        matching_groups = []
        
        for i in range(len(candles)):
            base_candle = candles[i]
            base_low = base_candle.low
            matching_candles = [base_candle]
            
            # Find other candles with similar lows
            for j in range(i + 1, len(candles)):
                other_candle = candles[j]
                deviation = abs(other_candle.low - base_low) / base_low * 100
                
                if deviation <= self.max_deviation_percentage:
                    matching_candles.append(other_candle)
                    
            if len(matching_candles) >= self.min_candles:
                # Calculate average level and max deviation
                lows = [candle.low for candle in matching_candles]
                avg_level = sum(lows) / len(lows)
                max_deviation = max(abs(low - avg_level) / avg_level * 100 for low in lows)
                
                matching_groups.append({
                    'candles': matching_candles,
                    'level': avg_level,
                    'deviation': max_deviation
                })
                
        return matching_groups
        
    def _find_matching_highs(self, candles: List[OHLCV]) -> List[Dict[str, Any]]:
        """Find groups of candles with matching highs"""
        matching_groups = []
        
        for i in range(len(candles)):
            base_candle = candles[i]
            base_high = base_candle.high
            matching_candles = [base_candle]
            
            # Find other candles with similar highs
            for j in range(i + 1, len(candles)):
                other_candle = candles[j]
                deviation = abs(other_candle.high - base_high) / base_high * 100
                
                if deviation <= self.max_deviation_percentage:
                    matching_candles.append(other_candle)
                    
            if len(matching_candles) >= self.min_candles:
                # Calculate average level and max deviation
                highs = [candle.high for candle in matching_candles]
                avg_level = sum(highs) / len(highs)
                max_deviation = max(abs(high - avg_level) / avg_level * 100 for high in highs)
                
                matching_groups.append({
                    'candles': matching_candles,
                    'level': avg_level,
                    'deviation': max_deviation
                })
                
        return matching_groups
        
    def _calculate_support_strength(self, candles: List[OHLCV], level: float) -> float:
        """Calculate strength of support level"""
        # Factors: number of touches, volume (if available), price action around level
        touch_count = len(candles)
        
        # Calculate how well the level held (lower deviation = stronger support)
        deviations = [abs(candle.low - level) / level * 100 for candle in candles]
        avg_deviation = sum(deviations) / len(deviations)
        deviation_score = max(0, 1.0 - (avg_deviation / self.max_deviation_percentage))
        
        # Calculate touch frequency bonus
        touch_bonus = min(1.0, touch_count / 5.0)  # Max bonus at 5 touches
        
        return (deviation_score + touch_bonus) / 2
        
    def _calculate_resistance_strength(self, candles: List[OHLCV], level: float) -> float:
        """Calculate strength of resistance level"""
        # Similar to support strength calculation
        touch_count = len(candles)
        
        # Calculate how well the level held
        deviations = [abs(candle.high - level) / level * 100 for candle in candles]
        avg_deviation = sum(deviations) / len(deviations)
        deviation_score = max(0, 1.0 - (avg_deviation / self.max_deviation_percentage))
        
        # Calculate touch frequency bonus
        touch_bonus = min(1.0, touch_count / 5.0)  # Max bonus at 5 touches
        
        return (deviation_score + touch_bonus) / 2
        
    def _is_potential_bullish_reversal(self, candle: OHLCV, support_level: float) -> bool:
        """Check if latest candle shows potential bullish reversal from support"""
        # Candle should touch or be near the support level
        if candle.low > support_level * 1.01:  # More than 1% above support
            return False
            
        # Candle should close above the support level (showing bounce)
        return candle.close > support_level
        
    def _is_potential_bearish_reversal(self, candle: OHLCV, resistance_level: float) -> bool:
        """Check if latest candle shows potential bearish reversal from resistance"""
        # Candle should touch or be near the resistance level
        if candle.high < resistance_level * 0.99:  # More than 1% below resistance
            return False
            
        # Candle should close below the resistance level (showing rejection)
        return candle.close < resistance_level
        
    def _calculate_pattern_strength(self, touch_count: int, deviation: float, sr_strength: float) -> str:
        """Calculate pattern strength based on touches, deviation, and support/resistance strength"""
        if touch_count >= 4 and deviation <= 0.05 and sr_strength >= 0.8:
            return 'strong'
        elif touch_count >= 3 and deviation <= 0.08 and sr_strength >= 0.6:
            return 'moderate'
        else:
            return 'weak'
            
    def _calculate_confidence(self, candles: List[OHLCV], deviation: float, sr_strength: float) -> float:
        """Calculate pattern confidence score"""
        confidence = 0.4  # Base confidence
        
        # Add confidence based on number of touches
        touch_count = len(candles)
        if touch_count >= 4:
            confidence += 0.3
        elif touch_count >= 3:
            confidence += 0.2
        else:
            confidence += 0.1
            
        # Add confidence based on price precision
        if deviation <= 0.05:
            confidence += 0.2
        elif deviation <= 0.08:
            confidence += 0.15
        else:
            confidence += 0.1
            
        # Add confidence based on support/resistance strength
        if sr_strength >= 0.8:
            confidence += 0.2
        elif sr_strength >= 0.6:
            confidence += 0.15
        else:
            confidence += 0.1
            
        return min(confidence, 0.9)  # Cap at 90%
        
    def generate_signal(self, pattern_data: Dict[str, Any]) -> MatchingSignal:
        """Generate trading signal from pattern detection"""
        candles = pattern_data['candles']
        latest_candle = max(candles, key=lambda c: c.timestamp)  # Get most recent candle
        matching_level = pattern_data['matching_level']
        
        if pattern_data['pattern_type'] == 'matching_low':
            entry_price = latest_candle.close
            stop_loss = matching_level * 0.995  # Just below support
            target_price = entry_price + (entry_price - stop_loss) * 2  # 2:1 R/R ratio
        else:  # matching_high
            entry_price = latest_candle.close
            stop_loss = matching_level * 1.005  # Just above resistance
            target_price = entry_price - (stop_loss - entry_price) * 2  # 2:1 R/R ratio
            
        return MatchingSignal(
            pattern_type=pattern_data['pattern_type'],
            confidence=pattern_data['confidence'],
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            timestamp=latest_candle.timestamp,
            candles_involved=candles,
            matching_level=matching_level,
            price_deviation=pattern_data['price_deviation'],
            support_resistance_strength=pattern_data['support_resistance_strength'],
            strength=pattern_data['strength']
        )
        
    def validate_pattern(self, pattern_data: Dict[str, Any]) -> bool:
        """Validate the detected pattern"""
        return self.validator.validate_matching_pattern(pattern_data)
