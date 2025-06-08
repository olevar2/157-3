import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

"""
Abandoned Baby Pattern Detection Engine

Abandoned Baby is a rare reversal pattern consisting of three candles with gaps:

Bullish Abandoned Baby (reversal at bottom):
- First candle: Long bearish candle
- Second candle: Doji that gaps down from first candle
- Third candle: Long bullish candle that gaps up from doji

Bearish Abandoned Baby (reversal at top):
- First candle: Long bullish candle
- Second candle: Doji that gaps up from first candle
- Third candle: Long bearish candle that gaps down from doji
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.market_data import OHLCV
from engines.base_pattern import BasePatternEngine


@dataclass
class AbandonedBabySignal:
    """Signal data for Abandoned Baby patterns"""
    pattern_type: str  # 'bullish_abandoned_baby' or 'bearish_abandoned_baby'
    confidence: float
    entry_price: float
    stop_loss: float
    target_price: float
    timestamp: str
    candles_involved: List[OHLCV]
    first_gap_size: float
    second_gap_size: float
    doji_quality: float
    strength: str  # 'weak', 'moderate', 'strong'


class AbandonedBabyPatternEngine(BasePatternEngine):
    """
    Abandoned Baby Pattern Detection and Signal Generation Engine
    
    Detects both bullish and bearish abandoned baby patterns with proper gap and doji validation    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 min_body_ratio: float = 0.6, max_doji_body_ratio: float = 0.1, min_gap_percentage: float = 0.2):
        super().__init__(config=config) # Pass config to BasePatternEngine
        self.min_body_ratio = min_body_ratio  # Minimum body ratio for first and third candles
        self.max_doji_body_ratio = max_doji_body_ratio  # Maximum body ratio for doji candle
        self.min_gap_percentage = min_gap_percentage  # Minimum gap size as % of candle range
        
    def detect_patterns(self, data: List[Dict[str, Any]]) -> List[AbandonedBabySignal]:
        """
        Detects Abandoned Baby patterns from a list of OHLCV-like dictionaries.
        Converts input data to List[OHLCV] before processing.
        """        if data is None or len(data) < 3:
            return []

        # Convert list of dictionaries to list of OHLCV objects
        ohlcv_candles: List[OHLCV] = []
        for item in data:
            try:
                # Ensure all necessary keys are present for OHLCV conversion
                # Adjust key names if they differ in the input dicts
                ohlcv_candles.append(OHLCV(
                    timestamp=item['timestamp'], # or item.get('timestamp') or item.get('date')
                    open=float(item['open']),
                    high=float(item['high']),
                    low=float(item['low']),
                    close=float(item['close']),
                    volume=float(item['volume'])
                ))
            except KeyError as e:
                # Handle missing keys, e.g., log an error or skip the item
                # print(f"Skipping item due to missing key: {e} in {item}")
                continue # Or raise an error, or handle more gracefully
            except (ValueError, TypeError) as e:
                # Handle conversion errors
                # print(f"Skipping item due to conversion error: {e} in {item}")
                continue


        if len(ohlcv_candles) < 3:
            return []

        signals = []
        # Iterate through the candles, considering 3 at a time
        for i in range(len(ohlcv_candles) - 2):
            current_batch = ohlcv_candles[i : i + 3]
            pattern_result = self.detect_abandoned_baby_pattern(current_batch)
            if pattern_result:
                signal = self.generate_signal(pattern_result)
                signals.append(signal)
        return signals

    def detect_abandoned_baby_pattern(self, candles: List[OHLCV]) -> Optional[Dict[str, Any]]:
        """
        Detect Abandoned Baby pattern in the given candles
        
        Args:
            candles: List of OHLCV data (minimum 3 candles needed)
            
        Returns:
            Pattern detection result or None if no pattern found
        """
        if len(candles) < 3:
            return None
            
        # Get the last three candles
        first_candle = candles[-3]
        doji_candle = candles[-2]
        third_candle = candles[-1]
        
        # Check for bullish abandoned baby pattern
        bullish_result = self._detect_bullish_abandoned_baby(first_candle, doji_candle, third_candle)
        if bullish_result:
            return bullish_result
            
        # Check for bearish abandoned baby pattern
        bearish_result = self._detect_bearish_abandoned_baby(first_candle, doji_candle, third_candle)
        if bearish_result:
            return bearish_result
            
        return None
        
    def _detect_bullish_abandoned_baby(self, first: OHLCV, doji: OHLCV, third: OHLCV) -> Optional[Dict[str, Any]]:
        """Detect bullish abandoned baby pattern"""
        # First candle should be long bearish
        if not self._is_long_bearish(first):
            return None
            
        # Middle candle should be doji-like
        if not self._is_doji_like(doji):
            return None
            
        # Third candle should be long bullish
        if not self._is_long_bullish(third):
            return None
            
        # Check gap down between first and doji
        first_gap = first.low - doji.high
        if first_gap <= 0:  # No gap or overlap
            return None
            
        # Check gap up between doji and third
        second_gap = third.low - doji.high
        if second_gap <= 0:  # No gap or overlap
            return None
            
        # Validate gap sizes
        first_range = first.high - first.low
        first_gap_percentage = (first_gap / first_range) * 100 if first_range > 0 else 0
        
        third_range = third.high - third.low
        second_gap_percentage = (second_gap / third_range) * 100 if third_range > 0 else 0
        
        if first_gap_percentage < self.min_gap_percentage or second_gap_percentage < self.min_gap_percentage:
            return None
            
        # Calculate doji quality
        doji_quality = self._calculate_doji_quality(doji)
        
        # Calculate pattern strength
        strength = self._calculate_pattern_strength(first_gap_percentage, second_gap_percentage, doji_quality)
        
        return {
            'pattern_type': 'bullish_abandoned_baby',
            'confidence': self._calculate_confidence(first, doji, third, first_gap_percentage, second_gap_percentage, doji_quality),
            'first_gap_size': first_gap,
            'second_gap_size': second_gap,
            'first_gap_percentage': first_gap_percentage,
            'second_gap_percentage': second_gap_percentage,
            'doji_quality': doji_quality,
            'strength': strength,
            'candles': [first, doji, third]
        }
        
    def _detect_bearish_abandoned_baby(self, first: OHLCV, doji: OHLCV, third: OHLCV) -> Optional[Dict[str, Any]]:
        """Detect bearish abandoned baby pattern"""
        # First candle should be long bullish
        if not self._is_long_bullish(first):
            return None
            
        # Middle candle should be doji-like
        if not self._is_doji_like(doji):
            return None
            
        # Third candle should be long bearish
        if not self._is_long_bearish(third):
            return None
            
        # Check gap up between first and doji
        first_gap = doji.low - first.high
        if first_gap <= 0:  # No gap or overlap
            return None
            
        # Check gap down between doji and third
        second_gap = doji.low - third.high
        if second_gap <= 0:  # No gap or overlap
            return None
            
        # Validate gap sizes
        first_range = first.high - first.low
        first_gap_percentage = (first_gap / first_range) * 100 if first_range > 0 else 0
        
        third_range = third.high - third.low
        second_gap_percentage = (second_gap / third_range) * 100 if third_range > 0 else 0
        
        if first_gap_percentage < self.min_gap_percentage or second_gap_percentage < self.min_gap_percentage:
            return None
            
        # Calculate doji quality
        doji_quality = self._calculate_doji_quality(doji)
        
        # Calculate pattern strength
        strength = self._calculate_pattern_strength(first_gap_percentage, second_gap_percentage, doji_quality)
        
        return {
            'pattern_type': 'bearish_abandoned_baby',
            'confidence': self._calculate_confidence(first, doji, third, first_gap_percentage, second_gap_percentage, doji_quality),
            'first_gap_size': first_gap,
            'second_gap_size': second_gap,
            'first_gap_percentage': first_gap_percentage,
            'second_gap_percentage': second_gap_percentage,
            'doji_quality': doji_quality,
            'strength': strength,
            'candles': [first, doji, third]
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
        
    def _is_doji_like(self, candle: OHLCV) -> bool:
        """Check if candle is doji-like (very small body)"""
        body_size = abs(candle.close - candle.open)
        total_range = candle.high - candle.low
        
        if total_range == 0:
            return True  # Perfect doji
            
        body_ratio = body_size / total_range
        return body_ratio <= self.max_doji_body_ratio
        
    def _calculate_doji_quality(self, doji: OHLCV) -> float:
        """Calculate quality of doji (lower body ratio = higher quality)"""
        body_size = abs(doji.close - doji.open)
        total_range = doji.high - doji.low
        
        if total_range == 0:
            return 1.0  # Perfect doji
            
        body_ratio = body_size / total_range
        return max(0.0, 1.0 - (body_ratio / self.max_doji_body_ratio))
        
    def _calculate_pattern_strength(self, first_gap_percentage: float, second_gap_percentage: float, doji_quality: float) -> str:
        """Calculate pattern strength based on gaps and doji quality"""
        avg_gap = (first_gap_percentage + second_gap_percentage) / 2
        
        if avg_gap >= 1.0 and doji_quality >= 0.8:
            return 'strong'
        elif avg_gap >= 0.5 and doji_quality >= 0.6:
            return 'moderate'
        else:
            return 'weak'
            
    def _calculate_confidence(self, first: OHLCV, doji: OHLCV, third: OHLCV,
                            first_gap_percentage: float, second_gap_percentage: float, doji_quality: float) -> float:
        """Calculate pattern confidence score"""
        confidence = 0.4  # Base confidence (lower due to rarity)
        
        # Add confidence based on gap sizes
        avg_gap = (first_gap_percentage + second_gap_percentage) / 2
        if avg_gap >= 1.0:
            confidence += 0.25
        elif avg_gap >= 0.5:
            confidence += 0.2
        else:
            confidence += 0.15
            
        # Add confidence based on doji quality
        if doji_quality >= 0.8:
            confidence += 0.2
        elif doji_quality >= 0.6:
            confidence += 0.15
        else:
            confidence += 0.1
            
        # Add confidence based on first and third candle strength
        first_body_ratio = abs(first.close - first.open) / (first.high - first.low)
        third_body_ratio = abs(third.close - third.open) / (third.high - third.low)
        
        avg_body_strength = (first_body_ratio + third_body_ratio) / 2
        if avg_body_strength >= 0.8:
            confidence += 0.15
        elif avg_body_strength >= 0.7:
            confidence += 0.1
        else:
            confidence += 0.05
            
        return min(confidence, 0.9)  # Cap at 90% due to pattern rarity
        
    def generate_signal(self, pattern_data: Dict[str, Any]) -> AbandonedBabySignal:
        """Generate trading signal from pattern detection"""
        candles = pattern_data['candles']
        third_candle = candles[-1]
        
        if pattern_data['pattern_type'] == 'bullish_abandoned_baby':
            entry_price = third_candle.close
            stop_loss = min(candle.low for candle in candles) * 0.995  # 0.5% buffer
            target_price = entry_price + (entry_price - stop_loss) * 3  # 3:1 R/R ratio (stronger pattern)
        else:  # bearish_abandoned_baby
            entry_price = third_candle.close
            stop_loss = max(candle.high for candle in candles) * 1.005  # 0.5% buffer
            target_price = entry_price - (stop_loss - entry_price) * 3  # 3:1 R/R ratio (stronger pattern)
            
        return AbandonedBabySignal(
            pattern_type=pattern_data['pattern_type'],
            confidence=pattern_data['confidence'],
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            timestamp=third_candle.timestamp,
            candles_involved=candles,
            first_gap_size=pattern_data['first_gap_size'],
            second_gap_size=pattern_data['second_gap_size'],
            doji_quality=pattern_data['doji_quality'],
            strength=pattern_data['strength']
        )
        
    def validate_pattern(self, pattern_data: Dict[str, Any]) -> bool:
        """Validate the detected pattern"""
        return self.validator.validate_abandoned_baby_pattern(pattern_data)
