import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

"""
Kicking Pattern Detection Engine

The Kicking pattern is a strong reversal pattern consisting of two consecutive candles:
- First candle: Strong directional candle (marubozu-like)
- Second candle: Opposite direction marubozu with gap
- Gap between the two candles (no overlap)

Bullish Kicking: Bearish marubozu followed by bullish marubozu with gap up
Bearish Kicking: Bullish marubozu followed by bearish marubozu with gap down
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent

from models.market_data import OHLCV
from engines.base_pattern import BasePatternEngine


@dataclass
class KickingSignal:
    """Signal data for Kicking pattern"""
    pattern_type: str  # 'bullish_kicking' or 'bearish_kicking'
    confidence: float
    entry_price: float
    stop_loss: float
    target_price: float
    timestamp: str
    candles_involved: List[OHLCV]
    gap_size: float
    strength: str  # 'weak', 'moderate', 'strong'


class KickingPatternEngine(BasePatternEngine):
    """
    Kicking Pattern Detection and Signal Generation Engine
    
    Detects both bullish and bearish kicking patterns with proper gap validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, min_gap_percentage: float = 0.5, min_body_ratio: float = 0.7):
        super().__init__(config)
        
        # Initialize from config if provided, otherwise use defaults
        if config:
            self.min_gap_percentage = config.get('min_gap_percentage', min_gap_percentage)
            self.min_body_ratio = config.get('min_body_ratio', min_body_ratio)
        else:
            self.min_gap_percentage = min_gap_percentage
            self.min_body_ratio = min_body_ratio
        
        self.logger.info(f"KickingPatternEngine initialized with min_gap_percentage: {self.min_gap_percentage}, min_body_ratio: {self.min_body_ratio}")

        
    def detect_kicking_pattern(self, candles: List[OHLCV]) -> Optional[Dict[str, Any]]:
        """
        Detect Kicking pattern in the given candles
        
        Args:
            candles: List of OHLCV data (minimum 2 candles needed)
            
        Returns:
            Pattern detection result or None if no pattern found
        """
        if len(candles) < 2:
            return None
            
        # Get the last two candles
        prev_candle = candles[-2]
        curr_candle = candles[-1]
        
        # Check for bullish kicking pattern
        bullish_result = self._detect_bullish_kicking(prev_candle, curr_candle)
        if bullish_result:
            return bullish_result
            
        # Check for bearish kicking pattern
        bearish_result = self._detect_bearish_kicking(prev_candle, curr_candle)
        if bearish_result:
            return bearish_result
            
        return None
        
    def _detect_bullish_kicking(self, prev_candle: OHLCV, curr_candle: OHLCV) -> Optional[Dict[str, Any]]:
        """Detect bullish kicking pattern"""
        # Previous candle should be bearish marubozu-like
        if not self._is_bearish_marubozu_like(prev_candle):
            return None
            
        # Current candle should be bullish marubozu-like
        if not self._is_bullish_marubozu_like(curr_candle):
            return None
            
        # Check for gap up (current low > previous high)
        if curr_candle.low <= prev_candle.high:
            return None
            
        # Calculate gap size
        gap_size = curr_candle.low - prev_candle.high
        gap_percentage = (gap_size / prev_candle.high) * 100
        
        if gap_percentage < self.min_gap_percentage:
            return None
            
        # Calculate pattern strength
        strength = self._calculate_pattern_strength(prev_candle, curr_candle, gap_percentage)
        
        return {
            'pattern_type': 'bullish_kicking',
            'confidence': self._calculate_confidence(prev_candle, curr_candle, gap_percentage),
            'gap_size': gap_size,
            'gap_percentage': gap_percentage,
            'strength': strength,
            'candles': [prev_candle, curr_candle]
        }
        
    def _detect_bearish_kicking(self, prev_candle: OHLCV, curr_candle: OHLCV) -> Optional[Dict[str, Any]]:
        """Detect bearish kicking pattern"""
        # Previous candle should be bullish marubozu-like
        if not self._is_bullish_marubozu_like(prev_candle):
            return None
            
        # Current candle should be bearish marubozu-like
        if not self._is_bearish_marubozu_like(curr_candle):
            return None
            
        # Check for gap down (current high < previous low)
        if curr_candle.high >= prev_candle.low:
            return None
            
        # Calculate gap size
        gap_size = prev_candle.low - curr_candle.high
        gap_percentage = (gap_size / prev_candle.low) * 100
        
        if gap_percentage < self.min_gap_percentage:
            return None
            
        # Calculate pattern strength
        strength = self._calculate_pattern_strength(prev_candle, curr_candle, gap_percentage)
        
        return {
            'pattern_type': 'bearish_kicking',
            'confidence': self._calculate_confidence(prev_candle, curr_candle, gap_percentage),
            'gap_size': gap_size,
            'gap_percentage': gap_percentage,
            'strength': strength,
            'candles': [prev_candle, curr_candle]
        }
        
    def _is_bullish_marubozu_like(self, candle: OHLCV) -> bool:
        """Check if candle is bullish marubozu-like"""
        if candle.close <= candle.open:
            return False
            
        body_size = candle.close - candle.open
        total_range = candle.high - candle.low
        
        if total_range == 0:
            return False
            
        body_ratio = body_size / total_range
        return body_ratio >= self.min_body_ratio
        
    def _is_bearish_marubozu_like(self, candle: OHLCV) -> bool:
        """Check if candle is bearish marubozu-like"""
        if candle.close >= candle.open:
            return False
            
        body_size = candle.open - candle.close
        total_range = candle.high - candle.low
        
        if total_range == 0:
            return False
            
        body_ratio = body_size / total_range
        return body_ratio >= self.min_body_ratio
        
    def _calculate_pattern_strength(self, prev_candle: OHLCV, curr_candle: OHLCV, gap_percentage: float) -> str:
        """Calculate pattern strength based on gap size and candle properties"""
        if gap_percentage >= 2.0:
            return 'strong'
        elif gap_percentage >= 1.0:
            return 'moderate'
        else:
            return 'weak'
            
    def _calculate_confidence(self, prev_candle: OHLCV, curr_candle: OHLCV, gap_percentage: float) -> float:
        """Calculate pattern confidence score"""
        confidence = 0.6  # Base confidence
        
        # Add confidence based on gap size
        if gap_percentage >= 2.0:
            confidence += 0.3
        elif gap_percentage >= 1.0:
            confidence += 0.2
        else:
            confidence += 0.1
            
        # Add confidence based on candle body strength
        prev_body_ratio = abs(prev_candle.close - prev_candle.open) / (prev_candle.high - prev_candle.low)
        curr_body_ratio = abs(curr_candle.close - curr_candle.open) / (curr_candle.high - curr_candle.low)
        
        avg_body_ratio = (prev_body_ratio + curr_body_ratio) / 2
        if avg_body_ratio >= 0.9:
            confidence += 0.1
        elif avg_body_ratio >= 0.8:
            confidence += 0.05
            
        return min(confidence, 0.95)  # Cap at 95%
        
    def generate_signal(self, pattern_data: Dict[str, Any]) -> KickingSignal:
        """Generate trading signal from pattern detection"""
        candles = pattern_data['candles']
        curr_candle = candles[-1]
        
        if pattern_data['pattern_type'] == 'bullish_kicking':
            entry_price = curr_candle.close
            stop_loss = min(candles[0].low, curr_candle.low) * 0.995  # 0.5% buffer
            target_price = entry_price + (entry_price - stop_loss) * 2  # 2:1 R/R ratio
        else:  # bearish_kicking
            entry_price = curr_candle.close
            stop_loss = max(candles[0].high, curr_candle.high) * 1.005  # 0.5% buffer
            target_price = entry_price - (stop_loss - entry_price) * 2  # 2:1 R/R ratio
            
        return KickingSignal(
            pattern_type=pattern_data['pattern_type'],
            confidence=pattern_data['confidence'],
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            timestamp=curr_candle.timestamp,
            candles_involved=candles,
            gap_size=pattern_data['gap_size'],
            strength=pattern_data['strength']
        )
        
    def validate_pattern(self, pattern_data: Dict[str, Any]) -> bool:
        """Validate the detected pattern"""
        return self.validator.validate_kicking_pattern(pattern_data)
