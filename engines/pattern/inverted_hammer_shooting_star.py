# -*- coding: utf-8 -*-
"""
Inverted Hammer & Shooting Star Pattern Detector
Platform3 Enhanced Technical Analysis Engine

Detects Inverted Hammer (bullish reversal) and Shooting Star (bearish reversal)
candlestick patterns with trend context analysis and strength scoring.

Pattern Characteristics:
- Small real body at lower end of trading range
- Long upper shadow (at least twice the body size)
- Little to no lower shadow
- Inverted Hammer appears in downtrends (bullish signal)
- Shooting Star appears in uptrends (bearish signal)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Fix imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent

from engines.indicator_base import IndicatorBase


@dataclass
class InvertedHammerShootingStarPattern:
    """Represents detected Inverted Hammer or Shooting Star pattern"""
    pattern_type: str  # 'inverted_hammer' or 'shooting_star'
    position: int
    candle_open: float
    candle_high: float
    candle_low: float
    candle_close: float
    strength: float
    confidence: float
    trend_context: str
    volume_confirmed: bool


class InvertedHammerShootingStarDetector(IndicatorBase):
    """
    Detects Inverted Hammer and Shooting Star candlestick patterns
    """
    
    def __init__(self,
                 body_ratio_max: float = 0.35,
                 upper_shadow_ratio_min: float = 2.0,
                 lower_shadow_ratio_max: float = 0.5,
                 trend_period: int = 10,
                 volume_factor: float = 1.2):
        """
        Initialize detector
        
        Args:
            body_ratio_max: Maximum body/range ratio (default 0.35)
            upper_shadow_ratio_min: Minimum upper shadow/body ratio (default 2.0)
            lower_shadow_ratio_max: Maximum lower shadow/body ratio (default 0.5)
            trend_period: Periods for trend analysis (default 10)
            volume_factor: Volume confirmation factor (default 1.2)
        """
        super().__init__()
        
        self.body_ratio_max = body_ratio_max
        self.upper_shadow_ratio_min = upper_shadow_ratio_min
        self.lower_shadow_ratio_max = lower_shadow_ratio_max
        self.trend_period = trend_period
        self.volume_factor = volume_factor
        
        self.patterns: List[InvertedHammerShootingStarPattern] = []
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Detect Inverted Hammer and Shooting Star patterns
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with detection results
        """
        try:
            # Validate data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            self._validate_data(data, required_columns)
            
            if len(data) < self.trend_period + 1:
                raise ValueError(f"Need at least {self.trend_period + 1} candles")
            
            # Clear previous results
            self.patterns.clear()
            
            # Extract data
            opens = data['open'].values
            highs = data['high'].values
            lows = data['low'].values
            closes = data['close'].values
            volumes = data['volume'].values
            
            # Detect patterns
            for i in range(self.trend_period, len(data)):
                pattern = self._check_pattern(
                    i, opens, highs, lows, closes, volumes
                )
                if pattern:
                    self.patterns.append(pattern)
            
            # Analyze results
            return self._create_results()
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {e}")
            raise
    
    def _check_pattern(self, idx: int, opens: np.ndarray, highs: np.ndarray,
                      lows: np.ndarray, closes: np.ndarray, 
                      volumes: np.ndarray) -> Optional[InvertedHammerShootingStarPattern]:
        """Check if candle at index forms pattern"""
        # Get candle data
        open_price = opens[idx]
        high = highs[idx]
        low = lows[idx]
        close = closes[idx]
        volume = volumes[idx]
        
        # Calculate components
        body_size = abs(close - open_price)
        total_range = high - low
        
        if total_range == 0:
            return None
        
        # Calculate shadows
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low
        
        # Check basic criteria
        body_ratio = body_size / total_range
        if body_ratio > self.body_ratio_max:
            return None
        
        # Check shadow criteria
        if body_size == 0:
            return None
        
        upper_shadow_ratio = upper_shadow / body_size
        lower_shadow_ratio = lower_shadow / body_size
        
        if upper_shadow_ratio < self.upper_shadow_ratio_min:
            return None
        
        if lower_shadow_ratio > self.lower_shadow_ratio_max:
            return None
        
        # Determine trend
        trend = self._determine_trend(closes, idx)
        
        # Determine pattern type based on trend
        if trend == 'downtrend':
            pattern_type = 'inverted_hammer'
        elif trend == 'uptrend':
            pattern_type = 'shooting_star'
        else:
            return None  # Pattern needs clear trend
        
        # Check volume confirmation
        avg_volume = np.mean(volumes[idx-5:idx])
        volume_confirmed = volume > avg_volume * self.volume_factor
        
        # Calculate strength and confidence
        strength = self._calculate_strength(
            body_ratio, upper_shadow_ratio, lower_shadow_ratio, trend
        )
        confidence = self._calculate_confidence(
            pattern_type, trend, volume_confirmed
        )
        
        return InvertedHammerShootingStarPattern(
            pattern_type=pattern_type,
            position=idx,
            candle_open=open_price,
            candle_high=high,
            candle_low=low,
            candle_close=close,
            strength=strength,
            confidence=confidence,
            trend_context=trend,
            volume_confirmed=volume_confirmed
        )
    
    def _determine_trend(self, closes: np.ndarray, idx: int) -> str:
        """Determine trend at given position"""
        if idx < self.trend_period:
            return 'unknown'
        
        # Get recent closes
        recent_closes = closes[idx-self.trend_period:idx]
        
        # Simple linear regression
        x = np.arange(len(recent_closes))
        slope = np.polyfit(x, recent_closes, 1)[0]
        
        # Normalize slope
        avg_price = np.mean(recent_closes)
        relative_slope = slope / avg_price
        
        if relative_slope > 0.001:
            return 'uptrend'
        elif relative_slope < -0.001:
            return 'downtrend'
        else:
            return 'sideways'
    
    def _calculate_strength(self, body_ratio: float, upper_shadow_ratio: float,
                          lower_shadow_ratio: float, trend: str) -> float:
        """Calculate pattern strength (0-100)"""
        strength = 50.0
        
        # Smaller body = stronger pattern
        strength += (self.body_ratio_max - body_ratio) / self.body_ratio_max * 20
        
        # Longer upper shadow = stronger pattern
        if upper_shadow_ratio > 3:
            strength += 20
        elif upper_shadow_ratio > 2:
            strength += 10
        
        # Smaller lower shadow = stronger pattern
        strength += (self.lower_shadow_ratio_max - lower_shadow_ratio) / self.lower_shadow_ratio_max * 10
        
        # Clear trend = stronger pattern
        if trend in ['uptrend', 'downtrend']:
            strength += 20
        
        return min(100, max(0, strength))
    
    def _calculate_confidence(self, pattern_type: str, trend: str,
                            volume_confirmed: bool) -> float:
        """Calculate pattern confidence (0-1)"""
        confidence = 0.5
        
        # Pattern-trend alignment
        if (pattern_type == 'inverted_hammer' and trend == 'downtrend') or \
           (pattern_type == 'shooting_star' and trend == 'uptrend'):
            confidence += 0.3
        
        # Volume confirmation
        if volume_confirmed:
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def _create_results(self) -> Dict:
        """Create results dictionary"""
        inverted_hammers = [p for p in self.patterns if p.pattern_type == 'inverted_hammer']
        shooting_stars = [p for p in self.patterns if p.pattern_type == 'shooting_star']
        
        return {
            'patterns': [
                {
                    'type': p.pattern_type,
                    'position': p.position,
                    'strength': p.strength,
                    'confidence': p.confidence,
                    'trend': p.trend_context,
                    'volume_confirmed': p.volume_confirmed,
                    'candle': {
                        'open': p.candle_open,
                        'high': p.candle_high,
                        'low': p.candle_low,
                        'close': p.candle_close
                    }
                }
                for p in self.patterns
            ],
            'summary': {
                'total_patterns': len(self.patterns),
                'inverted_hammers': len(inverted_hammers),
                'shooting_stars': len(shooting_stars),
                'avg_strength': np.mean([p.strength for p in self.patterns]) if self.patterns else 0,
                'avg_confidence': np.mean([p.confidence for p in self.patterns]) if self.patterns else 0
            }
        }
