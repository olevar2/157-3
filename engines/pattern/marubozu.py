# -*- coding: utf-8 -*-
"""
Marubozu Pattern Detector
Platform3 Enhanced Technical Analysis Engine

Detects Bullish and Bearish Marubozu candlestick patterns, which indicate
strong momentum and continuation signals.

Pattern Characteristics:
- Very long real body (close to 100% of range)
- Little or no upper shadow
- Little or no lower shadow
- Bullish Marubozu: Opens at low, closes at high
- Bearish Marubozu: Opens at high, closes at low
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

# Fix imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent

from engines.indicator_base import IndicatorBase


@dataclass
class MarubozuPattern:
    """Represents a detected Marubozu pattern"""
    pattern_type: str  # 'bullish_marubozu' or 'bearish_marubozu'
    position: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    body_ratio: float
    strength: float
    volume: float
    trend_alignment: bool


class MarubozuDetector(IndicatorBase):
    """
    Detects Bullish and Bearish Marubozu candlestick patterns
    """
    
    def __init__(self,
                 min_body_ratio: float = 0.90,
                 max_shadow_ratio: float = 0.05,
                 trend_period: int = 10):
        """
        Initialize Marubozu detector
        
        Args:
            min_body_ratio: Minimum body/range ratio (default 0.90)
            max_shadow_ratio: Maximum shadow/range ratio (default 0.05)
            trend_period: Periods for trend analysis (default 10)
        """
        super().__init__()
        
        self.min_body_ratio = min_body_ratio
        self.max_shadow_ratio = max_shadow_ratio
        self.trend_period = trend_period
        
        self.patterns: List[MarubozuPattern] = []
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Detect Marubozu patterns in candlestick data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with detection results
        """
        try:
            # Validate data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            self._validate_data(data, required_columns)
            
            # Clear previous results
            self.patterns.clear()
            
            # Extract data
            opens = data['open'].values
            highs = data['high'].values
            lows = data['low'].values
            closes = data['close'].values
            volumes = data['volume'].values
            
            # Detect patterns
            for i in range(len(data)):
                pattern = self._check_marubozu(
                    i, opens, highs, lows, closes, volumes
                )
                if pattern:
                    self.patterns.append(pattern)
            
            return self._create_results()
            
        except Exception as e:
            self.logger.error(f"Error detecting Marubozu patterns: {e}")
            raise
    
    def _check_marubozu(self, idx: int, opens: np.ndarray, highs: np.ndarray,
                       lows: np.ndarray, closes: np.ndarray,
                       volumes: np.ndarray) -> Optional[MarubozuPattern]:
        """Check if candle at index forms Marubozu pattern"""
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
        
        # Calculate body ratio
        body_ratio = body_size / total_range
        if body_ratio < self.min_body_ratio:
            return None
        
        # Calculate shadows
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low
        
        # Check shadow criteria
        upper_shadow_ratio = upper_shadow / total_range
        lower_shadow_ratio = lower_shadow / total_range
        
        if upper_shadow_ratio > self.max_shadow_ratio:
            return None
        if lower_shadow_ratio > self.max_shadow_ratio:
            return None
        
        # Determine pattern type
        is_bullish = close > open_price
        pattern_type = 'bullish_marubozu' if is_bullish else 'bearish_marubozu'
        
        # Check trend alignment
        trend_aligned = self._check_trend_alignment(
            closes, idx, is_bullish
        )
        
        # Calculate strength
        strength = self._calculate_strength(
            body_ratio, upper_shadow_ratio, lower_shadow_ratio, trend_aligned
        )
        
        return MarubozuPattern(
            pattern_type=pattern_type,
            position=idx,
            open_price=open_price,
            high_price=high,
            low_price=low,
            close_price=close,
            body_ratio=body_ratio,
            strength=strength,
            volume=volume,
            trend_alignment=trend_aligned
        )
    
    def _check_trend_alignment(self, closes: np.ndarray, idx: int,
                              is_bullish: bool) -> bool:
        """Check if pattern aligns with trend"""
        if idx < self.trend_period:
            return False
        
        # Get trend
        recent_closes = closes[idx-self.trend_period:idx]
        slope = np.polyfit(np.arange(len(recent_closes)), recent_closes, 1)[0]
        
        # Check alignment
        if is_bullish and slope > 0:
            return True
        elif not is_bullish and slope < 0:
            return True
        
        return False
    
    def _calculate_strength(self, body_ratio: float, upper_shadow_ratio: float,
                          lower_shadow_ratio: float, trend_aligned: bool) -> float:
        """Calculate pattern strength (0-100)"""
        strength = 70.0
        
        # Higher body ratio = stronger pattern
        strength += (body_ratio - self.min_body_ratio) / (1 - self.min_body_ratio) * 20
        
        # Smaller shadows = stronger pattern
        shadow_factor = 1 - (upper_shadow_ratio + lower_shadow_ratio) / (2 * self.max_shadow_ratio)
        strength += shadow_factor * 5
        
        # Trend alignment bonus
        if trend_aligned:
            strength += 5
        
        return min(100, max(0, strength))
    
    def _create_results(self) -> Dict:
        """Create results dictionary"""
        bullish_patterns = [p for p in self.patterns if p.pattern_type == 'bullish_marubozu']
        bearish_patterns = [p for p in self.patterns if p.pattern_type == 'bearish_marubozu']
        
        return {
            'patterns': [
                {
                    'type': p.pattern_type,
                    'position': p.position,
                    'body_ratio': p.body_ratio,
                    'strength': p.strength,
                    'volume': p.volume,
                    'trend_aligned': p.trend_alignment,
                    'candle': {
                        'open': p.open_price,
                        'high': p.high_price,
                        'low': p.low_price,
                        'close': p.close_price
                    }
                }
                for p in self.patterns
            ],
            'summary': {
                'total_patterns': len(self.patterns),
                'bullish_marubozu': len(bullish_patterns),
                'bearish_marubozu': len(bearish_patterns),
                'avg_strength': np.mean([p.strength for p in self.patterns]) if self.patterns else 0,
                'trend_aligned_count': sum(1 for p in self.patterns if p.trend_alignment)
            }
        }
