# -*- coding: utf-8 -*-
"""
Spinning Top Pattern Detector
Platform3 Enhanced Technical Analysis Engine

Detects Spinning Top candlestick patterns, which indicate market indecision
and potential trend reversals.

Pattern Characteristics:
- Small real body (10-30% of total range)
- Upper and lower shadows of similar length
- Shadows longer than body
- Can appear in any trend as reversal signal
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

# Fix imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from engines.indicator_base import IndicatorBase


@dataclass
class SpinningTopPattern:
    """Represents a detected Spinning Top pattern"""
    position: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    body_ratio: float
    upper_shadow: float
    lower_shadow: float
    shadow_balance: float
    strength: float
    trend_context: str


class SpinningTopDetector(IndicatorBase):
    """
    Detects Spinning Top candlestick patterns
    """
    
    def __init__(self,
                 min_body_ratio: float = 0.1,
                 max_body_ratio: float = 0.3,
                 min_shadow_body_ratio: float = 0.5,
                 max_shadow_imbalance: float = 2.0,
                 trend_period: int = 10):
        """
        Initialize Spinning Top detector
        
        Args:
            min_body_ratio: Minimum body/range ratio (default 0.1)
            max_body_ratio: Maximum body/range ratio (default 0.3)
            min_shadow_body_ratio: Minimum shadow/body ratio (default 0.5)
            max_shadow_imbalance: Maximum ratio between shadows (default 2.0)
            trend_period: Periods for trend analysis (default 10)
        """
        super().__init__()
        
        self.min_body_ratio = min_body_ratio
        self.max_body_ratio = max_body_ratio
        self.min_shadow_body_ratio = min_shadow_body_ratio
        self.max_shadow_imbalance = max_shadow_imbalance
        self.trend_period = trend_period
        
        self.patterns: List[SpinningTopPattern] = []
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Detect Spinning Top patterns
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with detection results
        """
        try:
            # Validate data
            required_columns = ['open', 'high', 'low', 'close']
            self._validate_data(data, required_columns)
            
            # Clear previous results
            self.patterns.clear()
            
            # Extract data
            opens = data['open'].values
            highs = data['high'].values
            lows = data['low'].values
            closes = data['close'].values
            
            # Detect patterns
            for i in range(self.trend_period, len(data)):
                pattern = self._check_spinning_top(
                    i, opens, highs, lows, closes
                )
                if pattern:
                    self.patterns.append(pattern)
            
            return self._create_results()
            
        except Exception as e:
            self.logger.error(f"Error detecting Spinning Top patterns: {e}")
            raise
    
    def _check_spinning_top(self, idx: int, opens: np.ndarray, highs: np.ndarray,
                          lows: np.ndarray, closes: np.ndarray) -> Optional[SpinningTopPattern]:
        """Check if candle forms Spinning Top pattern"""
        open_price = opens[idx]
        high = highs[idx]
        low = lows[idx]
        close = closes[idx]
        
        # Calculate components
        body_size = abs(close - open_price)
        total_range = high - low
        
        if total_range == 0:
            return None
        
        # Check body ratio
        body_ratio = body_size / total_range
        if body_ratio < self.min_body_ratio or body_ratio > self.max_body_ratio:
            return None
        
        # Calculate shadows
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low
        
        # Check shadow criteria
        if body_size == 0:
            return None
        
        upper_shadow_ratio = upper_shadow / body_size
        lower_shadow_ratio = lower_shadow / body_size
        
        # Both shadows should be significant
        if upper_shadow_ratio < self.min_shadow_body_ratio:
            return None
        if lower_shadow_ratio < self.min_shadow_body_ratio:
            return None
        
        # Check shadow balance
        if upper_shadow == 0 or lower_shadow == 0:
            return None
        
        shadow_imbalance = max(upper_shadow, lower_shadow) / min(upper_shadow, lower_shadow)
        if shadow_imbalance > self.max_shadow_imbalance:
            return None
        
        # Calculate shadow balance score
        shadow_balance = 1 - (shadow_imbalance - 1) / (self.max_shadow_imbalance - 1)
        
        # Determine trend context
        trend = self._determine_trend(closes, idx)
        
        # Calculate strength
        strength = self._calculate_strength(
            body_ratio, upper_shadow_ratio, lower_shadow_ratio,
            shadow_balance, trend
        )
        
        return SpinningTopPattern(
            position=idx,
            open_price=open_price,
            high_price=high,
            low_price=low,
            close_price=close,
            body_ratio=body_ratio,
            upper_shadow=upper_shadow,
            lower_shadow=lower_shadow,
            shadow_balance=shadow_balance,
            strength=strength,
            trend_context=trend
        )
    
    def _determine_trend(self, closes: np.ndarray, idx: int) -> str:
        """Determine trend at position"""
        if idx < self.trend_period:
            return 'unknown'
        
        recent_closes = closes[idx-self.trend_period:idx]
        x = np.arange(len(recent_closes))
        slope = np.polyfit(x, recent_closes, 1)[0]
        
        avg_price = np.mean(recent_closes)
        relative_slope = slope / avg_price
        
        if relative_slope > 0.001:
            return 'uptrend'
        elif relative_slope < -0.001:
            return 'downtrend'
        else:
            return 'sideways'
    
    def _calculate_strength(self, body_ratio: float, upper_shadow_ratio: float,
                          lower_shadow_ratio: float, shadow_balance: float,
                          trend: str) -> float:
        """Calculate pattern strength"""
        strength = 50.0
        
        # Ideal body ratio around 0.2
        ideal_body_ratio = 0.2
        body_factor = 1 - abs(body_ratio - ideal_body_ratio) / ideal_body_ratio
        strength += body_factor * 20
        
        # Longer shadows = stronger pattern
        avg_shadow_ratio = (upper_shadow_ratio + lower_shadow_ratio) / 2
        if avg_shadow_ratio > 2:
            strength += 15
        elif avg_shadow_ratio > 1:
            strength += 10
        
        # Better balance = stronger pattern
        strength += shadow_balance * 15
        
        # Trend context
        if trend in ['uptrend', 'downtrend']:
            strength += 10  # More significant in trending markets
        
        return min(100, max(0, strength))
    
    def _create_results(self) -> Dict:
        """Create results dictionary"""
        return {
            'patterns': [
                {
                    'position': p.position,
                    'body_ratio': p.body_ratio,
                    'upper_shadow': p.upper_shadow,
                    'lower_shadow': p.lower_shadow,
                    'shadow_balance': p.shadow_balance,
                    'strength': p.strength,
                    'trend': p.trend_context,
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
                'avg_strength': np.mean([p.strength for p in self.patterns]) if self.patterns else 0,
                'trend_distribution': self._get_trend_distribution()
            }
        }
    
    def _get_trend_distribution(self) -> Dict[str, int]:
        """Get distribution of patterns by trend"""
        distribution = {'uptrend': 0, 'downtrend': 0, 'sideways': 0, 'unknown': 0}
        for pattern in self.patterns:
            distribution[pattern.trend_context] += 1
        return distribution
