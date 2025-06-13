# -*- coding: utf-8 -*-
"""
Long-legged Doji Pattern Detector
Platform3 Enhanced Technical Analysis Engine

Detects Long-legged Doji patterns, which indicate extreme indecision with
very long shadows on both sides and minimal body.

Pattern Characteristics:
- Extremely small or no real body
- Very long upper and lower shadows
- Shadows at least 3x the body size
- Represents extreme market indecision
- Strong reversal signal in trends
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
class LongLeggedDojiPattern:
    """Represents a detected Long-legged Doji pattern"""
    position: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    body_size: float
    upper_shadow: float
    lower_shadow: float
    body_ratio: float
    shadow_symmetry: float
    strength: float
    trend_context: str


class LongLeggedDojiDetector(IndicatorBase):
    """
    Detects Long-legged Doji candlestick patterns
    """
    
    def __init__(self,
                 max_body_ratio: float = 0.1,
                 min_shadow_body_ratio: float = 3.0,
                 min_total_range: float = 0.005,
                 trend_period: int = 10):
        """
        Initialize Long-legged Doji detector
        
        Args:
            max_body_ratio: Maximum body/range ratio for doji (default 0.1)
            min_shadow_body_ratio: Minimum shadow/body ratio (default 3.0)
            min_total_range: Minimum range as price fraction (default 0.005)
            trend_period: Periods for trend analysis (default 10)
        """
        super().__init__()
        
        self.max_body_ratio = max_body_ratio
        self.min_shadow_body_ratio = min_shadow_body_ratio
        self.min_total_range = min_total_range
        self.trend_period = trend_period
        
        self.patterns: List[LongLeggedDojiPattern] = []
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Detect Long-legged Doji patterns
        
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
                pattern = self._check_long_legged_doji(
                    i, opens, highs, lows, closes
                )
                if pattern:
                    self.patterns.append(pattern)
            
            return self._create_results()
            
        except Exception as e:
            self.logger.error(f"Error detecting Long-legged Doji: {e}")
            raise
    
    def _check_long_legged_doji(self, idx: int, opens: np.ndarray,
                               highs: np.ndarray, lows: np.ndarray,
                               closes: np.ndarray) -> Optional[LongLeggedDojiPattern]:
        """Check if candle forms Long-legged Doji pattern"""
        open_price = opens[idx]
        high = highs[idx]
        low = lows[idx]
        close = closes[idx]
        
        # Calculate components
        body_size = abs(close - open_price)
        total_range = high - low
        
        # Check minimum range requirement
        avg_price = (high + low) / 2
        if total_range < avg_price * self.min_total_range:
            return None
        
        # Check doji criteria
        body_ratio = body_size / total_range if total_range > 0 else 0
        if body_ratio > self.max_body_ratio:
            return None
        
        # Calculate shadows
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low
        
        # For very small bodies, use total range for ratio calculation
        effective_body = max(body_size, total_range * 0.01)
        
        upper_shadow_ratio = upper_shadow / effective_body
        lower_shadow_ratio = lower_shadow / effective_body
        
        # Check long shadow criteria
        if upper_shadow_ratio < self.min_shadow_body_ratio:
            return None
        if lower_shadow_ratio < self.min_shadow_body_ratio:
            return None
        
        # Calculate shadow symmetry
        shadow_symmetry = self._calculate_shadow_symmetry(
            upper_shadow, lower_shadow
        )
        
        # Determine trend context
        trend = self._determine_trend(closes, idx)
        
        # Calculate strength
        strength = self._calculate_strength(
            body_ratio, upper_shadow_ratio, lower_shadow_ratio,
            shadow_symmetry, trend
        )
        
        return LongLeggedDojiPattern(
            position=idx,
            open_price=open_price,
            high_price=high,
            low_price=low,
            close_price=close,
            body_size=body_size,
            upper_shadow=upper_shadow,
            lower_shadow=lower_shadow,
            body_ratio=body_ratio,
            shadow_symmetry=shadow_symmetry,
            strength=strength,
            trend_context=trend
        )
    
    def _calculate_shadow_symmetry(self, upper_shadow: float,
                                 lower_shadow: float) -> float:
        """Calculate symmetry between shadows (0-1)"""
        if upper_shadow == 0 or lower_shadow == 0:
            return 0.0
        
        ratio = min(upper_shadow, lower_shadow) / max(upper_shadow, lower_shadow)
        return ratio
    
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
                          lower_shadow_ratio: float, shadow_symmetry: float,
                          trend: str) -> float:
        """Calculate pattern strength"""
        strength = 70.0
        
        # Smaller body = stronger doji
        strength += (self.max_body_ratio - body_ratio) / self.max_body_ratio * 10
        
        # Longer shadows = stronger pattern
        avg_shadow_ratio = (upper_shadow_ratio + lower_shadow_ratio) / 2
        if avg_shadow_ratio > 5:
            strength += 10
        elif avg_shadow_ratio > 4:
            strength += 5
        
        # Shadow symmetry bonus
        strength += shadow_symmetry * 5
        
        # Trend context - more significant in trends
        if trend in ['uptrend', 'downtrend']:
            strength += 5
        
        return min(100, max(0, strength))
    
    def _create_results(self) -> Dict:
        """Create results dictionary"""
        return {
            'patterns': [
                {
                    'position': p.position,
                    'body_size': p.body_size,
                    'body_ratio': p.body_ratio,
                    'upper_shadow': p.upper_shadow,
                    'lower_shadow': p.lower_shadow,
                    'shadow_symmetry': p.shadow_symmetry,
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
                'avg_symmetry': np.mean([p.shadow_symmetry for p in self.patterns]) if self.patterns else 0,
                'trend_distribution': self._get_trend_distribution()
            }
        }
    
    def _get_trend_distribution(self) -> Dict[str, int]:
        """Get distribution of patterns by trend"""
        distribution = {'uptrend': 0, 'downtrend': 0, 'sideways': 0, 'unknown': 0}
        for pattern in self.patterns:
            distribution[pattern.trend_context] += 1
        return distribution
