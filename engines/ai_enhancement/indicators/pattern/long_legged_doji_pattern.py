"""
Long-Legged Doji Pattern Indicator

A Long-Legged Doji is a candlestick pattern with a very small body and long shadows on both sides,
indicating indecision in the market. It suggests potential reversal when appearing at key levels.

Author: Platform3.AI
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd

from ..base_indicator import StandardIndicatorInterface


@dataclass
class LongLeggedDojiResult:
    """Result class for Long-Legged Doji Pattern analysis"""
    is_long_legged_doji: bool
    body_ratio: float
    upper_shadow_ratio: float
    lower_shadow_ratio: float
    total_shadow_ratio: float
    indecision_strength: float  # 0.0 to 1.0
    reversal_potential: float  # 0.0 to 1.0
    trend_context: str  # UPTREND, DOWNTREND, SIDEWAYS
    timestamp: Optional[str] = None


class LongLeggedDojiPattern(StandardIndicatorInterface):
    """
    Long-Legged Doji Pattern Detector
    
    Detects Long-Legged Doji patterns which indicate market indecision and potential reversals.
    
    Parameters:
    -----------
    max_body_ratio : float, default=0.1
        Maximum body ratio (body / total range) for doji classification
    min_shadow_ratio : float, default=0.4
        Minimum shadow ratio (each shadow / total range)
    min_total_shadow_ratio : float, default=0.8
        Minimum total shadow ratio (both shadows / total range)
    trend_lookback : int, default=5
        Number of periods to look back for trend determination
    """
    
    def __init__(self, 
                 max_body_ratio: float = 0.1,
                 min_shadow_ratio: float = 0.4,
                 min_total_shadow_ratio: float = 0.8,
                 trend_lookback: int = 5):
        super().__init__()
        self.max_body_ratio = max_body_ratio
        self.min_shadow_ratio = min_shadow_ratio
        self.min_total_shadow_ratio = min_total_shadow_ratio
        self.trend_lookback = trend_lookback
        self.last_values = []
        
    def _determine_trend(self, close_prices: List[float], index: int) -> str:
        """Determine the trend at a given index"""
        if index < self.trend_lookback:
            return "SIDEWAYS"
        
        start_idx = max(0, index - self.trend_lookback)
        trend_prices = close_prices[start_idx:index]
        
        if len(trend_prices) < 2:
            return "SIDEWAYS"
        
        # Calculate trend strength using linear regression slope
        x = np.arange(len(trend_prices))
        slope = np.polyfit(x, trend_prices, 1)[0]
        
        # Normalize slope by average price
        avg_price = np.mean(trend_prices)
        normalized_slope = slope / avg_price if avg_price > 0 else 0
        
        if normalized_slope > 0.01:  # 1% upward slope
            return "UPTREND"
        elif normalized_slope < -0.01:  # 1% downward slope
            return "DOWNTREND"
        else:
            return "SIDEWAYS"
        
    def calculate(self, 
                 high: List[float], 
                 low: List[float], 
                 open_price: List[float], 
                 close: List[float],
                 **kwargs) -> List[LongLeggedDojiResult]:
        """
        Calculate Long-Legged Doji pattern signals
        
        Returns:
            List of LongLeggedDojiResult objects
        """
        if len(high) < 1:
            return []
            
        results = []
        
        for i in range(len(high)):
            h, l, o, c = high[i], low[i], open_price[i], close[i]
            
            # Calculate ranges
            total_range = h - l
            body_size = abs(c - o)
            upper_shadow = h - max(o, c)
            lower_shadow = min(o, c) - l
            
            # Avoid division by zero
            if total_range == 0:
                results.append(LongLeggedDojiResult(
                    is_long_legged_doji=False,
                    body_ratio=0.0,
                    upper_shadow_ratio=0.0,
                    lower_shadow_ratio=0.0,
                    total_shadow_ratio=0.0,
                    indecision_strength=0.0,
                    reversal_potential=0.0,
                    trend_context="SIDEWAYS"
                ))
                continue
            
            # Calculate ratios
            body_ratio = body_size / total_range
            upper_shadow_ratio = upper_shadow / total_range
            lower_shadow_ratio = lower_shadow / total_range
            total_shadow_ratio = (upper_shadow + lower_shadow) / total_range
            
            # Determine trend context
            trend_context = self._determine_trend(close, i)
            
            # Check Long-Legged Doji conditions
            is_long_legged_doji = (
                body_ratio <= self.max_body_ratio and
                upper_shadow_ratio >= self.min_shadow_ratio and
                lower_shadow_ratio >= self.min_shadow_ratio and
                total_shadow_ratio >= self.min_total_shadow_ratio
            )
            
            # Calculate indecision strength
            if is_long_legged_doji:
                # Higher indecision when body is smaller and shadows are longer
                body_score = (self.max_body_ratio - body_ratio) / self.max_body_ratio
                shadow_score = min(upper_shadow_ratio, lower_shadow_ratio) / self.min_shadow_ratio
                balance_score = 1.0 - abs(upper_shadow_ratio - lower_shadow_ratio)
                
                indecision_strength = (body_score + shadow_score + balance_score) / 3
                indecision_strength = max(0.0, min(1.0, indecision_strength))
            else:
                indecision_strength = 0.0
            
            # Calculate reversal potential based on trend context
            if is_long_legged_doji and trend_context in ["UPTREND", "DOWNTREND"]:
                # Higher reversal potential in strong trends
                reversal_potential = indecision_strength * 1.2
                reversal_potential = max(0.0, min(1.0, reversal_potential))
            elif is_long_legged_doji:
                # Lower reversal potential in sideways markets
                reversal_potential = indecision_strength * 0.6
            else:
                reversal_potential = 0.0
            
            results.append(LongLeggedDojiResult(
                is_long_legged_doji=is_long_legged_doji,
                body_ratio=body_ratio,
                upper_shadow_ratio=upper_shadow_ratio,
                lower_shadow_ratio=lower_shadow_ratio,
                total_shadow_ratio=total_shadow_ratio,
                indecision_strength=indecision_strength,
                reversal_potential=reversal_potential,
                trend_context=trend_context
            ))
        
        self.last_values = results[-10:] if len(results) >= 10 else results
        return results
    
    def get_signal_strength(self) -> float:
        """Return the latest signal strength"""
        if not self.last_values:
            return 0.0
        return self.last_values[-1].indecision_strength
    
    def get_market_regime(self) -> str:
        """Determine market regime based on recent patterns"""
        if not self.last_values:
            return "NEUTRAL"
        
        recent = self.last_values[-3:] if len(self.last_values) >= 3 else self.last_values
        
        doji_count = sum(1 for r in recent if r.is_long_legged_doji)
        avg_reversal_potential = np.mean([r.reversal_potential for r in recent])
        
        if doji_count > 0 and avg_reversal_potential > 0.6:
            return "HIGH_INDECISION_REVERSAL_PENDING"
        elif doji_count > 0:
            return "MARKET_INDECISION"
        else:
            return "NEUTRAL"
    
    def generate_signals(self, 
                        high: List[float], 
                        low: List[float], 
                        open_price: List[float], 
                        close: List[float],
                        **kwargs) -> Dict[str, Any]:
        """
        Generate trading signals based on Long-Legged Doji patterns
        
        Returns:
            Dictionary containing signal information
        """
        results = self.calculate(high, low, open_price, close, **kwargs)
        
        if not results:
            return {"signal": "HOLD", "confidence": 0.0, "pattern": None}
        
        latest = results[-1]
        
        if latest.is_long_legged_doji and latest.reversal_potential > 0.7:
            # Long-Legged Doji suggests waiting for confirmation
            return {
                "signal": "WAIT_FOR_CONFIRMATION",
                "confidence": latest.reversal_potential,
                "pattern": "LONG_LEGGED_DOJI",
                "indecision_strength": latest.indecision_strength,
                "reversal_potential": latest.reversal_potential,
                "trend_context": latest.trend_context,
                "recommendation": "Wait for next candle confirmation"
            }
        elif latest.is_long_legged_doji and latest.indecision_strength > 0.6:
            return {
                "signal": "REDUCE_POSITION",
                "confidence": latest.indecision_strength,
                "pattern": "LONG_LEGGED_DOJI",
                "indecision_strength": latest.indecision_strength,
                "reversal_potential": latest.reversal_potential,
                "trend_context": latest.trend_context,
                "recommendation": "Market showing indecision, consider reducing exposure"
            }
        else:
            return {"signal": "HOLD", "confidence": 0.0, "pattern": None}


# Example usage and testing
if __name__ == "__main__":
    # Test data - simulating OHLC data with long-legged doji
    test_data = {
        'high': [100.0, 101.0, 102.0, 103.0, 105.0],  # Last candle: potential long-legged doji
        'low': [98.0, 99.0, 100.0, 101.0, 99.0],
        'open': [99.0, 100.0, 101.0, 102.0, 102.0],
        'close': [99.5, 100.5, 101.5, 102.5, 102.1]  # Small body with long shadows
    }
    
    indicator = LongLeggedDojiPattern()
    results = indicator.calculate(**test_data)
    
    print("Long-Legged Doji Pattern Analysis:")
    for i, result in enumerate(results):
        print(f"Period {i+1}: LongDoji={result.is_long_legged_doji}, "
              f"Indecision={result.indecision_strength:.3f}, "
              f"Reversal={result.reversal_potential:.3f}, "
              f"Trend={result.trend_context}")
    
    # Test signal generation
    signals = indicator.generate_signals(**test_data)
    print(f"\nLatest Signal: {signals}")
    print(f"Market Regime: {indicator.get_market_regime()}")