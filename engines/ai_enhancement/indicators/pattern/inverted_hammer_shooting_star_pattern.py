"""
Inverted Hammer and Shooting Star Pattern Indicator

The Inverted Hammer appears at the bottom of a downtrend and suggests a potential bullish reversal.
The Shooting Star appears at the top of an uptrend and suggests a potential bearish reversal.
Both patterns have small bodies and long upper shadows.

Author: Platform3.AI
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd

from ..base_indicator import StandardIndicatorInterface


@dataclass
class InvertedHammerShootingStarResult:
    """Result class for Inverted Hammer and Shooting Star Pattern analysis"""
    inverted_hammer: bool
    shooting_star: bool
    upper_shadow_ratio: float
    body_ratio: float
    lower_shadow_ratio: float
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    trend_context: str  # UPTREND, DOWNTREND, SIDEWAYS
    timestamp: Optional[str] = None


class InvertedHammerShootingStarPattern(StandardIndicatorInterface):
    """
    Inverted Hammer and Shooting Star Pattern Detector
    
    Detects both Inverted Hammer (bullish reversal) and Shooting Star (bearish reversal) patterns.
    
    Parameters:
    -----------
    min_upper_shadow_ratio : float, default=0.6
        Minimum upper shadow ratio (upper shadow / total range)
    max_body_ratio : float, default=0.3
        Maximum body ratio (body / total range)
    max_lower_shadow_ratio : float, default=0.2
        Maximum lower shadow ratio (lower shadow / total range)
    trend_lookback : int, default=5
        Number of periods to look back for trend determination
    """
    
    def __init__(self, 
                 min_upper_shadow_ratio: float = 0.6,
                 max_body_ratio: float = 0.3,
                 max_lower_shadow_ratio: float = 0.2,
                 trend_lookback: int = 5):
        super().__init__()
        self.min_upper_shadow_ratio = min_upper_shadow_ratio
        self.max_body_ratio = max_body_ratio
        self.max_lower_shadow_ratio = max_lower_shadow_ratio
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
        
        # Simple trend determination using price comparison
        start_price = trend_prices[0]
        end_price = trend_prices[-1]
        
        price_change_pct = (end_price - start_price) / start_price
        
        if price_change_pct > 0.02:  # 2% increase
            return "UPTREND"
        elif price_change_pct < -0.02:  # 2% decrease
            return "DOWNTREND"
        else:
            return "SIDEWAYS"
        
    def calculate(self, 
                 high: List[float], 
                 low: List[float], 
                 open_price: List[float], 
                 close: List[float],
                 **kwargs) -> List[InvertedHammerShootingStarResult]:
        """
        Calculate Inverted Hammer and Shooting Star pattern signals
        
        Returns:
            List of InvertedHammerShootingStarResult objects
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
                results.append(InvertedHammerShootingStarResult(
                    inverted_hammer=False,
                    shooting_star=False,
                    upper_shadow_ratio=0.0,
                    body_ratio=0.0,
                    lower_shadow_ratio=0.0,
                    strength=0.0,
                    confidence=0.0,
                    trend_context="SIDEWAYS"
                ))
                continue
            
            # Calculate ratios
            upper_shadow_ratio = upper_shadow / total_range
            body_ratio = body_size / total_range
            lower_shadow_ratio = lower_shadow / total_range
            
            # Determine trend context
            trend_context = self._determine_trend(close, i)
            
            # Check pattern conditions
            is_valid_pattern = (
                upper_shadow_ratio >= self.min_upper_shadow_ratio and
                body_ratio <= self.max_body_ratio and
                lower_shadow_ratio <= self.max_lower_shadow_ratio
            )
            
            # Determine specific patterns based on trend context
            inverted_hammer = is_valid_pattern and trend_context == "DOWNTREND"
            shooting_star = is_valid_pattern and trend_context == "UPTREND"
            
            # Calculate strength and confidence
            if is_valid_pattern:
                # Strength based on shadow ratios and trend context validity
                shadow_strength = upper_shadow_ratio - self.min_upper_shadow_ratio
                body_strength = self.max_body_ratio - body_ratio
                lower_shadow_strength = self.max_lower_shadow_ratio - lower_shadow_ratio
                
                strength = (shadow_strength + body_strength + lower_shadow_strength) / 3
                strength = max(0.0, min(1.0, strength))
                
                # Confidence based on how well pattern fits criteria
                confidence = 1.0 - body_ratio - lower_shadow_ratio + upper_shadow_ratio
                confidence = max(0.0, min(1.0, confidence))
                
                # Adjust confidence based on trend context
                if trend_context in ["UPTREND", "DOWNTREND"]:
                    confidence *= 1.2  # Boost confidence in trending markets
                else:
                    confidence *= 0.7  # Reduce confidence in sideways markets
                    
                confidence = max(0.0, min(1.0, confidence))
            else:
                strength = 0.0
                confidence = 0.0
            
            results.append(InvertedHammerShootingStarResult(
                inverted_hammer=inverted_hammer,
                shooting_star=shooting_star,
                upper_shadow_ratio=upper_shadow_ratio,
                body_ratio=body_ratio,
                lower_shadow_ratio=lower_shadow_ratio,
                strength=strength,
                confidence=confidence,
                trend_context=trend_context
            ))
        
        self.last_values = results[-10:] if len(results) >= 10 else results
        return results
    
    def get_signal_strength(self) -> float:
        """Return the latest signal strength"""
        if not self.last_values:
            return 0.0
        return self.last_values[-1].strength
    
    def get_market_regime(self) -> str:
        """Determine market regime based on recent patterns"""
        if not self.last_values:
            return "NEUTRAL"
        
        recent = self.last_values[-3:] if len(self.last_values) >= 3 else self.last_values
        
        inverted_hammer_count = sum(1 for r in recent if r.inverted_hammer)
        shooting_star_count = sum(1 for r in recent if r.shooting_star)
        
        if inverted_hammer_count > 0:
            return "POTENTIAL_BULLISH_REVERSAL"
        elif shooting_star_count > 0:
            return "POTENTIAL_BEARISH_REVERSAL"
        else:
            return "NEUTRAL"
    
    def generate_signals(self, 
                        high: List[float], 
                        low: List[float], 
                        open_price: List[float], 
                        close: List[float],
                        **kwargs) -> Dict[str, Any]:
        """
        Generate trading signals based on Inverted Hammer and Shooting Star patterns
        
        Returns:
            Dictionary containing signal information
        """
        results = self.calculate(high, low, open_price, close, **kwargs)
        
        if not results:
            return {"signal": "HOLD", "confidence": 0.0, "pattern": None}
        
        latest = results[-1]
        
        if latest.inverted_hammer and latest.confidence > 0.6:
            return {
                "signal": "BUY",
                "confidence": latest.confidence,
                "pattern": "INVERTED_HAMMER",
                "strength": latest.strength,
                "stop_loss_pct": 0.025,  # 2.5% below entry
                "take_profit_pct": 0.08,  # 8% above entry (reversal patterns have higher targets)
                "trend_context": latest.trend_context
            }
        elif latest.shooting_star and latest.confidence > 0.6:
            return {
                "signal": "SELL",
                "confidence": latest.confidence,
                "pattern": "SHOOTING_STAR",
                "strength": latest.strength,
                "stop_loss_pct": 0.025,  # 2.5% above entry
                "take_profit_pct": 0.08,  # 8% below entry
                "trend_context": latest.trend_context
            }
        else:
            return {"signal": "HOLD", "confidence": 0.0, "pattern": None}


# Example usage and testing
if __name__ == "__main__":
    # Test data - simulating OHLC data with reversal patterns
    test_data = {
        'high': [100.0, 99.0, 98.0, 97.0, 100.0],  # Last candle: potential inverted hammer
        'low': [98.0, 97.0, 96.0, 95.0, 96.0],
        'open': [99.0, 98.5, 97.5, 96.5, 96.5],
        'close': [98.5, 97.5, 96.5, 96.0, 97.0]
    }
    
    indicator = InvertedHammerShootingStarPattern()
    results = indicator.calculate(**test_data)
    
    print("Inverted Hammer / Shooting Star Pattern Analysis:")
    for i, result in enumerate(results):
        print(f"Period {i+1}: InvHammer={result.inverted_hammer}, "
              f"ShootingStar={result.shooting_star}, "
              f"Strength={result.strength:.3f}, "
              f"Confidence={result.confidence:.3f}, "
              f"Trend={result.trend_context}")
    
    # Test signal generation
    signals = indicator.generate_signals(**test_data)
    print(f"\nLatest Signal: {signals}")
    print(f"Market Regime: {indicator.get_market_regime()}")