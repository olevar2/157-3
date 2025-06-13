"""
Marubozu Pattern Indicator

A Marubozu is a candlestick pattern that has no shadows (wicks), indicating strong momentum.
There are two types: Bullish Marubozu (white/green) and Bearish Marubozu (black/red).

Author: Platform3.AI
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..base_indicator import StandardIndicatorInterface


@dataclass
class MarubozuResult:
    """Result class for Marubozu Pattern analysis"""

    bullish_marubozu: bool
    bearish_marubozu: bool
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    body_ratio: float  # Body size relative to total range
    timestamp: Optional[str] = None


class MarubozuPattern(StandardIndicatorInterface):
    """
    Marubozu Pattern Detector

    Detects Marubozu candlestick patterns with configurable tolerance for wicks.

    Parameters:
    -----------
    tolerance : float, default=0.02
        Maximum allowed wick size as percentage of total range (0.02 = 2%)
    min_body_size : float, default=0.005
        Minimum body size as percentage of price (0.005 = 0.5%)
    """

    def __init__(self, tolerance: float = 0.02, min_body_size: float = 0.005):
        super().__init__()
        self.tolerance = tolerance
        self.min_body_size = min_body_size
        self.last_values = []

    def calculate(
        self,
        high: List[float],
        low: List[float],
        open_price: List[float],
        close: List[float],
        **kwargs,
    ) -> List[MarubozuResult]:
        """
        Calculate Marubozu pattern signals

        Returns:
            List of MarubozuResult objects
        """
        if len(high) < 1:
            return []

        results = []

        for i in range(len(high)):
            h, l, o, c = high[i], low[i], open_price[i], close[i]

            # Calculate ranges
            total_range = h - l
            body_size = abs(c - o)
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l

            # Check for minimum body size
            if total_range == 0 or body_size < (o * self.min_body_size):
                results.append(
                    MarubozuResult(
                        bullish_marubozu=False,
                        bearish_marubozu=False,
                        strength=0.0,
                        confidence=0.0,
                        body_ratio=0.0,
                    )
                )
                continue

            # Calculate wick ratios
            upper_wick_ratio = upper_wick / total_range
            lower_wick_ratio = lower_wick / total_range
            body_ratio = body_size / total_range

            # Check for Marubozu conditions
            is_marubozu = (
                upper_wick_ratio <= self.tolerance
                and lower_wick_ratio <= self.tolerance
            )

            bullish_marubozu = is_marubozu and c > o
            bearish_marubozu = is_marubozu and c < o

            # Calculate strength and confidence
            if is_marubozu:
                # Strength based on body ratio and absence of wicks
                wick_penalty = (upper_wick_ratio + lower_wick_ratio) / (
                    2 * self.tolerance
                )
                strength = body_ratio * (1 - wick_penalty)

                # Confidence based on how close to perfect marubozu
                confidence = 1.0 - (upper_wick_ratio + lower_wick_ratio) / (
                    2 * self.tolerance
                )
                confidence = max(0.0, min(1.0, confidence))
            else:
                strength = 0.0
                confidence = 0.0

            results.append(
                MarubozuResult(
                    bullish_marubozu=bullish_marubozu,
                    bearish_marubozu=bearish_marubozu,
                    strength=strength,
                    confidence=confidence,
                    body_ratio=body_ratio,
                )
            )

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

        recent = (
            self.last_values[-5:] if len(self.last_values) >= 5 else self.last_values
        )

        bullish_count = sum(1 for r in recent if r.bullish_marubozu)
        bearish_count = sum(1 for r in recent if r.bearish_marubozu)

        if bullish_count > bearish_count:
            return "BULLISH_MOMENTUM"
        elif bearish_count > bullish_count:
            return "BEARISH_MOMENTUM"
        else:
            return "NEUTRAL"

    def generate_signals(
        self,
        high: List[float],
        low: List[float],
        open_price: List[float],
        close: List[float],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate trading signals based on Marubozu patterns

        Returns:
            Dictionary containing signal information
        """
        results = self.calculate(high, low, open_price, close, **kwargs)

        if not results:
            return {"signal": "HOLD", "confidence": 0.0, "pattern": None}

        latest = results[-1]

        if latest.bullish_marubozu and latest.confidence > 0.7:
            return {
                "signal": "BUY",
                "confidence": latest.confidence,
                "pattern": "BULLISH_MARUBOZU",
                "strength": latest.strength,
                "stop_loss_pct": 0.02,  # 2% below entry
                "take_profit_pct": 0.06,  # 6% above entry
            }
        elif latest.bearish_marubozu and latest.confidence > 0.7:
            return {
                "signal": "SELL",
                "confidence": latest.confidence,
                "pattern": "BEARISH_MARUBOZU",
                "strength": latest.strength,
                "stop_loss_pct": 0.02,  # 2% above entry
                "take_profit_pct": 0.06,  # 6% below entry
            }
        else:
            return {"signal": "HOLD", "confidence": 0.0, "pattern": None}


# Example usage and testing
if __name__ == "__main__":
    # Test data - simulating OHLC data
    test_data = {
        "high": [105.0, 107.5, 106.0, 108.0, 109.5],
        "low": [100.0, 105.0, 103.0, 105.0, 108.0],
        "open": [100.0, 105.0, 106.0, 105.0, 108.0],
        "close": [105.0, 107.5, 103.0, 108.0, 109.5],
    }

    indicator = MarubozuPattern(tolerance=0.01, min_body_size=0.01)
    results = indicator.calculate(**test_data)

    print("Marubozu Pattern Analysis:")
    for i, result in enumerate(results):
        print(
            f"Period {i+1}: Bullish={result.bullish_marubozu}, "
            f"Bearish={result.bearish_marubozu}, "
            f"Strength={result.strength:.3f}, "
            f"Confidence={result.confidence:.3f}"
        )

    # Test signal generation
    signals = indicator.generate_signals(**test_data)
    print(f"\nLatest Signal: {signals}")
    print(f"Market Regime: {indicator.get_market_regime()}")
