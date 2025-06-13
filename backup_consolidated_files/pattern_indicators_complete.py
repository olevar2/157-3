"""
Complete Pattern Indicators with Real Implementations for Platform3
Generated to complete the 157 indicator registry
"""

import numpy as np
import pandas as pd


def extract_ohlc_data(data):
    """Helper function to extract OHLC data from various input formats"""
    if hasattr(data, "iloc"):  # DataFrame
        opens = data["open"].values if "open" in data.columns else np.zeros(len(data))
        highs = data["high"].values if "high" in data.columns else np.zeros(len(data))
        lows = data["low"].values if "low" in data.columns else np.zeros(len(data))
        closes = (
            data["close"].values if "close" in data.columns else np.zeros(len(data))
        )
    else:  # Dict or array-like
        opens = np.array(data.get("open", np.zeros(len(data))))
        highs = np.array(data.get("high", np.zeros(len(data))))
        lows = np.array(data.get("low", np.zeros(len(data))))
        closes = np.array(data.get("close", np.zeros(len(data))))

    return opens, highs, lows, closes


class AbandonedBabySignal:
    """Real implementation for Abandoned Baby candlestick pattern"""

    def __init__(self, period=20, **kwargs):
        self.period = period
        self.kwargs = kwargs

    def calculate(self, data):
        """Calculate Abandoned Baby pattern signals"""
        if len(data) < 3:
            return np.zeros(len(data))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(2, len(data)):
            # Check for bullish abandoned baby
            if (
                closes[i - 2] < opens[i - 2]  # First candle is bearish
                and abs(closes[i - 1] - opens[i - 1])
                < (highs[i - 1] - lows[i - 1]) * 0.1  # Middle is doji-like
                and highs[i - 2] < lows[i - 1]  # Gap down
                and lows[i] > highs[i - 1]  # Gap up
                and closes[i] > opens[i]
            ):  # Third candle is bullish
                signals[i] = 1

            # Check for bearish abandoned baby
            elif (
                closes[i - 2] > opens[i - 2]  # First candle is bullish
                and abs(closes[i - 1] - opens[i - 1])
                < (highs[i - 1] - lows[i - 1]) * 0.1  # Middle is doji-like
                and lows[i - 2] > highs[i - 1]  # Gap up
                and highs[i] < lows[i - 1]  # Gap down
                and closes[i] < opens[i]
            ):  # Third candle is bearish
                signals[i] = -1

        return signals


class BeltHoldType:
    """Real implementation for Belt Hold candlestick pattern"""

    def __init__(self, period=20, **kwargs):
        self.period = period
        self.kwargs = kwargs

    def calculate(self, data):
        """Calculate Belt Hold pattern signals"""
        if len(data) < 1:
            return np.zeros(len(data))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(len(data)):
            body_size = abs(closes[i] - opens[i])
            upper_shadow = highs[i] - max(opens[i], closes[i])
            lower_shadow = min(opens[i], closes[i]) - lows[i]

            # Bullish belt hold (white/green belt hold)
            if (
                opens[i] == lows[i]  # Opens at low
                and closes[i] > opens[i]  # Bullish candle
                and body_size > (highs[i] - lows[i]) * 0.7  # Long body
                and upper_shadow < body_size * 0.1
            ):  # Minimal upper shadow
                signals[i] = 1

            # Bearish belt hold (black/red belt hold)
            elif (
                opens[i] == highs[i]  # Opens at high
                and closes[i] < opens[i]  # Bearish candle
                and body_size > (highs[i] - lows[i]) * 0.7  # Long body
                and lower_shadow < body_size * 0.1
            ):  # Minimal lower shadow
                signals[i] = -1

        return signals


class DarkCloudType:
    """Real implementation for Dark Cloud Cover pattern"""

    def __init__(self, period=20, **kwargs):
        self.period = period
        self.kwargs = kwargs

    def calculate(self, data):
        """Calculate Dark Cloud Cover pattern signals"""
        if len(data) < 2:
            return np.zeros(len(data))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(1, len(data)):
            # First candle should be bullish
            first_bullish = closes[i - 1] > opens[i - 1]
            first_body = closes[i - 1] - opens[i - 1]

            # Second candle should be bearish
            second_bearish = closes[i] < opens[i]

            if first_bullish and second_bearish:
                # Second candle opens above first candle's high
                gap_up = opens[i] > highs[i - 1]
                # Second candle closes below midpoint of first candle's body
                penetration = closes[i] < (opens[i - 1] + closes[i - 1]) / 2

                if gap_up and penetration:
                    signals[i] = -1  # Bearish signal

        return signals


class DojiType:
    """Real implementation for Doji pattern detection"""

    def __init__(self, period=20, **kwargs):
        self.period = period
        self.kwargs = kwargs

    def calculate(self, data):
        """Calculate Doji pattern types"""
        if len(data) < 1:
            return np.zeros(len(data))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(len(data)):
            body_size = abs(closes[i] - opens[i])
            total_range = highs[i] - lows[i]

            # Doji criteria: very small body relative to total range
            if total_range > 0 and body_size <= total_range * 0.1:
                upper_shadow = highs[i] - max(opens[i], closes[i])
                lower_shadow = min(opens[i], closes[i]) - lows[i]

                # Different doji types
                if upper_shadow > total_range * 0.6:  # Dragonfly doji
                    signals[i] = 1
                elif lower_shadow > total_range * 0.6:  # Gravestone doji
                    signals[i] = -1
                else:  # Standard doji
                    signals[i] = 0.5

        return signals


class EngulfingType:
    """Real implementation for Engulfing pattern detection"""

    def __init__(self, period=20, **kwargs):
        self.period = period
        self.kwargs = kwargs

    def calculate(self, data):
        """Calculate Engulfing pattern signals"""
        if len(data) < 2:
            return np.zeros(len(data))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(1, len(data)):
            # Previous candle real body
            prev_top = max(opens[i - 1], closes[i - 1])
            prev_bottom = min(opens[i - 1], closes[i - 1])

            # Current candle real body
            curr_top = max(opens[i], closes[i])
            curr_bottom = min(opens[i], closes[i])

            # Bullish engulfing
            if (
                closes[i - 1] < opens[i - 1]  # Previous candle bearish
                and closes[i] > opens[i]  # Current candle bullish
                and curr_bottom < prev_bottom  # Current engulfs previous
                and curr_top > prev_top
            ):
                signals[i] = 1

            # Bearish engulfing
            elif (
                closes[i - 1] > opens[i - 1]  # Previous candle bullish
                and closes[i] < opens[i]  # Current candle bearish
                and curr_bottom < prev_bottom  # Current engulfs previous
                and curr_top > prev_top
            ):
                signals[i] = -1

        return signals


class FibonacciType:
    """Simple implementation for Fibonacci retracement levels"""

    def __init__(self, period=20, **kwargs):
        self.period = period
        self.kwargs = kwargs

    def calculate(self, data):
        """Calculate Fibonacci retracement levels"""
        if len(data) < self.period:
            return np.zeros(len(data))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(self.period, len(data)):
            window = closes[i - self.period : i + 1]
            high_val = np.max(window)
            low_val = np.min(window)
            current = closes[i]

            # Calculate retracement levels
            diff = high_val - low_val
            if diff > 0:
                fib_618 = high_val - 0.618 * diff
                fib_382 = high_val - 0.382 * diff

                # Signal based on current price relative to fib levels
                if current <= fib_618:
                    signals[i] = 1  # Strong support level
                elif current <= fib_382:
                    signals[i] = 0.5  # Moderate support

        return signals


class GannAnglesTimeCycles:
    """Simple implementation for Gann angles and time cycles"""

    def __init__(self, period=20, **kwargs):
        self.period = period
        self.kwargs = kwargs

    def calculate(self, data):
        """Calculate simple Gann angle signals"""
        if len(data) < self.period:
            return np.zeros(len(data))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(self.period, len(data)):
            # Simple trend based on price movement over time
            price_change = closes[i] - closes[i - self.period]
            time_units = self.period

            # Gann 1x1 angle (45 degrees) - price = time
            if abs(price_change) >= time_units * 0.1:  # Scaled for practical use
                signals[i] = 1 if price_change > 0 else -1

        return signals


class HammerType:
    """Real implementation for Hammer and Hanging Man patterns"""

    def __init__(self, period=20, **kwargs):
        self.period = period
        self.kwargs = kwargs

    def calculate(self, data):
        """Calculate Hammer/Hanging Man pattern signals"""
        if len(data) < 1:
            return np.zeros(len(data))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(len(data)):
            body_size = abs(closes[i] - opens[i])
            upper_shadow = highs[i] - max(opens[i], closes[i])
            lower_shadow = min(opens[i], closes[i]) - lows[i]
            total_range = highs[i] - lows[i]

            if total_range > 0:
                # Hammer/Hanging man criteria
                if (
                    lower_shadow >= 2 * body_size  # Long lower shadow
                    and upper_shadow <= body_size * 0.1  # Small upper shadow
                    and body_size <= total_range * 0.3
                ):  # Small body

                    # Context determines hammer vs hanging man
                    # For simplicity, we'll use a basic signal
                    signals[i] = 1 if closes[i] >= opens[i] else -1

        return signals


class HaramiType:
    """Real implementation for Harami pattern detection"""

    def __init__(self, period=20, **kwargs):
        self.period = period
        self.kwargs = kwargs

    def calculate(self, data):
        """Calculate Harami pattern signals"""
        if len(data) < 2:
            return np.zeros(len(data))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(1, len(data)):
            # Previous candle (mother)
            prev_top = max(opens[i - 1], closes[i - 1])
            prev_bottom = min(opens[i - 1], closes[i - 1])
            prev_body = abs(closes[i - 1] - opens[i - 1])

            # Current candle (baby)
            curr_top = max(opens[i], closes[i])
            curr_bottom = min(opens[i], closes[i])
            curr_body = abs(closes[i] - opens[i])

            # Harami criteria: baby inside mother's body
            if (
                curr_top < prev_top
                and curr_bottom > prev_bottom
                and curr_body < prev_body
            ):

                # Bullish harami
                if closes[i - 1] < opens[i - 1] and closes[i] > opens[i]:
                    signals[i] = 1
                # Bearish harami
                elif closes[i - 1] > opens[i - 1] and closes[i] < opens[i]:
                    signals[i] = -1

        return signals


class HarmonicPoint:
    """Simple implementation for Harmonic pattern detection"""

    def __init__(self, period=20, **kwargs):
        self.period = period
        self.kwargs = kwargs

    def calculate(self, data):
        """Calculate basic harmonic pattern signals"""
        if len(data) < self.period:
            return np.zeros(len(data))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(self.period, len(data)):
            # Simple harmonic detection based on zigzag patterns
            window = closes[i - self.period : i + 1]
            if len(window) > 4:
                # Look for ABCD pattern approximation
                peaks = []
                for j in range(1, len(window) - 1):
                    if window[j] > window[j - 1] and window[j] > window[j + 1]:
                        peaks.append(j)

                if len(peaks) >= 2:
                    signals[i] = 0.5  # Potential harmonic pattern

        return signals


class HighWaveCandlePattern:
    """Real implementation for High Wave candle pattern"""

    def __init__(self, period=20, **kwargs):
        self.period = period
        self.kwargs = kwargs

    def calculate(self, data):
        """Calculate High Wave candle pattern signals"""
        if len(data) < 1:
            return np.zeros(len(data))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(len(data)):
            body_size = abs(closes[i] - opens[i])
            upper_shadow = highs[i] - max(opens[i], closes[i])
            lower_shadow = min(opens[i], closes[i]) - lows[i]
            total_range = highs[i] - lows[i]

            if total_range > 0:
                # High wave criteria: long shadows, small body
                if (
                    upper_shadow >= 2 * body_size
                    and lower_shadow >= 2 * body_size
                    and body_size <= total_range * 0.25
                ):
                    signals[i] = 1  # High wave pattern detected

        return signals


class InvertedHammerShootingStarPattern:
    """Real implementation for Inverted Hammer/Shooting Star patterns"""

    def __init__(self, period=20, **kwargs):
        self.period = period
        self.kwargs = kwargs

    def calculate(self, data):
        """Calculate Inverted Hammer/Shooting Star pattern signals"""
        if len(data) < 1:
            return np.zeros(len(data))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(len(data)):
            body_size = abs(closes[i] - opens[i])
            upper_shadow = highs[i] - max(opens[i], closes[i])
            lower_shadow = min(opens[i], closes[i]) - lows[i]
            total_range = highs[i] - lows[i]

            if total_range > 0:
                # Inverted hammer/shooting star criteria
                if (
                    upper_shadow >= 2 * body_size  # Long upper shadow
                    and lower_shadow <= body_size * 0.1  # Small lower shadow
                    and body_size <= total_range * 0.3
                ):  # Small body

                    # Context would determine inverted hammer vs shooting star
                    # For simplicity, use basic signal
                    signals[i] = 1 if closes[i] >= opens[i] else -1

        return signals


class KickingSignal:
    """Simple implementation for Kicking pattern"""

    def __init__(self, period=20, **kwargs):
        self.period = period
        self.kwargs = kwargs

    def calculate(self, data):
        """Calculate Kicking pattern signals"""
        if len(data) < 2:
            return np.zeros(len(data))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(1, len(data)):
            # Gap between candles with strong momentum
            if (
                closes[i - 1] < opens[i - 1]
                and closes[i] > opens[i]  # Bear to bull
                and opens[i] > closes[i - 1] * 1.01
            ):  # Gap up
                signals[i] = 1
            elif (
                closes[i - 1] > opens[i - 1]
                and closes[i] < opens[i]  # Bull to bear
                and opens[i] < closes[i - 1] * 0.99
            ):  # Gap down
                signals[i] = -1

        return signals


class LongLeggedDojiPattern:
    """Simple implementation for Long Legged Doji pattern"""

    def __init__(self, period=20, **kwargs):
        self.period = period
        self.kwargs = kwargs

    def calculate(self, data):
        """Calculate Long Legged Doji pattern signals"""
        if len(data) < 1:
            return np.zeros(len(data))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(len(data)):
            body_size = abs(closes[i] - opens[i])
            upper_shadow = highs[i] - max(opens[i], closes[i])
            lower_shadow = min(opens[i], closes[i]) - lows[i]
            total_range = highs[i] - lows[i]

            if (
                total_range > 0
                and body_size <= total_range * 0.1
                and upper_shadow >= total_range * 0.4
                and lower_shadow >= total_range * 0.4
            ):
                signals[i] = 1

        return signals


class MarubozuPattern:
    """Simple implementation for Marubozu pattern"""

    def __init__(self, period=20, **kwargs):
        self.period = period
        self.kwargs = kwargs

    def calculate(self, data):
        """Calculate Marubozu pattern signals"""
        if len(data) < 1:
            return np.zeros(len(data))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(len(data)):
            # White/bullish marubozu
            if opens[i] == lows[i] and closes[i] == highs[i] and closes[i] > opens[i]:
                signals[i] = 1
            # Black/bearish marubozu
            elif opens[i] == highs[i] and closes[i] == lows[i] and closes[i] < opens[i]:
                signals[i] = -1

        return signals


class MatchingSignal:
    """Simple implementation for Matching pattern"""

    def __init__(self, period=20, **kwargs):
        self.period = period
        self.kwargs = kwargs

    def calculate(self, data):
        """Calculate Matching pattern signals"""
        if len(data) < 2:
            return np.zeros(len(data))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(1, len(data)):
            # Simple matching close pattern
            if abs(closes[i] - closes[i - 1]) <= (
                closes[i] * 0.001
            ):  # Very close prices
                signals[i] = 0.5

        return signals


class PatternType:
    """Simple implementation for general Pattern detection"""

    def __init__(self, period=20, **kwargs):
        self.period = period
        self.kwargs = kwargs

    def calculate(self, data):
        """Calculate general pattern signals"""
        if len(data) < self.period:
            return np.zeros(len(data))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(self.period, len(data)):
            # Simple pattern detection based on price movement
            window = closes[i - self.period : i + 1]
            volatility = np.std(window)
            if volatility > np.mean(window) * 0.02:  # High volatility pattern
                signals[i] = 1

        return signals


class PiercingLineType:
    """Simple implementation for Piercing Line pattern"""

    def __init__(self, period=20, **kwargs):
        self.period = period
        self.kwargs = kwargs

    def calculate(self, data):
        """Calculate Piercing Line pattern signals"""
        if len(data) < 2:
            return np.zeros(len(data))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(1, len(data)):
            # First bearish, second bullish penetrating above midpoint
            if (
                closes[i - 1] < opens[i - 1]
                and closes[i] > opens[i]
                and opens[i] < closes[i - 1]
                and closes[i] > (opens[i - 1] + closes[i - 1]) / 2
            ):
                signals[i] = 1

        return signals


class SoldiersSignal:
    """Simple implementation for Three White Soldiers pattern"""

    def __init__(self, period=20, **kwargs):
        self.period = period
        self.kwargs = kwargs

    def calculate(self, data):
        """Calculate Three White Soldiers pattern signals"""
        if len(data) < 3:
            return np.zeros(len(data))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(2, len(data)):
            # Three consecutive bullish candles
            if (
                closes[i - 2] > opens[i - 2]
                and closes[i - 1] > opens[i - 1]
                and closes[i] > opens[i]
                and closes[i] > closes[i - 1]
                and closes[i - 1] > closes[i - 2]
            ):
                signals[i] = 1

        return signals


class SpinningTopPattern:
    """Simple implementation for Spinning Top pattern"""

    def __init__(self, period=20, **kwargs):
        self.period = period
        self.kwargs = kwargs

    def calculate(self, data):
        """Calculate Spinning Top pattern signals"""
        if len(data) < 1:
            return np.zeros(len(data))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(len(data)):
            body_size = abs(closes[i] - opens[i])
            upper_shadow = highs[i] - max(opens[i], closes[i])
            lower_shadow = min(opens[i], closes[i]) - lows[i]
            total_range = highs[i] - lows[i]

            # Small body with significant shadows on both sides
            if (
                total_range > 0
                and body_size <= total_range * 0.25
                and upper_shadow >= body_size
                and lower_shadow >= body_size
            ):
                signals[i] = 1

        return signals


class StarSignal:
    """Simple implementation for Star pattern"""

    def __init__(self, period=20, **kwargs):
        self.period = period
        self.kwargs = kwargs

    def calculate(self, data):
        """Calculate Star pattern signals"""
        if len(data) < 2:
            return np.zeros(len(data))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(1, len(data)):
            # Gap between candles (star pattern)
            prev_body_top = max(opens[i - 1], closes[i - 1])
            prev_body_bottom = min(opens[i - 1], closes[i - 1])
            curr_body_top = max(opens[i], closes[i])
            curr_body_bottom = min(opens[i], closes[i])

            # Evening star (bearish)
            if lows[i] > prev_body_top:
                signals[i] = -1
            # Morning star (bullish)
            elif highs[i] < prev_body_bottom:
                signals[i] = 1

        return signals


class ThreeInsideSignal:
    """Simple implementation for Three Inside Up/Down pattern"""

    def __init__(self, period=20, **kwargs):
        self.period = period
        self.kwargs = kwargs

    def calculate(self, data):
        """Calculate Three Inside pattern signals"""
        if len(data) < 3:
            return np.zeros(len(data))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(2, len(data)):
            # Three Inside Up: bearish, bullish harami, higher close
            if (
                closes[i - 2] < opens[i - 2]  # First bearish
                and closes[i - 1] > opens[i - 1]  # Second bullish
                and opens[i - 1] > closes[i - 2]
                and closes[i - 1] < opens[i - 2]  # Harami
                and closes[i] > closes[i - 1]
            ):  # Third higher
                signals[i] = 1
            # Three Inside Down: bullish, bearish harami, lower close
            elif (
                closes[i - 2] > opens[i - 2]  # First bullish
                and closes[i - 1] < opens[i - 1]  # Second bearish
                and opens[i - 1] < closes[i - 2]
                and closes[i - 1] > opens[i - 2]  # Harami
                and closes[i] < closes[i - 1]
            ):  # Third lower
                signals[i] = -1

        return signals


class ThreeLineStrikeSignal:
    """Simple implementation for Three Line Strike pattern"""

    def __init__(self, period=20, **kwargs):
        self.period = period
        self.kwargs = kwargs

    def calculate(self, data):
        """Calculate Three Line Strike pattern signals"""
        if len(data) < 4:
            return np.zeros(len(data))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(3, len(data)):
            # Three consecutive candles same direction, fourth reverses strongly
            same_dir = (
                closes[i - 3] > opens[i - 3]
                and closes[i - 2] > opens[i - 2]
                and closes[i - 1] > opens[i - 1]
            ) or (
                closes[i - 3] < opens[i - 3]
                and closes[i - 2] < opens[i - 2]
                and closes[i - 1] < opens[i - 1]
            )

            if same_dir:
                # Fourth candle engulfs all three
                if opens[i] > max(
                    closes[i - 3], closes[i - 2], closes[i - 1]
                ) and closes[i] < min(opens[i - 3], opens[i - 2], opens[i - 1]):
                    signals[i] = -1  # Bearish strike
                elif opens[i] < min(
                    closes[i - 3], closes[i - 2], closes[i - 1]
                ) and closes[i] > max(opens[i - 3], opens[i - 2], opens[i - 1]):
                    signals[i] = 1  # Bullish strike

        return signals


class ThreeOutsideSignal:
    """Simple implementation for Three Outside Up/Down pattern"""

    def __init__(self, period=20, **kwargs):
        self.period = period
        self.kwargs = kwargs

    def calculate(self, data):
        """Calculate Three Outside pattern signals"""
        if len(data) < 3:
            return np.zeros(len(data))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(2, len(data)):
            # Three Outside Up: bearish, bullish engulfing, higher close
            if (
                closes[i - 2] < opens[i - 2]  # First bearish
                and closes[i - 1] > opens[i - 1]  # Second bullish
                and opens[i - 1] < closes[i - 2]
                and closes[i - 1] > opens[i - 2]  # Engulfing
                and closes[i] > closes[i - 1]
            ):  # Third higher
                signals[i] = 1
            # Three Outside Down: bullish, bearish engulfing, lower close
            elif (
                closes[i - 2] > opens[i - 2]  # First bullish
                and closes[i - 1] < opens[i - 1]  # Second bearish
                and opens[i - 1] > closes[i - 2]
                and closes[i - 1] < opens[i - 2]  # Engulfing
                and closes[i] < closes[i - 1]
            ):  # Third lower
                signals[i] = -1

        return signals


class TweezerType:
    """Simple implementation for Tweezer Top/Bottom pattern"""

    def __init__(self, period=20, **kwargs):
        self.period = period
        self.kwargs = kwargs

    def calculate(self, data):
        """Calculate Tweezer pattern signals"""
        if len(data) < 2:
            return np.zeros(len(data))

        signals = np.zeros(len(data))

        for i in range(1, len(data)):
            # Tweezer tops - similar highs
            if abs(highs[i] - highs[i - 1]) <= highs[i] * 0.002:
                signals[i] = -1
            # Tweezer bottoms - similar lows
            elif abs(lows[i] - lows[i - 1]) <= lows[i] * 0.002:
                signals[i] = 1

        return signals
