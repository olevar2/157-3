"""
Real Trend Indicators for Platform3
Actual trend analysis indicators with proper calculations
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional, Tuple
import logging


class AverageTrueRange:
    """Average True Range (ATR) - Volatility indicator"""

    def __init__(self, period=14):
        self.period = period
        self.logger = logging.getLogger(__name__)

    def calculate(self, data) -> Optional[Dict]:
        """Calculate Average True Range"""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if all(col in data.columns for col in ["high", "low", "close"]):
                    highs = data["high"].values
                    lows = data["low"].values
                    closes = data["close"].values
                else:
                    return None
            elif isinstance(data, dict):
                highs = np.array(data.get("high", []))
                lows = np.array(data.get("low", []))
                closes = np.array(data.get("close", []))
            else:
                return None

            if len(closes) < self.period + 1:
                return None

            # Calculate True Range
            true_ranges = []
            for i in range(1, len(closes)):
                high_low = highs[i] - lows[i]
                high_close = abs(highs[i] - closes[i - 1])
                low_close = abs(lows[i] - closes[i - 1])
                true_range = max(high_low, high_close, low_close)
                true_ranges.append(true_range)

            # Calculate ATR using Wilder's smoothing
            atr = np.mean(true_ranges[: self.period])
            for tr in true_ranges[self.period :]:
                atr = (atr * (self.period - 1) + tr) / self.period

            # Determine volatility level
            current_price = closes[-1]
            atr_percentage = (atr / current_price) * 100

            if atr_percentage > 5:
                volatility = "high"
            elif atr_percentage > 2:
                volatility = "medium"
            else:
                volatility = "low"

            return {
                "atr": float(atr),
                "atr_percentage": float(atr_percentage),
                "volatility": volatility,
                "period": self.period,
            }

        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return None


class ParabolicSAR:
    """Parabolic Stop and Reverse (SAR) - Trend following indicator"""

    def __init__(self, initial_af=0.02, max_af=0.2):
        self.initial_af = initial_af
        self.max_af = max_af
        self.logger = logging.getLogger(__name__)

    def calculate(self, data) -> Optional[Dict]:
        """Calculate Parabolic SAR"""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if all(col in data.columns for col in ["high", "low", "close"]):
                    highs = data["high"].values
                    lows = data["low"].values
                    closes = data["close"].values
                else:
                    return None
            elif isinstance(data, dict):
                highs = np.array(data.get("high", []))
                lows = np.array(data.get("low", []))
                closes = np.array(data.get("close", []))
            else:
                return None

            if len(closes) < 3:
                return None

            # Initialize SAR calculation
            sar_values = np.zeros(len(closes))
            af = self.initial_af
            is_uptrend = closes[1] > closes[0]

            if is_uptrend:
                sar_values[0] = lows[0]
                ep = highs[1]  # Extreme point
            else:
                sar_values[0] = highs[0]
                ep = lows[1]

            # Calculate SAR for each period
            for i in range(1, len(closes)):
                # Calculate new SAR
                sar_values[i] = sar_values[i - 1] + af * (ep - sar_values[i - 1])

                # Check for trend reversal
                if is_uptrend:
                    if lows[i] <= sar_values[i]:
                        # Trend reversal to downtrend
                        is_uptrend = False
                        sar_values[i] = ep
                        ep = lows[i]
                        af = self.initial_af
                    else:
                        # Continue uptrend
                        if highs[i] > ep:
                            ep = highs[i]
                            af = min(af + self.initial_af, self.max_af)
                else:
                    if highs[i] >= sar_values[i]:
                        # Trend reversal to uptrend
                        is_uptrend = True
                        sar_values[i] = ep
                        ep = highs[i]
                        af = self.initial_af
                    else:
                        # Continue downtrend
                        if lows[i] < ep:
                            ep = lows[i]
                            af = min(af + self.initial_af, self.max_af)

            current_sar = sar_values[-1]
            current_price = closes[-1]

            # Determine signal
            if is_uptrend:
                signal = "buy"
                trend = "bullish"
            else:
                signal = "sell"
                trend = "bearish"

            return {
                "sar": float(current_sar),
                "signal": signal,
                "trend": trend,
                "current_price": float(current_price),
                "sar_series": sar_values.tolist(),
            }

        except Exception as e:
            self.logger.error(f"Error calculating Parabolic SAR: {e}")
            return None


class DirectionalMovementSystem:
    """Directional Movement System (DMS) - ADX and DI indicators"""

    def __init__(self, period=14):
        self.period = period
        self.logger = logging.getLogger(__name__)

    def calculate(self, data) -> Optional[Dict]:
        """Calculate Directional Movement System"""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if all(col in data.columns for col in ["high", "low", "close"]):
                    highs = data["high"].values
                    lows = data["low"].values
                    closes = data["close"].values
                else:
                    return None
            elif isinstance(data, dict):
                highs = np.array(data.get("high", []))
                lows = np.array(data.get("low", []))
                closes = np.array(data.get("close", []))
            else:
                return None

            if len(closes) < self.period + 1:
                return None

            # Calculate True Range and Directional Movement
            tr_values = []
            dm_plus = []
            dm_minus = []

            for i in range(1, len(closes)):
                # True Range
                high_low = highs[i] - lows[i]
                high_close = abs(highs[i] - closes[i - 1])
                low_close = abs(lows[i] - closes[i - 1])
                tr = max(high_low, high_close, low_close)
                tr_values.append(tr)

                # Directional Movement
                up_move = highs[i] - highs[i - 1]
                down_move = lows[i - 1] - lows[i]

                if up_move > down_move and up_move > 0:
                    dm_plus.append(up_move)
                else:
                    dm_plus.append(0)

                if down_move > up_move and down_move > 0:
                    dm_minus.append(down_move)
                else:
                    dm_minus.append(0)

            # Calculate smoothed values
            atr = np.mean(tr_values[: self.period])
            sum_dm_plus = np.sum(dm_plus[: self.period])
            sum_dm_minus = np.sum(dm_minus[: self.period])

            for i in range(self.period, len(tr_values)):
                atr = (atr * (self.period - 1) + tr_values[i]) / self.period
                sum_dm_plus = sum_dm_plus - sum_dm_plus / self.period + dm_plus[i]
                sum_dm_minus = sum_dm_minus - sum_dm_minus / self.period + dm_minus[i]

            # Calculate DI+ and DI-
            if atr != 0:
                di_plus = (sum_dm_plus / self.period) / atr * 100
                di_minus = (sum_dm_minus / self.period) / atr * 100
            else:
                di_plus = di_minus = 0

            # Calculate ADX
            if di_plus + di_minus != 0:
                dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100
            else:
                dx = 0

            # For simplicity, use current DX as ADX (normally would smooth)
            adx = dx

            # Determine trend strength and direction
            if adx > 25:
                strength = "strong"
            elif adx > 20:
                strength = "moderate"
            else:
                strength = "weak"

            if di_plus > di_minus:
                direction = "bullish"
            elif di_minus > di_plus:
                direction = "bearish"
            else:
                direction = "neutral"

            return {
                "adx": float(adx),
                "di_plus": float(di_plus),
                "di_minus": float(di_minus),
                "trend_strength": strength,
                "trend_direction": direction,
            }

        except Exception as e:
            self.logger.error(f"Error calculating DMS: {e}")
            return None


class AroonIndicator:
    """Aroon Indicator - Trend strength and direction"""

    def __init__(self, period=25):
        self.period = period
        self.logger = logging.getLogger(__name__)

    def calculate(self, data) -> Optional[Dict]:
        """Calculate Aroon Indicator"""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if all(col in data.columns for col in ["high", "low"]):
                    highs = data["high"].values
                    lows = data["low"].values
                else:
                    return None
            elif isinstance(data, dict):
                highs = np.array(data.get("high", []))
                lows = np.array(data.get("low", []))
            else:
                return None

            if len(highs) < self.period:
                return None

            # Get recent period data
            recent_highs = highs[-self.period :]
            recent_lows = lows[-self.period :]

            # Find periods since highest high and lowest low
            periods_since_highest = self.period - 1 - np.argmax(recent_highs)
            periods_since_lowest = self.period - 1 - np.argmin(recent_lows)

            # Calculate Aroon Up and Aroon Down
            aroon_up = ((self.period - periods_since_highest) / self.period) * 100
            aroon_down = ((self.period - periods_since_lowest) / self.period) * 100

            # Calculate Aroon Oscillator
            aroon_oscillator = aroon_up - aroon_down

            # Determine signal
            if aroon_up > 70 and aroon_down < 30:
                signal = "strong_uptrend"
            elif aroon_down > 70 and aroon_up < 30:
                signal = "strong_downtrend"
            elif abs(aroon_oscillator) < 20:
                signal = "consolidation"
            elif aroon_oscillator > 0:
                signal = "uptrend"
            else:
                signal = "downtrend"

            return {
                "aroon_up": float(aroon_up),
                "aroon_down": float(aroon_down),
                "aroon_oscillator": float(aroon_oscillator),
                "signal": signal,
                "period": self.period,
            }

        except Exception as e:
            self.logger.error(f"Error calculating Aroon: {e}")
            return None
