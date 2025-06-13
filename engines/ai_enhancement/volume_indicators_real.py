"""
Real Volume Indicators for Platform3
Actual volume analysis indicators with proper calculations
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional, Tuple
import logging


class OnBalanceVolume:
    """On Balance Volume (OBV) - Volume momentum indicator"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate(self, data) -> Optional[Dict]:
        """Calculate On Balance Volume"""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if all(col in data.columns for col in ["close", "volume"]):
                    prices = data["close"].values
                    volumes = data["volume"].values
                elif len(data.columns) >= 2:
                    prices = data.iloc[:, 0].values
                    volumes = data.iloc[:, 1].values
                else:
                    return None
            elif isinstance(data, dict):
                prices = np.array(data.get("close", []))
                volumes = np.array(data.get("volume", []))
            else:
                return None

            if len(prices) < 2 or len(volumes) < 2:
                return None

            # Calculate OBV
            obv = np.zeros(len(prices))
            obv[0] = volumes[0]

            for i in range(1, len(prices)):
                if prices[i] > prices[i - 1]:
                    obv[i] = obv[i - 1] + volumes[i]
                elif prices[i] < prices[i - 1]:
                    obv[i] = obv[i - 1] - volumes[i]
                else:
                    obv[i] = obv[i - 1]

            # Determine trend
            current_obv = obv[-1]
            if len(obv) >= 5:
                prev_obv = obv[-5]
                if current_obv > prev_obv:
                    trend = "bullish"
                elif current_obv < prev_obv:
                    trend = "bearish"
                else:
                    trend = "neutral"
            else:
                trend = "neutral"

            return {
                "obv": float(current_obv),
                "trend": trend,
                "obv_series": obv.tolist(),
            }

        except Exception as e:
            self.logger.error(f"Error calculating OBV: {e}")
            return None


class VolumeOscillator:
    """Volume Oscillator - Compares short and long volume moving averages"""

    def __init__(self, short_period=5, long_period=10):
        self.short_period = short_period
        self.long_period = long_period
        self.logger = logging.getLogger(__name__)

    def calculate(self, data) -> Optional[Dict]:
        """Calculate Volume Oscillator"""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if "volume" in data.columns:
                    volumes = data["volume"].values
                elif len(data.columns) >= 2:
                    volumes = data.iloc[:, 1].values
                else:
                    return None
            elif isinstance(data, dict):
                volumes = np.array(data.get("volume", []))
            else:
                return None

            if len(volumes) < self.long_period:
                return None

            # Calculate volume moving averages
            short_ma = np.mean(volumes[-self.short_period :])
            long_ma = np.mean(volumes[-self.long_period :])

            # Calculate oscillator
            if long_ma != 0:
                volume_osc = ((short_ma - long_ma) / long_ma) * 100
            else:
                volume_osc = 0

            # Determine signal
            if volume_osc > 10:
                signal = "high_volume"
            elif volume_osc < -10:
                signal = "low_volume"
            else:
                signal = "normal_volume"

            return {
                "volume_oscillator": float(volume_osc),
                "signal": signal,
                "short_ma": float(short_ma),
                "long_ma": float(long_ma),
            }

        except Exception as e:
            self.logger.error(f"Error calculating Volume Oscillator: {e}")
            return None


class VolumeWeightedAveragePrice:
    """Volume Weighted Average Price (VWAP)"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate(self, data) -> Optional[Dict]:
        """Calculate VWAP"""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if all(
                    col in data.columns for col in ["high", "low", "close", "volume"]
                ):
                    highs = data["high"].values
                    lows = data["low"].values
                    closes = data["close"].values
                    volumes = data["volume"].values
                elif len(data.columns) >= 4:
                    highs = data.iloc[:, 1].values
                    lows = data.iloc[:, 2].values
                    closes = data.iloc[:, 3].values
                    volumes = (
                        data.iloc[:, 4].values
                        if len(data.columns) > 4
                        else np.ones(len(highs))
                    )
                else:
                    return None
            elif isinstance(data, dict):
                highs = np.array(data.get("high", []))
                lows = np.array(data.get("low", []))
                closes = np.array(data.get("close", []))
                volumes = np.array(data.get("volume", []))
            else:
                return None

            if len(closes) == 0 or len(volumes) == 0:
                return None

            # Calculate typical price
            typical_prices = (highs + lows + closes) / 3

            # Calculate VWAP
            cumulative_price_volume = np.cumsum(typical_prices * volumes)
            cumulative_volume = np.cumsum(volumes)

            if cumulative_volume[-1] == 0:
                return None

            vwap = cumulative_price_volume[-1] / cumulative_volume[-1]

            # Compare current price to VWAP
            current_price = closes[-1]
            if current_price > vwap:
                position = "above_vwap"
            elif current_price < vwap:
                position = "below_vwap"
            else:
                position = "at_vwap"

            return {
                "vwap": float(vwap),
                "current_price": float(current_price),
                "position": position,
                "price_deviation": float((current_price - vwap) / vwap * 100),
            }

        except Exception as e:
            self.logger.error(f"Error calculating VWAP: {e}")
            return None


class ChaikinMoneyFlowSignal:
    """Chaikin Money Flow - Volume-weighted momentum indicator"""

    def __init__(self, period=20):
        self.period = period
        self.logger = logging.getLogger(__name__)

    def calculate(self, data) -> Optional[Dict]:
        """Calculate Chaikin Money Flow"""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if all(
                    col in data.columns for col in ["high", "low", "close", "volume"]
                ):
                    highs = data["high"].values
                    lows = data["low"].values
                    closes = data["close"].values
                    volumes = data["volume"].values
                else:
                    return None
            elif isinstance(data, dict):
                highs = np.array(data.get("high", []))
                lows = np.array(data.get("low", []))
                closes = np.array(data.get("close", []))
                volumes = np.array(data.get("volume", []))
            else:
                return None

            if len(closes) < self.period:
                return None

            # Calculate Money Flow Multiplier
            money_flow_multiplier = np.zeros(len(closes))
            for i in range(len(closes)):
                if highs[i] != lows[i]:
                    money_flow_multiplier[i] = (
                        (closes[i] - lows[i]) - (highs[i] - closes[i])
                    ) / (highs[i] - lows[i])
                else:
                    money_flow_multiplier[i] = 0

            # Calculate Money Flow Volume
            money_flow_volume = money_flow_multiplier * volumes

            # Calculate Chaikin Money Flow
            cmf = np.sum(money_flow_volume[-self.period :]) / np.sum(
                volumes[-self.period :]
            )

            # Determine signal
            if cmf > 0.1:
                signal = "buying_pressure"
            elif cmf < -0.1:
                signal = "selling_pressure"
            else:
                signal = "neutral"

            return {"cmf": float(cmf), "signal": signal, "period": self.period}

        except Exception as e:
            self.logger.error(f"Error calculating CMF: {e}")
            return None
