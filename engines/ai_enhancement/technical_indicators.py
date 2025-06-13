"""
Basic Technical Indicators - REAL IMPLEMENTATIONS for Platform3
Core technical analysis indicators with proper calculations
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass


@dataclass
class RSIResult:
    """Result structure for RSI indicator"""

    rsi: float
    signal: str  # 'oversold', 'overbought', 'neutral'
    trend: str  # 'bullish', 'bearish', 'neutral'


@dataclass
class MACDResult:
    """Result structure for MACD indicator"""

    macd: float
    signal: float
    histogram: float
    trend: str  # 'bullish', 'bearish', 'neutral'


@dataclass
class BollingerBandsResult:
    """Result structure for Bollinger Bands"""

    upper_band: float
    middle_band: float
    lower_band: float
    bandwidth: float
    position: float  # Price position within bands (-1 to 1)
    squeeze: bool  # True if bands are contracting


@dataclass
class DonchianChannelsResult:
    """Result structure for Donchian Channels"""

    upper_channel: float
    lower_channel: float
    middle_channel: float
    position: float  # Price position within channels (-1 to 1)
    breakout_signal: str  # 'upper_breakout', 'lower_breakout', 'none'


class RelativeStrengthIndex:
    """
    Relative Strength Index (RSI) - Momentum oscillator
    """

    def __init__(self, period=14):
        self.period = period
        self.logger = logging.getLogger(__name__)

    def calculate(self, data) -> Optional[RSIResult]:
        """Calculate RSI"""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if "close" in data.columns:
                    prices = data["close"].values
                else:
                    prices = data.iloc[:, 0].values
            elif isinstance(data, (list, tuple, np.ndarray)):
                prices = np.array(data)
            elif isinstance(data, dict):
                prices = np.array(data.get("close", []))
            else:
                return None

            if len(prices) < self.period + 1:
                return None

            # Calculate price changes
            deltas = np.diff(prices)

            # Separate gains and losses
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            # Calculate average gains and losses
            avg_gain = np.mean(gains[: self.period]) if len(gains) >= self.period else 0
            avg_loss = (
                np.mean(losses[: self.period]) if len(losses) >= self.period else 0
            )

            # Calculate subsequent averages using Wilder's smoothing
            for i in range(self.period, len(gains)):
                avg_gain = (avg_gain * (self.period - 1) + gains[i]) / self.period
                avg_loss = (avg_loss * (self.period - 1) + losses[i]) / self.period

            # Calculate RSI
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            # Determine signal
            if rsi <= 30:
                signal = "oversold"
            elif rsi >= 70:
                signal = "overbought"
            else:
                signal = "neutral"

            # Determine trend
            if len(prices) >= 5:
                recent_rsi_values = []
                for j in range(max(0, len(prices) - 5), len(prices)):
                    if j >= self.period:
                        period_gains = gains[j - self.period : j]
                        period_losses = losses[j - self.period : j]
                        period_avg_gain = (
                            np.mean(period_gains) if len(period_gains) > 0 else 0
                        )
                        period_avg_loss = (
                            np.mean(period_losses) if len(period_losses) > 0 else 0
                        )

                        if period_avg_loss == 0:
                            period_rsi = 100
                        else:
                            period_rs = period_avg_gain / period_avg_loss
                            period_rsi = 100 - (100 / (1 + period_rs))
                        recent_rsi_values.append(period_rsi)

                if len(recent_rsi_values) >= 2:
                    rsi_trend = recent_rsi_values[-1] - recent_rsi_values[0]
                    if rsi_trend > 5:
                        trend = "bullish"
                    elif rsi_trend < -5:
                        trend = "bearish"
                    else:
                        trend = "neutral"
                else:
                    trend = "neutral"
            else:
                trend = "neutral"

            return RSIResult(rsi=float(rsi), signal=signal, trend=trend)

        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return None


class MovingAverageConvergenceDivergence:
    """
    MACD - Moving Average Convergence Divergence
    """

    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.logger = logging.getLogger(__name__)

    def calculate(self, data) -> Optional[MACDResult]:
        """Calculate MACD"""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if "close" in data.columns:
                    prices = data["close"].values
                else:
                    prices = data.iloc[:, 0].values
            elif isinstance(data, (list, tuple, np.ndarray)):
                prices = np.array(data)
            elif isinstance(data, dict):
                prices = np.array(data.get("close", []))
            else:
                return None

            if len(prices) < self.slow_period + self.signal_period:
                return None

            # Calculate EMAs
            fast_ema = self._calculate_ema(prices, self.fast_period)
            slow_ema = self._calculate_ema(prices, self.slow_period)

            # Calculate MACD line
            macd_line = fast_ema - slow_ema

            # Calculate signal line (EMA of MACD)
            signal_line = self._calculate_ema(macd_line, self.signal_period)

            # Calculate histogram
            histogram = macd_line - signal_line

            # Get current values
            current_macd = macd_line[-1]
            current_signal = signal_line[-1]
            current_histogram = histogram[-1]

            # Determine trend
            if len(histogram) >= 3:
                recent_histogram = histogram[-3:]
                if all(
                    recent_histogram[i] > recent_histogram[i - 1]
                    for i in range(1, len(recent_histogram))
                ):
                    trend = "bullish"
                elif all(
                    recent_histogram[i] < recent_histogram[i - 1]
                    for i in range(1, len(recent_histogram))
                ):
                    trend = "bearish"
                else:
                    trend = "neutral"
            else:
                trend = "neutral"

            return MACDResult(
                macd=float(current_macd),
                signal=float(current_signal),
                histogram=float(current_histogram),
                trend=trend,
            )

        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return None

    def _calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]

        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

        return ema


class BollingerBands:
    """
    Bollinger Bands - Volatility-based price bands
    """

    def __init__(self, period=20, std_dev=2):
        self.period = period
        self.std_dev = std_dev
        self.logger = logging.getLogger(__name__)

    def calculate(self, data) -> Optional[BollingerBandsResult]:
        """Calculate Bollinger Bands"""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if "close" in data.columns:
                    prices = data["close"].values
                else:
                    prices = data.iloc[:, 0].values
            elif isinstance(data, (list, tuple, np.ndarray)):
                prices = np.array(data)
            elif isinstance(data, dict):
                prices = np.array(data.get("close", []))
            else:
                return None

            if len(prices) < self.period:
                return None

            # Calculate moving average (middle band)
            middle_band = np.mean(prices[-self.period :])

            # Calculate standard deviation
            std = np.std(prices[-self.period :])

            # Calculate upper and lower bands
            upper_band = middle_band + (self.std_dev * std)
            lower_band = middle_band - (self.std_dev * std)

            # Calculate bandwidth
            bandwidth = (upper_band - lower_band) / middle_band * 100

            # Calculate price position within bands
            current_price = prices[-1]
            if upper_band == lower_band:
                position = 0
            else:
                position = (current_price - lower_band) / (
                    upper_band - lower_band
                ) * 2 - 1  # -1 to 1 scale

            # Detect squeeze (low volatility)
            if len(prices) >= self.period * 2:
                historical_bandwidth = []
                for i in range(self.period, len(prices) - self.period + 1):
                    period_prices = prices[i : i + self.period]
                    period_mean = np.mean(period_prices)
                    period_std = np.std(period_prices)
                    period_bandwidth = (
                        (period_std * self.std_dev * 2) / period_mean * 100
                    )
                    historical_bandwidth.append(period_bandwidth)

                if historical_bandwidth:
                    avg_bandwidth = np.mean(historical_bandwidth)
                    squeeze = bandwidth < avg_bandwidth * 0.7
                else:
                    squeeze = False
            else:
                squeeze = False

            return BollingerBandsResult(
                upper_band=float(upper_band),
                middle_band=float(middle_band),
                lower_band=float(lower_band),
                bandwidth=float(bandwidth),
                position=float(position),
                squeeze=squeeze,
            )

        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            return None


class StochasticOscillator:
    """
    Stochastic Oscillator - Momentum indicator
    """

    def __init__(self, k_period=14, d_period=3):
        self.k_period = k_period
        self.d_period = d_period
        self.logger = logging.getLogger(__name__)

    def calculate(self, data) -> Optional[Dict]:
        """Calculate Stochastic Oscillator"""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if all(col in data.columns for col in ["high", "low", "close"]):
                    highs = data["high"].values
                    lows = data["low"].values
                    closes = data["close"].values
                else:
                    prices = data.iloc[:, 0].values
                    highs = lows = closes = prices
            elif isinstance(data, (list, tuple, np.ndarray)):
                prices = np.array(data)
                highs = lows = closes = prices
            elif isinstance(data, dict):
                highs = np.array(data.get("high", data.get("close", [])))
                lows = np.array(data.get("low", data.get("close", [])))
                closes = np.array(data.get("close", []))
            else:
                return None

            if len(closes) < self.k_period + self.d_period:
                return None

            # Calculate %K values
            k_values = []
            for i in range(self.k_period - 1, len(closes)):
                period_high = np.max(highs[i - self.k_period + 1 : i + 1])
                period_low = np.min(lows[i - self.k_period + 1 : i + 1])

                if period_high == period_low:
                    k = 50  # Default when no range
                else:
                    k = ((closes[i] - period_low) / (period_high - period_low)) * 100
                k_values.append(k)

            if len(k_values) < self.d_period:
                return None

            # Calculate %D (moving average of %K)
            d_values = []
            for i in range(self.d_period - 1, len(k_values)):
                d = np.mean(k_values[i - self.d_period + 1 : i + 1])
                d_values.append(d)

            if not d_values:
                return None

            current_k = k_values[-1]
            current_d = d_values[-1]

            # Determine signal
            if current_k <= 20 and current_d <= 20:
                signal = "oversold"
            elif current_k >= 80 and current_d >= 80:
                signal = "overbought"
            else:
                signal = "neutral"

            # Determine trend
            if len(k_values) >= 3:
                k_trend = k_values[-1] - k_values[-3]
                if k_trend > 10:
                    trend = "bullish"
                elif k_trend < -10:
                    trend = "bearish"
                else:
                    trend = "neutral"
            else:
                trend = "neutral"

            return {
                "k": float(current_k),
                "d": float(current_d),
                "signal": signal,
                "trend": trend,
            }

        except Exception as e:
            self.logger.error(f"Error calculating Stochastic Oscillator: {e}")
            return None


class CommodityChannelIndex:
    """
    Commodity Channel Index (CCI) - Momentum indicator
    """

    def __init__(self, period=20):
        self.period = period
        self.constant = 0.015
        self.logger = logging.getLogger(__name__)

    def calculate(self, data) -> Optional[Dict]:
        """Calculate CCI"""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if all(col in data.columns for col in ["high", "low", "close"]):
                    highs = data["high"].values
                    lows = data["low"].values
                    closes = data["close"].values
                else:
                    prices = data.iloc[:, 0].values
                    highs = lows = closes = prices
            elif isinstance(data, (list, tuple, np.ndarray)):
                prices = np.array(data)
                highs = lows = closes = prices
            elif isinstance(data, dict):
                highs = np.array(data.get("high", data.get("close", [])))
                lows = np.array(data.get("low", data.get("close", [])))
                closes = np.array(data.get("close", []))
            else:
                return None

            if len(closes) < self.period:
                return None

            # Calculate Typical Price
            typical_prices = (highs + lows + closes) / 3

            # Calculate CCI values
            cci_values = []
            for i in range(self.period - 1, len(typical_prices)):
                # Simple Moving Average of Typical Price
                sma_tp = np.mean(typical_prices[i - self.period + 1 : i + 1])

                # Mean Deviation
                mean_deviation = np.mean(
                    np.abs(typical_prices[i - self.period + 1 : i + 1] - sma_tp)
                )

                # CCI calculation
                if mean_deviation == 0:
                    cci = 0
                else:
                    cci = (typical_prices[i] - sma_tp) / (
                        self.constant * mean_deviation
                    )
                cci_values.append(cci)

            if not cci_values:
                return None

            current_cci = cci_values[-1]

            # Determine signal
            if current_cci > 100:
                signal = "overbought"
            elif current_cci < -100:
                signal = "oversold"
            else:
                signal = "neutral"

            # Determine trend
            if len(cci_values) >= 3:
                cci_trend = cci_values[-1] - cci_values[-3]
                if cci_trend > 50:
                    trend = "bullish"
                elif cci_trend < -50:
                    trend = "bearish"
                else:
                    trend = "neutral"
            else:
                trend = "neutral"

            return {"cci": float(current_cci), "signal": signal, "trend": trend}

        except Exception as e:
            self.logger.error(f"Error calculating CCI: {e}")
            return None


class SimpleMovingAverage:
    """Simple Moving Average (SMA)"""

    def __init__(self, period=20):
        self.period = period
        self.logger = logging.getLogger(__name__)

    def calculate(self, data) -> Optional[Dict]:
        """Calculate Simple Moving Average"""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if "close" in data.columns:
                    prices = data["close"].values
                else:
                    prices = data.iloc[:, 0].values
            elif isinstance(data, (list, tuple, np.ndarray)):
                prices = np.array(data)
            elif isinstance(data, dict):
                prices = np.array(data.get("close", []))
            else:
                return None

            if len(prices) < self.period:
                return None

            # Calculate SMA
            sma = np.mean(prices[-self.period :])

            # Determine trend
            if len(prices) >= self.period + 1:
                prev_sma = np.mean(prices[-self.period - 1 : -1])
                if sma > prev_sma:
                    trend = "bullish"
                elif sma < prev_sma:
                    trend = "bearish"
                else:
                    trend = "neutral"
            else:
                trend = "neutral"

            return {"sma": float(sma), "trend": trend, "period": self.period}

        except Exception as e:
            self.logger.error(f"Error calculating SMA: {e}")
            return None


class ExponentialMovingAverage:
    """Exponential Moving Average (EMA)"""

    def __init__(self, period=20):
        self.period = period
        self.logger = logging.getLogger(__name__)

    def calculate(self, data) -> Optional[Dict]:
        """Calculate Exponential Moving Average"""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if "close" in data.columns:
                    prices = data["close"].values
                else:
                    prices = data.iloc[:, 0].values
            elif isinstance(data, (list, tuple, np.ndarray)):
                prices = np.array(data)
            elif isinstance(data, dict):
                prices = np.array(data.get("close", []))
            else:
                return None

            if len(prices) < self.period:
                return None

            # Calculate EMA
            alpha = 2.0 / (self.period + 1)
            ema = prices[0]

            for price in prices[1:]:
                ema = alpha * price + (1 - alpha) * ema

            # Determine trend
            if len(prices) >= 2:
                prev_ema = prices[0]
                for price in prices[1:-1]:
                    prev_ema = alpha * price + (1 - alpha) * prev_ema

                if ema > prev_ema:
                    trend = "bullish"
                elif ema < prev_ema:
                    trend = "bearish"
                else:
                    trend = "neutral"
            else:
                trend = "neutral"

            return {"ema": float(ema), "trend": trend, "period": self.period}

        except Exception as e:
            self.logger.error(f"Error calculating EMA: {e}")
            return None


class WeightedMovingAverage:
    """Weighted Moving Average (WMA)"""

    def __init__(self, period=20):
        self.period = period
        self.logger = logging.getLogger(__name__)

    def calculate(self, data) -> Optional[Dict]:
        """Calculate Weighted Moving Average"""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if "close" in data.columns:
                    prices = data["close"].values
                else:
                    prices = data.iloc[:, 0].values
            elif isinstance(data, (list, tuple, np.ndarray)):
                prices = np.array(data)
            elif isinstance(data, dict):
                prices = np.array(data.get("close", []))
            else:
                return None

            if len(prices) < self.period:
                return None

            # Calculate WMA
            recent_prices = prices[-self.period :]
            weights = np.arange(1, self.period + 1)
            wma = np.sum(recent_prices * weights) / np.sum(weights)  # Determine trend
            if len(prices) >= self.period + 1:
                prev_prices = prices[-self.period - 1 : -1]
                prev_wma = np.sum(prev_prices * weights) / np.sum(weights)
                if wma > prev_wma:
                    trend = "bullish"
                elif wma < prev_wma:
                    trend = "bearish"
                else:
                    trend = "neutral"
            else:
                trend = "neutral"

            return {"wma": float(wma), "trend": trend, "period": self.period}

        except Exception as e:
            self.logger.error(f"Error calculating WMA: {e}")
            return None


class DonchianChannels:
    """
    Donchian Channels - Trend following indicator
    Shows highest high and lowest low over a period
    """

    def __init__(self, period=20):
        self.period = period
        self.logger = logging.getLogger(__name__)

    def calculate(self, data) -> Optional[DonchianChannelsResult]:
        """Calculate Donchian Channels"""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if (
                    "high" in data.columns
                    and "low" in data.columns
                    and "close" in data.columns
                ):
                    highs = data["high"].values
                    lows = data["low"].values
                    closes = data["close"].values
                else:
                    # Use close as proxy for high/low if not available
                    closes = (
                        data["close"].values
                        if "close" in data.columns
                        else data.iloc[:, 0].values
                    )
                    highs = closes
                    lows = closes
            elif isinstance(data, dict):
                highs = np.array(data.get("high", data.get("close", [])))
                lows = np.array(data.get("low", data.get("close", [])))
                closes = np.array(data.get("close", []))
            else:
                # Assume single price series
                closes = np.array(data)
                highs = closes
                lows = closes

            if len(closes) < self.period:
                return None

            # Calculate Donchian Channels
            recent_highs = highs[-self.period :]
            recent_lows = lows[-self.period :]

            upper_channel = np.max(recent_highs)
            lower_channel = np.min(recent_lows)
            middle_channel = (upper_channel + lower_channel) / 2

            # Calculate current position within channels
            current_price = closes[-1]
            if upper_channel != lower_channel:
                position = (current_price - middle_channel) / (
                    (upper_channel - lower_channel) / 2
                )
                position = max(-1, min(1, position))  # Clamp to [-1, 1]
            else:
                position = 0

            # Detect breakout signals
            breakout_signal = "none"
            if current_price >= upper_channel:
                breakout_signal = "upper_breakout"
            elif current_price <= lower_channel:
                breakout_signal = "lower_breakout"

            return DonchianChannelsResult(
                upper_channel=float(upper_channel),
                lower_channel=float(lower_channel),
                middle_channel=float(middle_channel),
                position=float(position),
                breakout_signal=breakout_signal,
            )

        except Exception as e:
            self.logger.error(f"Error calculating Donchian Channels: {e}")
            return None
