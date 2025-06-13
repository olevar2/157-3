"""
Volatility Indicators - REAL IMPLEMENTATIONS for Platform3
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional
import logging
from dataclasses import dataclass


@dataclass
class VolatilityResult:
    """Standard result structure for volatility indicators"""

    value: float
    signal: str  # 'high', 'low', 'normal'
    percentile: float
    trend: str  # 'increasing', 'decreasing', 'stable'


class ChaikinVolatility:
    """
    Chaikin Volatility - Measures price volatility using EMA of high-low spread
    """

    def __init__(self, period=10, ema_period=10):
        self.period = period
        self.ema_period = ema_period
        self.logger = logging.getLogger(__name__)

    def calculate(self, data) -> Optional[VolatilityResult]:
        """Calculate Chaikin Volatility"""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if "high" in data.columns and "low" in data.columns:
                    highs = data["high"].values
                    lows = data["low"].values
                else:
                    # Use close prices as proxy
                    prices = data.iloc[:, 0].values
                    highs = lows = prices
            elif isinstance(data, (list, tuple, np.ndarray)):
                prices = np.array(data)
                highs = lows = prices
            elif isinstance(data, dict):
                highs = np.array(data.get("high", data.get("close", [])))
                lows = np.array(data.get("low", data.get("close", [])))
            else:
                return None

            if len(highs) < self.period + self.ema_period:
                return None

            # Calculate high-low spread
            hl_spread = highs - lows

            # Calculate EMA of high-low spread
            ema_spread = self._calculate_ema(hl_spread, self.ema_period)

            # Calculate volatility as percentage change in EMA
            if len(ema_spread) < self.period:
                return None

            volatility = (
                (ema_spread[-1] - ema_spread[-self.period]) / ema_spread[-self.period]
            ) * 100

            # Determine signal
            recent_vol = np.std(ema_spread[-self.period :])
            percentile = np.percentile(ema_spread, 50) if len(ema_spread) > 0 else 0

            if abs(volatility) > recent_vol * 2:
                signal = "high"
            elif abs(volatility) < recent_vol * 0.5:
                signal = "low"
            else:
                signal = "normal"

            # Determine trend
            if len(ema_spread) >= 3:
                recent_trend = np.polyfit(range(3), ema_spread[-3:], 1)[0]
                if recent_trend > 0.01:
                    trend = "increasing"
                elif recent_trend < -0.01:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            return VolatilityResult(
                value=float(volatility),
                signal=signal,
                percentile=float(percentile),
                trend=trend,
            )

        except Exception as e:
            self.logger.error(f"Error calculating Chaikin Volatility: {e}")
            return None

    def _calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]

        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

        return ema


class HistoricalVolatility:
    """
    Historical Volatility - Measures price volatility using standard deviation of returns
    """

    def __init__(self, period=20, annualize=True):
        self.period = period
        self.annualize = annualize
        self.logger = logging.getLogger(__name__)

    def calculate(self, data) -> Optional[VolatilityResult]:
        """Calculate Historical Volatility"""
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

            # Calculate log returns
            returns = np.diff(np.log(prices))

            # Calculate rolling volatility
            volatilities = []
            for i in range(self.period - 1, len(returns)):
                period_returns = returns[i - self.period + 1 : i + 1]
                vol = np.std(period_returns)
                if self.annualize:
                    vol *= np.sqrt(252)  # Annualize assuming 252 trading days
                volatilities.append(vol)

            if not volatilities:
                return None

            current_vol = volatilities[-1] * 100  # Convert to percentage

            # Calculate percentile within recent history
            percentile = (
                np.sum(np.array(volatilities) <= volatilities[-1]) / len(volatilities)
            ) * 100

            # Determine signal
            avg_vol = np.mean(volatilities)
            if current_vol > avg_vol * 1.5:
                signal = "high"
            elif current_vol < avg_vol * 0.5:
                signal = "low"
            else:
                signal = "normal"

            # Determine trend
            if len(volatilities) >= 3:
                recent_trend = np.polyfit(range(3), volatilities[-3:], 1)[0]
                if recent_trend > avg_vol * 0.1:
                    trend = "increasing"
                elif recent_trend < -avg_vol * 0.1:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            return VolatilityResult(
                value=float(current_vol),
                signal=signal,
                percentile=float(percentile),
                trend=trend,
            )

        except Exception as e:
            self.logger.error(f"Error calculating Historical Volatility: {e}")
            return None


class RelativeVolatilityIndex:
    """
    Relative Volatility Index - Combines price direction with volatility
    """

    def __init__(self, period=14):
        self.period = period
        self.logger = logging.getLogger(__name__)

    def calculate(self, data) -> Optional[VolatilityResult]:
        """Calculate Relative Volatility Index"""
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

            # Calculate standard deviation for up and down moves
            changes = np.diff(prices)

            up_changes = np.where(changes > 0, changes, 0)
            down_changes = np.where(changes < 0, abs(changes), 0)

            # Calculate rolling standard deviations
            up_volatility = []
            down_volatility = []

            for i in range(self.period - 1, len(changes)):
                up_vol = np.std(up_changes[i - self.period + 1 : i + 1])
                down_vol = np.std(down_changes[i - self.period + 1 : i + 1])
                up_volatility.append(up_vol)
                down_volatility.append(down_vol)

            if not up_volatility or not down_volatility:
                return None

            # Calculate RVI
            up_avg = np.mean(up_volatility)
            down_avg = np.mean(down_volatility)

            if down_avg == 0:
                rvi = 100
            else:
                rvi = 100 * up_avg / (up_avg + down_avg)

            # Determine signal
            if rvi > 70:
                signal = "high"
            elif rvi < 30:
                signal = "low"
            else:
                signal = "normal"

            # Calculate percentile
            percentile = rvi

            # Determine trend
            if len(up_volatility) >= 3 and len(down_volatility) >= 3:
                recent_up_trend = np.polyfit(range(3), up_volatility[-3:], 1)[0]
                recent_down_trend = np.polyfit(range(3), down_volatility[-3:], 1)[0]

                if recent_up_trend > recent_down_trend:
                    trend = "increasing"
                elif recent_up_trend < recent_down_trend:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            return VolatilityResult(
                value=float(rvi),
                signal=signal,
                percentile=float(percentile),
                trend=trend,
            )

        except Exception as e:
            self.logger.error(f"Error calculating Relative Volatility Index: {e}")
            return None


class VolatilityIndex:
    """
    Volatility Index - General volatility measure using multiple methods
    """

    def __init__(self, period=14):
        self.period = period
        self.logger = logging.getLogger(__name__)

    def calculate(self, data) -> Optional[VolatilityResult]:
        """Calculate Volatility Index"""
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

            # Calculate multiple volatility measures
            returns = np.diff(np.log(prices))

            # 1. Standard deviation volatility
            std_vol = np.std(returns[-self.period :])

            # 2. Mean absolute deviation
            mad_vol = np.mean(
                np.abs(returns[-self.period :] - np.mean(returns[-self.period :]))
            )

            # 3. Range-based volatility (Parkinson estimator)
            if len(prices) >= self.period:
                range_vol = np.sqrt(
                    np.mean(
                        (
                            np.log(np.maximum.accumulate(prices[-self.period :]))
                            - np.log(np.minimum.accumulate(prices[-self.period :]))
                        )
                        ** 2
                    )
                    / (4 * np.log(2))
                )
            else:
                range_vol = std_vol

            # Combine volatilities
            volatility_index = (std_vol + mad_vol + range_vol) / 3 * 100

            # Calculate historical percentile
            all_volatilities = []
            for i in range(self.period, len(returns)):
                period_returns = returns[i - self.period : i]
                vol = np.std(period_returns)
                all_volatilities.append(vol)

            if all_volatilities:
                percentile = (
                    np.sum(np.array(all_volatilities) <= std_vol)
                    / len(all_volatilities)
                ) * 100
            else:
                percentile = 50.0

            # Determine signal
            if percentile > 80:
                signal = "high"
            elif percentile < 20:
                signal = "low"
            else:
                signal = "normal"

            # Determine trend
            if len(all_volatilities) >= 3:
                recent_trend = np.polyfit(range(3), all_volatilities[-3:], 1)[0]
                avg_vol = np.mean(all_volatilities)
                if recent_trend > avg_vol * 0.1:
                    trend = "increasing"
                elif recent_trend < -avg_vol * 0.1:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            return VolatilityResult(
                value=float(volatility_index),
                signal=signal,
                percentile=float(percentile),
                trend=trend,
            )

        except Exception as e:
            self.logger.error(f"Error calculating Volatility Index: {e}")
            return None


class MassIndex:
    """
    Mass Index - Identifies trend reversals using high-low range analysis
    """

    def __init__(self, period=25, ema_period=9):
        self.period = period
        self.ema_period = ema_period
        self.logger = logging.getLogger(__name__)

    def calculate(self, data) -> Optional[VolatilityResult]:
        """Calculate Mass Index"""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if "high" in data.columns and "low" in data.columns:
                    highs = data["high"].values
                    lows = data["low"].values
                else:
                    # Use close prices as proxy
                    prices = data.iloc[:, 0].values
                    highs = lows = prices
            elif isinstance(data, (list, tuple, np.ndarray)):
                prices = np.array(data)
                highs = lows = prices
            elif isinstance(data, dict):
                highs = np.array(data.get("high", data.get("close", [])))
                lows = np.array(data.get("low", data.get("close", [])))
            else:
                return None

            if len(highs) < self.period + (self.ema_period * 2):
                return None

            # Calculate high-low range
            hl_range = highs - lows

            # Calculate single EMA
            ema1 = self._calculate_ema(hl_range, self.ema_period)

            # Calculate double EMA (EMA of EMA)
            ema2 = self._calculate_ema(ema1, self.ema_period)

            # Calculate ratio
            ratio = np.divide(ema1, ema2, out=np.ones_like(ema1), where=ema2 != 0)

            # Calculate Mass Index
            mass_index_values = []
            for i in range(self.period - 1, len(ratio)):
                mi = np.sum(ratio[i - self.period + 1 : i + 1])
                mass_index_values.append(mi)

            if not mass_index_values:
                return None

            current_mi = mass_index_values[-1]

            # Determine signal (Mass Index > 27 indicates potential reversal)
            if current_mi > 27:
                signal = "high"
            elif current_mi < 26.5:
                signal = "low"
            else:
                signal = "normal"

            # Calculate percentile
            percentile = (
                np.sum(np.array(mass_index_values) <= current_mi)
                / len(mass_index_values)
            ) * 100

            # Determine trend
            if len(mass_index_values) >= 3:
                recent_trend = np.polyfit(range(3), mass_index_values[-3:], 1)[0]
                if recent_trend > 0.1:
                    trend = "increasing"
                elif recent_trend < -0.1:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            return VolatilityResult(
                value=float(current_mi),
                signal=signal,
                percentile=float(percentile),
                trend=trend,
            )

        except Exception as e:
            self.logger.error(f"Error calculating Mass Index: {e}")
            return None

    def _calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]

        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

        return ema
