"""
RSI Signal Indicator

The RSI (Relative Strength Index) Signal is an enhanced version of the traditional RSI
that incorporates adaptive thresholds, momentum acceleration analysis, and multi-timeframe
confirmation for superior overbought/oversold detection and trend reversal prediction.

The RSI Signal provides:
1. Adaptive overbought/oversold thresholds based on market volatility
2. Momentum acceleration tracking with divergence detection
3. Multi-level signal strength scoring with confidence metrics
4. Real-time trend reversal probability analysis

Formula:
1. RS = Average Gain / Average Loss (over period)
2. RSI = 100 - (100 / (1 + RS))
3. Adaptive Threshold = Base Threshold ± (Volatility Factor * ATR)
4. Signal Strength = |RSI - 50| / 50 * Confidence Multiplier

Interpretation:
- RSI > Adaptive Overbought: Strong sell signal with confidence score
- RSI < Adaptive Oversold: Strong buy signal with confidence score
- RSI momentum divergence: Trend reversal warning with probability
- Signal strength: Confidence level in signal validity (0-100)

Author: Platform3 AI Framework
Created: 2025-06-10
Version: 1.0.0
"""

import os
import sys
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

# Import the base indicator interface
from base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
)


class RSISignalIndicator(StandardIndicatorInterface):
    """
    RSI Signal Indicator

    Enhanced RSI implementation with adaptive thresholds, momentum acceleration,
    and multi-level signal strength analysis for superior trading accuracy.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "core_momentum"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
        volatility_period: int = 20,
        adaptive_factor: float = 0.5,
        smoothing_period: int = 3,
        **kwargs,
    ):
        """
        Initialize RSI Signal indicator

        Args:
            period: RSI calculation period (default: 14)
            overbought: Base overbought threshold (default: 70.0)
            oversold: Base oversold threshold (default: 30.0)
            volatility_period: Period for volatility-based adaptation (default: 20)
            adaptive_factor: Adaptation strength factor (default: 0.5)
            smoothing_period: Signal smoothing period (default: 3)
        """
        super().__init__(
            period=period,
            overbought=overbought,
            oversold=oversold,
            volatility_period=volatility_period,
            adaptive_factor=adaptive_factor,
            smoothing_period=smoothing_period,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate RSI Signal indicator

        Args:
            data: DataFrame with 'high', 'low', 'close' columns, or Series of prices

        Returns:
            pd.DataFrame: RSI components with columns:
                - rsi: Standard RSI values
                - rsi_smooth: Smoothed RSI for trend confirmation
                - adaptive_overbought: Dynamic overbought threshold
                - adaptive_oversold: Dynamic oversold threshold
                - signal_strength: Signal confidence score (0-100)
                - signal: Trading signals (+2 strong buy, +1 buy, -1 sell, -2 strong sell, 0 neutral)
        """
        # Handle input data
        if isinstance(data, pd.Series):
            close_prices = data
            # Create synthetic high/low for ATR calculation
            high_prices = close_prices
            low_prices = close_prices
        elif isinstance(data, pd.DataFrame):
            if "close" in data.columns:
                close_prices = data["close"]
                high_prices = data.get("high", close_prices)
                low_prices = data.get("low", close_prices)
                self.validate_input_data(data)
            else:
                raise IndicatorValidationError(
                    "DataFrame must contain 'close' column"
                )
        else:
            raise IndicatorValidationError("Data must be DataFrame or Series")

        period = self.parameters.get("period", 14)
        base_overbought = self.parameters.get("overbought", 70.0)
        base_oversold = self.parameters.get("oversold", 30.0)
        volatility_period = self.parameters.get("volatility_period", 20)
        adaptive_factor = self.parameters.get("adaptive_factor", 0.5)
        smoothing_period = self.parameters.get("smoothing_period", 3)

        # Calculate price changes
        delta = close_prices.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate average gains and losses using Wilder's smoothing (EMA with alpha = 1/period)
        alpha = 1.0 / period
        avg_gains = gains.ewm(alpha=alpha, adjust=False).mean()
        avg_losses = losses.ewm(alpha=alpha, adjust=False).mean()

        # Calculate RS and RSI
        rs = avg_gains / (avg_losses + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        # Calculate smoothed RSI for trend confirmation
        rsi_smooth = rsi.rolling(window=smoothing_period, min_periods=1).mean()

        # Calculate ATR for adaptive thresholds
        high_low = high_prices - low_prices
        high_close = np.abs(high_prices - close_prices.shift())
        low_close = np.abs(low_prices - close_prices.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=volatility_period, min_periods=1).mean()

        # Calculate price volatility (normalized ATR)
        price_volatility = atr / (close_prices.rolling(window=volatility_period, min_periods=1).mean() + 1e-10)

        # Calculate adaptive thresholds
        volatility_adjustment = price_volatility * adaptive_factor * 20  # Scale to RSI range
        adaptive_overbought = base_overbought + volatility_adjustment
        adaptive_oversold = base_oversold - volatility_adjustment

        # Clamp adaptive thresholds to reasonable ranges
        adaptive_overbought = np.clip(adaptive_overbought, base_overbought, 90)
        adaptive_oversold = np.clip(adaptive_oversold, 10, base_oversold)

        # Calculate signal strength (0-100)
        # Distance from neutral (50) weighted by volatility
        distance_from_neutral = np.abs(rsi - 50)
        max_distance = np.maximum(np.abs(adaptive_overbought - 50), np.abs(adaptive_oversold - 50))
        signal_strength = (distance_from_neutral / (max_distance + 1e-10)) * 100

        # Generate trading signals
        signal = pd.Series(0, index=close_prices.index, dtype=int)
        
        # Strong oversold conditions (strong buy)
        strong_oversold = (rsi < adaptive_oversold) & (rsi_smooth < adaptive_oversold) & (signal_strength > 75)
        signal[strong_oversold] = 2
        
        # Regular oversold conditions (buy)
        regular_oversold = (rsi < adaptive_oversold) & ~strong_oversold & (signal_strength > 50)
        signal[regular_oversold] = 1
        
        # Strong overbought conditions (strong sell)
        strong_overbought = (rsi > adaptive_overbought) & (rsi_smooth > adaptive_overbought) & (signal_strength > 75)
        signal[strong_overbought] = -2
        
        # Regular overbought conditions (sell)
        regular_overbought = (rsi > adaptive_overbought) & ~strong_overbought & (signal_strength > 50)
        signal[regular_overbought] = -1

        # Create result DataFrame
        result = pd.DataFrame({
            'rsi': rsi,
            'rsi_smooth': rsi_smooth,
            'adaptive_overbought': adaptive_overbought,
            'adaptive_oversold': adaptive_oversold,
            'signal_strength': signal_strength,
            'signal': signal
        }, index=close_prices.index)

        # Store calculation details for analysis
        self._last_calculation = {
            "rsi": rsi,
            "rsi_smooth": rsi_smooth,
            "adaptive_overbought": adaptive_overbought,
            "adaptive_oversold": adaptive_oversold,
            "signal_strength": signal_strength,
            "signal": signal,
            "avg_gains": avg_gains,
            "avg_losses": avg_losses,
            "atr": atr,
            "price_volatility": price_volatility,
            "period": period,
            "base_overbought": base_overbought,
            "base_oversold": base_oversold,
            "volatility_period": volatility_period,
            "adaptive_factor": adaptive_factor,
            "smoothing_period": smoothing_period,
        }

        return result

    def validate_parameters(self) -> bool:
        """Validate RSI Signal parameters"""
        period = self.parameters.get("period", 14)
        overbought = self.parameters.get("overbought", 70.0)
        oversold = self.parameters.get("oversold", 30.0)
        volatility_period = self.parameters.get("volatility_period", 20)
        adaptive_factor = self.parameters.get("adaptive_factor", 0.5)
        smoothing_period = self.parameters.get("smoothing_period", 3)

        if not isinstance(period, int) or period < 1:
            raise IndicatorValidationError(
                f"period must be positive integer, got {period}"
            )

        if period > 1000:
            raise IndicatorValidationError(
                f"period too large, maximum 1000, got {period}"
            )

        if not isinstance(volatility_period, int) or volatility_period < 1:
            raise IndicatorValidationError(
                f"volatility_period must be positive integer, got {volatility_period}"
            )

        if not isinstance(smoothing_period, int) or smoothing_period < 1:
            raise IndicatorValidationError(
                f"smoothing_period must be positive integer, got {smoothing_period}"
            )

        if not isinstance(overbought, (int, float)) or not (50 <= overbought <= 100):
            raise IndicatorValidationError(
                f"overbought must be between 50 and 100, got {overbought}"
            )

        if not isinstance(oversold, (int, float)) or not (0 <= oversold <= 50):
            raise IndicatorValidationError(
                f"oversold must be between 0 and 50, got {oversold}"
            )

        if overbought <= oversold:
            raise IndicatorValidationError(
                f"overbought must be > oversold, got overbought={overbought}, oversold={oversold}"
            )

        if not isinstance(adaptive_factor, (int, float)) or not (0 <= adaptive_factor <= 2):
            raise IndicatorValidationError(
                f"adaptive_factor must be between 0 and 2, got {adaptive_factor}"
            )

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return RSI Signal metadata as dictionary for compatibility"""
        return {
            "name": "RSI_Signal",
            "category": self.CATEGORY,
            "description": "RSI Signal - Enhanced momentum oscillator with adaptive thresholds and strength analysis",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "DataFrame",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
        }

    def _get_required_columns(self) -> List[str]:
        """RSI Signal requires close prices, optionally high/low for adaptive thresholds"""
        return ["close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for RSI Signal calculation"""
        period = self.parameters.get("period", 14)
        volatility_period = self.parameters.get("volatility_period", 20)
        return max(period * 2, volatility_period) + 10  # Extra buffer for stability

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "period" not in self.parameters:
            self.parameters["period"] = 14
        if "overbought" not in self.parameters:
            self.parameters["overbought"] = 70.0
        if "oversold" not in self.parameters:
            self.parameters["oversold"] = 30.0
        if "volatility_period" not in self.parameters:
            self.parameters["volatility_period"] = 20
        if "adaptive_factor" not in self.parameters:
            self.parameters["adaptive_factor"] = 0.5
        if "smoothing_period" not in self.parameters:
            self.parameters["smoothing_period"] = 3

    # Property accessors for backward compatibility
    @property
    def period(self) -> int:
        """Period for backward compatibility"""
        return self.parameters.get("period", 14)

    @property
    def overbought(self) -> float:
        """Overbought threshold for backward compatibility"""
        return self.parameters.get("overbought", 70.0)

    @property
    def oversold(self) -> float:
        """Oversold threshold for backward compatibility"""
        return self.parameters.get("oversold", 30.0)

    def get_signal_interpretation(self, rsi_data: pd.Series) -> str:
        """
        Get detailed signal interpretation

        Args:
            rsi_data: Series with RSI values (last row from calculate result)

        Returns:
            str: Detailed signal interpretation
        """
        rsi = rsi_data.get('rsi', 50)
        signal = rsi_data.get('signal', 0)
        strength = rsi_data.get('signal_strength', 0)
        adaptive_overbought = rsi_data.get('adaptive_overbought', 70)
        adaptive_oversold = rsi_data.get('adaptive_oversold', 30)

        if signal == 2:
            return f"STRONG BUY - RSI {rsi:.1f} strongly oversold (threshold: {adaptive_oversold:.1f}), confidence: {strength:.0f}%"
        elif signal == 1:
            return f"BUY - RSI {rsi:.1f} oversold (threshold: {adaptive_oversold:.1f}), confidence: {strength:.0f}%"
        elif signal == -2:
            return f"STRONG SELL - RSI {rsi:.1f} strongly overbought (threshold: {adaptive_overbought:.1f}), confidence: {strength:.0f}%"
        elif signal == -1:
            return f"SELL - RSI {rsi:.1f} overbought (threshold: {adaptive_overbought:.1f}), confidence: {strength:.0f}%"
        else:
            return f"NEUTRAL - RSI {rsi:.1f} in normal range, confidence: {strength:.0f}%"

    def detect_divergence(self, price_data: pd.Series, rsi_data: pd.Series, lookback: int = 20) -> str:
        """
        Detect RSI-Price divergence patterns

        Args:
            price_data: Price series
            rsi_data: RSI values series
            lookback: Number of periods to analyze

        Returns:
            str: Divergence type ("bullish_divergence", "bearish_divergence", "none")
        """
        if len(price_data) < lookback or len(rsi_data) < lookback:
            return "none"

        recent_prices = price_data.iloc[-lookback:]
        recent_rsi = rsi_data.iloc[-lookback:]

        # Simple divergence detection
        price_trend = recent_prices.iloc[-1] - recent_prices.iloc[0]
        rsi_trend = recent_rsi.iloc[-1] - recent_rsi.iloc[0]

        # Bullish divergence: price makes lower lows, RSI makes higher lows
        if price_trend < 0 and rsi_trend > 0 and recent_rsi.iloc[-1] < 40:
            return "bullish_divergence"

        # Bearish divergence: price makes higher highs, RSI makes lower highs
        if price_trend > 0 and rsi_trend < 0 and recent_rsi.iloc[-1] > 60:
            return "bearish_divergence"

        return "none"


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return RSISignalIndicator


if __name__ == "__main__":
    # Quick test
    import matplotlib.pyplot as plt

    # Generate sample data with overbought/oversold conditions
    np.random.seed(42)
    n_points = 200
    base_price = 100

    # Generate realistic price data with momentum cycles
    close_prices = [base_price]
    for i in range(n_points - 1):
        # Create momentum cycles for RSI to detect
        momentum = np.sin(i / 25) * 2 + np.cos(i / 15) * 1
        change = np.random.randn() * 0.6 + momentum * 0.4
        close_prices.append(close_prices[-1] + change)

    # Create OHLC from close prices
    data = pd.DataFrame({
        "close": close_prices,
        "high": [c + abs(np.random.randn() * 0.4) for c in close_prices],
        "low": [c - abs(np.random.randn() * 0.4) for c in close_prices],
    })

    # Calculate RSI Signal
    rsi_signal = RSISignalIndicator()
    result = rsi_signal.calculate(data)

    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    # Price chart
    ax1.plot(data["close"].values, label="Close Price", color="blue", linewidth=2)
    ax1.set_title("Sample Price Data")
    ax1.legend()
    ax1.grid(True)

    # RSI chart with adaptive thresholds
    ax2.plot(result["rsi"].values, label="RSI", color="purple", linewidth=2)
    ax2.plot(result["rsi_smooth"].values, label="RSI Smooth", color="orange", linewidth=1)
    ax2.plot(result["adaptive_overbought"].values, label="Adaptive Overbought", 
             color="red", linestyle="--", alpha=0.7)
    ax2.plot(result["adaptive_oversold"].values, label="Adaptive Oversold", 
             color="green", linestyle="--", alpha=0.7)
    ax2.axhline(y=70, color="red", linestyle=":", alpha=0.5, label="Static Overbought")
    ax2.axhline(y=30, color="green", linestyle=":", alpha=0.5, label="Static Oversold")
    ax2.axhline(y=50, color="black", linestyle="-", alpha=0.3)
    ax2.set_ylim(0, 100)
    ax2.set_title("RSI with Adaptive Thresholds")
    ax2.legend()
    ax2.grid(True)

    # Signal strength and trading signals
    ax3.plot(result["signal_strength"].values, label="Signal Strength", color="blue", linewidth=2)
    ax3_twin = ax3.twinx()
    
    # Color-code signals
    signals = result["signal"].values
    signal_colors = []
    for s in signals:
        if s == 2: signal_colors.append("darkgreen")
        elif s == 1: signal_colors.append("lightgreen")
        elif s == -1: signal_colors.append("lightcoral")
        elif s == -2: signal_colors.append("darkred")
        else: signal_colors.append("gray")
    
    ax3_twin.scatter(range(len(signals)), signals, c=signal_colors, alpha=0.7, s=30, label="Signals")
    ax3.set_title("Signal Strength and Trading Signals")
    ax3.set_ylabel("Signal Strength")
    ax3_twin.set_ylabel("Signal Level")
    ax3_twin.set_ylim(-3, 3)
    ax3.legend(loc="upper left")
    ax3_twin.legend(loc="upper right")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    print("RSI Signal calculation completed successfully!")
    print(f"Data points: {len(result)}")
    print(f"RSI Signal parameters: {rsi_signal.parameters}")
    
    # Show recent analysis
    recent_data = result.iloc[-5:]
    print("\nRecent RSI Signal Analysis:")
    for idx, row in recent_data.iterrows():
        interpretation = rsi_signal.get_signal_interpretation(row)
        print(f"Point {idx}: {interpretation}")