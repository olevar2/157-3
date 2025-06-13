"""
Fisher Signal Type Indicator

Advanced Fisher Transform-based signal detection indicator that converts price data into a
Gaussian normal distribution for improved signal clarity and reduced noise. The Fisher Transform
is particularly effective at identifying potential reversal points and trend changes by
transforming the price data to emphasize extreme values.

The Fisher Transform formula:
Fisher = 0.5 * ln((1 + X) / (1 - X))

Where X is the normalized price value between -1 and +1.

This implementation includes:
1. Multiple Fisher Transform variants (Price, Momentum, RSI-based)
2. Signal type classification (reversal, continuation, breakout)
3. Confidence scoring based on signal strength
4. Multi-timeframe signal confluence
5. Adaptive threshold adjustment
6. Noise filtering and signal smoothing

Author: Platform3 AI Framework  
Created: 2025-06-10
"""

import os
import sys
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

# Import the base indicator interface
from ai_enhancement.indicators.base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
)


class FisherSignalTypeIndicator(StandardIndicatorInterface):
    """
    Fisher Signal Type Indicator
    
    Advanced signal detection using Fisher Transform with multiple variants
    and intelligent signal classification.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "cycle"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 14,
        smooth_period: int = 3,
        signal_threshold: float = 1.5,
        confidence_period: int = 20,
        fisher_type: str = "price",  # price, momentum, rsi, hybrid
        adaptive_threshold: bool = True,
        **kwargs,
    ):
        """
        Initialize Fisher Signal Type indicator

        Args:
            period: Period for Fisher Transform calculation (default: 14)
            smooth_period: Period for signal smoothing (default: 3)
            signal_threshold: Threshold for signal generation (default: 1.5)
            confidence_period: Period for confidence calculation (default: 20)
            fisher_type: Type of Fisher Transform ('price', 'momentum', 'rsi', 'hybrid') (default: 'price')
            adaptive_threshold: Whether to use adaptive thresholds (default: True)
        """
        super().__init__(
            period=period,
            smooth_period=smooth_period,
            signal_threshold=signal_threshold,
            confidence_period=confidence_period,
            fisher_type=fisher_type,
            adaptive_threshold=adaptive_threshold,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate Fisher Signal Type indicator

        Args:
            data: DataFrame with OHLC data or Series of prices

        Returns:
            pd.DataFrame: Fisher signal analysis results
        """
        # Handle input data
        if isinstance(data, pd.Series):
            price = data
            # Create a dummy DataFrame for OHLC calculations
            temp_df = pd.DataFrame(index=data.index)
            temp_df["close"] = price
            temp_df["high"] = price
            temp_df["low"] = price
            data = temp_df
        elif isinstance(data, pd.DataFrame):
            self.validate_input_data(data)
            price = data["close"]
        else:
            raise IndicatorValidationError("Data must be DataFrame or Series")

        # Get parameters
        period = self.parameters.get("period", 14)
        smooth_period = self.parameters.get("smooth_period", 3)
        signal_threshold = self.parameters.get("signal_threshold", 1.5)
        confidence_period = self.parameters.get("confidence_period", 20)
        fisher_type = self.parameters.get("fisher_type", "price")
        adaptive_threshold = self.parameters.get("adaptive_threshold", True)

        # Calculate different Fisher Transform types
        if fisher_type == "price":
            fisher_values = self._calculate_price_fisher(data, period)
        elif fisher_type == "momentum":
            fisher_values = self._calculate_momentum_fisher(price, period)
        elif fisher_type == "rsi":
            fisher_values = self._calculate_rsi_fisher(price, period)
        elif fisher_type == "hybrid":
            fisher_values = self._calculate_hybrid_fisher(data, price, period)
        else:
            raise IndicatorValidationError(f"Unknown fisher_type: {fisher_type}")

        # Smooth the Fisher Transform
        fisher_smooth = fisher_values.rolling(window=smooth_period, min_periods=1).mean()
        
        # Calculate Fisher derivative (signal line)
        fisher_signal = fisher_smooth - fisher_smooth.shift(1)
        
        # Calculate adaptive thresholds if enabled
        if adaptive_threshold:
            upper_threshold, lower_threshold = self._calculate_adaptive_thresholds(
                fisher_smooth, confidence_period, signal_threshold
            )
        else:
            upper_threshold = pd.Series(signal_threshold, index=fisher_smooth.index)
            lower_threshold = pd.Series(-signal_threshold, index=fisher_smooth.index)

        # Generate signal types
        signal_type = self._classify_signal_types(fisher_smooth, fisher_signal, upper_threshold, lower_threshold)
        
        # Calculate signal strength and confidence
        signal_strength = self._calculate_signal_strength(fisher_smooth, upper_threshold, lower_threshold)
        signal_confidence = self._calculate_signal_confidence(fisher_smooth, fisher_signal, confidence_period)
        
        # Calculate trend direction and momentum
        trend_direction = self._calculate_trend_direction(fisher_smooth, fisher_signal)
        momentum_score = self._calculate_momentum_score(fisher_signal, confidence_period)
        
        # Calculate reversal probability
        reversal_probability = self._calculate_reversal_probability(
            fisher_smooth, fisher_signal, signal_strength
        )

        # Create result DataFrame
        result = pd.DataFrame(index=price.index)
        result["fisher_raw"] = fisher_values
        result["fisher_smooth"] = fisher_smooth
        result["fisher_signal"] = fisher_signal
        result["signal_type"] = signal_type
        result["signal_strength"] = signal_strength
        result["signal_confidence"] = signal_confidence
        result["trend_direction"] = trend_direction
        result["momentum_score"] = momentum_score
        result["reversal_probability"] = reversal_probability
        result["upper_threshold"] = upper_threshold
        result["lower_threshold"] = lower_threshold

        # Store calculation details
        self._last_calculation = {
            "price": price,
            "fisher_type": fisher_type,
            "parameters": self.parameters,
            "signal_stats": {
                "total_signals": (signal_type != "neutral").sum(),
                "buy_signals": (signal_type == "buy").sum(),
                "sell_signals": (signal_type == "sell").sum(),
                "reversal_signals": (signal_type.str.contains("reversal", na=False)).sum(),
            }
        }

        return result

    def _calculate_price_fisher(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Fisher Transform based on price (HL2)"""
        # Use median price
        if "high" in data.columns and "low" in data.columns:
            median_price = (data["high"] + data["low"]) / 2
        else:
            median_price = data["close"]
        
        # Calculate min and max over period
        rolling_min = median_price.rolling(window=period, min_periods=1).min()
        rolling_max = median_price.rolling(window=period, min_periods=1).max()
        
        # Normalize to -1 to +1 range
        range_val = rolling_max - rolling_min
        normalized = 2 * (median_price - rolling_min) / (range_val + 1e-10) - 1
        
        # Clip to avoid log errors
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Apply Fisher Transform
        fisher = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
        
        return pd.Series(fisher, index=data.index)

    def _calculate_momentum_fisher(self, price: pd.Series, period: int) -> pd.Series:
        """Calculate Fisher Transform based on momentum"""
        # Calculate momentum
        momentum = price.pct_change(period)
        
        # Normalize momentum
        rolling_min = momentum.rolling(window=period*2, min_periods=1).min()
        rolling_max = momentum.rolling(window=period*2, min_periods=1).max()
        
        range_val = rolling_max - rolling_min
        normalized = 2 * (momentum - rolling_min) / (range_val + 1e-10) - 1
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Apply Fisher Transform
        fisher = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
        
        return pd.Series(fisher, index=price.index)

    def _calculate_rsi_fisher(self, price: pd.Series, period: int) -> pd.Series:
        """Calculate Fisher Transform based on RSI"""
        # Calculate RSI
        delta = price.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Normalize RSI to -1 to +1 range
        normalized = 2 * (rsi / 100) - 1
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Apply Fisher Transform
        fisher = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
        
        return pd.Series(fisher, index=price.index)

    def _calculate_hybrid_fisher(self, data: pd.DataFrame, price: pd.Series, period: int) -> pd.Series:
        """Calculate hybrid Fisher Transform combining multiple methods"""
        price_fisher = self._calculate_price_fisher(data, period)
        momentum_fisher = self._calculate_momentum_fisher(price, period)
        rsi_fisher = self._calculate_rsi_fisher(price, period)
        
        # Weighted combination
        hybrid_fisher = (0.5 * price_fisher + 0.3 * momentum_fisher + 0.2 * rsi_fisher)
        
        return hybrid_fisher

    def _calculate_adaptive_thresholds(
        self, fisher: pd.Series, period: int, base_threshold: float
    ) -> tuple:
        """Calculate adaptive thresholds based on volatility"""
        # Calculate rolling volatility of Fisher values
        fisher_volatility = fisher.rolling(window=period, min_periods=1).std()
        
        # Adaptive threshold based on volatility
        volatility_factor = fisher_volatility / fisher_volatility.rolling(window=period*2, min_periods=1).mean()
        
        # Adjust base threshold
        upper_threshold = base_threshold * (1 + volatility_factor * 0.5)
        lower_threshold = -base_threshold * (1 + volatility_factor * 0.5)
        
        return upper_threshold, lower_threshold

    def _classify_signal_types(
        self, fisher: pd.Series, fisher_signal: pd.Series, 
        upper_threshold: pd.Series, lower_threshold: pd.Series
    ) -> pd.Series:
        """Classify different types of Fisher signals"""
        signals = pd.Series("neutral", index=fisher.index)
        
        # Strong reversal signals (extreme Fisher values with signal line reversal)
        strong_buy_reversal = (
            (fisher < lower_threshold) & 
            (fisher_signal > 0) & 
            (fisher_signal.shift(1) <= 0)
        )
        signals.loc[strong_buy_reversal] = "strong_buy_reversal"
        
        strong_sell_reversal = (
            (fisher > upper_threshold) & 
            (fisher_signal < 0) & 
            (fisher_signal.shift(1) >= 0)
        )
        signals.loc[strong_sell_reversal] = "strong_sell_reversal"
        
        # Regular reversal signals
        buy_reversal = (
            (fisher < lower_threshold * 0.7) & 
            (fisher_signal > 0)
        )
        signals.loc[buy_reversal & ~strong_buy_reversal] = "buy_reversal"
        
        sell_reversal = (
            (fisher > upper_threshold * 0.7) & 
            (fisher_signal < 0)
        )
        signals.loc[sell_reversal & ~strong_sell_reversal] = "sell_reversal"
        
        # Continuation signals
        buy_continuation = (
            (fisher > 0) & 
            (fisher_signal > 0) & 
            (fisher_signal > fisher_signal.shift(1))
        )
        signals.loc[buy_continuation & (signals == "neutral")] = "buy_continuation"
        
        sell_continuation = (
            (fisher < 0) & 
            (fisher_signal < 0) & 
            (fisher_signal < fisher_signal.shift(1))
        )
        signals.loc[sell_continuation & (signals == "neutral")] = "sell_continuation"
        
        # Simple buy/sell signals
        simple_buy = (fisher_signal > 0) & (fisher > -0.5)
        signals.loc[simple_buy & (signals == "neutral")] = "buy"
        
        simple_sell = (fisher_signal < 0) & (fisher < 0.5)
        signals.loc[simple_sell & (signals == "neutral")] = "sell"
        
        return signals

    def _calculate_signal_strength(
        self, fisher: pd.Series, upper_threshold: pd.Series, lower_threshold: pd.Series
    ) -> pd.Series:
        """Calculate signal strength (0-100)"""
        # Normalize Fisher values to 0-100 scale
        strength = np.abs(fisher) / (np.maximum(np.abs(upper_threshold), np.abs(lower_threshold)) + 1e-10)
        strength = np.clip(strength * 100, 0, 100)
        
        return pd.Series(strength, index=fisher.index)

    def _calculate_signal_confidence(
        self, fisher: pd.Series, fisher_signal: pd.Series, period: int
    ) -> pd.Series:
        """Calculate signal confidence based on consistency"""
        # Calculate signal consistency over period
        signal_direction = np.sign(fisher_signal)
        consistency = signal_direction.rolling(window=period, min_periods=1).apply(
            lambda x: np.sum(x == x.iloc[-1]) / len(x) if len(x) > 0 else 0
        )
        
        # Signal magnitude factor
        magnitude_factor = np.abs(fisher_signal) / (np.abs(fisher_signal).rolling(window=period, min_periods=1).max() + 1e-10)
        
        # Combine consistency and magnitude
        confidence = (consistency * 0.7 + magnitude_factor * 0.3) * 100
        
        return pd.Series(confidence, index=fisher.index)

    def _calculate_trend_direction(self, fisher: pd.Series, fisher_signal: pd.Series) -> pd.Series:
        """Calculate overall trend direction"""
        # Combine Fisher level and signal direction
        trend = np.sign(fisher) * 0.6 + np.sign(fisher_signal) * 0.4
        
        return pd.Series(trend, index=fisher.index)

    def _calculate_momentum_score(self, fisher_signal: pd.Series, period: int) -> pd.Series:
        """Calculate momentum score based on signal acceleration"""
        # Calculate signal change acceleration
        signal_change = fisher_signal.diff()
        signal_acceleration = signal_change.diff()
        
        # Normalize momentum score
        momentum = signal_acceleration.rolling(window=period, min_periods=1).mean()
        momentum_std = signal_acceleration.rolling(window=period, min_periods=1).std()
        
        momentum_score = momentum / (momentum_std + 1e-10)
        
        return pd.Series(momentum_score, index=fisher_signal.index)

    def _calculate_reversal_probability(
        self, fisher: pd.Series, fisher_signal: pd.Series, signal_strength: pd.Series
    ) -> pd.Series:
        """Calculate probability of price reversal"""
        # High Fisher values with declining signal suggest reversal
        extreme_fisher = np.abs(fisher) > 1.5
        declining_signal = fisher_signal < fisher_signal.shift(1)
        high_strength = signal_strength > 70
        
        # Calculate reversal probability
        reversal_prob = pd.Series(0.0, index=fisher.index)
        
        # High probability conditions
        high_prob_condition = extreme_fisher & declining_signal & high_strength
        reversal_prob.loc[high_prob_condition] = 0.8
        
        # Medium probability conditions
        medium_prob_condition = (extreme_fisher & declining_signal) | (extreme_fisher & high_strength)
        reversal_prob.loc[medium_prob_condition & ~high_prob_condition] = 0.5
        
        # Low probability conditions
        low_prob_condition = extreme_fisher & ~declining_signal & ~high_strength
        reversal_prob.loc[low_prob_condition & (reversal_prob == 0)] = 0.2
        
        return reversal_prob

    def validate_parameters(self) -> bool:
        """Validate parameters"""
        period = self.parameters.get("period", 14)
        smooth_period = self.parameters.get("smooth_period", 3)
        signal_threshold = self.parameters.get("signal_threshold", 1.5)
        confidence_period = self.parameters.get("confidence_period", 20)
        fisher_type = self.parameters.get("fisher_type", "price")

        if not isinstance(period, int) or period < 1:
            raise IndicatorValidationError(f"period must be positive integer, got {period}")
        
        if period > 200:
            raise IndicatorValidationError(f"period too large, maximum 200, got {period}")

        if not isinstance(smooth_period, int) or smooth_period < 1:
            raise IndicatorValidationError(f"smooth_period must be positive integer, got {smooth_period}")

        if not isinstance(signal_threshold, (int, float)) or signal_threshold <= 0:
            raise IndicatorValidationError(f"signal_threshold must be positive, got {signal_threshold}")

        if not isinstance(confidence_period, int) or confidence_period < 1:
            raise IndicatorValidationError(f"confidence_period must be positive integer, got {confidence_period}")

        valid_types = ["price", "momentum", "rsi", "hybrid"]
        if fisher_type not in valid_types:
            raise IndicatorValidationError(f"fisher_type must be one of {valid_types}, got {fisher_type}")

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata"""
        return {
            "name": "FisherSignalType",
            "category": self.CATEGORY,
            "description": "Advanced Fisher Transform signal detection with multiple variants and signal classification",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "DataFrame",
            "output_columns": [
                "fisher_raw", "fisher_smooth", "fisher_signal", "signal_type",
                "signal_strength", "signal_confidence", "trend_direction",
                "momentum_score", "reversal_probability", "upper_threshold", "lower_threshold"
            ],
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
        }

    def _get_required_columns(self) -> List[str]:
        """Required columns"""
        return ["close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed"""
        return max(self.parameters.get("period", 14) * 2, 20)

    def _setup_defaults(self):
        """Setup default parameter values"""
        defaults = {
            "period": 14,
            "smooth_period": 3,
            "signal_threshold": 1.5,
            "confidence_period": 20,
            "fisher_type": "price",
            "adaptive_threshold": True,
        }
        
        for key, value in defaults.items():
            if key not in self.parameters:
                self.parameters[key] = value

    # Backward compatibility properties
    @property
    def minimum_periods(self) -> int:
        return self._get_minimum_data_points()

    def get_config(self) -> Dict[str, Any]:
        """Get indicator configuration"""
        return {
            "indicator": "FisherSignalType",
            "category": self.CATEGORY,
            "parameters": self.parameters,
            "version": self.VERSION,
        }


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return FisherSignalTypeIndicator