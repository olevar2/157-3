"""
Correlation Matrix - Real Implementation
Analyzes correlations between different price series or indicators for momentum analysis
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from ..base_indicator import StandardIndicatorInterface as BaseIndicator
except ImportError:
    # Fallback if base_indicator is not available
    class BaseIndicator:
        def __init__(self, **kwargs):
            pass


class CorrelationMatrixIndicator(BaseIndicator):
    """
    Correlation Matrix - Real Implementation

    Calculates correlation coefficients between multiple price series or indicators
    to identify momentum relationships and market regime changes
    """

    def __init__(self, period=20, correlation_threshold=0.7, **kwargs):
        super().__init__(**kwargs)
        self.period = period
        self.correlation_threshold = correlation_threshold
        self.logger = logging.getLogger(__name__)

    def _calculate_correlation(self, series1, series2):
        """Calculate Pearson correlation coefficient between two series"""
        if len(series1) != len(series2) or len(series1) < 2:
            return 0.0

        # Convert to numpy arrays
        x = np.array(series1)
        y = np.array(series2)

        # Calculate correlation coefficient
        correlation_matrix = np.corrcoef(x, y)
        return (
            correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0
        )

    def _calculate_price_changes(self, prices):
        """Calculate price changes (returns) for correlation analysis"""
        if len(prices) < 2:
            return []

        changes = []
        for i in range(1, len(prices)):
            if prices[i - 1] != 0:
                change = (prices[i] - prices[i - 1]) / prices[i - 1]
                changes.append(change)
            else:
                changes.append(0)

        return changes

    def calculate(self, data) -> Optional[Dict]:
        """Calculate Correlation Matrix analysis"""
        try:
            # Parse input data - expecting multiple series
            if isinstance(data, pd.DataFrame):
                # Extract price columns for correlation analysis
                price_columns = (
                    ["close", "high", "low", "open"] if "close" in data.columns else []
                )
                if len(price_columns) < 2:
                    # If we only have close, create synthetic series for analysis
                    if "close" in data.columns:
                        closes = data["close"].values
                        if len(closes) < self.period:
                            return None

                        # Create synthetic momentum series for correlation analysis
                        series_dict = {
                            "price": closes,
                            "sma_5": self._calculate_sma(closes, 5),
                            "sma_10": self._calculate_sma(closes, 10),
                            "momentum": self._calculate_momentum(closes, 5),
                        }
                    else:
                        return None
                else:
                    # Use available price columns
                    series_dict = {}
                    for col in price_columns:
                        if col in data.columns:
                            series_dict[col] = data[col].values

            elif isinstance(data, dict):
                # Extract multiple series from dictionary
                series_dict = {}
                for key, values in data.items():
                    if (
                        isinstance(values, (list, np.ndarray))
                        and len(values) >= self.period
                    ):
                        series_dict[key] = np.array(values)

                if len(series_dict) < 2:
                    # Create synthetic series if insufficient data
                    if "close" in data:
                        closes = np.array(data["close"])
                        series_dict = {
                            "price": closes,
                            "returns": self._calculate_price_changes(closes),
                            "momentum": self._calculate_momentum(closes, 5),
                        }
                    else:
                        return None
            else:
                return None

            if len(series_dict) < 2:
                return None

            # Ensure all series have the same length
            min_length = min(len(series) for series in series_dict.values())
            if min_length < self.period:
                return None

            # Trim all series to the same length
            trimmed_series = {}
            for key, series in series_dict.items():
                trimmed_series[key] = series[-min_length:]

            # Calculate correlation matrix for the most recent period
            recent_length = min(self.period, min_length)
            correlation_matrix = {}
            series_names = list(trimmed_series.keys())

            for i, name1 in enumerate(series_names):
                correlation_matrix[name1] = {}
                series1 = trimmed_series[name1][-recent_length:]

                for j, name2 in enumerate(series_names):
                    series2 = trimmed_series[name2][-recent_length:]
                    correlation = self._calculate_correlation(series1, series2)
                    correlation_matrix[name1][name2] = float(correlation)

            # Analyze correlation patterns
            high_correlations = []
            low_correlations = []

            for name1 in series_names:
                for name2 in series_names:
                    if name1 != name2:
                        corr = correlation_matrix[name1][name2]
                        if abs(corr) >= self.correlation_threshold:
                            high_correlations.append(
                                {
                                    "pair": f"{name1}-{name2}",
                                    "correlation": corr,
                                    "type": "positive" if corr > 0 else "negative",
                                }
                            )
                        elif abs(corr) <= 0.3:
                            low_correlations.append(
                                {"pair": f"{name1}-{name2}", "correlation": corr}
                            )

            # Calculate average correlation strength
            all_correlations = []
            for name1 in series_names:
                for name2 in series_names:
                    if name1 != name2:
                        all_correlations.append(abs(correlation_matrix[name1][name2]))

            avg_correlation = np.mean(all_correlations) if all_correlations else 0.0

            # Determine market regime based on correlations
            if avg_correlation > 0.7:
                market_regime = "high_correlation"
                signal = "risk_off"  # High correlations often indicate stress
                trend = "synchronized"
            elif avg_correlation > 0.4:
                market_regime = "moderate_correlation"
                signal = "neutral"
                trend = "mixed"
            else:
                market_regime = "low_correlation"
                signal = "risk_on"  # Low correlations indicate diversification
                trend = "dispersed"

            # Calculate signal strength based on correlation extremes
            max_correlation = max(all_correlations) if all_correlations else 0
            strength = max_correlation * 100

            # Calculate confidence based on consistency of correlations
            correlation_std = (
                np.std(all_correlations) if len(all_correlations) > 1 else 0
            )
            confidence = max(
                30, 100 - (correlation_std * 200)
            )  # Lower std = higher confidence

            return {
                "correlation_matrix": correlation_matrix,
                "average_correlation": float(avg_correlation),
                "max_correlation": float(max_correlation),
                "market_regime": market_regime,
                "signal": signal,
                "trend": trend,
                "strength": float(strength),
                "confidence": int(confidence),
                "high_correlations": high_correlations,
                "low_correlations": low_correlations,
                "correlation_count": len(all_correlations),
                "period": self.period,
                "threshold": self.correlation_threshold,
            }

        except Exception as e:
            self.logger.error(f"Error calculating Correlation Matrix: {e}")
            return None

    def _calculate_sma(self, data, period):
        """Calculate Simple Moving Average"""
        if len(data) < period:
            return np.array([])

        sma_values = []
        for i in range(period - 1, len(data)):
            sma = np.mean(data[i - period + 1 : i + 1])
            sma_values.append(sma)

        return np.array(sma_values)

    def _calculate_momentum(self, data, period):
        """Calculate Momentum (Rate of Change)"""
        if len(data) < period + 1:
            return np.array([])

        momentum_values = []
        for i in range(period, len(data)):
            if data[i - period] != 0:
                momentum = (data[i] - data[i - period]) / data[i - period]
                momentum_values.append(momentum)
            else:
                momentum_values.append(0)

        return np.array(momentum_values)

    def get_signals(self, data) -> Dict:
        """Get trading signals from Correlation Matrix analysis"""
        result = self.calculate(data)
        if not result:
            return {"action": "hold", "reason": "insufficient_data"}

        market_regime = result["market_regime"]
        avg_correlation = result["average_correlation"]
        signal = result["signal"]

        if market_regime == "high_correlation" and signal == "risk_off":
            return {
                "action": "reduce_risk",
                "reason": "high_correlation_regime",
                "confidence": result["confidence"],
                "avg_correlation": avg_correlation,
                "regime": market_regime,
            }
        elif market_regime == "low_correlation" and signal == "risk_on":
            return {
                "action": "increase_diversification",
                "reason": "low_correlation_regime",
                "confidence": result["confidence"],
                "avg_correlation": avg_correlation,
                "regime": market_regime,
            }
        elif len(result["high_correlations"]) > 2:
            return {
                "action": "monitor_risk",
                "reason": "multiple_high_correlations",
                "confidence": result["confidence"],
                "avg_correlation": avg_correlation,
                "high_corr_count": len(result["high_correlations"]),
            }
        else:
            return {
                "action": "normal_allocation",
                "reason": "balanced_correlations",
                "confidence": result["confidence"],
                "avg_correlation": avg_correlation,
                "regime": market_regime,
            }
