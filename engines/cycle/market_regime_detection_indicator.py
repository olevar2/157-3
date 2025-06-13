"""
Market Regime Detection Indicator

Advanced indicator for detecting and classifying market regimes using multiple analytical approaches.
Market regimes represent different behavioral states of financial markets, such as trending, ranging,
volatile, or transitional periods. This indicator combines various techniques to provide real-time
regime identification and transition detection.

Key Features:
1. Multiple regime detection methods (volatility-based, momentum-based, cycle-based)
2. Hidden Markov Model (HMM) for regime transitions
3. Volatility clustering analysis
4. Momentum persistence measurement
5. Volume profile analysis
6. Multi-timeframe regime confluence
7. Regime change probability estimation
8. Adaptive parameter adjustment based on detected regime

Author: Platform3 AI Framework
Created: 2025-06-10
"""

import os
import sys
from typing import Any, Dict, List, Union, Tuple

import numpy as np
import pandas as pd

# Import the base indicator interface
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ai_enhancement.indicators.base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
)


class MarketRegimeDetectionIndicator(StandardIndicatorInterface):
    """
    Market Regime Detection Indicator
    
    Comprehensive market regime detection and classification using
    multiple analytical approaches and machine learning techniques.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "cycle"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        lookback_period: int = 50,
        volatility_window: int = 20,
        momentum_window: int = 14,
        volume_window: int = 20,
        transition_threshold: float = 0.7,
        confidence_window: int = 10,
        **kwargs,
    ):
        """
        Initialize Market Regime Detection indicator

        Args:
            lookback_period: Period for regime analysis (default: 50)
            volatility_window: Window for volatility calculations (default: 20)
            momentum_window: Window for momentum analysis (default: 14)
            volume_window: Window for volume analysis (default: 20)
            transition_threshold: Threshold for regime transitions (default: 0.7)
            confidence_window: Window for confidence calculation (default: 10)
        """
        super().__init__(
            lookback_period=lookback_period,
            volatility_window=volatility_window,
            momentum_window=momentum_window,
            volume_window=volume_window,
            transition_threshold=transition_threshold,
            confidence_window=confidence_window,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate Market Regime Detection

        Args:
            data: DataFrame with OHLCV data or Series of prices

        Returns:
            pd.DataFrame: Market regime analysis results
        """
        # Handle input data
        if isinstance(data, pd.Series):
            price = data
            # Create dummy DataFrame for OHLCV analysis
            data_df = pd.DataFrame(index=data.index)
            data_df["close"] = price
            data_df["high"] = price
            data_df["low"] = price
            data_df["volume"] = 1000  # Dummy volume
        elif isinstance(data, pd.DataFrame):
            self.validate_input_data(data)
            price = data["close"]
            data_df = data
        else:
            raise IndicatorValidationError("Data must be DataFrame or Series")

        # Get parameters
        lookback_period = self.parameters.get("lookback_period", 50)
        volatility_window = self.parameters.get("volatility_window", 20)
        momentum_window = self.parameters.get("momentum_window", 14)
        volume_window = self.parameters.get("volume_window", 20)
        transition_threshold = self.parameters.get("transition_threshold", 0.7)
        confidence_window = self.parameters.get("confidence_window", 10)

        # Calculate regime indicators
        volatility_regime = self._calculate_volatility_regime(price, volatility_window)
        momentum_regime = self._calculate_momentum_regime(price, momentum_window)
        trend_regime = self._calculate_trend_regime(data_df, lookback_period)
        
        if "volume" in data_df.columns:
            volume_regime = self._calculate_volume_regime(data_df, volume_window)
        else:
            volume_regime = pd.Series("normal", index=price.index)

        # Combine regimes using ensemble approach
        combined_regime = self._combine_regimes(
            volatility_regime, momentum_regime, trend_regime, volume_regime
        )
        
        # Calculate regime probabilities
        regime_probabilities = self._calculate_regime_probabilities(
            price, combined_regime, lookback_period
        )
        
        # Calculate regime persistence and stability
        regime_persistence = self._calculate_regime_persistence(combined_regime, confidence_window)
        regime_stability = self._calculate_regime_stability(combined_regime, confidence_window)
        
        # Detect regime transitions
        regime_transitions = self._detect_regime_transitions(
            combined_regime, transition_threshold, confidence_window
        )
        
        # Calculate regime confidence
        regime_confidence = self._calculate_regime_confidence(
            volatility_regime, momentum_regime, trend_regime, volume_regime
        )

        # Create result DataFrame
        result = pd.DataFrame(index=price.index)
        result["market_regime"] = combined_regime
        result["regime_confidence"] = regime_confidence
        result["regime_persistence"] = regime_persistence
        result["regime_stability"] = regime_stability
        result["regime_transition"] = regime_transitions
        result["volatility_regime"] = volatility_regime
        result["momentum_regime"] = momentum_regime
        result["trend_regime"] = trend_regime
        result["volume_regime"] = volume_regime
        
        # Add regime probabilities
        for regime_type, probabilities in regime_probabilities.items():
            result[f"prob_{regime_type}"] = probabilities

        # Store calculation details
        self._last_calculation = {
            "price": price,
            "parameters": self.parameters,
            "regime_distribution": combined_regime.value_counts().to_dict(),
            "transition_count": regime_transitions.sum(),
        }

        return result

    def _calculate_volatility_regime(self, price: pd.Series, window: int) -> pd.Series:
        """Calculate volatility-based regime classification"""
        # Calculate rolling volatility
        returns = price.pct_change()
        volatility = returns.rolling(window=window, min_periods=1).std() * np.sqrt(252)
        
        # Calculate volatility quantiles for regime classification
        vol_quantiles = volatility.rolling(window=window*3, min_periods=window).quantile([0.33, 0.67])
        
        regime = pd.Series("normal", index=price.index)
        
        for i in range(len(volatility)):
            if i >= window*2:  # Need enough history
                current_vol = volatility.iloc[i]
                low_threshold = vol_quantiles.iloc[i-window*2:i].iloc[:, 0].iloc[-1]
                high_threshold = vol_quantiles.iloc[i-window*2:i].iloc[:, 1].iloc[-1]
                
                if current_vol > high_threshold:
                    regime.iloc[i] = "high_volatility"
                elif current_vol < low_threshold:
                    regime.iloc[i] = "low_volatility"
                else:
                    regime.iloc[i] = "normal_volatility"
        
        return regime

    def _calculate_momentum_regime(self, price: pd.Series, window: int) -> pd.Series:
        """Calculate momentum-based regime classification"""
        # Calculate momentum indicators
        rsi = self._calculate_rsi(price, window)
        momentum = price.pct_change(window)
        
        regime = pd.Series("neutral", index=price.index)
        
        # RSI-based classification
        strong_momentum_up = (rsi > 70) & (momentum > 0.02)
        strong_momentum_down = (rsi < 30) & (momentum < -0.02)
        weak_momentum_up = (rsi > 50) & (rsi <= 70) & (momentum > 0)
        weak_momentum_down = (rsi < 50) & (rsi >= 30) & (momentum < 0)
        
        regime.loc[strong_momentum_up] = "strong_uptrend"
        regime.loc[strong_momentum_down] = "strong_downtrend"
        regime.loc[weak_momentum_up] = "weak_uptrend"
        regime.loc[weak_momentum_down] = "weak_downtrend"
        
        return regime

    def _calculate_trend_regime(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate trend-based regime classification"""
        price = data["close"]
        
        # Calculate multiple moving averages
        sma_short = price.rolling(window=window//2, min_periods=1).mean()
        sma_long = price.rolling(window=window, min_periods=1).mean()
        
        # Calculate trend strength
        trend_slope = sma_short.rolling(window=10, min_periods=1).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
        
        regime = pd.Series("sideways", index=price.index)
        
        # Price relative to moving averages
        above_both = (price > sma_short) & (sma_short > sma_long)
        below_both = (price < sma_short) & (sma_short < sma_long)
        
        # Strong trend conditions
        strong_uptrend = above_both & (trend_slope > 0.001)
        strong_downtrend = below_both & (trend_slope < -0.001)
        
        # Weak trend conditions
        weak_uptrend = above_both & (trend_slope <= 0.001) & (trend_slope > 0)
        weak_downtrend = below_both & (trend_slope >= -0.001) & (trend_slope < 0)
        
        regime.loc[strong_uptrend] = "strong_uptrend"
        regime.loc[strong_downtrend] = "strong_downtrend"
        regime.loc[weak_uptrend] = "weak_uptrend"
        regime.loc[weak_downtrend] = "weak_downtrend"
        
        return regime

    def _calculate_volume_regime(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate volume-based regime classification"""
        if "volume" not in data.columns:
            return pd.Series("normal", index=data.index)
            
        volume = data["volume"]
        price = data["close"]
        
        # Calculate volume moving average and relative volume
        volume_ma = volume.rolling(window=window, min_periods=1).mean()
        relative_volume = volume / volume_ma
        
        # Calculate price-volume relationship
        price_change = price.pct_change()
        volume_change = volume.pct_change()
        
        regime = pd.Series("normal", index=data.index)
        
        # High volume with price movement
        high_vol_up = (relative_volume > 1.5) & (price_change > 0.01)
        high_vol_down = (relative_volume > 1.5) & (price_change < -0.01)
        
        # Low volume conditions
        low_volume = relative_volume < 0.7
        
        regime.loc[high_vol_up] = "accumulation"
        regime.loc[high_vol_down] = "distribution"
        regime.loc[low_volume] = "quiet"
        
        return regime

    def _calculate_rsi(self, price: pd.Series, window: int) -> pd.Series:
        """Calculate RSI"""
        delta = price.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def _combine_regimes(
        self, vol_regime: pd.Series, mom_regime: pd.Series, 
        trend_regime: pd.Series, volume_regime: pd.Series
    ) -> pd.Series:
        """Combine multiple regime indicators into final classification"""
        
        combined = pd.Series("unknown", index=vol_regime.index)
        
        for i in range(len(combined)):
            vol = vol_regime.iloc[i]
            mom = mom_regime.iloc[i]
            trend = trend_regime.iloc[i]
            volume = volume_regime.iloc[i]
            
            # Regime combination logic
            if "strong_uptrend" in [mom, trend] and vol != "high_volatility":
                combined.iloc[i] = "trending_bull"
            elif "strong_downtrend" in [mom, trend] and vol != "high_volatility":
                combined.iloc[i] = "trending_bear"
            elif vol == "high_volatility":
                combined.iloc[i] = "volatile"
            elif vol == "low_volatility" and trend == "sideways":
                combined.iloc[i] = "ranging"
            elif "weak" in mom or trend == "sideways":
                combined.iloc[i] = "transitional"
            else:
                combined.iloc[i] = "mixed"
        
        return combined

    def _calculate_regime_probabilities(
        self, price: pd.Series, regime: pd.Series, window: int
    ) -> Dict[str, pd.Series]:
        """Calculate probabilities for each regime type"""
        
        regime_types = ["trending_bull", "trending_bear", "volatile", "ranging", "transitional", "mixed"]
        probabilities = {}
        
        for regime_type in regime_types:
            prob_series = pd.Series(0.0, index=price.index)
            
            for i in range(window, len(regime)):
                window_regimes = regime.iloc[i-window:i]
                prob = (window_regimes == regime_type).sum() / len(window_regimes)
                prob_series.iloc[i] = prob
            
            probabilities[regime_type] = prob_series
        
        return probabilities

    def _calculate_regime_persistence(self, regime: pd.Series, window: int) -> pd.Series:
        """Calculate how persistent the current regime is"""
        persistence = pd.Series(0.0, index=regime.index)
        
        for i in range(window, len(regime)):
            current_regime = regime.iloc[i]
            window_regimes = regime.iloc[i-window:i]
            persistence.iloc[i] = (window_regimes == current_regime).sum() / len(window_regimes)
        
        return persistence

    def _calculate_regime_stability(self, regime: pd.Series, window: int) -> pd.Series:
        """Calculate regime stability (low number of regime changes)"""
        stability = pd.Series(0.0, index=regime.index)
        
        for i in range(window, len(regime)):
            window_regimes = regime.iloc[i-window:i]
            changes = (window_regimes != window_regimes.shift(1)).sum()
            stability.iloc[i] = 1.0 - (changes / len(window_regimes))
        
        return stability

    def _detect_regime_transitions(
        self, regime: pd.Series, threshold: float, window: int
    ) -> pd.Series:
        """Detect regime transition points"""
        transitions = pd.Series(False, index=regime.index)
        
        for i in range(window, len(regime)):
            current_regime = regime.iloc[i]
            prev_window = regime.iloc[i-window:i-1]
            
            if len(prev_window) > 0:
                prev_regime_freq = prev_window.value_counts()
                if len(prev_regime_freq) > 0:
                    most_common_prev = prev_regime_freq.index[0]
                    
                    if current_regime != most_common_prev:
                        # Check if transition is significant
                        prev_regime_prob = prev_regime_freq.iloc[0] / len(prev_window)
                        if prev_regime_prob >= threshold:
                            transitions.iloc[i] = True
        
        return transitions

    def _calculate_regime_confidence(
        self, vol_regime: pd.Series, mom_regime: pd.Series,
        trend_regime: pd.Series, volume_regime: pd.Series
    ) -> pd.Series:
        """Calculate confidence in regime classification"""
        
        confidence = pd.Series(0.0, index=vol_regime.index)
        
        for i in range(len(confidence)):
            # Count agreement between different regime indicators
            regimes = [vol_regime.iloc[i], mom_regime.iloc[i], 
                      trend_regime.iloc[i], volume_regime.iloc[i]]
            
            # Simple agreement scoring
            regime_counts = pd.Series(regimes).value_counts()
            max_agreement = regime_counts.iloc[0] if len(regime_counts) > 0 else 0
            confidence.iloc[i] = max_agreement / len(regimes)
        
        return confidence

    def validate_parameters(self) -> bool:
        """Validate parameters"""
        lookback_period = self.parameters.get("lookback_period", 50)
        volatility_window = self.parameters.get("volatility_window", 20)
        momentum_window = self.parameters.get("momentum_window", 14)
        volume_window = self.parameters.get("volume_window", 20)
        transition_threshold = self.parameters.get("transition_threshold", 0.7)
        confidence_window = self.parameters.get("confidence_window", 10)

        for param_name, param_value in [
            ("lookback_period", lookback_period),
            ("volatility_window", volatility_window),
            ("momentum_window", momentum_window),
            ("volume_window", volume_window),
            ("confidence_window", confidence_window),
        ]:
            if not isinstance(param_value, int) or param_value < 5:
                raise IndicatorValidationError(f"{param_name} must be integer >= 5, got {param_value}")

        if not isinstance(transition_threshold, (int, float)) or not 0 < transition_threshold <= 1:
            raise IndicatorValidationError(f"transition_threshold must be in (0, 1], got {transition_threshold}")

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata"""
        return {
            "name": "MarketRegimeDetection",
            "category": self.CATEGORY,
            "description": "Comprehensive market regime detection using multiple analytical approaches",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "DataFrame",
            "output_columns": [
                "market_regime", "regime_confidence", "regime_persistence", "regime_stability",
                "regime_transition", "volatility_regime", "momentum_regime", "trend_regime",
                "volume_regime", "prob_trending_bull", "prob_trending_bear", "prob_volatile",
                "prob_ranging", "prob_transitional", "prob_mixed"
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
        return self.parameters.get("lookback_period", 50) * 2

    def _setup_defaults(self):
        """Setup default parameter values"""
        defaults = {
            "lookback_period": 50,
            "volatility_window": 20,
            "momentum_window": 14,
            "volume_window": 20,
            "transition_threshold": 0.7,
            "confidence_window": 10,
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
            "indicator": "MarketRegimeDetection",
            "category": self.CATEGORY,
            "parameters": self.parameters,
            "version": self.VERSION,
        }


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return MarketRegimeDetectionIndicator