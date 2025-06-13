"""
FractalChaosOscillator - Individual Implementation
Real chaos theory-based market analysis using fractal dimensions
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class FractalChaosResult:
    """Result structure for Fractal Chaos analysis"""

    chaos_value: float
    fractal_dimension: float
    market_regime: str  # 'ranging', 'trending', 'chaotic'
    complexity_score: float
    predictability_index: float
    regime_strength: float


class FractalChaosOscillator:
    """
    Real Fractal Chaos Oscillator Implementation

    Applies chaos theory to financial markets to detect regime changes and measure
    market complexity. Uses fractal dimension calculation, Lyapunov exponents,
    and chaos metrics to identify market states.

    Features:
    - Chaos theory-based market analysis
    - Fractal dimension calculation using box-counting method
    - Market regime detection (chaotic/trending/ranging)
    - Complexity scoring and predictability assessment
    - Real-time regime strength measurement
    """

    def __init__(
        self,
        period=20,
        chaos_window=20,
        dimension_scales=10,
        regime_threshold=0.5,
        **kwargs,
    ):
        """Initialize Fractal Chaos Oscillator."""
        self.period = period
        self.chaos_window = chaos_window
        self.dimension_scales = dimension_scales
        self.regime_threshold = regime_threshold
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)

    def calculate_fractal_dimension(self, data: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method."""
        if len(data) < 10:
            return 1.5

        try:
            # Normalize data to [0, 1] range
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-10)

            # Create different box sizes
            scales = np.logspace(0.01, 1, self.dimension_scales)
            counts = []

            for scale in scales:
                # Create grid with given scale
                grid_size = int(1 / scale)
                if grid_size < 2:
                    grid_size = 2

                # Count boxes that contain data points
                boxes_with_data = set()
                for i, value in enumerate(data_norm):
                    x_box = int(i * grid_size / len(data_norm))
                    y_box = int(value * grid_size)
                    boxes_with_data.add((x_box, y_box))

                counts.append(len(boxes_with_data))

            # Calculate fractal dimension from log-log slope
            log_scales = np.log(scales)
            log_counts = np.log(np.array(counts) + 1e-10)

            # Linear regression to find slope
            if len(log_scales) > 1:
                slope, _ = np.polyfit(log_scales, log_counts, 1)
                fractal_dim = -slope
            else:
                fractal_dim = 1.5

            return np.clip(fractal_dim, 1.0, 2.0)

        except Exception as e:
            self.logger.error(f"Error calculating fractal dimension: {e}")
            return 1.5

    def calculate_lyapunov_exponent(self, data: np.ndarray) -> float:
        """Estimate Lyapunov exponent for chaos measurement."""
        if len(data) < 10:
            return 0.0

        try:
            # Calculate log returns
            returns = np.diff(np.log(data + 1e-10))

            # Estimate Lyapunov exponent using nearest neighbor method
            lyap_sum = 0.0
            count = 0

            for i in range(len(returns) - 5):
                # Find nearest neighbor
                distances = np.abs(returns[i] - returns)
                sorted_indices = np.argsort(distances)

                # Skip self and find nearest neighbor
                for j in sorted_indices[1:]:
                    if abs(j - i) > 1:  # Avoid temporal correlation
                        initial_distance = distances[j]
                        if initial_distance > 1e-10:
                            # Calculate divergence after one step
                            if i + 1 < len(returns) and j + 1 < len(returns):
                                final_distance = abs(returns[i + 1] - returns[j + 1])
                                if final_distance > 1e-10:
                                    lyap_sum += np.log(
                                        final_distance / initial_distance
                                    )
                                    count += 1
                        break

            return lyap_sum / max(count, 1)

        except Exception as e:
            self.logger.error(f"Error calculating Lyapunov exponent: {e}")
            return 0.0

    def calculate_complexity_score(self, data: np.ndarray) -> float:
        """Calculate market complexity score."""
        try:
            # Calculate various complexity measures
            returns = np.diff(np.log(data + 1e-10))

            # Variance-based complexity
            variance_complexity = np.std(returns)

            # Autocorrelation-based complexity
            if len(returns) > 1:
                autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                autocorr_complexity = (
                    1 - abs(autocorr) if not np.isnan(autocorr) else 0.5
                )
            else:
                autocorr_complexity = 0.5

            # Range-based complexity
            range_complexity = (np.max(data) - np.min(data)) / (np.mean(data) + 1e-10)

            # Combine measures
            complexity = np.mean(
                [
                    np.clip(variance_complexity * 10, 0, 1),
                    autocorr_complexity,
                    np.clip(range_complexity / 10, 0, 1),
                ]
            )

            return float(complexity)

        except Exception as e:
            self.logger.error(f"Error calculating complexity score: {e}")
            return 0.5

    def determine_market_regime(
        self, fractal_dim: float, chaos_value: float, complexity: float
    ) -> Tuple[str, float]:
        """Determine market regime based on chaos metrics."""
        # Combine metrics for regime detection
        chaos_score = abs(chaos_value)
        regime_metric = (fractal_dim - 1.0) * 0.4 + chaos_score * 0.3 + complexity * 0.3

        if regime_metric < self.regime_threshold * 0.7:
            return "ranging", regime_metric
        elif regime_metric > self.regime_threshold * 1.3:
            return "chaotic", regime_metric
        else:
            return "trending", regime_metric

    def calculate(self, data: Union[Dict, pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """Calculate Fractal Chaos Oscillator analysis."""
        try:
            # Extract price data
            if isinstance(data, dict):
                prices = np.array(data.get("close", data.get("Close", [])))
            elif isinstance(data, pd.DataFrame):
                prices = (
                    data["close"].values
                    if "close" in data.columns
                    else data["Close"].values
                )
            else:
                prices = data.flatten() if hasattr(data, "flatten") else np.array(data)

            if len(prices) < self.chaos_window:
                return {"error": "Insufficient data for chaos analysis"}

            # Use recent window for analysis
            recent_prices = prices[-self.chaos_window :]

            # Calculate chaos metrics
            fractal_dimension = self.calculate_fractal_dimension(recent_prices)
            lyapunov_exp = self.calculate_lyapunov_exponent(recent_prices)
            complexity_score = self.calculate_complexity_score(recent_prices)

            # Calculate main chaos value
            chaos_value = (fractal_dimension - 1.5) * 2 + lyapunov_exp

            # Determine market regime
            market_regime, regime_strength = self.determine_market_regime(
                fractal_dimension, chaos_value, complexity_score
            )

            # Calculate predictability index (inverse of chaos)
            predictability_index = max(0, 1 - abs(chaos_value))

            return {
                "chaos_value": float(chaos_value),
                "fractal_dimension": float(fractal_dimension),
                "market_regime": market_regime,
                "complexity_score": float(complexity_score),
                "predictability_index": float(predictability_index),
                "regime_strength": float(regime_strength),
                "lyapunov_exponent": float(lyapunov_exp),
            }

        except Exception as e:
            self.logger.error(f"Error calculating Fractal Chaos Oscillator: {e}")
            return {"error": str(e)}
