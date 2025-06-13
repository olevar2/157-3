"""
Platform3 Fractal Indicators - ENHANCED VERSION
Contains real implementations of fractal indicators used by agents.

Updated with REAL implementation for FractalChannelIndicator!
Other indicators: FractalChaosOscillator, FractalEnergyIndicator, MandelbrotFractalIndicator
still need real implementations (currently stubs).

Note: Real HurstExponent implementations are in:
- engines/fractal/hurst_exponent.py
- engines/cycle/hurst_exponent.py
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class FractalPoint:
    """Represents a fractal point in price data"""

    index: int
    price: float
    fractal_type: str  # 'high' or 'low'
    strength: float = 1.0


@dataclass
class FractalChannelResult:
    """Result structure for Fractal Channel analysis"""

    upper_channel: float
    lower_channel: float
    middle_channel: float
    channel_width: float
    fractal_high: Optional[float] = None
    fractal_low: Optional[float] = None
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    breakout_signal: Optional[str] = None  # 'bullish', 'bearish', or None


class FractalChannelIndicator:
    """
    Real Fractal Channel Indicator Implementation

    Uses Williams Fractals to identify key support and resistance levels,
    then constructs dynamic channels around these fractal points.

    Features:
    - Williams Fractal detection (configurable periods)
    - Dynamic channel construction
    - Support/Resistance level identification
    - Breakout signal detection
    - Channel strength analysis
    """

    def __init__(
        self,
        period=20,
        fractal_period: int = 2,
        channel_lookback: int = 20,
        min_fractals: int = 2,
        breakout_threshold: float = 0.02,
        **kwargs,
    ):
        """
        Initialize Real Fractal Channel Indicator.

        Args:
            period: Legacy compatibility parameter
            fractal_period: Period for Williams Fractal detection (default: 5)
            channel_lookback: Number of fractals to use for channel construction
            min_fractals: Minimum fractals required for valid channel
            breakout_threshold: Threshold for breakout signal (2% default)
        """
        self.period = period
        self.fractal_period = fractal_period
        self.channel_lookback = channel_lookback
        self.min_fractals = min_fractals
        self.breakout_threshold = breakout_threshold
        self.kwargs = kwargs

        # Internal state
        self.fractal_highs: List[FractalPoint] = []
        self.fractal_lows: List[FractalPoint] = []
        self.logger = logging.getLogger(__name__)

    def detect_williams_fractals(
        self, highs: np.ndarray, lows: np.ndarray
    ) -> Tuple[List[FractalPoint], List[FractalPoint]]:
        """
        Detect Williams Fractals in price data.

        A Williams Fractal High occurs when the high of the current bar is higher
        than the highs of the specified number of bars before and after it.
        A Williams Fractal Low occurs when the low of the current bar is lower
        than the lows of the specified number of bars before and after it.

        Args:
            highs: Array of high prices
            lows: Array of low prices

        Returns:
            Tuple of (fractal_highs, fractal_lows)
        """
        fractal_highs = []
        fractal_lows = []

        period = self.fractal_period

        # Need at least 2*period+1 bars for fractal detection
        if len(highs) < 2 * period + 1:
            return fractal_highs, fractal_lows

        # Check each potential fractal point
        for i in range(period, len(highs) - period):
            # Check for fractal high
            current_high = highs[i]
            is_fractal_high = True

            # Check bars before and after
            for j in range(1, period + 1):
                if highs[i - j] >= current_high or highs[i + j] >= current_high:
                    is_fractal_high = False
                    break

            if is_fractal_high:
                # Calculate fractal strength based on how much it exceeds nearby highs
                nearby_highs = np.concatenate(
                    [highs[i - period : i], highs[i + 1 : i + period + 1]]
                )
                strength = (current_high - np.max(nearby_highs)) / current_high
                fractal_highs.append(FractalPoint(i, current_high, "high", strength))

            # Check for fractal low
            current_low = lows[i]
            is_fractal_low = True

            # Check bars before and after
            for j in range(1, period + 1):
                if lows[i - j] <= current_low or lows[i + j] <= current_low:
                    is_fractal_low = False
                    break

            if is_fractal_low:
                # Calculate fractal strength
                nearby_lows = np.concatenate(
                    [lows[i - period : i], lows[i + 1 : i + period + 1]]
                )
                strength = (np.min(nearby_lows) - current_low) / current_low
                fractal_lows.append(FractalPoint(i, current_low, "low", strength))

        return fractal_highs, fractal_lows

    def construct_channel(
        self, fractal_highs: List[FractalPoint], fractal_lows: List[FractalPoint]
    ) -> Optional[FractalChannelResult]:
        """
        Construct fractal channel from detected fractal points.

        Args:
            fractal_highs: List of fractal high points
            fractal_lows: List of fractal low points

        Returns:
            FractalChannelResult or None if insufficient data
        """
        if (
            len(fractal_highs) < self.min_fractals
            or len(fractal_lows) < self.min_fractals
        ):
            return None

        # Get recent fractals for channel construction
        recent_highs = (
            fractal_highs[-self.channel_lookback :]
            if len(fractal_highs) >= self.channel_lookback
            else fractal_highs
        )
        recent_lows = (
            fractal_lows[-self.channel_lookback :]
            if len(fractal_lows) >= self.channel_lookback
            else fractal_lows
        )

        # Calculate weighted averages (giving more weight to stronger fractals)
        if recent_highs:
            high_prices = [fp.price for fp in recent_highs]
            high_weights = [
                fp.strength + 1.0 for fp in recent_highs
            ]  # +1 to avoid zero weights
            upper_channel = np.average(high_prices, weights=high_weights)
            resistance_level = max(high_prices)
        else:
            upper_channel = None
            resistance_level = None

        if recent_lows:
            low_prices = [fp.price for fp in recent_lows]
            low_weights = [fp.strength + 1.0 for fp in recent_lows]
            lower_channel = np.average(low_prices, weights=low_weights)
            support_level = min(low_prices)
        else:
            lower_channel = None
            support_level = None

        if upper_channel is None or lower_channel is None:
            return None

        # Calculate channel metrics
        middle_channel = (upper_channel + lower_channel) / 2
        channel_width = upper_channel - lower_channel

        # Get most recent fractal points
        fractal_high = recent_highs[-1].price if recent_highs else None
        fractal_low = recent_lows[-1].price if recent_lows else None

        return FractalChannelResult(
            upper_channel=upper_channel,
            lower_channel=lower_channel,
            middle_channel=middle_channel,
            channel_width=channel_width,
            fractal_high=fractal_high,
            fractal_low=fractal_low,
            support_level=support_level,
            resistance_level=resistance_level,
        )

    def detect_breakout_signals(
        self, current_price: float, channel_result: FractalChannelResult
    ) -> Optional[str]:
        """
        Detect breakout signals based on current price vs channel.

        Args:
            current_price: Current price to analyze
            channel_result: Current channel structure

        Returns:
            'bullish', 'bearish', or None
        """
        if not channel_result:
            return None

        # Calculate breakout thresholds
        upper_breakout = channel_result.upper_channel * (1 + self.breakout_threshold)
        lower_breakout = channel_result.lower_channel * (1 - self.breakout_threshold)

        if current_price > upper_breakout:
            return "bullish"
        elif current_price < lower_breakout:
            return "bearish"
        else:
            return None

    def calculate(
        self, data: Union[pd.DataFrame, np.ndarray, Dict]
    ) -> Optional[FractalChannelResult]:
        """
        Calculate Fractal Channel Indicator for given data.

        Args:
            data: Price data (DataFrame with OHLC, array, or dict)

        Returns:
            FractalChannelResult with channel analysis, or None if calculation fails
        """
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if "high" in data.columns and "low" in data.columns:
                    highs = data["high"].values
                    lows = data["low"].values
                    current_price = (
                        data["close"].iloc[-1] if "close" in data.columns else highs[-1]
                    )
                else:
                    # Assume single column is close price
                    prices = data.iloc[:, 0].values
                    highs = lows = prices
                    current_price = prices[-1]
            elif isinstance(data, np.ndarray):
                if data.ndim == 2 and data.shape[1] >= 2:
                    highs = data[:, 1]  # Assume second column is high
                    lows = (
                        data[:, 2] if data.shape[1] > 2 else data[:, 1]
                    )  # Third column is low
                    current_price = data[-1, 0] if data.shape[1] > 0 else highs[-1]
                else:
                    prices = data.flatten()
                    highs = lows = prices
                    current_price = prices[-1]
            elif isinstance(data, dict):
                highs = np.array(data.get("high", data.get("close", [])))
                lows = np.array(data.get("low", data.get("close", [])))
                current_price = (
                    data.get("close", [])[-1] if data.get("close") else highs[-1]
                )
            elif isinstance(data, (list, tuple)):
                # Handle Python lists/tuples as price data
                prices = np.array(data)
                highs = lows = prices
                current_price = prices[-1] if len(prices) > 0 else None
            else:
                self.logger.error(f"Unsupported data type: {type(data)}")
                return None

            if len(highs) == 0 or len(lows) == 0:
                return None

            # Detect fractals
            fractal_highs, fractal_lows = self.detect_williams_fractals(highs, lows)

            # Update internal fractal tracking
            self.fractal_highs = fractal_highs
            self.fractal_lows = fractal_lows

            # Construct channel
            channel_result = self.construct_channel(fractal_highs, fractal_lows)

            if channel_result:
                # Detect breakout signals
                breakout_signal = self.detect_breakout_signals(
                    current_price, channel_result
                )
                channel_result.breakout_signal = breakout_signal

            return channel_result

        except Exception as e:
            self.logger.error(f"Error calculating Fractal Channel Indicator: {e}")
            return None

    def get_fractal_summary(self) -> Dict[str, Any]:
        """Get summary of detected fractals."""
        return {
            "fractal_highs_count": len(self.fractal_highs),
            "fractal_lows_count": len(self.fractal_lows),
            "latest_fractal_high": (
                self.fractal_highs[-1].price if self.fractal_highs else None
            ),
            "latest_fractal_low": (
                self.fractal_lows[-1].price if self.fractal_lows else None
            ),
            "avg_fractal_strength_high": (
                np.mean([fp.strength for fp in self.fractal_highs])
                if self.fractal_highs
                else 0
            ),
            "avg_fractal_strength_low": (
                np.mean([fp.strength for fp in self.fractal_lows])
                if self.fractal_lows
                else 0
            ),
        }


@dataclass
class FractalChaosResult:
    """Result structure for Fractal Chaos Oscillator analysis"""

    chaos_value: float
    fractal_dimension: float
    market_regime: str  # 'chaotic', 'trending', 'ranging'
    complexity_score: float
    predictability_index: float
    regime_strength: float


@dataclass
class FractalEnergyResult:
    """Result structure for Fractal Energy Indicator analysis"""

    energy_level: float
    momentum_strength: float
    energy_direction: str  # 'bullish', 'bearish', 'neutral'
    energy_sustainability: float
    power_ratio: float
    kinetic_energy: float
    potential_energy: float


@dataclass
class MandelbrotResult:
    """Result structure for Mandelbrot Fractal Indicator analysis"""

    mandelbrot_value: float
    iteration_count: int
    convergence_rate: float
    pattern_complexity: float
    self_similarity_index: float
    fractal_pattern: str  # 'convergent', 'divergent', 'oscillating'


class FractalChaosOscillator:
    """
    Real Fractal Chaos Oscillator Implementation

    Applies chaos theory to financial markets to detect regime changes and measure
    market complexity. Uses fractal dimension calculation, Lyapunov exponents,
    and chaos metrics to identify market states.

    Features:
    - Chaos theory-based market analysis    - Fractal dimension calculation using box-counting method
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
        """
        Initialize Fractal Chaos Oscillator.

        Args:
            period: Legacy compatibility parameter
            chaos_window: Window size for chaos analysis
            dimension_scales: Number of scales for fractal dimension calculation
            regime_threshold: Threshold for regime classification
        """
        self.period = period
        self.chaos_window = chaos_window
        self.dimension_scales = dimension_scales
        self.regime_threshold = regime_threshold
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)

    def calculate_fractal_dimension(self, data: np.ndarray) -> float:
        """
        Calculate fractal dimension using box-counting method.

        Args:
            data: Price data array

        Returns:
            Fractal dimension value
        """
        # Quick return for small datasets
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
                fractal_dim = 1.5  # Default fallback

            # Clamp to reasonable range
            return np.clip(fractal_dim, 1.0, 2.0)

        except Exception as e:
            self.logger.error(f"Error calculating fractal dimension: {e}")
            return 1.5

    def calculate_lyapunov_exponent(self, data: np.ndarray) -> float:
        # Quick return for small datasets
        if len(data) < 10:
            return 0.0
        """
        Estimate Lyapunov exponent for chaos measurement.

        Args:
            data: Price data array

        Returns:
            Lyapunov exponent estimate
        """
        try:
            if len(data) < 10:
                return 0.0

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
        """
        Calculate market complexity score.

        Args:
            data: Price data array

        Returns:
            Complexity score [0, 1]
        """
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
        """
        Determine market regime based on chaos metrics.

        Args:
            fractal_dim: Fractal dimension value
            chaos_value: Chaos oscillator value
            complexity: Complexity score

        Returns:
            Tuple of (regime_name, regime_strength)
        """
        # Combine metrics for regime detection
        chaos_score = abs(chaos_value)
        regime_metric = (fractal_dim - 1.0) * 0.4 + chaos_score * 0.3 + complexity * 0.3

        if regime_metric < self.regime_threshold * 0.7:
            return "ranging", regime_metric
        elif regime_metric > self.regime_threshold * 1.3:
            return "chaotic", regime_metric
        else:
            return "trending", regime_metric

    def calculate(
        self, data: Union[pd.DataFrame, np.ndarray, Dict]
    ) -> Optional[FractalChaosResult]:
        """
        Calculate Fractal Chaos Oscillator for given data.

        Args:
            data: Price data (DataFrame with OHLC, array, or dict)

        Returns:
            FractalChaosResult with chaos analysis
        """
        try:  # Parse input data
            if isinstance(data, pd.DataFrame):
                if "close" in data.columns:
                    prices = data["close"].values
                else:
                    prices = data.iloc[:, 0].values
            elif isinstance(data, np.ndarray):
                prices = data.flatten()
            elif isinstance(data, dict):
                prices = np.array(data.get("close", []))
            elif isinstance(data, (list, tuple)):
                # Handle Python lists/tuples as price data
                prices = np.array(data)
            else:
                self.logger.error(f"Unsupported data type: {type(data)}")
                return None

            if len(prices) < self.chaos_window:
                return None

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

            return FractalChaosResult(
                chaos_value=float(chaos_value),
                fractal_dimension=float(fractal_dimension),
                market_regime=market_regime,
                complexity_score=float(complexity_score),
                predictability_index=float(predictability_index),
                regime_strength=float(regime_strength),
            )

        except Exception as e:
            self.logger.error(f"Error calculating Fractal Chaos Oscillator: {e}")
            return None


class FractalEnergyIndicator:
    """
    Real Fractal Energy Indicator Implementation

    Measures market energy based on fractal geometry and momentum analysis.
    Energy is calculated using price movements, volatility patterns, and
    fractal structures to determine market momentum strength and sustainability.

    Features:
    - Energy-based momentum analysis
    - Kinetic vs potential energy separation
    - Energy direction and sustainability measurement
    - Power ratio analysis for trend strength
    - Fractal-based energy calculations
    """

    def __init__(
        self,
        period=20,
        energy_window=14,
        momentum_period=10,
        volatility_period=20,
        energy_threshold=0.6,
        **kwargs,
    ):
        """
        Initialize Real Fractal Energy Indicator.

        Args:
            period: Legacy compatibility parameter
            energy_window: Window for energy calculations
            momentum_period: Period for momentum analysis
            volatility_period: Period for volatility measurement
            energy_threshold: Threshold for energy direction classification
        """
        self.period = period
        self.energy_window = energy_window
        self.momentum_period = momentum_period
        self.volatility_period = volatility_period
        self.energy_threshold = energy_threshold
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)

    def calculate_kinetic_energy(self, prices: np.ndarray) -> float:
        """
        Calculate kinetic energy based on price momentum.

        Kinetic energy represents active price movement and velocity.

        Args:
            prices: Array of price values

        Returns:
            Kinetic energy value
        """
        try:
            if len(prices) < 2:
                return 0.0

            # Calculate price velocity (rate of change)
            velocity = np.diff(prices)

            # Kinetic energy proportional to velocity squared
            kinetic_energy = np.mean(velocity**2)

            # Normalize by price level to make it scale-independent
            price_level = np.mean(prices)
            if price_level > 0:
                kinetic_energy = kinetic_energy / (price_level**2)

            return float(kinetic_energy)

        except Exception as e:
            self.logger.error(f"Error calculating kinetic energy: {e}")
            return 0.0

    def calculate_potential_energy(self, prices: np.ndarray) -> float:
        """
        Calculate potential energy based on price position relative to range.

        Potential energy represents stored energy from price compression.

        Args:
            prices: Array of price values

        Returns:
            Potential energy value
        """
        try:
            if len(prices) < 2:
                return 0.0

            # Calculate price range
            price_range = np.max(prices) - np.min(prices)
            current_price = prices[-1]

            # Position within range (0 = at minimum, 1 = at maximum)
            if price_range > 0:
                position = (current_price - np.min(prices)) / price_range
            else:
                position = 0.5

            # Potential energy highest at extremes (compression points)
            # Use parabolic function: highest at 0 and 1, lowest at 0.5
            potential_energy = 4 * position * (1 - position)

            # Scale by volatility (higher volatility = more potential energy)
            volatility = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
            potential_energy *= volatility

            return float(potential_energy)

        except Exception as e:
            self.logger.error(f"Error calculating potential energy: {e}")
            return 0.0

    def calculate_momentum_strength(self, prices: np.ndarray) -> float:
        """
        Calculate momentum strength using fractal-based analysis.

        Args:
            prices: Array of price values

        Returns:
            Momentum strength [0, 1]
        """
        try:
            if len(prices) < self.momentum_period:
                return 0.0

            # Calculate multiple timeframe momentum
            short_momentum = (
                (prices[-1] - prices[-self.momentum_period // 2])
                / prices[-self.momentum_period // 2]
                if prices[-self.momentum_period // 2] != 0
                else 0
            )
            long_momentum = (
                (prices[-1] - prices[-self.momentum_period])
                / prices[-self.momentum_period]
                if prices[-self.momentum_period] != 0
                else 0
            )

            # Calculate momentum consistency (how consistent is the direction)
            price_changes = np.diff(prices[-self.momentum_period :])
            positive_changes = np.sum(price_changes > 0)
            consistency = abs(positive_changes - len(price_changes) / 2) / (
                len(price_changes) / 2
            )

            # Combine momentum measures
            momentum_magnitude = np.sqrt(short_momentum**2 + long_momentum**2)
            momentum_strength = momentum_magnitude * consistency

            # Normalize to [0, 1] range
            return float(np.clip(momentum_strength * 10, 0, 1))

        except Exception as e:
            self.logger.error(f"Error calculating momentum strength: {e}")
            return 0.0

    def calculate_energy_sustainability(
        self, prices: np.ndarray, volume: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate energy sustainability index.

        Args:
            prices: Array of price values
            volume: Optional volume data for enhanced calculation

        Returns:
            Sustainability index [0, 1]
        """
        try:
            if len(prices) < self.energy_window:
                return 0.5

            # Calculate trend consistency
            returns = np.diff(np.log(prices + 1e-10))
            trend_strength = abs(np.mean(returns)) / (np.std(returns) + 1e-10)

            # Calculate energy decay rate
            recent_volatility = np.std(returns[-self.energy_window // 2 :])
            past_volatility = np.std(
                returns[-self.energy_window : -self.energy_window // 2]
            )

            if past_volatility > 0:
                volatility_ratio = recent_volatility / past_volatility
                # Stable or decreasing volatility suggests sustainability
                volatility_sustainability = 1 / (1 + max(0, volatility_ratio - 1))
            else:
                volatility_sustainability = 0.5

            # Volume-based sustainability (if available)
            volume_sustainability = 0.5
            if volume is not None and len(volume) >= self.energy_window:
                recent_volume = np.mean(volume[-self.energy_window // 2 :])
                past_volume = np.mean(
                    volume[-self.energy_window : -self.energy_window // 2]
                )
                if past_volume > 0:
                    volume_ratio = recent_volume / past_volume
                    # Increasing volume with trend suggests sustainability
                    volume_sustainability = min(1.0, volume_ratio)

            # Combine measures
            sustainability = np.mean(
                [
                    np.clip(trend_strength, 0, 1),
                    volatility_sustainability,
                    volume_sustainability,
                ]
            )

            return float(sustainability)

        except Exception as e:
            self.logger.error(f"Error calculating energy sustainability: {e}")
            return 0.5

    def determine_energy_direction(
        self, kinetic_energy: float, momentum_strength: float, prices: np.ndarray
    ) -> str:
        """
        Determine energy direction based on momentum and price action.

        Args:
            kinetic_energy: Calculated kinetic energy
            momentum_strength: Momentum strength value
            prices: Price array for trend analysis

        Returns:
            Energy direction: 'bullish', 'bearish', or 'neutral'
        """
        try:
            # Calculate recent price trend
            if len(prices) >= 5:
                recent_trend = (
                    (prices[-1] - prices[-5]) / prices[-5] if prices[-5] != 0 else 0
                )
            else:
                recent_trend = 0

            # Combine kinetic energy, momentum, and trend
            energy_score = momentum_strength * np.sign(recent_trend)

            if energy_score > self.energy_threshold:
                return "bullish"
            elif energy_score < -self.energy_threshold:
                return "bearish"
            else:
                return "neutral"

        except Exception as e:
            self.logger.error(f"Error determining energy direction: {e}")
            return "neutral"

    def calculate(
        self, data: Union[pd.DataFrame, np.ndarray, Dict]
    ) -> Optional[FractalEnergyResult]:
        """
        Calculate Fractal Energy Indicator for given data.

        Args:
            data: Price data (DataFrame with OHLC, array, or dict)

        Returns:
            FractalEnergyResult with energy analysis
        """
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if "close" in data.columns:
                    prices = data["close"].values
                else:
                    prices = data.iloc[:, 0].values  # Try to get volume if available
                volume = data["volume"].values if "volume" in data.columns else None
            elif isinstance(data, np.ndarray):
                prices = data.flatten()
                volume = None
            elif isinstance(data, dict):
                prices = np.array(data.get("close", []))
                volume = np.array(data.get("volume", [])) if "volume" in data else None
            elif isinstance(data, (list, tuple)):
                # Handle Python lists/tuples as price data
                prices = np.array(data)
                volume = None
            else:
                self.logger.error(f"Unsupported data type: {type(data)}")
                return None

            if len(prices) < self.energy_window:
                return None

            # Use recent window for analysis
            recent_prices = prices[-self.energy_window :]
            recent_volume = (
                volume[-self.energy_window :] if volume is not None else None
            )

            # Calculate energy components
            kinetic_energy = self.calculate_kinetic_energy(recent_prices)
            potential_energy = self.calculate_potential_energy(recent_prices)

            # Calculate total energy level
            energy_level = kinetic_energy + potential_energy

            # Calculate momentum strength
            momentum_strength = self.calculate_momentum_strength(recent_prices)

            # Calculate energy sustainability
            energy_sustainability = self.calculate_energy_sustainability(
                recent_prices, recent_volume
            )

            # Determine energy direction
            energy_direction = self.determine_energy_direction(
                kinetic_energy, momentum_strength, recent_prices
            )

            # Calculate power ratio (kinetic vs potential energy)
            total_energy = kinetic_energy + potential_energy
            power_ratio = kinetic_energy / total_energy if total_energy > 0 else 0.5

            return FractalEnergyResult(
                energy_level=float(energy_level),
                momentum_strength=float(momentum_strength),
                energy_direction=energy_direction,
                energy_sustainability=float(energy_sustainability),
                power_ratio=float(power_ratio),
                kinetic_energy=float(kinetic_energy),
                potential_energy=float(potential_energy),
            )

        except Exception as e:
            self.logger.error(f"Error calculating Fractal Energy Indicator: {e}")
            return None


class MandelbrotFractalIndicator:
    """
    Real Mandelbrot Fractal Indicator Implementation

    Applies Mandelbrot set mathematics to financial market analysis for
    advanced pattern recognition and complexity measurement. Uses iterative
    complex number calculations to identify self-similar patterns and
    market structure characteristics.

    Features:
    - Mandelbrot set-based pattern analysis
    - Self-similarity index calculation
    - Convergence/divergence pattern detection
    - Market complexity measurement using fractal mathematics
    - Pattern classification and strength assessment
    """

    def __init__(
        self,
        period=20,
        max_iterations=20,
        escape_radius=2.0,
        grid_resolution=50,
        complexity_threshold=0.5,
        pattern_memory=10,
        **kwargs,
    ):
        """
        Initialize Real Mandelbrot Fractal Indicator.

        Args:
            period: Legacy compatibility parameter
            max_iterations: Maximum iterations for Mandelbrot calculation
            escape_radius: Escape radius for convergence testing
            grid_resolution: Resolution for fractal grid analysis            complexity_threshold: Threshold for pattern classification
            pattern_memory: Number of periods to remember for pattern analysis
        """
        self.period = (
            getattr(period, "value", period)
            if hasattr(period, "value")
            else int(period) if period is not None else 20
        )
        self.max_iterations = (
            getattr(max_iterations, "value", max_iterations)
            if hasattr(max_iterations, "value")
            else int(max_iterations) if max_iterations is not None else 20
        )
        self.escape_radius = (
            getattr(escape_radius, "value", escape_radius)
            if hasattr(escape_radius, "value")
            else float(escape_radius) if escape_radius is not None else 2.0
        )
        self.grid_resolution = (
            getattr(grid_resolution, "value", grid_resolution)
            if hasattr(grid_resolution, "value")
            else int(grid_resolution) if grid_resolution is not None else 50
        )
        self.complexity_threshold = (
            getattr(complexity_threshold, "value", complexity_threshold)
            if hasattr(complexity_threshold, "value")
            else (
                float(complexity_threshold) if complexity_threshold is not None else 0.5
            )
        )
        self.pattern_memory = (
            getattr(pattern_memory, "value", pattern_memory)
            if hasattr(pattern_memory, "value")
            else int(pattern_memory) if pattern_memory is not None else 10
        )
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)

        # Pattern history for continuity analysis
        self.pattern_history = []

    def mandelbrot_iteration(
        self, c: complex, max_iter: int = None
    ) -> Tuple[int, float]:
        """
        Perform Mandelbrot set iteration for a complex number.

        Args:
            c: Complex number to test
            max_iter: Maximum iterations (uses self.max_iterations if None)

        Returns:
            Tuple of (iterations_to_escape, final_magnitude)
        """
        if max_iter is None:
            max_iter = self.max_iterations

        z = 0 + 0j
        for i in range(max_iter):
            if abs(z) > self.escape_radius:
                return i, abs(z)
            z = z * z + c

        return max_iter, abs(z)

    def price_to_complex(self, prices: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """
        Convert price data to complex numbers for Mandelbrot analysis.

        Args:
            prices: Array of price values
            returns: Array of price returns

        Returns:
            Array of complex numbers representing market state
        """
        try:
            # Normalize prices and returns to suitable complex plane range
            price_norm = (prices - np.min(prices)) / (
                np.max(prices) - np.min(prices) + 1e-10
            )
            return_norm = returns / (np.std(returns) + 1e-10)

            # Map to complex plane: real = normalized price, imaginary = normalized return
            # Scale to [-2, 2] range for Mandelbrot analysis
            real_part = (price_norm - 0.5) * 4
            imag_part = np.clip(return_norm, -2, 2)

            complex_points = real_part + 1j * imag_part
            return complex_points

        except Exception as e:
            self.logger.error(f"Error converting prices to complex numbers: {e}")
            return np.array([0 + 0j])

    def calculate_fractal_complexity(self, complex_points: np.ndarray) -> float:
        """
        Calculate fractal complexity using Mandelbrot iterations.

        Args:
            complex_points: Array of complex numbers representing market states

        Returns:
            Complexity measure [0, 1]
        """
        try:
            total_iterations = 0
            convergent_points = 0

            for point in complex_points:
                iterations, final_mag = self.mandelbrot_iteration(point)
                total_iterations += iterations

                if iterations == self.max_iterations:
                    convergent_points += 1

            # Complexity based on average iterations and convergence ratio
            avg_iterations = total_iterations / len(complex_points)
            convergence_ratio = convergent_points / len(complex_points)

            # Normalize complexity measure
            iteration_complexity = avg_iterations / self.max_iterations
            structure_complexity = (
                1 - abs(convergence_ratio - 0.5) * 2
            )  # Most complex at 50% convergence

            complexity = (iteration_complexity + structure_complexity) / 2
            return float(np.clip(complexity, 0, 1))

        except Exception as e:
            self.logger.error(f"Error calculating fractal complexity: {e}")
            return 0.5

    def calculate_self_similarity(self, prices: np.ndarray) -> float:
        """
        Calculate self-similarity index using fractal analysis.

        Args:
            prices: Array of price values

        Returns:
            Self-similarity index [0, 1]
        """
        try:
            if len(prices) < 8:
                return 0.0

            # Compare patterns at different scales
            similarities = []

            # Test multiple scale ratios
            for scale in [2, 3, 4]:
                if len(prices) >= scale * 4:
                    # Extract patterns at different scales
                    full_pattern = prices[-scale * 4 :]
                    half_pattern = prices[-scale * 2 :]

                    # Normalize patterns
                    full_norm = (full_pattern - np.mean(full_pattern)) / (
                        np.std(full_pattern) + 1e-10
                    )
                    half_norm = (half_pattern - np.mean(half_pattern)) / (
                        np.std(half_pattern) + 1e-10
                    )

                    # Calculate correlation between scaled patterns
                    if len(full_norm) >= len(half_norm) * 2:
                        # Downsample full pattern to match half pattern length
                        downsampled = full_norm[::2][: len(half_norm)]
                        if len(downsampled) == len(half_norm):
                            correlation = np.corrcoef(downsampled, half_norm)[0, 1]
                            if not np.isnan(correlation):
                                similarities.append(abs(correlation))

            # Return average similarity across scales
            if similarities:
                return float(np.mean(similarities))
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating self-similarity: {e}")
            return 0.0

    def calculate_convergence_rate(self, complex_points: np.ndarray) -> float:
        """
        Calculate convergence rate for the complex points.

        Args:
            complex_points: Array of complex numbers

        Returns:
            Convergence rate measure
        """
        try:
            convergence_speeds = []

            for point in complex_points:
                iterations, final_mag = self.mandelbrot_iteration(point)

                # Calculate convergence speed
                if iterations < self.max_iterations:
                    # Faster escape = higher convergence speed
                    convergence_speed = 1.0 / (iterations + 1)
                else:
                    # No escape = very slow convergence
                    convergence_speed = 1.0 / self.max_iterations

                convergence_speeds.append(convergence_speed)

            # Return average convergence rate
            return float(np.mean(convergence_speeds))

        except Exception as e:
            self.logger.error(f"Error calculating convergence rate: {e}")
            return 0.0

    def classify_fractal_pattern(
        self, complexity: float, convergence_rate: float, similarity: float
    ) -> str:
        """
        Classify the fractal pattern based on calculated metrics.

        Args:
            complexity: Fractal complexity measure
            convergence_rate: Convergence rate measure
            similarity: Self-similarity index

        Returns:
            Pattern classification: 'convergent', 'divergent', or 'oscillating'
        """
        try:
            # Weight the different measures
            pattern_score = complexity * 0.4 + convergence_rate * 0.3 + similarity * 0.3

            # Add pattern history for stability
            self.pattern_history.append(pattern_score)
            if len(self.pattern_history) > self.pattern_memory:
                self.pattern_history.pop(0)

            # Use historical average for more stable classification
            avg_pattern_score = np.mean(self.pattern_history)

            # Classify based on thresholds
            if avg_pattern_score > self.complexity_threshold * 1.2:
                return "divergent"
            elif avg_pattern_score < self.complexity_threshold * 0.8:
                return "convergent"
            else:
                return "oscillating"

        except Exception as e:
            self.logger.error(f"Error classifying fractal pattern: {e}")
            return "oscillating"

    def calculate_mandelbrot_value(self, complex_points: np.ndarray) -> float:
        """
        Calculate main Mandelbrot indicator value.

        Args:
            complex_points: Array of complex numbers

        Returns:
            Mandelbrot indicator value
        """
        try:
            # Calculate average escape iterations
            total_iterations = 0
            for point in complex_points:
                iterations, _ = self.mandelbrot_iteration(point)
                total_iterations += iterations

            avg_iterations = total_iterations / len(complex_points)

            # Normalize to [-1, 1] range
            normalized_value = (avg_iterations / self.max_iterations) * 2 - 1

            return float(normalized_value)

        except Exception as e:
            self.logger.error(f"Error calculating Mandelbrot value: {e}")
            return 0.0

    def calculate(
        self, data: Union[pd.DataFrame, np.ndarray, Dict]
    ) -> Optional[MandelbrotResult]:
        """
        Calculate Mandelbrot Fractal Indicator for given data.

        Args:
            data: Price data (DataFrame with OHLC, array, or dict)

        Returns:
            MandelbrotResult with fractal analysis
        """
        try:  # Parse input data
            if isinstance(data, pd.DataFrame):
                if "close" in data.columns:
                    prices = data["close"].values
                else:
                    prices = data.iloc[:, 0].values
            elif isinstance(data, np.ndarray):
                prices = data.flatten()
            elif isinstance(data, dict):
                prices = np.array(data.get("close", []))
            elif isinstance(data, (list, tuple)):
                # Handle Python lists/tuples as price data
                prices = np.array(data)
            else:
                self.logger.error(f"Unsupported data type: {type(data)}")
                return None

            if len(prices) < max(self.period, 10):
                return None

            # Calculate returns for complex number mapping
            returns = np.diff(np.log(prices + 1e-10))

            # Use recent data for analysis
            recent_length = min(len(prices), self.period * 2)
            recent_prices = prices[-recent_length:]
            recent_returns = (
                returns[-recent_length + 1 :]
                if len(returns) >= recent_length - 1
                else returns
            )

            # Convert to complex plane
            complex_points = self.price_to_complex(recent_prices[1:], recent_returns)

            # Calculate Mandelbrot metrics
            mandelbrot_value = self.calculate_mandelbrot_value(complex_points)
            complexity = self.calculate_fractal_complexity(complex_points)
            convergence_rate = self.calculate_convergence_rate(complex_points)
            similarity = self.calculate_self_similarity(recent_prices)

            # Classify pattern
            pattern_type = self.classify_fractal_pattern(
                complexity, convergence_rate, similarity
            )

            # Calculate average iteration count for the result
            total_iterations = 0
            for point in complex_points:
                iterations, _ = self.mandelbrot_iteration(point)
                total_iterations += iterations
            avg_iterations = int(total_iterations / len(complex_points))

            return MandelbrotResult(
                mandelbrot_value=float(mandelbrot_value),
                iteration_count=avg_iterations,
                convergence_rate=float(convergence_rate),
                pattern_complexity=float(complexity),
                self_similarity_index=float(similarity),
                fractal_pattern=pattern_type,
            )
        except Exception as e:
            self.logger.error(f"Error calculating Mandelbrot Fractal Indicator: {e}")
            return None


@dataclass
class HurstExponentResult:
    """Result structure for Hurst Exponent analysis"""

    hurst_value: float
    memory_type: str  # 'trending', 'random', 'mean_reverting'
    confidence: float
    trend_strength: float
    persistence_score: float
    rs_hurst: float
    dfa_hurst: float


class HurstExponentIndicator:
    """
    Real Hurst Exponent Indicator Implementation

    Measures the long-term memory of price series to determine if movements are:
    - Trending/persistent (H > 0.5)
    - Random walk (H ≈ 0.5)
    - Mean-reverting (H < 0.5)

    Uses multiple calculation methods for robust estimation.
    """

    def __init__(
        self, period=20, min_window=50, max_window=200, num_divisions=10, **kwargs
    ):
        """
        Initialize Hurst Exponent Indicator.

        Args:
            period: Legacy compatibility parameter
            min_window: Minimum window size for analysis
            max_window: Maximum window size for analysis
            num_divisions: Number of divisions for R/S analysis
        """
        self.period = period
        self.min_window = min_window
        self.max_window = max_window
        self.num_divisions = num_divisions
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)

    def rs_hurst(self, data: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S (Range/Standard deviation) method."""
        try:
            if len(data) < 10:
                return 0.5

            log_returns = np.diff(np.log(data + 1e-10))
            n = len(log_returns)

            # Calculate mean-adjusted cumulative returns
            mean_return = np.mean(log_returns)
            cumulative_deviations = np.cumsum(log_returns - mean_return)

            # Calculate range and standard deviation
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            S = np.std(log_returns)

            if S == 0:
                return 0.5

            # R/S ratio
            rs_ratio = R / S

            # Hurst exponent approximation
            hurst = np.log(rs_ratio) / np.log(n)

            return np.clip(hurst, 0.0, 1.0)

        except Exception as e:
            self.logger.error(f"Error in R/S Hurst calculation: {e}")
            return 0.5

    def dfa_hurst(self, data: np.ndarray) -> float:
        """Calculate Hurst exponent using Detrended Fluctuation Analysis."""
        try:
            if len(data) < 20:
                return 0.5

            log_returns = np.diff(np.log(data + 1e-10))

            # Integrate the series
            integrated_series = np.cumsum(log_returns - np.mean(log_returns))

            # Define window sizes
            min_win = max(4, len(integrated_series) // 20)
            max_win = len(integrated_series) // 4
            window_sizes = np.logspace(np.log10(min_win), np.log10(max_win), 10).astype(
                int
            )

            fluctuations = []

            for win_size in window_sizes:
                if win_size >= len(integrated_series):
                    continue

                # Divide series into non-overlapping windows
                n_windows = len(integrated_series) // win_size

                detrended_fluctuation = 0
                for i in range(n_windows):
                    start_idx = i * win_size
                    end_idx = start_idx + win_size
                    window_data = integrated_series[start_idx:end_idx]

                    # Linear detrending
                    x = np.arange(len(window_data))
                    slope, intercept = np.polyfit(x, window_data, 1)
                    trend = slope * x + intercept

                    # Calculate fluctuation
                    detrended = window_data - trend
                    detrended_fluctuation += np.mean(detrended**2)

                avg_fluctuation = np.sqrt(detrended_fluctuation / n_windows)
                fluctuations.append(avg_fluctuation)

            if len(fluctuations) < 2:
                return 0.5

            # Calculate Hurst exponent from log-log slope
            log_windows = np.log(window_sizes[: len(fluctuations)])
            log_fluctuations = np.log(fluctuations)

            slope, _ = np.polyfit(log_windows, log_fluctuations, 1)

            return np.clip(slope, 0.0, 1.0)

        except Exception as e:
            self.logger.error(f"Error in DFA Hurst calculation: {e}")
            return 0.5

    def calculate(
        self, data: Union[pd.DataFrame, np.ndarray, Dict]
    ) -> Optional[HurstExponentResult]:
        """Calculate Hurst Exponent Indicator for given data."""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if "close" in data.columns:
                    prices = data["close"].values
                else:
                    prices = data.iloc[:, 0].values
            elif isinstance(data, np.ndarray):
                prices = data.flatten()
            elif isinstance(data, dict):
                prices = np.array(data.get("close", []))
            elif isinstance(data, (list, tuple)):
                prices = np.array(data)
            else:
                self.logger.error(f"Unsupported data type: {type(data)}")
                return None

            if len(prices) < self.min_window:
                return None

            # Use appropriate window size
            window_size = min(len(prices), self.max_window)
            recent_prices = prices[-window_size:]

            # Calculate Hurst exponents using different methods
            rs_hurst = self.rs_hurst(recent_prices)
            dfa_hurst = self.dfa_hurst(recent_prices)

            # Average the methods for final Hurst value
            hurst_value = (rs_hurst + dfa_hurst) / 2

            # Determine memory type
            if hurst_value > 0.6:
                memory_type = "trending"
                trend_strength = (hurst_value - 0.5) * 2
            elif hurst_value < 0.4:
                memory_type = "mean_reverting"
                trend_strength = (0.5 - hurst_value) * 2
            else:
                memory_type = "random"
                trend_strength = 1 - abs(hurst_value - 0.5) * 2

            # Calculate confidence based on method agreement
            confidence = 1 - abs(rs_hurst - dfa_hurst)

            # Persistence score
            persistence_score = abs(hurst_value - 0.5) * 2

            return HurstExponentResult(
                hurst_value=float(hurst_value),
                memory_type=memory_type,
                confidence=float(confidence),
                trend_strength=float(trend_strength),
                persistence_score=float(persistence_score),
                rs_hurst=float(rs_hurst),
                dfa_hurst=float(dfa_hurst),
            )

        except Exception as e:
            self.logger.error(f"Error calculating Hurst Exponent Indicator: {e}")
            return None


@dataclass
class HurstExponentResult:
    """Result structure for Hurst Exponent analysis"""

    hurst_value: float
    memory_type: str  # 'trending', 'random', 'mean_reverting'
    confidence: float
    trend_strength: float
    persistence_score: float
    rs_hurst: float
    dfa_hurst: float


@dataclass
class FractalAdaptiveResult:
    """Result structure for Fractal Adaptive Moving Average"""

    frama_value: float
    fractal_dimension: float
    smoothing_factor: float
    trend_direction: str  # 'bullish', 'bearish', 'neutral'
    adaptation_speed: float
    signal_strength: float


@dataclass
class FractalDimensionResult:
    """Result structure for Fractal Dimension analysis"""

    dimension_value: float
    complexity_level: str  # 'low', 'medium', 'high'
    market_state: str  # 'trending', 'ranging', 'chaotic'
    dimension_stability: float
    pattern_strength: float


@dataclass
class FractalBreakoutResult:
    """Result structure for Fractal Breakout analysis"""

    breakout_signal: str  # 'bullish', 'bearish', 'none'
    breakout_strength: float
    support_level: float
    resistance_level: float
    fractal_high: float
    fractal_low: float
    signal_confidence: float


@dataclass
class FractalVolumeResult:
    """Result structure for Fractal Volume analysis"""

    volume_fractal_dimension: float
    volume_profile: Dict[str, float]
    volume_flow: str  # 'bullish', 'bearish', 'neutral'
    volume_strength: float
    volume_sustainability: float
    average_volume: float


class HurstExponentIndicator:
    """
    Real Hurst Exponent Indicator Implementation

    Measures the long-term memory of price series to determine if movements are:
    - Trending/persistent (H > 0.5)
    - Random walk (H ≈ 0.5)
    - Mean-reverting (H < 0.5)

    Uses multiple calculation methods for robust estimation.
    """

    def __init__(
        self, period=20, min_window=50, max_window=200, num_divisions=10, **kwargs
    ):
        """
        Initialize Hurst Exponent Indicator.

        Args:
            period: Legacy compatibility parameter
            min_window: Minimum window size for analysis
            max_window: Maximum window size for analysis
            num_divisions: Number of divisions for R/S analysis
        """
        self.period = period
        self.min_window = min_window
        self.max_window = max_window
        self.num_divisions = num_divisions
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)

    def rs_hurst(self, data: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S (Range/Standard deviation) method."""
        try:
            if len(data) < 10:
                return 0.5

            log_returns = np.diff(np.log(data + 1e-10))
            n = len(log_returns)

            # Calculate mean-adjusted cumulative returns
            mean_return = np.mean(log_returns)
            cumulative_deviations = np.cumsum(log_returns - mean_return)

            # Calculate range and standard deviation
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            S = np.std(log_returns)

            if S == 0:
                return 0.5

            # R/S ratio
            rs_ratio = R / S

            # Hurst exponent approximation
            hurst = np.log(rs_ratio) / np.log(n)

            return np.clip(hurst, 0.0, 1.0)

        except Exception as e:
            self.logger.error(f"Error in R/S Hurst calculation: {e}")
            return 0.5

    def dfa_hurst(self, data: np.ndarray) -> float:
        """Calculate Hurst exponent using Detrended Fluctuation Analysis."""
        try:
            if len(data) < 20:
                return 0.5

            log_returns = np.diff(np.log(data + 1e-10))

            # Integrate the series
            integrated_series = np.cumsum(log_returns - np.mean(log_returns))

            # Define window sizes
            min_win = max(4, len(integrated_series) // 20)
            max_win = len(integrated_series) // 4
            window_sizes = np.logspace(np.log10(min_win), np.log10(max_win), 10).astype(
                int
            )

            fluctuations = []

            for win_size in window_sizes:
                if win_size >= len(integrated_series):
                    continue

                # Divide series into non-overlapping windows
                n_windows = len(integrated_series) // win_size

                detrended_fluctuation = 0
                for i in range(n_windows):
                    start_idx = i * win_size
                    end_idx = start_idx + win_size
                    window_data = integrated_series[start_idx:end_idx]

                    # Linear detrending
                    x = np.arange(len(window_data))
                    slope, intercept = np.polyfit(x, window_data, 1)
                    trend = slope * x + intercept

                    # Calculate fluctuation
                    detrended = window_data - trend
                    detrended_fluctuation += np.mean(detrended**2)

                avg_fluctuation = np.sqrt(detrended_fluctuation / n_windows)
                fluctuations.append(avg_fluctuation)

            if len(fluctuations) < 2:
                return 0.5

            # Calculate Hurst exponent from log-log slope
            log_windows = np.log(window_sizes[: len(fluctuations)])
            log_fluctuations = np.log(fluctuations)

            slope, _ = np.polyfit(log_windows, log_fluctuations, 1)

            return np.clip(slope, 0.0, 1.0)

        except Exception as e:
            self.logger.error(f"Error in DFA Hurst calculation: {e}")
            return 0.5

    def calculate(
        self, data: Union[pd.DataFrame, np.ndarray, Dict]
    ) -> Optional[HurstExponentResult]:
        """Calculate Hurst Exponent Indicator for given data."""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if "close" in data.columns:
                    prices = data["close"].values
                else:
                    prices = data.iloc[:, 0].values
            elif isinstance(data, np.ndarray):
                prices = data.flatten()
            elif isinstance(data, dict):
                prices = np.array(data.get("close", []))
            elif isinstance(data, (list, tuple)):
                prices = np.array(data)
            else:
                self.logger.error(f"Unsupported data type: {type(data)}")
                return None

            if len(prices) < self.min_window:
                return None

            # Use appropriate window size
            window_size = min(len(prices), self.max_window)
            recent_prices = prices[-window_size:]

            # Calculate Hurst exponents using different methods
            rs_hurst = self.rs_hurst(recent_prices)
            dfa_hurst = self.dfa_hurst(recent_prices)

            # Average the methods for final Hurst value
            hurst_value = (rs_hurst + dfa_hurst) / 2

            # Determine memory type
            if hurst_value > 0.6:
                memory_type = "trending"
                trend_strength = (hurst_value - 0.5) * 2
            elif hurst_value < 0.4:
                memory_type = "mean_reverting"
                trend_strength = (0.5 - hurst_value) * 2
            else:
                memory_type = "random"
                trend_strength = 1 - abs(hurst_value - 0.5) * 2

            # Calculate confidence based on method agreement
            confidence = 1 - abs(rs_hurst - dfa_hurst)

            # Persistence score
            persistence_score = abs(hurst_value - 0.5) * 2

            return HurstExponentResult(
                hurst_value=float(hurst_value),
                memory_type=memory_type,
                confidence=float(confidence),
                trend_strength=float(trend_strength),
                persistence_score=float(persistence_score),
                rs_hurst=float(rs_hurst),
                dfa_hurst=float(dfa_hurst),
            )

        except Exception as e:
            self.logger.error(f"Error calculating Hurst Exponent Indicator: {e}")
            return None


class FractalAdaptiveMovingAverage:
    """
    Real Fractal Adaptive Moving Average (FRAMA) Implementation

    Adaptive moving average that adjusts smoothing based on fractal dimension.
    Provides faster response in trending markets and slower in choppy markets.
    """

    def __init__(self, period=20, fast_limit=0.67, slow_limit=0.03, **kwargs):
        """
        Initialize FRAMA indicator.

        Args:
            period: Period for fractal dimension calculation
            fast_limit: Fast smoothing limit (trending markets)
            slow_limit: Slow smoothing limit (choppy markets)
        """
        self.period = period
        self.fast_limit = fast_limit
        self.slow_limit = slow_limit
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)

        # Internal state
        self.frama_values = []

    def calculate_fractal_dimension(self, highs: np.ndarray, lows: np.ndarray) -> float:
        """Calculate fractal dimension for FRAMA."""
        try:
            n = len(highs)
            if n < 2:
                return 1.5

            # Calculate the length of the price path
            total_length = 0
            for i in range(1, n):
                high_diff = abs(highs[i] - highs[i - 1])
                low_diff = abs(lows[i] - lows[i - 1])
                total_length += max(high_diff, low_diff)

            # Calculate straight line distance
            straight_distance = abs(highs[-1] - highs[0]) + abs(lows[-1] - lows[0])

            if straight_distance == 0:
                return 1.5

            # Fractal dimension
            dimension = np.log(total_length / straight_distance) / np.log(n)

            return np.clip(dimension, 1.0, 2.0)

        except Exception as e:
            self.logger.error(f"Error calculating fractal dimension: {e}")
            return 1.5

    def calculate(
        self, data: Union[pd.DataFrame, np.ndarray, Dict]
    ) -> Optional[FractalAdaptiveResult]:
        """Calculate FRAMA for given data."""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if "high" in data.columns and "low" in data.columns:
                    highs = data["high"].values
                    lows = data["low"].values
                    closes = data["close"].values if "close" in data.columns else highs
                else:
                    prices = data.iloc[:, 0].values
                    highs = lows = closes = prices
            elif isinstance(data, np.ndarray):
                if data.ndim == 2 and data.shape[1] >= 3:
                    highs = data[:, 1]
                    lows = data[:, 2]
                    closes = data[:, 0]
                else:
                    prices = data.flatten()
                    highs = lows = closes = prices
            elif isinstance(data, dict):
                highs = np.array(data.get("high", data.get("close", [])))
                lows = np.array(data.get("low", data.get("close", [])))
                closes = np.array(data.get("close", []))
            elif isinstance(data, (list, tuple)):
                prices = np.array(data)
                highs = lows = closes = prices
            else:
                self.logger.error(f"Unsupported data type: {type(data)}")
                return None

            if len(closes) < self.period:
                return None

            # Calculate fractal dimension for recent period
            recent_highs = highs[-self.period :]
            recent_lows = lows[-self.period :]

            fractal_dim = self.calculate_fractal_dimension(recent_highs, recent_lows)

            # Calculate smoothing factor
            # Higher dimension (choppy) -> slower smoothing
            # Lower dimension (trending) -> faster smoothing
            alpha = (fractal_dim - 1.0) * (
                self.slow_limit - self.fast_limit
            ) + self.fast_limit
            alpha = np.clip(alpha, self.slow_limit, self.fast_limit)

            # Calculate FRAMA
            current_price = closes[-1]
            if len(self.frama_values) == 0:
                frama_value = current_price
            else:
                prev_frama = self.frama_values[-1]
                frama_value = alpha * current_price + (1 - alpha) * prev_frama

            self.frama_values.append(frama_value)

            # Keep only recent values
            if len(self.frama_values) > self.period * 2:
                self.frama_values = self.frama_values[-self.period :]

            # Determine trend direction
            if len(self.frama_values) >= 2:
                if frama_value > self.frama_values[-2]:
                    trend_direction = "bullish"
                elif frama_value < self.frama_values[-2]:
                    trend_direction = "bearish"
                else:
                    trend_direction = "neutral"
            else:
                trend_direction = "neutral"

            # Adaptation speed (how fast the indicator is adapting)
            adaptation_speed = alpha / self.fast_limit

            # Signal strength based on price vs FRAMA distance
            price_distance = abs(current_price - frama_value) / current_price
            signal_strength = min(1.0, price_distance * 10)

            return FractalAdaptiveResult(
                frama_value=float(frama_value),
                fractal_dimension=float(fractal_dim),
                smoothing_factor=float(alpha),
                trend_direction=trend_direction,
                adaptation_speed=float(adaptation_speed),
                signal_strength=float(signal_strength),
            )

        except Exception as e:
            self.logger.error(f"Error calculating FRAMA: {e}")
            return None


class FractalDimensionIndicator:
    """
    Real Fractal Dimension Indicator Implementation

    Measures the complexity and self-similarity of price movements using
    fractal geometry principles. Higher dimensions indicate more complex,
    chaotic price action while lower dimensions suggest trending behavior.
    """

    def __init__(self, period=20, box_sizes=10, dimension_threshold=1.5, **kwargs):
        """
        Initialize Fractal Dimension Indicator.

        Args:
            period: Period for dimension calculation
            box_sizes: Number of different box sizes for calculation
            dimension_threshold: Threshold for market state classification
        """
        self.period = period
        self.box_sizes = box_sizes
        self.dimension_threshold = dimension_threshold
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)

    def box_counting_dimension(self, data: np.ndarray) -> float:
        """Calculate fractal dimension using box counting method."""
        try:
            if len(data) < 4:
                return 1.5

            # Normalize data to [0, 1] range
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-10)

            # Create different box sizes
            max_box_size = len(data) // 4
            box_sizes = np.logspace(0, np.log10(max_box_size), self.box_sizes).astype(
                int
            )
            box_sizes = box_sizes[box_sizes > 0]

            counts = []

            for box_size in box_sizes:
                if box_size >= len(data):
                    continue

                # Create grid
                grid_size = int(1.0 / box_size * len(data))
                if grid_size < 2:
                    grid_size = 2

                # Count boxes containing data points
                boxes = set()
                for i, value in enumerate(data_norm):
                    x_box = int(i * grid_size / len(data_norm))
                    y_box = int(value * grid_size)
                    boxes.add((min(x_box, grid_size - 1), min(y_box, grid_size - 1)))

                counts.append(len(boxes))

            if len(counts) < 2:
                return 1.5

            # Calculate dimension from log-log slope
            log_sizes = np.log(box_sizes[: len(counts)])
            log_counts = np.log(np.array(counts) + 1e-10)

            slope, _ = np.polyfit(log_sizes, log_counts, 1)
            dimension = -slope

            return np.clip(dimension, 1.0, 2.0)

        except Exception as e:
            self.logger.error(f"Error in box counting dimension: {e}")
            return 1.5

    def calculate(
        self, data: Union[pd.DataFrame, np.ndarray, Dict]
    ) -> Optional[FractalDimensionResult]:
        """Calculate Fractal Dimension Indicator for given data."""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if "close" in data.columns:
                    prices = data["close"].values
                else:
                    prices = data.iloc[:, 0].values
            elif isinstance(data, np.ndarray):
                prices = data.flatten()
            elif isinstance(data, dict):
                prices = np.array(data.get("close", []))
            elif isinstance(data, (list, tuple)):
                prices = np.array(data)
            else:
                self.logger.error(f"Unsupported data type: {type(data)}")
                return None

            if len(prices) < self.period:
                return None

            # Use recent window
            recent_prices = prices[-self.period :]

            # Calculate fractal dimension
            dimension_value = self.box_counting_dimension(recent_prices)

            # Classify complexity level
            if dimension_value < 1.3:
                complexity_level = "low"
                market_state = "trending"
            elif dimension_value > 1.7:
                complexity_level = "high"
                market_state = "chaotic"
            else:
                complexity_level = "medium"
                market_state = "ranging"

            # Calculate dimension stability (consistency over time)
            if len(prices) >= self.period * 2:
                prev_window = prices[-self.period * 2 : -self.period]
                prev_dimension = self.box_counting_dimension(prev_window)
                dimension_stability = 1 - abs(dimension_value - prev_dimension)
            else:
                dimension_stability = 0.5

            # Pattern strength based on deviation from random walk (1.5)
            pattern_strength = abs(dimension_value - 1.5) * 2

            return FractalDimensionResult(
                dimension_value=float(dimension_value),
                complexity_level=complexity_level,
                market_state=market_state,
                dimension_stability=float(dimension_stability),
                pattern_strength=float(pattern_strength),
            )

        except Exception as e:
            self.logger.error(f"Error calculating Fractal Dimension Indicator: {e}")
            return None


class FractalBreakoutIndicator:
    """
    Real Fractal Breakout Indicator Implementation

    Identifies potential breakout points using Williams Fractals and
    dynamic support/resistance levels. Provides early warning signals
    for trend changes and momentum shifts.
    """

    def __init__(
        self,
        period=20,
        fractal_period=5,
        breakout_threshold=0.02,
        confirmation_bars=3,
        **kwargs,
    ):
        """
        Initialize Fractal Breakout Indicator.

        Args:
            period: Period for overall analysis
            fractal_period: Period for fractal detection
            breakout_threshold: Threshold for breakout confirmation (2%)
            confirmation_bars: Bars needed for breakout confirmation
        """
        self.period = period
        self.fractal_period = fractal_period
        self.breakout_threshold = breakout_threshold
        self.confirmation_bars = confirmation_bars
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)

    def find_fractals(
        self, highs: np.ndarray, lows: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """Find Williams Fractal points."""
        try:
            fractal_highs = []
            fractal_lows = []

            period = self.fractal_period

            for i in range(period, len(highs) - period):
                # Check for fractal high
                is_fractal_high = True
                for j in range(1, period + 1):
                    if highs[i - j] >= highs[i] or highs[i + j] >= highs[i]:
                        is_fractal_high = False
                        break

                if is_fractal_high:
                    fractal_highs.append(i)

                # Check for fractal low
                is_fractal_low = True
                for j in range(1, period + 1):
                    if lows[i - j] <= lows[i] or lows[i + j] <= lows[i]:
                        is_fractal_low = False
                        break

                if is_fractal_low:
                    fractal_lows.append(i)

            return fractal_highs, fractal_lows

        except Exception as e:
            self.logger.error(f"Error finding fractals: {e}")
            return [], []

    def calculate(
        self, data: Union[pd.DataFrame, np.ndarray, Dict]
    ) -> Optional[FractalBreakoutResult]:
        """Calculate Fractal Breakout Indicator for given data."""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if "high" in data.columns and "low" in data.columns:
                    highs = data["high"].values
                    lows = data["low"].values
                    closes = data["close"].values if "close" in data.columns else highs
                else:
                    prices = data.iloc[:, 0].values
                    highs = lows = closes = prices
            elif isinstance(data, np.ndarray):
                if data.ndim == 2 and data.shape[1] >= 3:
                    closes = data[:, 0]
                    highs = data[:, 1]
                    lows = data[:, 2]
                else:
                    prices = data.flatten()
                    highs = lows = closes = prices
            elif isinstance(data, dict):
                highs = np.array(data.get("high", data.get("close", [])))
                lows = np.array(data.get("low", data.get("close", [])))
                closes = np.array(data.get("close", []))
            elif isinstance(data, (list, tuple)):
                prices = np.array(data)
                highs = lows = closes = prices
            else:
                self.logger.error(f"Unsupported data type: {type(data)}")
                return None

            if len(closes) < self.period:
                return None

            # Find fractal points
            fractal_high_indices, fractal_low_indices = self.find_fractals(highs, lows)

            if not fractal_high_indices or not fractal_low_indices:
                return FractalBreakoutResult(
                    breakout_signal="none",
                    breakout_strength=0.0,
                    support_level=lows[-1] if len(lows) > 0 else 0.0,
                    resistance_level=highs[-1] if len(highs) > 0 else 0.0,
                    fractal_high=highs[-1] if len(highs) > 0 else 0.0,
                    fractal_low=lows[-1] if len(lows) > 0 else 0.0,
                    signal_confidence=0.0,
                )

            # Get most recent fractal levels
            latest_fractal_high_idx = fractal_high_indices[-1]
            latest_fractal_low_idx = fractal_low_indices[-1]

            resistance_level = highs[latest_fractal_high_idx]
            support_level = lows[latest_fractal_low_idx]

            # Current price
            current_price = closes[-1]

            # Calculate breakout thresholds
            resistance_breakout = resistance_level * (1 + self.breakout_threshold)
            support_breakout = support_level * (1 - self.breakout_threshold)

            # Detect breakout signals
            breakout_signal = "none"
            breakout_strength = 0.0

            if current_price > resistance_breakout:
                breakout_signal = "bullish"
                breakout_strength = (
                    current_price - resistance_level
                ) / resistance_level
            elif current_price < support_breakout:
                breakout_signal = "bearish"
                breakout_strength = (support_level - current_price) / support_level

            # Calculate signal confidence based on volume and momentum
            if len(closes) >= self.confirmation_bars:
                recent_closes = closes[-self.confirmation_bars :]
                if breakout_signal == "bullish":
                    # Check if price has been consistently moving up
                    upward_moves = sum(
                        1
                        for i in range(1, len(recent_closes))
                        if recent_closes[i] > recent_closes[i - 1]
                    )
                    signal_confidence = upward_moves / (len(recent_closes) - 1)
                elif breakout_signal == "bearish":
                    # Check if price has been consistently moving down
                    downward_moves = sum(
                        1
                        for i in range(1, len(recent_closes))
                        if recent_closes[i] < recent_closes[i - 1]
                    )
                    signal_confidence = downward_moves / (len(recent_closes) - 1)
                else:
                    signal_confidence = 0.0
            else:
                signal_confidence = 0.5

            return FractalBreakoutResult(
                breakout_signal=breakout_signal,
                breakout_strength=float(breakout_strength),
                support_level=float(support_level),
                resistance_level=float(resistance_level),
                fractal_high=float(resistance_level),
                fractal_low=float(support_level),
                signal_confidence=float(signal_confidence),
            )

        except Exception as e:
            self.logger.error(f"Error calculating Fractal Breakout Indicator: {e}")
            return None


class FractalVolumeIndicator:
    """
    Real Fractal Volume Indicator Implementation

    Analyzes volume patterns using fractal geometry to identify
    accumulation/distribution phases and volume-price synchronization.
    Provides insights into institutional activity and market participation.
    """

    def __init__(
        self,
        period=20,
        volume_window=14,
        complexity_threshold=1.5,
        sync_threshold=0.6,
        **kwargs,
    ):
        """
        Initialize Fractal Volume Indicator.

        Args:
            period: Period for overall analysis
            volume_window: Window for volume analysis
            complexity_threshold: Threshold for volume complexity
            sync_threshold: Threshold for price-volume synchronization
        """
        self.period = period
        self.volume_window = volume_window
        self.complexity_threshold = complexity_threshold
        self.sync_threshold = sync_threshold
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)

    def calculate_volume_profile(
        self, prices: np.ndarray, volume: np.ndarray
    ) -> Dict[str, float]:
        """Calculate volume profile for price levels."""
        try:
            if len(prices) != len(volume) or len(prices) < 2:
                return {
                    "high_volume_price": 0.0,
                    "volume_spread": 0.0,
                    "concentration_index": 0.0,
                }

            # Create price bins
            price_min, price_max = np.min(prices), np.max(prices)
            if price_max - price_min == 0:
                return {
                    "high_volume_price": price_min,
                    "volume_spread": 0.0,
                    "concentration_index": 1.0,
                }

            num_bins = min(20, len(prices) // 3)
            bins = np.linspace(price_min, price_max, num_bins)

            # Aggregate volume by price level
            volume_by_price = np.zeros(len(bins) - 1)
            for i in range(len(prices)):
                bin_idx = np.searchsorted(bins, prices[i]) - 1
                bin_idx = np.clip(bin_idx, 0, len(volume_by_price) - 1)
                volume_by_price[bin_idx] += volume[i]

            # Find high volume price level
            max_volume_idx = np.argmax(volume_by_price)
            high_volume_price = (bins[max_volume_idx] + bins[max_volume_idx + 1]) / 2

            # Calculate volume spread (how concentrated volume is)
            total_volume = np.sum(volume_by_price)
            if total_volume > 0:
                volume_spread = np.std(volume_by_price) / np.mean(volume_by_price)
                concentration_index = np.max(volume_by_price) / total_volume
            else:
                volume_spread = 0.0
                concentration_index = 0.0

            return {
                "high_volume_price": float(high_volume_price),
                "volume_spread": float(volume_spread),
                "concentration_index": float(concentration_index),
            }

        except Exception as e:
            self.logger.error(f"Error calculating volume profile: {e}")
            return {
                "high_volume_price": 0.0,
                "volume_spread": 0.0,
                "concentration_index": 0.0,
            }

    def calculate_volume_fractal_dimension(self, volume: np.ndarray) -> float:
        """Calculate fractal dimension of volume data with improved method."""
        try:
            if len(volume) < 4:
                return 1.5

            # Use box counting method
            min_vol, max_vol = np.min(volume), np.max(volume)
            if max_vol - min_vol == 0:
                return 1.5

            # Normalize volume to [0, 1]
            volume_norm = (volume - min_vol) / (max_vol - min_vol)

            # Different scale sizes for box counting
            scales = [2, 4, 8, min(16, len(volume) // 2)]
            scales = [s for s in scales if s < len(volume)]

            if len(scales) < 2:
                return 1.5

            box_counts = []
            for scale in scales:
                grid_x = scale
                grid_y = scale

                boxes = set()
                for i, vol in enumerate(volume_norm):
                    x_idx = int((i / len(volume_norm)) * grid_x)
                    y_idx = int(vol * grid_y)
                    x_idx = min(x_idx, grid_x - 1)
                    y_idx = min(y_idx, grid_y - 1)
                    boxes.add((x_idx, y_idx))

                box_counts.append(len(boxes))

            # Calculate fractal dimension using log-log regression
            log_scales = np.log(scales)
            log_counts = np.log(box_counts)

            # Linear regression
            slope, _ = np.polyfit(log_scales, log_counts, 1)
            fractal_dim = slope  # For box counting, dimension is positive slope

            return np.clip(fractal_dim, 1.0, 2.0)

        except Exception as e:
            self.logger.error(f"Error calculating volume fractal dimension: {e}")
            return 1.5

    def calculate_volume_fractal(self, volume: np.ndarray) -> float:
        """Calculate fractal dimension of volume data."""
        try:
            if len(volume) < 4:
                return 1.5

            # Normalize volume
            volume_norm = (volume - np.min(volume)) / (
                np.max(volume) - np.min(volume) + 1e-10
            )

            # Box counting for volume fractal
            scales = np.logspace(0, np.log10(len(volume) // 4), 5).astype(int)
            scales = scales[scales > 0]

            counts = []
            for scale in scales:
                grid_size = max(2, len(volume) // scale)
                boxes = set()

                for i, vol in enumerate(volume_norm):
                    x_box = int(i * grid_size / len(volume_norm))
                    y_box = int(vol * grid_size)
                    boxes.add((min(x_box, grid_size - 1), min(y_box, grid_size - 1)))

                counts.append(len(boxes))

            if len(counts) < 2:
                return 1.5

            # Calculate fractal dimension
            log_scales = np.log(scales[: len(counts)])
            log_counts = np.log(np.array(counts) + 1e-10)

            # Linear regression to find slope
            if len(log_scales) > 1:
                slope, _ = np.polyfit(log_scales, log_counts, 1)
                fractal_dim = -slope
            else:
                fractal_dim = 1.5

            return np.clip(fractal_dim, 1.0, 2.0)

        except Exception as e:
            self.logger.error(f"Error calculating volume fractal dimension: {e}")
            return 1.5

    def calculate(
        self, data: Union[pd.DataFrame, np.ndarray, Dict]
    ) -> Optional[FractalVolumeResult]:
        """
        Calculate Fractal Volume Indicator for given data.

        Args:
            data: Price and volume data

        Returns:
            FractalVolumeResult with analysis
        """
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if "volume" in data.columns and "close" in data.columns:
                    volume = data["volume"].values
                    prices = data["close"].values
                else:
                    self.logger.warning("Volume or close column not found")
                    return None
            elif isinstance(data, dict):
                volume = np.array(data.get("volume", []))
                prices = np.array(data.get("close", []))
            else:
                self.logger.error(f"Unsupported data type: {type(data)}")
                return None

            if len(volume) < self.period or len(prices) < self.period:
                return None
            # Calculate volume metrics
            volume_profile = self.calculate_volume_profile(prices, volume)
            fractal_dimension = self.calculate_volume_fractal_dimension(volume)

            # Calculate volume flow
            recent_volume = volume[-self.period :]
            avg_volume = np.mean(recent_volume)
            volume_trend = (recent_volume[-1] - recent_volume[0]) / (
                recent_volume[0] + 1e-10
            )

            # Determine volume flow direction
            if volume_trend > 0.1:
                volume_flow = "bullish"
            elif volume_trend < -0.1:
                volume_flow = "bearish"
            else:
                volume_flow = "neutral"

            # Calculate strength metrics
            volume_strength = min(abs(volume_trend), 1.0)
            sustainability = max(0, 1 - abs(fractal_dimension - 1.5) * 2)

            return FractalVolumeResult(
                volume_fractal_dimension=float(fractal_dimension),
                volume_profile=volume_profile,
                volume_flow=volume_flow,
                volume_strength=float(volume_strength),
                volume_sustainability=float(sustainability),
                average_volume=float(avg_volume),
            )

        except Exception as e:
            self.logger.error(f"Error calculating Fractal Volume Indicator: {e}")
            return None
