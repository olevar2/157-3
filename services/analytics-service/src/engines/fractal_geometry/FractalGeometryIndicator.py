"""
Fractal Geometry Indicator Module for Advanced Pattern Recognition

This module implements fractal geometry analysis for advanced pattern recognition
and market structure analysis in forex trading. It provides mathematical tools
for analyzing market fractals, calculating fractal dimensions, and identifying
geometric patterns in price data.

Key Features:
- Fractal dimension calculation using box-counting method
- Hurst exponent analysis for trend persistence
- Fractal pattern recognition (Williams Fractals, custom fractals)
- Market structure analysis using fractal geometry
- Multi-timeframe fractal analysis
- Geometric price analysis and forecasting

Author: Platform3 Analytics Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FractalType(Enum):
    """Types of fractal patterns"""
    WILLIAMS_UP = "williams_up"
    WILLIAMS_DOWN = "williams_down"
    CUSTOM_UP = "custom_up"
    CUSTOM_DOWN = "custom_down"
    GEOMETRIC_UP = "geometric_up"
    GEOMETRIC_DOWN = "geometric_down"

class TrendPersistence(Enum):
    """Trend persistence levels based on Hurst exponent"""
    MEAN_REVERTING = "mean_reverting"  # H < 0.5
    RANDOM_WALK = "random_walk"        # H ≈ 0.5
    TRENDING = "trending"              # H > 0.5
    STRONG_TRENDING = "strong_trending" # H > 0.7

@dataclass
class FractalPoint:
    """Represents a fractal point in the market"""
    timestamp: datetime
    price: float
    fractal_type: FractalType
    strength: float
    dimension: float
    confidence: float

@dataclass
class FractalDimension:
    """Fractal dimension analysis result"""
    dimension: float
    confidence: float
    method: str
    timeframe: str
    complexity: float

@dataclass
class HurstAnalysis:
    """Hurst exponent analysis result"""
    hurst_exponent: float
    persistence: TrendPersistence
    confidence: float
    lookback_period: int
    trend_strength: float

@dataclass
class GeometricPattern:
    """Geometric pattern identified through fractal analysis"""
    pattern_type: str
    start_point: FractalPoint
    end_point: FractalPoint
    geometric_ratio: float
    fractal_dimension: float
    prediction_target: Optional[float]
    confidence: float

class FractalGeometryIndicator:
    """
    Advanced Fractal Geometry Indicator for market structure analysis

    This class implements sophisticated fractal geometry techniques for:
    - Calculating fractal dimensions of price movements
    - Identifying fractal patterns and market structure
    - Analyzing trend persistence through Hurst exponent
    - Geometric pattern recognition and forecasting
    """

    def __init__(self,
                 lookback_period: int = 100,
                 fractal_period: int = 5,
                 min_fractal_strength: float = 0.6,
                 hurst_window: int = 50):
        """
        Initialize Fractal Geometry Indicator

        Args:
            lookback_period: Number of periods for analysis
            fractal_period: Period for fractal identification
            min_fractal_strength: Minimum strength for valid fractals
            hurst_window: Window size for Hurst exponent calculation
        """
        self.lookback_period = lookback_period
        self.fractal_period = fractal_period
        self.min_fractal_strength = min_fractal_strength
        self.hurst_window = hurst_window

        # Analysis cache
        self.fractal_cache = {}
        self.dimension_cache = {}
        self.hurst_cache = {}

        # Performance tracking
        self.analysis_count = 0
        self.cache_hits = 0

        logger.info(f"✅ FractalGeometryIndicator initialized with lookback={lookback_period}")

    def calculate_fractal_dimension(self,
                                  price_data: np.ndarray,
                                  method: str = "box_counting") -> FractalDimension:
        """
        Calculate fractal dimension of price series

        Args:
            price_data: Array of price values
            method: Method for calculation ('box_counting', 'correlation', 'variance')

        Returns:
            FractalDimension object with analysis results
        """
        try:
            if method == "box_counting":
                dimension = self._box_counting_dimension(price_data)
            elif method == "correlation":
                dimension = self._correlation_dimension(price_data)
            elif method == "variance":
                dimension = self._variance_dimension(price_data)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Calculate confidence based on data quality
            confidence = self._calculate_dimension_confidence(price_data, dimension)

            # Assess complexity
            complexity = self._assess_complexity(price_data)

            return FractalDimension(
                dimension=dimension,
                confidence=confidence,
                method=method,
                timeframe="current",
                complexity=complexity
            )

        except Exception as e:
            logger.error(f"Error calculating fractal dimension: {e}")
            return FractalDimension(0.0, 0.0, method, "current", 0.0)

    def _box_counting_dimension(self, price_data: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method"""
        try:
            # Normalize price data
            normalized_prices = (price_data - np.min(price_data)) / (np.max(price_data) - np.min(price_data))

            # Create different box sizes
            box_sizes = np.logspace(-2, 0, 20)
            box_counts = []

            for box_size in box_sizes:
                # Count boxes needed to cover the curve
                x_boxes = int(1.0 / box_size) + 1
                y_boxes = int(1.0 / box_size) + 1

                covered_boxes = set()

                for i in range(len(normalized_prices) - 1):
                    x1, y1 = i / len(normalized_prices), normalized_prices[i]
                    x2, y2 = (i + 1) / len(normalized_prices), normalized_prices[i + 1]

                    # Bresenham-like algorithm for line coverage
                    steps = max(abs(int(x2 * x_boxes) - int(x1 * x_boxes)),
                               abs(int(y2 * y_boxes) - int(y1 * y_boxes))) + 1

                    for step in range(steps):
                        t = step / max(steps - 1, 1)
                        x = x1 + t * (x2 - x1)
                        y = y1 + t * (y2 - y1)

                        box_x = int(x * x_boxes)
                        box_y = int(y * y_boxes)
                        covered_boxes.add((box_x, box_y))

                box_counts.append(len(covered_boxes))

            # Calculate fractal dimension from log-log slope
            log_box_sizes = np.log(box_sizes)
            log_box_counts = np.log(box_counts)

            # Linear regression to find slope
            slope, _ = np.polyfit(log_box_sizes, log_box_counts, 1)

            # Fractal dimension is negative slope
            fractal_dimension = -slope

            # Clamp to reasonable range
            return max(1.0, min(2.0, fractal_dimension))

        except Exception as e:
            logger.error(f"Error in box counting: {e}")
            return 1.5  # Default middle value

    def _correlation_dimension(self, price_data: np.ndarray) -> float:
        """Calculate fractal dimension using correlation method"""
        try:
            # Embed the time series in higher dimensions
            embedding_dim = 3
            delay = 1

            # Create embedded vectors
            embedded = []
            for i in range(len(price_data) - (embedding_dim - 1) * delay):
                vector = []
                for j in range(embedding_dim):
                    vector.append(price_data[i + j * delay])
                embedded.append(vector)

            embedded = np.array(embedded)

            # Calculate correlation dimension
            distances = []
            for i in range(len(embedded)):
                for j in range(i + 1, len(embedded)):
                    dist = np.linalg.norm(embedded[i] - embedded[j])
                    distances.append(dist)

            distances = np.array(distances)

            # Calculate correlation integral for different radii
            radii = np.logspace(-3, 0, 20)
            correlations = []

            for radius in radii:
                correlation = np.sum(distances < radius) / len(distances)
                correlations.append(max(correlation, 1e-10))  # Avoid log(0)

            # Calculate dimension from slope
            log_radii = np.log(radii)
            log_correlations = np.log(correlations)

            # Find linear region and calculate slope
            valid_indices = np.isfinite(log_correlations)
            if np.sum(valid_indices) > 5:
                slope, _ = np.polyfit(log_radii[valid_indices], log_correlations[valid_indices], 1)
                return max(1.0, min(2.0, slope))
            else:
                return 1.5

        except Exception as e:
            logger.error(f"Error in correlation dimension: {e}")
            return 1.5

    def _variance_dimension(self, price_data: np.ndarray) -> float:
        """Calculate fractal dimension using variance method"""
        try:
            # Calculate variance at different scales
            scales = np.logspace(0, np.log10(len(price_data) // 4), 10).astype(int)
            variances = []

            for scale in scales:
                if scale >= len(price_data):
                    continue

                # Downsample data
                downsampled = price_data[::scale]
                if len(downsampled) < 2:
                    continue

                # Calculate variance of differences
                differences = np.diff(downsampled)
                variance = np.var(differences)
                variances.append(variance)

            if len(variances) < 3:
                return 1.5

            # Calculate Hurst exponent from variance scaling
            log_scales = np.log(scales[:len(variances)])
            log_variances = np.log(variances)

            slope, _ = np.polyfit(log_scales, log_variances, 1)
            hurst = slope / 2

            # Convert Hurst to fractal dimension
            fractal_dimension = 2 - hurst

            return max(1.0, min(2.0, fractal_dimension))

        except Exception as e:
            logger.error(f"Error in variance dimension: {e}")
            return 1.5

    def _calculate_dimension_confidence(self, price_data: np.ndarray, dimension: float) -> float:
        """Calculate confidence score for fractal dimension"""
        try:
            # Base confidence on data length and quality
            data_length_score = min(1.0, len(price_data) / 100)

            # Check for reasonable dimension value
            dimension_score = 1.0 - abs(dimension - 1.5) / 0.5
            dimension_score = max(0.0, min(1.0, dimension_score))

            # Check data variability
            variability = np.std(price_data) / np.mean(price_data) if np.mean(price_data) != 0 else 0
            variability_score = min(1.0, variability * 10)

            # Combined confidence
            confidence = (data_length_score + dimension_score + variability_score) / 3
            return max(0.1, min(1.0, confidence))

        except Exception:
            return 0.5

    def _assess_complexity(self, price_data: np.ndarray) -> float:
        """Assess market complexity based on price movements"""
        try:
            # Calculate various complexity measures
            returns = np.diff(price_data) / price_data[:-1]

            # Volatility clustering
            volatility = np.abs(returns)
            vol_autocorr = np.corrcoef(volatility[:-1], volatility[1:])[0, 1]
            vol_autocorr = 0 if np.isnan(vol_autocorr) else abs(vol_autocorr)

            # Return distribution kurtosis
            kurtosis = self._calculate_kurtosis(returns)
            kurtosis_score = min(1.0, abs(kurtosis - 3) / 10)

            # Trend changes
            trend_changes = np.sum(np.diff(np.sign(returns)) != 0) / len(returns)

            # Combined complexity score
            complexity = (vol_autocorr + kurtosis_score + trend_changes) / 3
            return max(0.0, min(1.0, complexity))

        except Exception:
            return 0.5

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 3.0

            normalized = (data - mean) / std
            kurtosis = np.mean(normalized ** 4)
            return kurtosis

        except Exception:
            return 3.0

    def calculate_hurst_exponent(self, price_data: np.ndarray) -> HurstAnalysis:
        """
        Calculate Hurst exponent for trend persistence analysis

        Args:
            price_data: Array of price values

        Returns:
            HurstAnalysis object with results
        """
        try:
            # Use R/S analysis method
            hurst = self._rs_hurst_exponent(price_data)

            # Determine persistence type
            if hurst < 0.45:
                persistence = TrendPersistence.MEAN_REVERTING
            elif hurst < 0.55:
                persistence = TrendPersistence.RANDOM_WALK
            elif hurst < 0.7:
                persistence = TrendPersistence.TRENDING
            else:
                persistence = TrendPersistence.STRONG_TRENDING

            # Calculate confidence
            confidence = self._calculate_hurst_confidence(price_data, hurst)

            # Calculate trend strength
            trend_strength = abs(hurst - 0.5) * 2

            return HurstAnalysis(
                hurst_exponent=hurst,
                persistence=persistence,
                confidence=confidence,
                lookback_period=len(price_data),
                trend_strength=trend_strength
            )

        except Exception as e:
            logger.error(f"Error calculating Hurst exponent: {e}")
            return HurstAnalysis(0.5, TrendPersistence.RANDOM_WALK, 0.0, len(price_data), 0.0)

    def _rs_hurst_exponent(self, price_data: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis"""
        try:
            # Convert prices to log returns
            log_returns = np.diff(np.log(price_data))

            # Calculate R/S for different time scales
            scales = np.unique(np.logspace(1, np.log10(len(log_returns) // 4), 10).astype(int))
            rs_values = []

            for scale in scales:
                if scale >= len(log_returns):
                    continue

                # Split data into non-overlapping windows
                n_windows = len(log_returns) // scale
                rs_window_values = []

                for i in range(n_windows):
                    window = log_returns[i * scale:(i + 1) * scale]

                    # Calculate mean
                    mean_return = np.mean(window)

                    # Calculate cumulative deviations
                    deviations = np.cumsum(window - mean_return)

                    # Calculate range
                    R = np.max(deviations) - np.min(deviations)

                    # Calculate standard deviation
                    S = np.std(window)

                    # R/S ratio
                    if S > 0:
                        rs_window_values.append(R / S)

                if rs_window_values:
                    rs_values.append(np.mean(rs_window_values))

            if len(rs_values) < 3:
                return 0.5

            # Calculate Hurst exponent from log-log regression
            log_scales = np.log(scales[:len(rs_values)])
            log_rs = np.log(rs_values)

            # Remove any infinite or NaN values
            valid_mask = np.isfinite(log_scales) & np.isfinite(log_rs)
            if np.sum(valid_mask) < 3:
                return 0.5

            hurst, _ = np.polyfit(log_scales[valid_mask], log_rs[valid_mask], 1)

            # Clamp to reasonable range
            return max(0.1, min(0.9, hurst))

        except Exception as e:
            logger.error(f"Error in R/S Hurst calculation: {e}")
            return 0.5

    def _calculate_hurst_confidence(self, price_data: np.ndarray, hurst: float) -> float:
        """Calculate confidence in Hurst exponent calculation"""
        try:
            # Base confidence on data length
            length_score = min(1.0, len(price_data) / 200)

            # Penalize extreme values
            extreme_penalty = abs(hurst - 0.5) * 0.5
            hurst_score = max(0.0, 1.0 - extreme_penalty)

            # Check data quality
            returns = np.diff(price_data) / price_data[:-1]
            quality_score = 1.0 - min(0.5, np.sum(np.isnan(returns)) / len(returns))

            confidence = (length_score + hurst_score + quality_score) / 3
            return max(0.1, min(1.0, confidence))

        except Exception:
            return 0.5

    def identify_fractal_patterns(self,
                                ohlc_data: pd.DataFrame,
                                pattern_type: str = "williams") -> List[FractalPoint]:
        """
        Identify fractal patterns in OHLC data

        Args:
            ohlc_data: DataFrame with OHLC columns
            pattern_type: Type of fractal pattern ('williams', 'custom', 'geometric')

        Returns:
            List of FractalPoint objects
        """
        try:
            if pattern_type == "williams":
                return self._identify_williams_fractals(ohlc_data)
            elif pattern_type == "custom":
                return self._identify_custom_fractals(ohlc_data)
            elif pattern_type == "geometric":
                return self._identify_geometric_fractals(ohlc_data)
            else:
                raise ValueError(f"Unknown pattern type: {pattern_type}")

        except Exception as e:
            logger.error(f"Error identifying fractal patterns: {e}")
            return []

    def _identify_williams_fractals(self, ohlc_data: pd.DataFrame) -> List[FractalPoint]:
        """Identify Williams Fractal patterns"""
        fractals = []

        try:
            highs = ohlc_data['high'].values
            lows = ohlc_data['low'].values
            timestamps = ohlc_data.index

            # Look for fractal highs (up fractals)
            for i in range(self.fractal_period, len(highs) - self.fractal_period):
                is_fractal_high = True
                center_high = highs[i]

                # Check if center is highest in the window
                for j in range(i - self.fractal_period, i + self.fractal_period + 1):
                    if j != i and highs[j] >= center_high:
                        is_fractal_high = False
                        break

                if is_fractal_high:
                    strength = self._calculate_fractal_strength(highs, i, True)
                    if strength >= self.min_fractal_strength:
                        dimension = self._calculate_local_dimension(highs, i)
                        confidence = self._calculate_fractal_confidence(highs, i, True)

                        fractals.append(FractalPoint(
                            timestamp=timestamps[i],
                            price=center_high,
                            fractal_type=FractalType.WILLIAMS_UP,
                            strength=strength,
                            dimension=dimension,
                            confidence=confidence
                        ))

            # Look for fractal lows (down fractals)
            for i in range(self.fractal_period, len(lows) - self.fractal_period):
                is_fractal_low = True
                center_low = lows[i]

                # Check if center is lowest in the window
                for j in range(i - self.fractal_period, i + self.fractal_period + 1):
                    if j != i and lows[j] <= center_low:
                        is_fractal_low = False
                        break

                if is_fractal_low:
                    strength = self._calculate_fractal_strength(lows, i, False)
                    if strength >= self.min_fractal_strength:
                        dimension = self._calculate_local_dimension(lows, i)
                        confidence = self._calculate_fractal_confidence(lows, i, False)

                        fractals.append(FractalPoint(
                            timestamp=timestamps[i],
                            price=center_low,
                            fractal_type=FractalType.WILLIAMS_DOWN,
                            strength=strength,
                            dimension=dimension,
                            confidence=confidence
                        ))

            return sorted(fractals, key=lambda x: x.timestamp)

        except Exception as e:
            logger.error(f"Error in Williams fractal identification: {e}")
            return []

    def _calculate_fractal_strength(self, price_data: np.ndarray, index: int, is_high: bool) -> float:
        """Calculate strength of fractal point"""
        try:
            center_price = price_data[index]
            window_start = max(0, index - self.fractal_period)
            window_end = min(len(price_data), index + self.fractal_period + 1)

            window_prices = price_data[window_start:window_end]

            if is_high:
                # For highs, strength is how much higher than surrounding prices
                other_prices = np.concatenate([window_prices[:index-window_start],
                                             window_prices[index-window_start+1:]])
                if len(other_prices) > 0:
                    max_other = np.max(other_prices)
                    strength = (center_price - max_other) / center_price if center_price > 0 else 0
                else:
                    strength = 0
            else:
                # For lows, strength is how much lower than surrounding prices
                other_prices = np.concatenate([window_prices[:index-window_start],
                                             window_prices[index-window_start+1:]])
                if len(other_prices) > 0:
                    min_other = np.min(other_prices)
                    strength = (min_other - center_price) / center_price if center_price > 0 else 0
                else:
                    strength = 0

            return max(0.0, min(1.0, strength * 100))  # Scale to 0-1 range

        except Exception:
            return 0.0

    def _calculate_local_dimension(self, price_data: np.ndarray, index: int) -> float:
        """Calculate local fractal dimension around a point"""
        try:
            window_start = max(0, index - 10)
            window_end = min(len(price_data), index + 11)
            local_data = price_data[window_start:window_end]

            if len(local_data) < 5:
                return 1.5

            # Use box counting on local window
            return self._box_counting_dimension(local_data)

        except Exception:
            return 1.5

    def _calculate_fractal_confidence(self, price_data: np.ndarray, index: int, is_high: bool) -> float:
        """Calculate confidence in fractal identification"""
        try:
            # Check how clear the fractal is
            center_price = price_data[index]
            window_start = max(0, index - self.fractal_period)
            window_end = min(len(price_data), index + self.fractal_period + 1)

            window_prices = price_data[window_start:window_end]

            if is_high:
                # Count how many prices are clearly below
                clear_below = np.sum(window_prices < center_price * 0.999)
                total_prices = len(window_prices) - 1  # Exclude center
            else:
                # Count how many prices are clearly above
                clear_above = np.sum(window_prices > center_price * 1.001)
                total_prices = len(window_prices) - 1  # Exclude center

            if total_prices > 0:
                if is_high:
                    confidence = clear_below / total_prices
                else:
                    confidence = clear_above / total_prices
            else:
                confidence = 0.0

            return max(0.1, min(1.0, confidence))

        except Exception:
            return 0.5

    def _identify_custom_fractals(self, ohlc_data: pd.DataFrame) -> List[FractalPoint]:
        """Identify custom fractal patterns with enhanced criteria"""
        fractals = []

        try:
            highs = ohlc_data['high'].values
            lows = ohlc_data['low'].values
            closes = ohlc_data['close'].values
            timestamps = ohlc_data.index

            # Enhanced fractal detection with volume and momentum
            for i in range(self.fractal_period * 2, len(highs) - self.fractal_period * 2):
                # Check for enhanced fractal high
                if self._is_enhanced_fractal_high(highs, closes, i):
                    strength = self._calculate_enhanced_strength(highs, closes, i, True)
                    if strength >= self.min_fractal_strength:
                        dimension = self._calculate_local_dimension(highs, i)
                        confidence = self._calculate_enhanced_confidence(highs, closes, i, True)

                        fractals.append(FractalPoint(
                            timestamp=timestamps[i],
                            price=highs[i],
                            fractal_type=FractalType.CUSTOM_UP,
                            strength=strength,
                            dimension=dimension,
                            confidence=confidence
                        ))

                # Check for enhanced fractal low
                if self._is_enhanced_fractal_low(lows, closes, i):
                    strength = self._calculate_enhanced_strength(lows, closes, i, False)
                    if strength >= self.min_fractal_strength:
                        dimension = self._calculate_local_dimension(lows, i)
                        confidence = self._calculate_enhanced_confidence(lows, closes, i, False)

                        fractals.append(FractalPoint(
                            timestamp=timestamps[i],
                            price=lows[i],
                            fractal_type=FractalType.CUSTOM_DOWN,
                            strength=strength,
                            dimension=dimension,
                            confidence=confidence
                        ))

            return sorted(fractals, key=lambda x: x.timestamp)

        except Exception as e:
            logger.error(f"Error in custom fractal identification: {e}")
            return []

    def _is_enhanced_fractal_high(self, highs: np.ndarray, closes: np.ndarray, index: int) -> bool:
        """Check if point is an enhanced fractal high"""
        try:
            center_high = highs[index]
            period = self.fractal_period

            # Basic fractal check
            for j in range(index - period, index + period + 1):
                if j != index and j >= 0 and j < len(highs):
                    if highs[j] >= center_high:
                        return False

            # Additional momentum check
            if index >= 3 and index < len(closes) - 3:
                momentum_before = closes[index-1] - closes[index-3]
                momentum_after = closes[index+3] - closes[index+1]

                # Look for momentum divergence
                if momentum_before > 0 and momentum_after < 0:
                    return True

            return True

        except Exception:
            return False

    def _is_enhanced_fractal_low(self, lows: np.ndarray, closes: np.ndarray, index: int) -> bool:
        """Check if point is an enhanced fractal low"""
        try:
            center_low = lows[index]
            period = self.fractal_period

            # Basic fractal check
            for j in range(index - period, index + period + 1):
                if j != index and j >= 0 and j < len(lows):
                    if lows[j] <= center_low:
                        return False

            # Additional momentum check
            if index >= 3 and index < len(closes) - 3:
                momentum_before = closes[index-1] - closes[index-3]
                momentum_after = closes[index+3] - closes[index+1]

                # Look for momentum divergence
                if momentum_before < 0 and momentum_after > 0:
                    return True

            return True

        except Exception:
            return False

    def _calculate_enhanced_strength(self, price_data: np.ndarray, closes: np.ndarray,
                                   index: int, is_high: bool) -> float:
        """Calculate enhanced fractal strength including momentum"""
        try:
            basic_strength = self._calculate_fractal_strength(price_data, index, is_high)

            # Add momentum component
            if index >= 3 and index < len(closes) - 3:
                momentum_strength = abs(closes[index+1] - closes[index-1]) / closes[index]
                momentum_strength = min(1.0, momentum_strength * 100)
            else:
                momentum_strength = 0.0

            # Combine strengths
            combined_strength = (basic_strength + momentum_strength) / 2
            return max(0.0, min(1.0, combined_strength))

        except Exception:
            return 0.0

    def _calculate_enhanced_confidence(self, price_data: np.ndarray, closes: np.ndarray,
                                     index: int, is_high: bool) -> float:
        """Calculate enhanced confidence including momentum analysis"""
        try:
            basic_confidence = self._calculate_fractal_confidence(price_data, index, is_high)

            # Add momentum confidence
            if index >= 5 and index < len(closes) - 5:
                # Check for momentum divergence
                price_momentum = price_data[index] - price_data[index-5]
                close_momentum = closes[index] - closes[index-5]

                if is_high:
                    momentum_confidence = 1.0 if price_momentum > 0 and close_momentum < 0 else 0.5
                else:
                    momentum_confidence = 1.0 if price_momentum < 0 and close_momentum > 0 else 0.5
            else:
                momentum_confidence = 0.5

            # Combine confidences
            combined_confidence = (basic_confidence + momentum_confidence) / 2
            return max(0.1, min(1.0, combined_confidence))

        except Exception:
            return 0.5

    def _identify_geometric_fractals(self, ohlc_data: pd.DataFrame) -> List[FractalPoint]:
        """Identify geometric fractal patterns using mathematical ratios"""
        fractals = []

        try:
            highs = ohlc_data['high'].values
            lows = ohlc_data['low'].values
            timestamps = ohlc_data.index

            # Golden ratio and other geometric ratios
            golden_ratio = 1.618
            fibonacci_ratios = [0.382, 0.618, 1.0, 1.618, 2.618]

            for i in range(10, len(highs) - 10):
                # Check for geometric high patterns
                if self._is_geometric_high(highs, i, fibonacci_ratios):
                    strength = self._calculate_geometric_strength(highs, i, True, fibonacci_ratios)
                    if strength >= self.min_fractal_strength:
                        dimension = self._calculate_local_dimension(highs, i)
                        confidence = self._calculate_geometric_confidence(highs, i, True)

                        fractals.append(FractalPoint(
                            timestamp=timestamps[i],
                            price=highs[i],
                            fractal_type=FractalType.GEOMETRIC_UP,
                            strength=strength,
                            dimension=dimension,
                            confidence=confidence
                        ))

                # Check for geometric low patterns
                if self._is_geometric_low(lows, i, fibonacci_ratios):
                    strength = self._calculate_geometric_strength(lows, i, False, fibonacci_ratios)
                    if strength >= self.min_fractal_strength:
                        dimension = self._calculate_local_dimension(lows, i)
                        confidence = self._calculate_geometric_confidence(lows, i, False)

                        fractals.append(FractalPoint(
                            timestamp=timestamps[i],
                            price=lows[i],
                            fractal_type=FractalType.GEOMETRIC_DOWN,
                            strength=strength,
                            dimension=dimension,
                            confidence=confidence
                        ))

            return sorted(fractals, key=lambda x: x.timestamp)

        except Exception as e:
            logger.error(f"Error in geometric fractal identification: {e}")
            return []

    def _is_geometric_high(self, highs: np.ndarray, index: int, ratios: List[float]) -> bool:
        """Check if point forms geometric high pattern"""
        try:
            center_high = highs[index]

            # Look for geometric relationships in surrounding prices
            for lookback in [3, 5, 8, 13]:  # Fibonacci numbers
                if index >= lookback and index < len(highs) - lookback:
                    left_high = highs[index - lookback]
                    right_high = highs[index + lookback] if index + lookback < len(highs) else center_high

                    # Check if center forms geometric relationship
                    for ratio in ratios:
                        expected_high = left_high * ratio
                        tolerance = center_high * 0.01  # 1% tolerance

                        if abs(center_high - expected_high) < tolerance:
                            return True

            return False

        except Exception:
            return False

    def _is_geometric_low(self, lows: np.ndarray, index: int, ratios: List[float]) -> bool:
        """Check if point forms geometric low pattern"""
        try:
            center_low = lows[index]

            # Look for geometric relationships in surrounding prices
            for lookback in [3, 5, 8, 13]:  # Fibonacci numbers
                if index >= lookback and index < len(lows) - lookback:
                    left_low = lows[index - lookback]
                    right_low = lows[index + lookback] if index + lookback < len(lows) else center_low

                    # Check if center forms geometric relationship
                    for ratio in ratios:
                        expected_low = left_low / ratio
                        tolerance = center_low * 0.01  # 1% tolerance

                        if abs(center_low - expected_low) < tolerance:
                            return True

            return False

        except Exception:
            return False

    def _calculate_geometric_strength(self, price_data: np.ndarray, index: int,
                                    is_high: bool, ratios: List[float]) -> float:
        """Calculate strength of geometric pattern"""
        try:
            center_price = price_data[index]
            max_strength = 0.0

            # Find best geometric relationship
            for lookback in [3, 5, 8, 13]:
                if index >= lookback:
                    reference_price = price_data[index - lookback]

                    for ratio in ratios:
                        if is_high:
                            expected_price = reference_price * ratio
                        else:
                            expected_price = reference_price / ratio

                        # Calculate how close to expected ratio
                        if expected_price > 0:
                            accuracy = 1.0 - abs(center_price - expected_price) / expected_price
                            strength = max(0.0, accuracy)
                            max_strength = max(max_strength, strength)

            return min(1.0, max_strength)

        except Exception:
            return 0.0

    def _calculate_geometric_confidence(self, price_data: np.ndarray, index: int, is_high: bool) -> float:
        """Calculate confidence in geometric pattern"""
        try:
            # Check multiple geometric relationships
            confidence_scores = []

            for lookback in [3, 5, 8, 13]:
                if index >= lookback and index < len(price_data) - lookback:
                    left_price = price_data[index - lookback]
                    center_price = price_data[index]
                    right_price = price_data[index + lookback]

                    # Check symmetry and geometric progression
                    if left_price > 0 and right_price > 0:
                        left_ratio = center_price / left_price
                        right_ratio = right_price / center_price

                        # Good geometric pattern should have consistent ratios
                        ratio_consistency = 1.0 - abs(left_ratio - right_ratio) / max(left_ratio, right_ratio)
                        confidence_scores.append(max(0.0, ratio_consistency))

            if confidence_scores:
                return sum(confidence_scores) / len(confidence_scores)
            else:
                return 0.5

        except Exception:
            return 0.5

    def analyze_market_structure(self, ohlc_data: pd.DataFrame) -> Dict:
        """
        Comprehensive market structure analysis using fractal geometry

        Args:
            ohlc_data: DataFrame with OHLC data

        Returns:
            Dictionary with comprehensive analysis results
        """
        try:
            # Get all fractal types
            williams_fractals = self.identify_fractal_patterns(ohlc_data, "williams")
            custom_fractals = self.identify_fractal_patterns(ohlc_data, "custom")
            geometric_fractals = self.identify_fractal_patterns(ohlc_data, "geometric")

            # Calculate fractal dimensions
            close_prices = ohlc_data['close'].values
            fractal_dimension = self.calculate_fractal_dimension(close_prices)

            # Calculate Hurst exponent
            hurst_analysis = self.calculate_hurst_exponent(close_prices)

            # Analyze fractal distribution
            fractal_density = len(williams_fractals) / len(ohlc_data) if len(ohlc_data) > 0 else 0

            # Market regime classification
            market_regime = self._classify_market_regime(fractal_dimension, hurst_analysis)

            # Performance metrics
            self.analysis_count += 1

            return {
                'fractal_dimension': fractal_dimension,
                'hurst_analysis': hurst_analysis,
                'williams_fractals': williams_fractals,
                'custom_fractals': custom_fractals,
                'geometric_fractals': geometric_fractals,
                'fractal_density': fractal_density,
                'market_regime': market_regime,
                'total_fractals': len(williams_fractals) + len(custom_fractals) + len(geometric_fractals),
                'analysis_timestamp': datetime.now(),
                'performance_stats': {
                    'analysis_count': self.analysis_count,
                    'cache_hits': self.cache_hits,
                    'cache_hit_rate': self.cache_hits / max(1, self.analysis_count)
                }
            }

        except Exception as e:
            logger.error(f"Error in market structure analysis: {e}")
            return {}

    def _classify_market_regime(self, fractal_dim: FractalDimension, hurst: HurstAnalysis) -> str:
        """Classify market regime based on fractal analysis"""
        try:
            # Combine fractal dimension and Hurst exponent
            if hurst.persistence == TrendPersistence.STRONG_TRENDING:
                if fractal_dim.dimension < 1.3:
                    return "strong_trend_low_complexity"
                else:
                    return "strong_trend_high_complexity"
            elif hurst.persistence == TrendPersistence.TRENDING:
                if fractal_dim.dimension < 1.4:
                    return "trending_low_complexity"
                else:
                    return "trending_high_complexity"
            elif hurst.persistence == TrendPersistence.MEAN_REVERTING:
                if fractal_dim.dimension > 1.6:
                    return "mean_reverting_high_complexity"
                else:
                    return "mean_reverting_low_complexity"
            else:
                if fractal_dim.dimension > 1.5:
                    return "random_walk_high_complexity"
                else:
                    return "random_walk_low_complexity"

        except Exception:
            return "unknown_regime"

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            'analysis_count': self.analysis_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / max(1, self.analysis_count),
            'cache_size': len(self.fractal_cache) + len(self.dimension_cache) + len(self.hurst_cache)
        }
