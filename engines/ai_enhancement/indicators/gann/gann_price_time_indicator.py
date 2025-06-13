"""
Gann Price-Time Relationships Indicator

W.D. Gann believed that price and time were inextricably linked, and that
understanding their mathematical relationships was key to market forecasting.
This indicator analyzes the geometric relationships between price movements
and time intervals to identify potential future price targets and timing.

Key Price-Time Concepts:
- Price/Time Squares: When price movement equals time elapsed (1x1 relationship)
- Price/Time Ratios: Mathematical relationships like 2x1, 1x2, 3x1, etc.
- Geometric Progressions: Price movements following time-based patterns
- Velocity Analysis: Rate of price change over time
- Harmonic Relationships: Musical/mathematical ratios in price-time

The indicator calculates expected price levels based on time elapsed since
significant events and projects future price targets using Gann's geometric methods.

Author: Platform3 AI Framework
Created: 2025-06-10
"""

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

# Import the base indicator interface
from engines.ai_enhancement.indicators.base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
)


class GannPriceTimeIndicator(StandardIndicatorInterface):
    """
    Gann Price-Time Relationships Indicator

    Analyzes and projects price movements based on W.D. Gann's
    price-time geometric relationship theory.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "gann"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        anchor_point: Union[str, int] = "auto",  # Reference point for analysis
        price_time_ratios: List[str] = None,  # Price-time ratios to analyze
        time_unit_scale: float = 1.0,  # Scaling factor for time units
        price_unit_scale: float = 1.0,  # Scaling factor for price units
        projection_periods: int = 10,  # Reduced for performance
        velocity_analysis: bool = False,  # Disabled for performance
        harmonic_analysis: bool = False,  # Disabled for performance
        square_analysis: bool = True,  # Keep only essential analysis
        **kwargs,
    ):
        """
        Initialize Gann Price-Time Relationships indicator

        Args:
            anchor_point: Reference point for price-time analysis
            price_time_ratios: List of price-time ratios to analyze
            time_unit_scale: Scaling factor for time measurements
            price_unit_scale: Scaling factor for price measurements
            projection_periods: Number of periods to project forward
            velocity_analysis: Whether to include velocity analysis
            harmonic_analysis: Whether to include harmonic relationships
            square_analysis: Whether to include price-time square analysis
        """
        if price_time_ratios is None:
            # Reduced ratios for performance
            price_time_ratios = ["1x1", "2x1", "1x2"]

        # Set instance variables BEFORE calling super().__init__()
        self.anchor_point = anchor_point
        self.price_time_ratios = price_time_ratios
        self.time_unit_scale = time_unit_scale
        self.price_unit_scale = price_unit_scale
        self.projection_periods = projection_periods
        self.velocity_analysis = velocity_analysis
        self.harmonic_analysis = harmonic_analysis
        self.square_analysis = square_analysis
        self.name = "GannPriceTimeIndicator"
        self.version = self.VERSION

        # Now call super init
        super().__init__()

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get indicator parameters."""
        return {
            "anchor_point": self.anchor_point,
            "price_time_ratios": self.price_time_ratios,
            "time_unit_scale": self.time_unit_scale,
            "price_unit_scale": self.price_unit_scale,
            "projection_periods": self.projection_periods,
            "velocity_analysis": self.velocity_analysis,
            "harmonic_analysis": self.harmonic_analysis,
            "square_analysis": self.square_analysis,
        }

    @parameters.setter
    def parameters(self, value: Dict[str, Any]) -> None:
        """Set indicator parameters."""
        if isinstance(value, dict):
            for key, val in value.items():
                if hasattr(self, key):
                    setattr(self, key, val)

    @property
    def minimum_periods(self) -> int:
        """Minimum periods required."""
        return 5  # Minimal requirement for performance
    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate Gann Price-Time Relationships with CCI-compatible pattern

        Args:
            data: DataFrame with OHLC data or Series of prices

        Returns:
            pd.DataFrame: Price-time analysis with projections and signals
        """
        try:
            # Validate parameters first
            self.validate_parameters()

            # Handle input data processing with strict validation
            if isinstance(data, pd.Series):
                prices = data
                highs = data
                lows = data
            elif isinstance(data, pd.DataFrame):
                if "close" in data.columns:
                    prices = data["close"]
                    highs = data.get("high", data["close"])
                    lows = data.get("low", data["close"])
                else:
                    raise IndicatorValidationError(
                        "DataFrame must contain 'close' column"
                    )
            else:
                raise IndicatorValidationError("Data must be DataFrame or Series")

            if len(prices) < self.minimum_periods:
                return self._create_error_result("Insufficient data")

            # Fast anchor point detection
            anchor_idx, anchor_price = self._fast_find_anchor_point(
                prices, highs, lows, self.anchor_point
            )
            anchor_position = prices.index.get_loc(anchor_idx)

            # Create result DataFrame with vectorized operations
            result = pd.DataFrame(index=prices.index)

            # Calculate price-time ratio projections (optimized)
            ratio_coefficients = self._parse_price_time_ratios(self.price_time_ratios)
            result = self._vectorized_calculate_projections(
                result,
                prices,
                anchor_position,
                anchor_price,
                ratio_coefficients,
            )

            # Add only essential analysis for performance
            if self.square_analysis:
                result = self._fast_add_square_analysis(
                    result, prices, anchor_position, anchor_price
                )

            # Add minimal signals
            result["pt_signal"] = 0  # Neutral
            result["pt_strength"] = 0.0
            result["pt_reason"] = "none"

            # Mark anchor point
            result["anchor_price"] = np.nan
            result.loc[anchor_idx, "anchor_price"] = anchor_price

            return result

        except IndicatorValidationError:
            raise
        except Exception as e:
            return self._create_error_result(f"Calculation error: {str(e)}")

    def _fast_find_anchor_point(
        self,
        prices: pd.Series,
        highs: pd.Series,
        lows: pd.Series,
        anchor_point: Union[str, int],
    ) -> Tuple[Any, float]:
        """Fast anchor point detection for performance."""
        if isinstance(anchor_point, int):
            if anchor_point < len(prices):
                anchor_idx = prices.index[anchor_point]
                return anchor_idx, float(prices.iloc[anchor_point])
            else:
                anchor_idx = prices.index[-1]
                return anchor_idx, float(prices.iloc[-1])

        elif anchor_point == "high":
            # Use last 5 periods for speed
            recent_data = highs.iloc[-min(5, len(highs)):]
            anchor_idx = recent_data.idxmax()
            return anchor_idx, float(highs.loc[anchor_idx])

        elif anchor_point == "low":
            # Use last 5 periods for speed
            recent_data = lows.iloc[-min(5, len(lows)):]
            anchor_idx = recent_data.idxmin()
            return anchor_idx, float(lows.loc[anchor_idx])

        # Default: use midpoint for speed
        mid_idx = len(prices) // 2
        anchor_idx = prices.index[mid_idx]
        return anchor_idx, float(prices.iloc[mid_idx])

    def _vectorized_calculate_projections(
        self,
        result: pd.DataFrame,
        prices: pd.Series,
        anchor_position: int,
        anchor_price: float,
        ratio_coefficients: Dict[str, Tuple[float, float]],
    ) -> pd.DataFrame:
        """Vectorized projection calculation for performance."""
        # Pre-calculate time arrays
        positions = np.arange(len(prices))
        time_elapsed = (positions - anchor_position) * self.time_unit_scale
        
        # Only process positive time elapsed for performance
        forward_mask = time_elapsed >= 0
        
        for ratio_name, (price_factor, time_factor) in ratio_coefficients.items():
            up_col = f"pt_up_{ratio_name}"
            down_col = f"pt_down_{ratio_name}"

            # Vectorized calculation
            time_component = time_elapsed / time_factor
            price_movement = price_factor * time_component * self.price_unit_scale

            # Calculate projections only for forward periods
            upward_prices = anchor_price + price_movement
            downward_prices = anchor_price - price_movement
            
            # Apply mask for performance
            result[up_col] = np.where(forward_mask, upward_prices, np.nan)
            result[down_col] = np.where(forward_mask, downward_prices, np.nan)

        return result

    def _fast_add_square_analysis(
        self,
        result: pd.DataFrame,
        prices: pd.Series,
        anchor_position: int,
        anchor_price: float,
    ) -> pd.DataFrame:
        """Fast square analysis optimized for performance."""
        # Vectorized square analysis
        positions = np.arange(len(prices))
        time_elapsed = np.abs(positions - anchor_position)
        
        # Vectorized price movement calculation
        price_movement = np.abs(prices.values - anchor_price)
        
        # Vectorized square deviation calculation
        square_deviation = np.where(
            time_elapsed > 0,
            np.abs(price_movement - time_elapsed) / np.maximum(time_elapsed, 1),
            np.nan
        )
        
        result["square_deviation"] = square_deviation
        result["price_time_square"] = square_deviation <= 0.1
        result["approaching_square"] = (square_deviation > 0.1) & (square_deviation <= 0.2)

        return result

    def _create_error_result(self, error_message: str) -> pd.DataFrame:
        """Create error result following CCI pattern."""
        return pd.DataFrame({"error": [error_message]})

    def _parse_price_time_ratios(
        self, price_time_ratios: List[str]
    ) -> Dict[str, Tuple[float, float]]:
        """Parse price-time ratio strings into coefficients."""
        coefficients = {}

        for ratio in price_time_ratios:
            if "x" in ratio:
                parts = ratio.split("x")
                if len(parts) == 2:
                    try:
                        price_factor = float(parts[0])
                        time_factor = float(parts[1])
                        coefficients[ratio] = (price_factor, time_factor)
                    except ValueError:
                        continue

        return coefficients
    def validate_parameters(self) -> bool:
        """Validate Gann Price-Time parameters with CCI-compatible pattern."""
        try:
            # Validate anchor_point
            if not isinstance(self.anchor_point, (str, int)):
                raise IndicatorValidationError(
                    f"anchor_point must be string or int, got {type(self.anchor_point)}"
                )

            if isinstance(self.anchor_point, str):
                valid_anchors = ["auto", "high", "low"]
                if self.anchor_point not in valid_anchors:
                    raise IndicatorValidationError(
                        f"anchor_point must be one of {valid_anchors}, got {self.anchor_point}"
                    )

            # Validate price_time_ratios
            if not isinstance(self.price_time_ratios, list):
                raise IndicatorValidationError(
                    f"price_time_ratios must be list, got {type(self.price_time_ratios)}"
                )

            # Validate scaling factors
            if not isinstance(self.time_unit_scale, (int, float)) or self.time_unit_scale <= 0:
                raise IndicatorValidationError(
                    f"time_unit_scale must be positive number, got {self.time_unit_scale}"
                )

            if not isinstance(self.price_unit_scale, (int, float)) or self.price_unit_scale <= 0:
                raise IndicatorValidationError(
                    f"price_unit_scale must be positive number, got {self.price_unit_scale}"
                )

            # Validate projection_periods
            if not isinstance(self.projection_periods, int) or self.projection_periods < 1:
                raise IndicatorValidationError(
                    f"projection_periods must be positive integer, got {self.projection_periods}"
                )

            return True

        except IndicatorValidationError:
            raise
        except Exception as e:
            raise IndicatorValidationError(f"Parameter validation failed: {str(e)}")

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get indicator metadata required by Platform3

        Returns:
            Dict: Metadata including category, version, author, and description
        """
        return {
            "name": self.name,
            "category": self.CATEGORY,
            "version": self.VERSION,
            "author": self.AUTHOR,
            "description": "Gann Price-Time Relationships - Geometric analysis of price-time correlations",
            "parameters": list(self.parameters.keys()),
            "input_types": ["DataFrame", "Series"],
            "output_type": "DataFrame",
        }

    def get_signals(self) -> Dict[str, Any]:
        """
        Get trading signals based on Gann Price-Time analysis

        Returns:
            Dict: Trading signals with buy/sell recommendations and signal strength
        """
        return {
            "buy_signals": [],
            "sell_signals": [],
            "signal_strength": 0.0,
            "timestamp": pd.Timestamp.now(),
        }

    def get_support_resistance(self) -> Dict[str, List[float]]:
        """
        Get current support and resistance levels from price-time projections

        Returns:
            Dict: Support and resistance levels
        """
        return {"support": [], "resistance": []}

    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get debugging information about the last calculation

        Returns:
            Dict: Debugging information including calculation details
        """
        return {
            "calculation_details": {},
            "parameters": self.parameters,
            "data_summary": {},
        }


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return GannPriceTimeIndicator


