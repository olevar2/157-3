"""
Fractal Dimension Indicator

Advanced fractal dimension calculation for market complexity analysis using
box-counting methods to measure self-similarity and chaos in price movements.

Formula:
- Uses box-counting algorithm to measure fractal dimension
- Calculates Hausdorff dimension approximation
- Provides complexity and chaos analysis metrics
- Measures market self-similarity patterns

Author: Platform3 AI Framework
Created: 2025-06-10
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List

# Import the base indicator interface
import sys
import os

from base_indicator import (
    StandardIndicatorInterface,
    IndicatorMetadata,
    IndicatorValidationError,
)


class FractalDimensionIndicator(StandardIndicatorInterface):
    """
    Fractal Dimension Indicator

    Advanced fractal dimension calculation for market complexity analysis using
    box-counting methods to measure self-similarity and chaos in price movements.

    Key Features:
    - Box-counting algorithm for fractal dimension calculation
    - Hausdorff dimension approximation for complexity analysis
    - Market chaos and self-similarity measurement
    - Complexity trending analysis for market state identification
    - Multi-scale analysis for robust dimension estimation

    Mathematical Approach:
    Uses the box-counting method to estimate the Hausdorff dimension of price
    time series, providing insights into market complexity, chaos levels, and
    the degree of self-similarity in price movement patterns.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "fractal"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 50,
        scales: int = 10,
        min_scale: int = 2,
        max_scale: int = 20,
        **kwargs,
    ):
        """
        Initialize Fractal Dimension Indicator

        Args:
            period: Analysis period for dimension calculation (default: 50)
            scales: Number of scales for box-counting (default: 10)
            min_scale: Minimum scale for box-counting (default: 2)
            max_scale: Maximum scale for box-counting (default: 20)
            **kwargs: Additional parameters
        """
        # Validate critical parameters before calling super()
        if period <= 0:
            raise ValueError(f"period must be positive, got {period}")
        if scales <= 0:
            raise ValueError(f"scales must be positive, got {scales}")
        
        # REQUIRED: Call parent constructor with all parameters
        super().__init__(
            period=period,
            scales=scales,
            min_scale=min_scale,
            max_scale=max_scale,
            **kwargs,
        )

    def calculate_fractal_dimension(self, prices: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method."""
        try:
            if len(prices) < 3:
                return 1.0  # Default dimension for insufficient data

            # Normalize prices to 0-1 range
            prices_norm = (prices - np.min(prices)) / (np.max(prices) - np.min(prices) + 1e-10)
            
            min_scale = self.parameters.get("min_scale", 2)
            max_scale = self.parameters.get("max_scale", 20)
            scales = self.parameters.get("scales", 10)
            
            # Generate logarithmically spaced scales
            scale_values = np.logspace(
                np.log10(min_scale), 
                np.log10(min(max_scale, len(prices)//2)), 
                scales
            ).astype(int)
            
            # Remove duplicates and sort
            scale_values = np.unique(scale_values)
            
            if len(scale_values) < 2:
                return 1.5  # Default dimension
            
            box_counts = []
            
            for scale in scale_values:
                # Count boxes that contain the price curve
                x_boxes = int(np.ceil(len(prices) / scale))
                y_boxes = int(np.ceil(1.0 / (1.0 / scale)))
                
                box_count = 0
                for i in range(x_boxes):
                    x_start = i * scale
                    x_end = min((i + 1) * scale, len(prices))
                    
                    if x_end > x_start:
                        y_min = np.min(prices_norm[x_start:x_end])
                        y_max = np.max(prices_norm[x_start:x_end])
                        
                        # Count boxes in y-direction that the curve passes through
                        y_start_box = int(y_min * scale)
                        y_end_box = int(y_max * scale)
                        box_count += max(1, y_end_box - y_start_box + 1)
                
                box_counts.append(max(1, box_count))
            
            # Calculate fractal dimension using linear regression
            log_scales = np.log(1.0 / scale_values)
            log_counts = np.log(box_counts)
            
            # Perform linear regression to find slope (fractal dimension)
            if len(log_scales) >= 2:
                A = np.vstack([log_scales, np.ones(len(log_scales))]).T
                dimension, _ = np.linalg.lstsq(A, log_counts, rcond=None)[0]
                
                # Clamp dimension to reasonable bounds
                dimension = max(1.0, min(2.0, abs(dimension)))
            else:
                dimension = 1.5
                
            return float(dimension)

        except Exception as e:
            raise IndicatorValidationError(f"Error calculating fractal dimension: {e}")

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fractal Dimension analysis.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: Fractal dimension analysis results
        """
        try:
            # Validate input data first
            self.validate_input_data(data)
            
            # Extract close prices
            closes = data["close"].values
            period = self.parameters.get("period", 50)

            if len(closes) < period:
                raise IndicatorValidationError(f"Insufficient data: need {period}, got {len(closes)}")

            # Initialize result arrays
            result_length = len(data)
            fractal_dimension = np.full(result_length, np.nan)
            complexity_level = np.full(result_length, np.nan)
            chaos_indicator = np.full(result_length, np.nan)

            # Calculate fractal dimension for sliding windows
            for i in range(period - 1, result_length):
                window_data = closes[i - period + 1:i + 1]
                
                # Calculate fractal dimension
                dimension = self.calculate_fractal_dimension(window_data)
                fractal_dimension[i] = dimension
                
                # Calculate complexity level (deviation from Euclidean dimension)
                complexity_level[i] = abs(dimension - 1.0) 
                
                # Calculate chaos indicator (higher dimension = more chaotic)
                chaos_indicator[i] = min(1.0, (dimension - 1.0) / 1.0)

            # Create result DataFrame
            result_df = pd.DataFrame({
                "fractal_dimension": fractal_dimension,
                "complexity_level": complexity_level,
                "chaos_indicator": chaos_indicator,
                "market_state": np.where(
                    fractal_dimension > 1.5, 1.0,  # Chaotic
                    np.where(fractal_dimension < 1.2, -1.0, 0.0)  # Ordered vs Neutral
                )
            }, index=data.index)

            # Store calculation details for debugging
            self._last_calculation = {
                "final_dimension": float(fractal_dimension[-1]) if not np.isnan(fractal_dimension[-1]) else None,
                "avg_complexity": float(np.nanmean(complexity_level)),
                "parameters_used": self.parameters
            }

            return result_df

        except Exception as e:
            raise IndicatorValidationError(f"Error in FractalDimensionIndicator calculation: {e}")

    def validate_parameters(self) -> bool:
        """
        Validate indicator parameters for correctness and trading suitability.
        
        Returns:
            bool: True if parameters are valid
            
        Raises:
            IndicatorValidationError: If parameters are invalid
        """
        period = self.parameters.get("period", 50)
        scales = self.parameters.get("scales", 10)
        min_scale = self.parameters.get("min_scale", 2)
        max_scale = self.parameters.get("max_scale", 20)
        
        # Validate parameter ranges
        if not isinstance(period, int) or period <= 0:
            raise IndicatorValidationError(f"period must be positive integer, got {period}")
        if not isinstance(scales, int) or scales <= 0:
            raise IndicatorValidationError(f"scales must be positive integer, got {scales}")
        if not isinstance(min_scale, int) or min_scale < 1:
            raise IndicatorValidationError(f"min_scale must be positive integer, got {min_scale}")
        if not isinstance(max_scale, int) or max_scale <= min_scale:
            raise IndicatorValidationError(f"max_scale must be > min_scale, got {max_scale} <= {min_scale}")
            
        # Validate logical relationships
        if max_scale >= period:
            raise IndicatorValidationError(f"max_scale ({max_scale}) must be less than period ({period})")
            
        return True

    def get_metadata(self) -> IndicatorMetadata:
        """
        Return comprehensive metadata about the indicator.
        
        Returns:
            IndicatorMetadata: Complete indicator specification
        """
        return IndicatorMetadata(
            name="FractalDimensionIndicator",
            category=self.CATEGORY,
            description="Advanced fractal dimension calculation for market complexity analysis",
            parameters=self.parameters,
            input_requirements=["close"],
            output_type="DataFrame",
            version=self.VERSION,
            author=self.AUTHOR,
            trading_grade=True,
            performance_tier="standard",
            min_data_points=self._get_minimum_data_points(),
            max_lookback_period=self.parameters.get("period", 50)
        )

    def _get_required_columns(self) -> List[str]:
        """Get list of required data columns for this indicator."""
        return ["close"]

    def _get_minimum_data_points(self) -> int:
        """Get minimum number of data points required for calculation."""
        return self.parameters.get("period", 50)

    # Backward compatibility properties
    @property
    def period(self) -> int:
        """Period for backward compatibility"""
        return self.parameters.get("period", 50)

    @property
    def scales(self) -> int:
        """Scales for backward compatibility"""
        return self.parameters.get("scales", 10)


def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return FractalDimensionIndicator