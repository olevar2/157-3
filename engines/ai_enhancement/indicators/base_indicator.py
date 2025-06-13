"""
Platform3 Standard Indicator Interface
Base class and interface standards for all individual indicator implementations.

This module defines the contract that all indicators must follow to ensure:
- Consistent API across all indicators
- Trading-grade accuracy and reliability
- Proper error handling and validation
- Performance standards compliance
- Comprehensive testing capabilities
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IndicatorMetadata:
    """Metadata structure for individual indicators"""

    name: str
    category: str
    description: str
    parameters: Dict[str, Any]
    input_requirements: List[str]
    output_type: str
    version: str = "1.0.0"
    author: str = "Platform3"
    trading_grade: bool = True
    performance_tier: str = "standard"  # fast, standard, slow
    min_data_points: int = 1
    max_lookback_period: int = 1000


class IndicatorValidationError(Exception):
    """Custom exception for indicator validation errors"""

    pass


class StandardIndicatorInterface(ABC):
    """
    Abstract base class defining the standard interface for all Platform3 indicators.

    All individual indicator implementations must inherit from this class and implement
    the required methods to ensure consistency, reliability, and trading-grade accuracy.

    Key Requirements:
    - Mathematical precision suitable for trading decisions
    - Robust error handling for edge cases
    - Parameter validation and range checking
    - Performance optimization for real-time usage
    - Comprehensive testing support
    """

    # Class-level metadata (to be overridden by implementations)
    CATEGORY: str = "unknown"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(self, **kwargs):
        """
        Initialize the indicator with parameters.

        Args:
            **kwargs: Indicator-specific parameters
        """
        self.parameters = kwargs
        self.metadata = None
        self._is_initialized = False
        self._last_calculation = None
        self._performance_stats = {}  # Setup defaults first, then validate parameters
        self._setup_defaults()
        self.validate_parameters()
        self._is_initialized = True

        logger.debug(
            f"Initialized {self.__class__.__name__} with parameters: {self.parameters}"
        )

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """
        Main calculation method - must be implemented by all indicators.

        Args:
            data: DataFrame with OHLCV data (minimum required columns depend on indicator)
                  Expected columns: ['open', 'high', 'low', 'close', 'volume']
                  Index should be datetime-based for proper time series handling

        Returns:
            pd.Series or pd.DataFrame: Calculated indicator values
            - For simple indicators: pd.Series with same index as input
            - For complex indicators: pd.DataFrame with multiple columns

        Raises:
            IndicatorValidationError: If input data is invalid
            ValueError: If calculation parameters are invalid
        """
        pass

    @abstractmethod
    def validate_parameters(self) -> bool:
        """
        Validate indicator parameters for correctness and trading suitability.

        Returns:
            bool: True if parameters are valid

        Raises:
            IndicatorValidationError: If parameters are invalid
        """
        pass

    @abstractmethod
    def get_metadata(self) -> IndicatorMetadata:
        """
        Return comprehensive metadata about the indicator.

        Returns:
            IndicatorMetadata: Complete indicator specification
        """
        pass

    def validate_input_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data structure and content.

        Args:
            data: Input DataFrame to validate

        Returns:
            bool: True if data is valid

        Raises:
            IndicatorValidationError: If data is invalid
        """
        if not isinstance(data, pd.DataFrame):
            raise IndicatorValidationError("Input data must be a pandas DataFrame")

        if data.empty:
            raise IndicatorValidationError("Input data cannot be empty")

        # Check for required columns based on indicator requirements
        required_columns = self._get_required_columns()
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise IndicatorValidationError(
                f"Missing required columns: {missing_columns}. "
                f"Required: {required_columns}, Available: {list(data.columns)}"
            )

        # Check for minimum data points
        min_points = self._get_minimum_data_points()
        if len(data) < min_points:
            raise IndicatorValidationError(
                f"Insufficient data points. Required: {min_points}, Available: {len(data)}"
            )

        # Check for NaN values in critical columns
        for col in required_columns:
            if data[col].isna().all():
                raise IndicatorValidationError(
                    f"Column '{col}' contains only NaN values"
                )

        return True

    def _get_required_columns(self) -> List[str]:
        """
        Get list of required data columns for this indicator.
        Override in subclasses to specify requirements.
        """
        return ["close"]  # Default: most indicators need at least close prices

    def _get_minimum_data_points(self) -> int:
        """
        Get minimum number of data points required for calculation.
        Override in subclasses to specify requirements.
        """
        return 1  # Default minimum

    def _setup_defaults(self):
        """
        Setup default parameter values.
        Override in subclasses to set indicator-specific defaults.
        """
        pass

    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about indicator parameters.

        Returns:
            Dict: Parameter specifications with validation rules
        """
        return {
            param_name: {
                "value": param_value,
                "type": type(param_value).__name__,
                "description": f"Parameter {param_name} for {self.__class__.__name__}",
            }
            for param_name, param_value in self.parameters.items()
        }

    def benchmark_performance(
        self, data: pd.DataFrame, iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark indicator performance for optimization.

        Args:
            data: Test data for benchmarking
            iterations: Number of calculation iterations

        Returns:
            Dict: Performance metrics
        """
        import time

        # Validate data first
        self.validate_input_data(data)

        start_time = time.time()
        for _ in range(iterations):
            result = self.calculate(data)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time = total_time / iterations

        self._performance_stats = {
            "total_time_seconds": total_time,
            "average_time_seconds": avg_time,
            "iterations": iterations,
            "data_points": len(data),
            "throughput_points_per_second": len(data) * iterations / total_time,
        }

        return self._performance_stats

    def get_calculation_summary(self) -> Dict[str, Any]:
        """
        Get summary of last calculation performed.

        Returns:
            Dict: Calculation summary and statistics
        """
        if self._last_calculation is None:
            return {"status": "no_calculation_performed"}

        return {
            "status": "calculation_completed",
            "timestamp": datetime.now(),
            "parameters": self.parameters,
            "performance_stats": self._performance_stats,
            "data_shape": getattr(self._last_calculation, "shape", "unknown"),
        }

    def __str__(self) -> str:
        """String representation of the indicator"""
        return f"{self.__class__.__name__}({self.parameters})"

    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"{self.__class__.__name__}(parameters={self.parameters}, initialized={self._is_initialized})"


class TradingGradeValidator:
    """
    Utility class for validating trading-grade accuracy and reliability.
    """

    @staticmethod
    def validate_numerical_precision(values: pd.Series, min_precision: int = 6) -> bool:
        """
        Validate that calculated values have sufficient numerical precision.

        Args:
            values: Calculated indicator values
            min_precision: Minimum decimal precision required

        Returns:
            bool: True if precision is adequate
        """
        # Check for numerical stability
        if values.isna().any():
            logger.warning(
                "Calculated values contain NaN - may indicate numerical instability"
            )
            return False

        if np.isinf(values).any():
            logger.warning(
                "Calculated values contain infinity - numerical overflow detected"
            )
            return False

        return True

    @staticmethod
    def validate_consistency(values: pd.Series, tolerance: float = 1e-10) -> bool:
        """
        Validate calculation consistency (no unexpected jumps or discontinuities).

        Args:
            values: Calculated indicator values
            tolerance: Maximum allowed relative change between consecutive values

        Returns:
            bool: True if values are consistent
        """
        if len(values) < 2:
            return True

        # Check for unrealistic jumps
        diff = values.diff().abs()
        max_change = diff.max()

        if np.isnan(max_change):
            return True  # No valid differences to check

        # Relative to the magnitude of values
        value_magnitude = values.abs().median()
        if value_magnitude > 0:
            relative_change = max_change / value_magnitude
            if relative_change > tolerance:
                logger.warning(f"Large relative change detected: {relative_change}")
                return False

        return True


# Export key classes and functions
__all__ = [
    "StandardIndicatorInterface",
    "IndicatorMetadata",
    "IndicatorValidationError",
    "TradingGradeValidator",
]
