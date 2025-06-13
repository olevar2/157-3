"""
Base Test Class for Gann Indicators

Provides shared testing infrastructure following Platform3 standards,
adapted from Fibonacci test patterns with Gann-specific precision validation.
"""

import time
import unittest
from typing import Dict, Tuple

import pandas as pd

from .gann_test_utils import GannDataGenerator, GannMathUtils, GannPrecisionValidator


class GannTestBase(unittest.TestCase):
    """
    Base test class for all Gann indicators providing shared infrastructure,
    data generators, and validation utilities following Platform3 standards.
    """

    def setUp(self):
        """Set up test data and Gann mathematical constants"""
        # Initialize data generator
        self.data_generator = GannDataGenerator()

        # Generate test datasets of various sizes
        self.small_data = self._generate_small_test_data()
        self.test_data = self._generate_medium_test_data()
        self.large_data = self._generate_large_test_data()

        # Gann mathematical constants with high precision
        self.gann_constants = self._initialize_gann_constants()

        # Geometric precision for angle calculations (6+ decimal places)
        self.geometric_precision = 1e-6
        self.angle_precision = 1e-4  # Degrees precision

        # Initialize precision validator
        self.precision_validator = GannPrecisionValidator(
            geometric_precision=self.geometric_precision,
            angle_precision=self.angle_precision,
        )

        # Mathematical utilities
        self.math_utils = GannMathUtils()

    def _generate_small_test_data(self) -> pd.DataFrame:
        """Generate small dataset for hand-calculable mathematical verification"""
        return self.data_generator.generate_controlled_data(
            n_points=20, base_price=100.0, trend_type="upward", volatility=0.02
        )

    def _generate_medium_test_data(self) -> pd.DataFrame:
        """Generate medium dataset for general testing (100 points)"""
        return self.data_generator.generate_realistic_market_data(
            n_points=100, base_price=100.0, include_trends=True, include_reversals=True
        )

    def _generate_large_test_data(self) -> pd.DataFrame:
        """Generate large dataset for performance testing (10K points)"""
        return self.data_generator.generate_realistic_market_data(
            n_points=10000,
            base_price=100.0,
            include_trends=True,
            include_reversals=True,
            random_seed=42,
        )

    def _initialize_gann_constants(self) -> Dict[str, float]:
        """Initialize Gann mathematical constants with high precision"""
        return {
            # Sacred Gann angles in degrees
            "ANGLE_1x1": 45.0,
            "ANGLE_2x1": 63.43494882292201,  # arctan(2)
            "ANGLE_3x1": 71.56505117707799,  # arctan(3)
            "ANGLE_4x1": 75.96375653207353,  # arctan(4)
            "ANGLE_8x1": 82.87498365109785,  # arctan(8)
            "ANGLE_1x2": 26.56505117707799,  # arctan(0.5)
            "ANGLE_1x3": 18.43494882292201,  # arctan(1/3)
            "ANGLE_1x4": 14.03624346792648,  # arctan(0.25)
            "ANGLE_1x8": 7.125016348902153,  # arctan(0.125)
            # Gann square geometric ratios
            "SQUARE_ROOT_2": 1.4142135623730950488016887242097,
            "SQUARE_ROOT_3": 1.7320508075688772935274463415059,
            "SQUARE_ROOT_5": 2.2360679774997896964091736687313,
            # Circle divisions for time cycles
            "CIRCLE_360": 360.0,
            "CIRCLE_180": 180.0,
            "CIRCLE_90": 90.0,
            "CIRCLE_45": 45.0,
            # Geometric progression constants
            "PHI": 1.6180339887498948482045868343656,  # Golden ratio
            "PI": 3.1415926535897932384626433832795,  # Pi
        }

    def validate_angle_precision(
        self, calculated_angle: float, expected_angle: float, test_name: str = ""
    ) -> None:
        """
        Validate angle calculation precision to required decimal places

        Args:
            calculated_angle: Calculated angle in degrees
            expected_angle: Expected angle in degrees
            test_name: Optional test identifier for error messages
        """
        self.precision_validator.validate_angle(
            calculated_angle, expected_angle, test_name
        )

    def validate_geometric_precision(
        self, calculated_value: float, expected_value: float, test_name: str = ""
    ) -> None:
        """
        Validate geometric calculation precision (6+ decimal places)

        Args:
            calculated_value: Calculated geometric value
            expected_value: Expected geometric value
            test_name: Optional test identifier for error messages
        """
        self.precision_validator.validate_geometric(
            calculated_value, expected_value, test_name
        )

    def validate_ratio_precision(
        self, calculated_ratio: float, expected_ratio: float, test_name: str = ""
    ) -> None:
        """
        Validate Gann ratio calculations with high precision

        Args:
            calculated_ratio: Calculated ratio value
            expected_ratio: Expected ratio value
            test_name: Optional test identifier for error messages
        """
        self.precision_validator.validate_ratio(
            calculated_ratio, expected_ratio, test_name
        )

    def generate_trend_data(
        self, trend_type: str, n_points: int = 100, base_price: float = 100.0
    ) -> pd.DataFrame:
        """
        Generate test data with specific trend characteristics

        Args:
            trend_type: 'upward', 'downward', 'sideways', 'reversal'
            n_points: Number of data points
            base_price: Starting price

        Returns:
            DataFrame with OHLC data exhibiting the specified trend
        """
        return self.data_generator.generate_trend_data(
            trend_type=trend_type, n_points=n_points, base_price=base_price
        )

    def generate_geometric_progression_data(
        self, progression_type: str, n_points: int = 50
    ) -> pd.DataFrame:
        """
        Generate data following geometric progressions for Gann square testing

        Args:
            progression_type: 'fibonacci', 'square_root', 'custom'
            n_points: Number of data points

        Returns:
            DataFrame with geometrically progressive price data
        """
        return self.data_generator.generate_geometric_progression_data(
            progression_type=progression_type, n_points=n_points
        )

    def create_angle_test_scenario(
        self, start_price: float, end_price: float, time_units: int
    ) -> Tuple[float, Dict[str, float]]:
        """
        Create test scenario for angle calculations

        Args:
            start_price: Starting price point
            end_price: Ending price point
            time_units: Number of time units

        Returns:
            Tuple of (calculated_angle, expected_gann_angles)
        """
        return self.math_utils.create_angle_test_scenario(
            start_price, end_price, time_units
        )

    def test_mathematical_accuracy_base(self):
        """Base mathematical accuracy test - should be overridden by subclasses"""
        self.skipTest("Base test - should be implemented in subclasses")

    def test_parameter_validation_base(self):
        """Base parameter validation test - should be overridden by subclasses"""
        self.skipTest("Base test - should be implemented in subclasses")

    def test_data_validation_base(self):
        """Base data validation test - should be overridden by subclasses"""
        self.skipTest("Base test - should be implemented in subclasses")

    def test_edge_cases_base(self):
        """Base edge cases test - should be overridden by subclasses"""
        self.skipTest("Base test - should be implemented in subclasses")

    def test_performance_base(self):
        """Base performance test - should be overridden by subclasses"""
        self.skipTest("Base test - should be implemented in subclasses")

    def test_interface_compliance_base(self):
        """Base interface compliance test - should be overridden by subclasses"""
        self.skipTest("Base test - should be implemented in subclasses")

    def run_performance_benchmark(
        self, indicator, test_data: pd.DataFrame, max_execution_time: float = 5.0
    ) -> Dict[str, float]:
        """
        Run performance benchmark following Platform3 standards

        Args:
            indicator: Indicator instance to test
            test_data: Test data for performance testing
            max_execution_time: Maximum allowed execution time in seconds

        Returns:
            Dictionary containing performance metrics
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()

        # Execute indicator calculation
        result = indicator.calculate(test_data)

        execution_time = time.time() - start_time
        end_memory = self._get_memory_usage()
        memory_delta = end_memory - start_memory

        # Validate performance requirements
        self.assertLess(
            execution_time,
            max_execution_time,
            f"Execution time {execution_time:.3f}s exceeds limit {max_execution_time}s",
        )

        # Validate result integrity
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0, "Result should not be empty")

        return {
            "execution_time": execution_time,
            "memory_delta": memory_delta,
            "data_points": len(test_data),
            "result_points": len(result),
        }

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0  # Skip memory tracking if psutil not available
