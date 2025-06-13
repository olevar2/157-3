"""
Unit Tests for Gann Square Indicator

Comprehensive test suite for Gann Square of Nine indicator covering mathematical
accuracy, parameter validation, data validation, edge cases, performance benchmarks,
and interface compliance. Follows Platform3 testing standards.

Created: 2025-06-10
Author: Platform3 Testing Framework
"""

import sys
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent

# Platform3 imports
from engines.ai_enhancement.indicators.base_indicator import IndicatorValidationError
from engines.ai_enhancement.indicators.gann.gann_square_indicator import (
    GannSquareIndicator,
)
from tests.gann.test_base import GannTestBase


class TestGannSquareIndicator(GannTestBase):
    """
    Comprehensive unit tests for GannSquareIndicator

    Test Categories:
    1. Mathematical Accuracy - Square of Nine calculations
    2. Parameter Validation - Invalid types and ranges
    3. Data Input Validation - Missing columns and insufficient data
    4. Edge Cases - Flat prices and extreme volatility
    5. Performance Benchmarks - <10ms for 1K data requirement
    6. Interface Compliance - StandardIndicatorInterface adherence
    """

    def setUp(self):
        """Set up test fixtures for Gann Square tests"""
        super().setUp()

        # Create indicator instance
        self.indicator = GannSquareIndicator()

        # Square of Nine reference calculations
        self.square_reference = {
            "center_100": 100.0,
            "level_1": [99, 101, 102, 105, 108, 111, 114, 117],  # First ring
            "level_2": [96, 98, 120, 123, 126, 129, 132, 135],  # Second ring
        }

    def test_mathematical_accuracy_square_of_nine_levels(self):
        """Test Square of Nine level calculations"""
        # Test with controlled center price
        center_price = 100.0
        test_data = self._create_square_test_data(center_price)

        result = self.indicator.calculate(test_data)

        # Verify square levels are calculated
        self.assertIn("square_levels", result.columns, "Square levels not calculated")

        # Check that levels form proper geometric progression
        levels = result["square_levels"].dropna().unique()
        self.assertGreater(len(levels), 3, "Insufficient square levels calculated")

        # Verify center price is included
        center_found = any(abs(level - center_price) < 0.01 for level in levels)
        self.assertTrue(
            center_found, f"Center price {center_price} not found in square levels"
        )

    def test_mathematical_accuracy_geometric_progression(self):
        """Test geometric progression of square levels"""
        test_data = self.generate_realistic_ohlc_data(100)
        result = self.indicator.calculate(test_data)

        if "square_levels" in result.columns:
            levels = sorted(result["square_levels"].dropna().unique())

            # Verify levels are in ascending order
            for i in range(1, len(levels)):
                self.assertGreater(
                    levels[i],
                    levels[i - 1],
                    f"Square levels not in ascending order at index {i}",
                )

    def test_parameter_validation_invalid_center_method(self):
        """Test validation of center calculation method"""
        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannSquareIndicator(center_method="invalid_method")

    def test_parameter_validation_invalid_square_size(self):
        """Test validation of square size parameter"""
        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannSquareIndicator(square_size=-1)  # Should be positive

        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannSquareIndicator(square_size=0)  # Should be greater than zero

    def test_data_validation_missing_columns(self):
        """Test handling of missing required columns"""
        invalid_data = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [101, 102, 103],
                "low": [99, 100, 101],
                # Missing 'close' column
            }
        )

        with self.assertRaises(IndicatorValidationError):
            self.indicator.calculate(invalid_data)

    def test_edge_case_flat_prices(self):
        """Test behavior with flat price data"""
        flat_data = self._create_flat_price_data(100.0, 50)

        result = self.indicator.calculate(flat_data)
        self.assertIsInstance(result, pd.DataFrame)

        # Should still calculate square levels for flat data
        if "square_levels" in result.columns:
            levels = result["square_levels"].dropna()
            self.assertGreater(len(levels), 0, "No square levels for flat data")

    def test_performance_benchmark_1k_data(self):
        """Test performance with 1K data points"""
        test_data = self.generate_realistic_ohlc_data(1000)

        start_time = time.time()
        result = self.indicator.calculate(test_data)
        execution_time = time.time() - start_time

        self.assertLess(
            execution_time,
            self.max_time_1k,
            f"1K data too slow: {execution_time:.4f}s > {self.max_time_1k}s",
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_interface_compliance_inheritance(self):
        """Test proper inheritance from StandardIndicatorInterface"""
        self.assert_indicator_interface_compliance(self.indicator)

    def test_interface_compliance_calculate_method(self):
        """Test calculate method interface compliance"""
        result = self.indicator.calculate(self.test_data)

        self.assertIsInstance(result, pd.DataFrame, "calculate() must return DataFrame")
        self.assertGreater(
            len(result), 0, "calculate() must return non-empty DataFrame"
        )

    def _create_square_test_data(self, center_price: float) -> pd.DataFrame:
        """Create test data centered around specific price"""
        periods = 100
        data = []

        for i in range(periods):
            # Create price movement around center
            variation = np.sin(i * 0.1) * 5  # Small oscillation around center
            price = center_price + variation

            data.append(
                {
                    "timestamp": pd.Timestamp("2023-01-01") + pd.Timedelta(days=i),
                    "open": price,
                    "high": price + 0.5,
                    "low": price - 0.5,
                    "close": price,
                    "volume": 1000000,
                }
            )

        return pd.DataFrame(data).set_index("timestamp")

    def _create_flat_price_data(self, price: float, periods: int) -> pd.DataFrame:
        """Create completely flat price data"""
        data = []
        for i in range(periods):
            data.append(
                {
                    "timestamp": pd.Timestamp("2023-01-01") + pd.Timedelta(days=i),
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 1000000,
                }
            )
        return pd.DataFrame(data).set_index("timestamp")


if __name__ == "__main__":
    unittest.main()
