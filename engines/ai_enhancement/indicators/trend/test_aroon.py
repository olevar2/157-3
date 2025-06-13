"""
Unit tests for Aroon Indicator
Following Platform3 testing standards
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add the indicators directory to the path
from aroon import AroonIndicator


class TestAroonIndicator(unittest.TestCase):
    """Test suite for Aroon indicator"""

    def setUp(self):
        """Set up test data and indicator instances"""
        # Create test OHLC data
        np.random.seed(42)
        n_points = 50
        base_price = 100

        # Generate realistic OHLC data
        close_prices = [base_price]
        for _ in range(n_points - 1):
            change = np.random.randn() * 0.5
            close_prices.append(close_prices[-1] + change)

        self.test_data = pd.DataFrame(
            {
                "close": close_prices,
                "high": [c + abs(np.random.randn() * 0.3) for c in close_prices],
                "low": [c - abs(np.random.randn() * 0.3) for c in close_prices],
            }
        )

        # Create test Series
        self.test_series = pd.Series(close_prices)

        # Create Aroon instance with default parameters
        self.aroon = AroonIndicator()

        # Create Aroon with custom parameters
        self.aroon_custom = AroonIndicator(period=10, include_oscillator=False)

    def test_initialization(self):
        """Test Aroon initialization with various parameters"""
        # Test default initialization
        aroon_default = AroonIndicator()
        self.assertEqual(aroon_default.period, 14)
        self.assertEqual(aroon_default.include_oscillator, True)

        # Test custom initialization
        aroon_custom = AroonIndicator(period=21, include_oscillator=False)
        self.assertEqual(aroon_custom.period, 21)
        self.assertEqual(aroon_custom.include_oscillator, False)

    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test valid parameters
        self.assertTrue(self.aroon.validate_parameters())

        # Test invalid period
        with self.assertRaises(Exception):
            AroonIndicator(period=0)

        with self.assertRaises(Exception):
            AroonIndicator(period=-5)

        with self.assertRaises(Exception):
            AroonIndicator(period=1001)

        # Test invalid include_oscillator
        with self.assertRaises(Exception):
            AroonIndicator(include_oscillator="yes")

    def test_data_validation(self):
        """Test input data validation"""
        # Test valid DataFrame data
        self.assertTrue(self.aroon.validate_input_data(self.test_data))

        # Test empty DataFrame
        empty_df = pd.DataFrame()
        with self.assertRaises(Exception):
            self.aroon.calculate(empty_df)

        # Test DataFrame without required columns (should work with close)
        close_only_df = pd.DataFrame({"close": [1, 2, 3, 4, 5] * 4})
        result = self.aroon.calculate(close_only_df)
        self.assertIsInstance(result, pd.DataFrame)

        # Test insufficient data
        short_data = pd.DataFrame(
            {"high": [1, 2, 3], "low": [0.8, 1.8, 2.8], "close": [0.9, 1.9, 2.9]}
        )
        with self.assertRaises(Exception):
            self.aroon.validate_input_data(short_data)

    def test_basic_calculation_dataframe(self):
        """Test basic Aroon calculation with DataFrame"""
        result = self.aroon.calculate(self.test_data)

        # Check result type and structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.test_data))

        # Check required columns
        expected_columns = ["AROON_UP", "AROON_DOWN", "AROON_OSC"]
        for col in expected_columns:
            self.assertIn(col, result.columns)

        # Check Aroon range (0-100)
        valid_up = result["AROON_UP"].dropna()
        valid_down = result["AROON_DOWN"].dropna()
        self.assertTrue(all(valid_up >= 0))
        self.assertTrue(all(valid_up <= 100))
        self.assertTrue(all(valid_down >= 0))
        self.assertTrue(all(valid_down <= 100))

        # Check Aroon Oscillator range (-100 to 100)
        valid_osc = result["AROON_OSC"].dropna()
        self.assertTrue(all(valid_osc >= -100))
        self.assertTrue(all(valid_osc <= 100))

        # Check that we have NaN values at the beginning (insufficient data)
        period = self.aroon.period
        self.assertTrue(result["AROON_UP"].iloc[: period - 1].isna().any())

    def test_basic_calculation_series(self):
        """Test basic Aroon calculation with Series"""
        result = self.aroon.calculate(self.test_series)

        # Check result type and structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.test_series))

        # Check required columns
        expected_columns = ["AROON_UP", "AROON_DOWN", "AROON_OSC"]
        for col in expected_columns:
            self.assertIn(col, result.columns)

    def test_without_oscillator(self):
        """Test Aroon calculation without oscillator"""
        aroon_no_osc = AroonIndicator(include_oscillator=False)
        result = aroon_no_osc.calculate(self.test_data)

        # Check that oscillator column is not present
        self.assertNotIn("AROON_OSC", result.columns)

        # Check that up and down are present
        self.assertIn("AROON_UP", result.columns)
        self.assertIn("AROON_DOWN", result.columns)

    def test_calculation_edge_cases(self):
        """Test Aroon calculation with edge cases"""
        # Test with consistently increasing data (should have high Aroon Up)
        increasing_data = pd.DataFrame(
            {
                "high": pd.Series(range(1, 31)) + 0.1,
                "low": pd.Series(range(1, 31)) - 0.1,
                "close": pd.Series(range(1, 31)),
            }
        )
        aroon_up = AroonIndicator(period=14)
        result_up = aroon_up.calculate(increasing_data)

        # Final Aroon Up should be high (100 since highest high is most recent)
        final_aroon_up = result_up["AROON_UP"].iloc[-1]
        self.assertGreaterEqual(final_aroon_up, 90)

        # Test with consistently decreasing data (should have high Aroon Down)
        decreasing_data = pd.DataFrame(
            {
                "high": pd.Series(range(30, 0, -1)) + 0.1,
                "low": pd.Series(range(30, 0, -1)) - 0.1,
                "close": pd.Series(range(30, 0, -1)),
            }
        )
        aroon_down = AroonIndicator(period=14)
        result_down = aroon_down.calculate(decreasing_data)

        # Final Aroon Down should be high (100 since lowest low is most recent)
        final_aroon_down = result_down["AROON_DOWN"].iloc[-1]
        self.assertGreaterEqual(final_aroon_down, 90)

    def test_different_periods(self):
        """Test Aroon with different periods"""
        periods = [7, 14, 21]

        for period in periods:
            aroon = AroonIndicator(period=period)
            # Generate enough data for each period
            test_data = pd.DataFrame(
                {
                    "high": np.random.randn(period + 10) + 100.5,
                    "low": np.random.randn(period + 10) + 99.5,
                    "close": np.random.randn(period + 10) + 100,
                }
            )

            result = aroon.calculate(test_data)

            # Check that result length matches input length
            self.assertEqual(len(result), len(test_data))

            # Check that we have NaN values for insufficient data
            self.assertTrue(result["AROON_UP"].iloc[: period - 1].isna().any())

    def test_metadata(self):
        """Test Aroon metadata"""
        # Test with oscillator
        metadata = self.aroon.get_metadata()

        # Check required metadata fields
        required_fields = [
            "name",
            "category",
            "description",
            "parameters",
            "input_requirements",
            "output_type",
            "version",
            "author",
        ]
        for field in required_fields:
            self.assertIn(field, metadata)

        # Check specific values
        self.assertEqual(metadata["name"], "Aroon")
        self.assertEqual(metadata["category"], "trend")
        self.assertEqual(metadata["output_type"], "DataFrame")
        self.assertEqual(metadata["version"], "1.0.0")
        self.assertIn("output_columns", metadata)

        # Check output columns (with oscillator)
        expected_output_cols = ["AROON_UP", "AROON_DOWN", "AROON_OSC"]
        for col in expected_output_cols:
            self.assertIn(col, metadata["output_columns"])

        # Test without oscillator
        aroon_no_osc = AroonIndicator(include_oscillator=False)
        metadata_no_osc = aroon_no_osc.get_metadata()
        expected_no_osc = ["AROON_UP", "AROON_DOWN"]
        self.assertEqual(len(metadata_no_osc["output_columns"]), 2)
        for col in expected_no_osc:
            self.assertIn(col, metadata_no_osc["output_columns"])

    def test_backward_compatibility(self):
        """Test backward compatibility property accessors"""
        aroon = AroonIndicator(period=21, include_oscillator=False)

        # Test property access
        self.assertEqual(aroon.period, 21)
        self.assertEqual(aroon.include_oscillator, False)

    def test_export_function(self):
        """Test indicator export function for registry"""
        from aroon import get_indicator_class

        indicator_class = get_indicator_class()
        self.assertEqual(indicator_class, AroonIndicator)

        # Test that we can create an instance from the exported class
        instance = indicator_class()
        self.assertIsInstance(instance, AroonIndicator)

    def test_signal_methods(self):
        """Test signal generation methods"""
        # Test trend signal classification
        self.assertEqual(self.aroon.get_trend_signal(90, 10), "bullish")
        self.assertEqual(self.aroon.get_trend_signal(10, 90), "bearish")
        self.assertEqual(self.aroon.get_trend_signal(50, 50), "consolidating")
        self.assertEqual(self.aroon.get_trend_signal(30, 30), "neutral")

        # Test oscillator signal classification
        self.assertEqual(self.aroon.get_oscillator_signal(30), "bullish")
        self.assertEqual(self.aroon.get_oscillator_signal(-30), "bearish")
        self.assertEqual(self.aroon.get_oscillator_signal(10), "neutral")

    def test_required_columns(self):
        """Test required columns specification"""
        required = self.aroon._get_required_columns()
        expected = ["high", "low"]
        self.assertEqual(required, expected)

    def test_minimum_data_points(self):
        """Test minimum data points calculation"""
        min_points = self.aroon._get_minimum_data_points()
        expected = self.aroon.period
        self.assertEqual(min_points, expected)

    def test_mathematical_accuracy(self):
        """Test mathematical accuracy with known values"""
        # Create simple test case where we can verify the calculation
        test_data = pd.DataFrame(
            {
                "high": [10, 11, 12, 13, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4],
                "low": [8, 9, 10, 11, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2],
                "close": [9, 10, 11, 12, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3],
            }
        )

        aroon = AroonIndicator(period=5)
        result = aroon.calculate(test_data)

        # At index 4 (5th element), we should have first valid calculation
        # The highest high in first 5 elements is at index 4 (value 14)
        # The lowest low in first 5 elements is at index 0 (value 8)
        # Aroon Up = ((5 - 0) / 5) * 100 = 100 (most recent is highest)
        # Aroon Down = ((5 - 4) / 5) * 100 = 20 (lowest is 4 periods ago)
        self.assertAlmostEqual(result["AROON_UP"].iloc[4], 100.0, places=1)
        self.assertAlmostEqual(result["AROON_DOWN"].iloc[4], 20.0, places=1)

    def test_performance(self):
        """Test Aroon performance with larger dataset"""
        # Create larger dataset
        np.random.seed(42)
        n_points = 1000
        large_data = pd.DataFrame(
            {
                "high": np.random.randn(n_points) + 100.5,
                "low": np.random.randn(n_points) + 99.5,
                "close": np.random.randn(n_points) + 100,
            }
        )

        # Time the calculation
        import time

        start_time = time.time()
        result = self.aroon.calculate(large_data)
        end_time = time.time()

        # Should complete quickly (less than 1 second for 1000 points)
        self.assertLess(end_time - start_time, 1.0)

        # Result should have correct length
        self.assertEqual(len(result), len(large_data))


if __name__ == "__main__":
    unittest.main()
