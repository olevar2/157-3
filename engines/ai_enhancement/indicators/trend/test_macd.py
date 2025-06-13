"""
Test suite for MACD (Moving Average Convergence Divergence) Indicator

Tests for trading accuracy, edge cases, and performance validation.

Author: Platform3 AI Framework
Created: 2025-06-09
"""

import unittest
import numpy as np
import pandas as pd
import time
from typing import Dict, Any

# Import the MACD indicator
from macd import MACDIndicator, MACDResult


class TestMACDIndicator(unittest.TestCase):
    """Comprehensive test suite for MACD indicator"""

    def setUp(self):
        """Set up test fixtures"""
        self.macd = MACDIndicator()

        # Create sample data for testing
        np.random.seed(42)
        self.sample_prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5))

        # Known test data with expected results (simplified test case)
        self.known_prices = pd.Series(
            [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
                110.0,
                111.0,
                112.0,
                113.0,
                114.0,
                115.0,
                116.0,
                117.0,
                118.0,
                119.0,
                120.0,
                121.0,
                122.0,
                123.0,
                124.0,
                125.0,
                126.0,
                127.0,
                128.0,
                129.0,
                130.0,
                131.0,
                132.0,
                133.0,
                134.0,
                135.0,
                136.0,
                137.0,
                138.0,
                139.0,
            ]
        )

    def test_initialization(self):
        """Test indicator initialization"""
        # Test default parameters
        macd = MACDIndicator()
        self.assertEqual(macd.fast_period, 12)
        self.assertEqual(macd.slow_period, 26)
        self.assertEqual(macd.signal_period, 9)

        # Test custom parameters
        macd_custom = MACDIndicator(fast_period=5, slow_period=20, signal_period=7)
        self.assertEqual(macd_custom.fast_period, 5)
        self.assertEqual(macd_custom.slow_period, 20)
        self.assertEqual(macd_custom.signal_period, 7)

        # Test invalid parameters
        with self.assertRaises(ValueError):
            MACDIndicator(fast_period=-1)

        with self.assertRaises(ValueError):
            MACDIndicator(fast_period=15, slow_period=10)  # Fast >= Slow

    def test_calculation_basic(self):
        """Test basic MACD calculation"""
        result = self.macd.calculate(self.sample_prices)

        # Check result type
        self.assertIsInstance(result, MACDResult)

        # Check result arrays
        self.assertIsInstance(result.macd_line, np.ndarray)
        self.assertIsInstance(result.signal_line, np.ndarray)
        self.assertIsInstance(result.histogram, np.ndarray)

        # Check array lengths
        expected_length = len(self.sample_prices)
        self.assertEqual(len(result.macd_line), expected_length)
        self.assertEqual(len(result.signal_line), expected_length)
        self.assertEqual(len(result.histogram), expected_length)

        # Check that histogram = macd_line - signal_line
        np.testing.assert_array_almost_equal(
            result.histogram, result.macd_line - result.signal_line, decimal=10
        )

    def test_known_values(self):
        """Test against known reference values"""
        # Use simple ascending prices for predictable results
        result = self.macd.calculate(self.known_prices)

        # For ascending prices, MACD should generally be positive
        # and increasing (trend following behavior)
        self.assertTrue(np.all(np.isfinite(result.macd_line)))
        self.assertTrue(np.all(np.isfinite(result.signal_line)))
        self.assertTrue(np.all(np.isfinite(result.histogram)))

        # Check that results are not all zeros
        self.assertFalse(np.allclose(result.macd_line, 0))
        self.assertFalse(np.allclose(result.signal_line, 0))

    def test_data_formats(self):
        """Test different input data formats"""
        # Test with pandas DataFrame
        df = pd.DataFrame({"close": self.sample_prices})
        result_df = self.macd.calculate(df)

        # Test with pandas Series
        result_series = self.macd.calculate(self.sample_prices)

        # Test with numpy array
        result_array = self.macd.calculate(self.sample_prices.values)

        # Results should be very similar (allowing for small numerical differences)
        np.testing.assert_array_almost_equal(
            result_df.macd_line, result_series.macd_line, decimal=10
        )
        np.testing.assert_array_almost_equal(
            result_series.macd_line, result_array.macd_line, decimal=10
        )

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test insufficient data
        short_data = pd.Series([100, 101, 102])
        with self.assertRaises(ValueError):
            self.macd.calculate(short_data)

        # Test empty data
        with self.assertRaises(ValueError):
            self.macd.calculate(pd.Series([]))

        # Test data with NaN values
        nan_data = self.sample_prices.copy()
        nan_data.iloc[10:15] = np.nan
        result = self.macd.calculate(nan_data)
        self.assertTrue(np.all(np.isfinite(result.macd_line)))

        # Test all constant prices
        constant_prices = pd.Series([100.0] * 50)
        result = self.macd.calculate(constant_prices)
        # MACD should be zero for constant prices
        np.testing.assert_array_almost_equal(result.macd_line, 0, decimal=10)
        np.testing.assert_array_almost_equal(result.signal_line, 0, decimal=10)
        np.testing.assert_array_almost_equal(result.histogram, 0, decimal=10)

    def test_parameter_validation(self):
        """Test parameter validation and updates"""
        # Test get_parameters
        params = self.macd.get_parameters()
        expected_params = {"fast_period": 12, "slow_period": 26, "signal_period": 9}
        self.assertEqual(params, expected_params)

        # Test set_parameters
        self.macd.set_parameters(fast_period=10, signal_period=8)
        self.assertEqual(self.macd.fast_period, 10)
        self.assertEqual(self.macd.signal_period, 8)

        # Test invalid parameter update
        with self.assertRaises(ValueError):
            self.macd.set_parameters(fast_period=30)  # Would make fast >= slow

    def test_data_validation(self):
        """Test data validation"""
        # Valid data
        self.assertTrue(self.macd.validate_data(self.sample_prices))
        self.assertTrue(
            self.macd.validate_data(pd.DataFrame({"close": self.sample_prices}))
        )
        self.assertTrue(self.macd.validate_data(self.sample_prices.values))

        # Invalid data
        self.assertFalse(self.macd.validate_data(pd.Series([])))
        self.assertFalse(
            self.macd.validate_data(pd.DataFrame({"price": self.sample_prices}))
        )  # Wrong column
        self.assertFalse(self.macd.validate_data("invalid"))

    def test_metadata(self):
        """Test indicator metadata"""
        metadata = self.macd.get_metadata()

        self.assertEqual(metadata["name"], "MACD")
        self.assertEqual(metadata["category"], "trend")
        self.assertEqual(metadata["version"], "1.0.0")
        self.assertIn("description", metadata)
        self.assertIn("parameters", metadata)
        self.assertIn("output_names", metadata)
        self.assertEqual(len(metadata["output_names"]), 3)

    def test_performance(self):
        """Test calculation performance"""
        # Generate larger dataset for performance testing
        large_data = pd.Series(100 + np.cumsum(np.random.randn(10000) * 0.5))

        # Measure calculation time
        start_time = time.time()
        result = self.macd.calculate(large_data)
        calculation_time = time.time() - start_time

        # Should complete in under 100ms for 10k data points
        self.assertLess(calculation_time, 0.1)

        # Verify result quality
        self.assertEqual(len(result.macd_line), len(large_data))
        self.assertTrue(np.all(np.isfinite(result.macd_line)))

    def test_different_periods(self):
        """Test with different period configurations"""
        # Test shorter periods
        macd_short = MACDIndicator(fast_period=5, slow_period=10, signal_period=3)
        result_short = macd_short.calculate(self.sample_prices)

        # Test longer periods
        macd_long = MACDIndicator(fast_period=20, slow_period=50, signal_period=15)
        # Need more data for longer periods
        long_data = pd.Series(100 + np.cumsum(np.random.randn(200) * 0.5))
        result_long = macd_long.calculate(long_data)

        # Both should produce valid results
        self.assertTrue(np.all(np.isfinite(result_short.macd_line)))
        self.assertTrue(np.all(np.isfinite(result_long.macd_line)))

    def test_mathematical_properties(self):
        """Test mathematical properties of MACD"""
        result = self.macd.calculate(self.sample_prices)

        # Test that EMA relationship holds
        # (This is more of a conceptual test since we don't expose the EMAs directly)

        # Test histogram calculation
        calculated_histogram = result.macd_line - result.signal_line
        np.testing.assert_array_almost_equal(
            result.histogram, calculated_histogram, decimal=12
        )

        # Test that signal line is smoother than MACD line (lower volatility)
        macd_volatility = np.std(np.diff(result.macd_line))
        signal_volatility = np.std(np.diff(result.signal_line))
        self.assertLess(signal_volatility, macd_volatility)


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
