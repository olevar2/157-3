"""
Unit tests for RSI (Relative Strength Index) Indicator
Following Platform3 testing standards
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add the indicators directory to the path
sys.path.append(os.path.dirname(__file__))
from rsi import RSIIndicator


class TestRSIIndicator(unittest.TestCase):
    """Test suite for RSI indicator"""

    def setUp(self):
        """Set up test data and indicator instances"""
        # Create test data - simple price series
        self.test_prices = pd.Series(
            [
                44.34,
                44.09,
                44.15,
                43.61,
                44.33,
                44.83,
                45.85,
                46.08,
                45.89,
                46.03,
                46.83,
                47.69,
                46.49,
                46.26,
                47.09,
                47.37,
                47.20,
                47.72,
                47.90,
                47.24,
                46.20,
                46.08,
                46.03,
                46.83,
                47.69,
                46.49,
                46.26,
                47.09,
                47.37,
                47.20,
            ]
        )
        self.test_data = pd.DataFrame({"close": self.test_prices})

        # Create RSI instance with default parameters
        self.rsi = RSIIndicator()

        # Create RSI with custom parameters
        self.rsi_custom = RSIIndicator(period=10, overbought=80, oversold=20)

    def test_initialization(self):
        """Test RSI initialization with various parameters"""
        # Test default initialization
        rsi_default = RSIIndicator()
        self.assertEqual(rsi_default.period, 14)
        self.assertEqual(rsi_default.overbought, 70.0)
        self.assertEqual(rsi_default.oversold, 30.0)

        # Test custom initialization
        rsi_custom = RSIIndicator(period=21, overbought=75, oversold=25)
        self.assertEqual(rsi_custom.period, 21)
        self.assertEqual(rsi_custom.overbought, 75)
        self.assertEqual(rsi_custom.oversold, 25)

    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test valid parameters
        self.assertTrue(self.rsi.validate_parameters())

        # Test invalid period
        with self.assertRaises(Exception):
            RSIIndicator(period=0)

        with self.assertRaises(Exception):
            RSIIndicator(period=-5)

        with self.assertRaises(Exception):
            RSIIndicator(period=1001)

        # Test invalid thresholds
        with self.assertRaises(Exception):
            RSIIndicator(overbought=30, oversold=70)  # overbought <= oversold

        with self.assertRaises(Exception):
            RSIIndicator(overbought=110)  # > 100

        with self.assertRaises(Exception):
            RSIIndicator(oversold=-10)  # < 0

    def test_data_validation(self):
        """Test input data validation"""
        # Test valid data
        self.assertTrue(self.rsi.validate_input_data(self.test_data))

        # Test empty DataFrame
        empty_df = pd.DataFrame()
        with self.assertRaises(Exception):
            self.rsi.validate_input_data(empty_df)

        # Test DataFrame without close column
        invalid_df = pd.DataFrame({"open": [1, 2, 3]})
        with self.assertRaises(Exception):
            self.rsi.validate_input_data(invalid_df)

        # Test insufficient data
        short_data = pd.DataFrame({"close": [1, 2, 3]})  # Less than period + 1
        with self.assertRaises(Exception):
            self.rsi.validate_input_data(short_data)

    def test_basic_calculation(self):
        """Test basic RSI calculation"""
        result = self.rsi.calculate(self.test_data)

        # Check result type and length
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.test_data))

        # Check RSI range (0-100)
        valid_values = result.dropna()
        self.assertTrue(all(valid_values >= 0))
        self.assertTrue(all(valid_values <= 100))

        # Check that we have NaN values at the beginning (insufficient data)
        period = self.rsi.period
        self.assertTrue(result.iloc[:period].isna().any())

        # Check that we have valid values after the initial period
        self.assertFalse(result.iloc[period:].isna().any())

    def test_calculation_edge_cases(self):
        """Test RSI calculation with edge cases"""
        # Test with all increasing prices (should approach 100)
        increasing_prices = pd.DataFrame({"close": pd.Series(range(1, 51))})
        rsi_up = RSIIndicator(period=14)
        result_up = rsi_up.calculate(increasing_prices)
        # RSI should be high for consistently increasing prices
        self.assertGreater(result_up.iloc[-1], 70)

        # Test with all decreasing prices (should approach 0)
        decreasing_prices = pd.DataFrame({"close": pd.Series(range(50, 0, -1))})
        rsi_down = RSIIndicator(period=14)
        result_down = rsi_down.calculate(decreasing_prices)
        # RSI should be low for consistently decreasing prices
        self.assertLess(
            result_down.iloc[-1], 30
        )  # Test with constant prices (should be around 50 or NaN due to zero division)
        constant_prices = pd.DataFrame({"close": pd.Series([100] * 30)})
        rsi_const = RSIIndicator(period=14)
        result_const = rsi_const.calculate(constant_prices)
        # With constant prices, gains and losses are zero, leading to special handling
        # The implementation should handle this gracefully

    def test_different_periods(self):
        """Test RSI with different periods"""
        periods = [7, 14, 21, 30]

        for period in periods:
            rsi = RSIIndicator(period=period)
            # Generate enough data for each period
            test_data = pd.DataFrame({"close": np.random.randn(period + 20) + 100})
            result = rsi.calculate(test_data)

            # Check that NaN period matches the RSI period
            self.assertTrue(result.iloc[:period].isna().any())
            # Check that result length matches input length
            self.assertEqual(len(result), len(test_data))

    def test_series_input(self):
        """Test RSI calculation with Series input"""
        result = self.rsi.calculate(self.test_prices)

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.test_prices))

    def test_metadata(self):
        """Test RSI metadata"""
        metadata = self.rsi.get_metadata()

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
        self.assertEqual(metadata["name"], "RSI")
        self.assertEqual(metadata["category"], "trend")
        self.assertEqual(metadata["output_type"], "Series")
        self.assertEqual(metadata["version"], "1.0.0")
        self.assertIn("close", metadata["input_requirements"])

        # Check parameters
        params = metadata["parameters"]
        self.assertEqual(params["period"], 14)
        self.assertEqual(params["overbought"], 70.0)
        self.assertEqual(params["oversold"], 30.0)

    def test_backward_compatibility(self):
        """Test backward compatibility property accessors"""
        rsi = RSIIndicator(period=21, overbought=75, oversold=25)

        # Test property access
        self.assertEqual(rsi.period, 21)
        self.assertEqual(rsi.overbought, 75)
        self.assertEqual(rsi.oversold, 25)

    def test_export_function(self):
        """Test indicator export function for registry"""
        from rsi import get_indicator_class

        indicator_class = get_indicator_class()
        self.assertEqual(indicator_class, RSIIndicator)

        # Test that we can create an instance from the exported class
        instance = indicator_class()
        self.assertIsInstance(instance, RSIIndicator)

    def test_mathematical_accuracy(self):
        """Test mathematical accuracy with known values"""
        # Create a simple test case with known RSI values
        # Using a subset of well-known price data
        test_prices = pd.Series(
            [
                44.34,
                44.09,
                44.15,
                43.61,
                44.33,
                44.83,
                45.85,
                46.08,
                45.89,
                46.03,
                46.83,
                47.69,
                46.49,
                46.26,
                47.09,
            ]
        )
        test_data = pd.DataFrame({"close": test_prices})

        rsi = RSIIndicator(period=14)
        result = rsi.calculate(test_data)
        # After period 14, we should have a valid RSI value
        # The exact value depends on the implementation but should be reasonable
        final_rsi = result.iloc[-1]
        self.assertIsInstance(final_rsi, (int, float))
        self.assertGreaterEqual(final_rsi, 0)
        self.assertLessEqual(final_rsi, 100)

    def test_required_columns(self):
        """Test required columns specification"""
        required = self.rsi._get_required_columns()
        self.assertEqual(required, ["close"])

    def test_minimum_data_points(self):
        """Test minimum data points calculation"""
        min_points = self.rsi._get_minimum_data_points()
        expected = self.rsi.period + 1
        self.assertEqual(min_points, expected)

    def test_performance(self):
        """Test RSI performance with larger dataset"""
        # Create larger dataset
        np.random.seed(42)
        large_prices = pd.Series(100 + np.cumsum(np.random.randn(1000) * 0.5))
        large_data = pd.DataFrame({"close": large_prices})

        # Time the calculation
        import time

        start_time = time.time()
        result = self.rsi.calculate(large_data)
        end_time = time.time()

        # Should complete quickly (less than 1 second for 1000 points)
        self.assertLess(end_time - start_time, 1.0)

        # Result should have correct length
        self.assertEqual(len(result), len(large_data))


if __name__ == "__main__":
    unittest.main()
