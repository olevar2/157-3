"""
Unit tests for CCI (Commodity Channel Index) Indicator
Following Platform3 testing standards
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add the indicators directory to the path
from cci import CCIIndicator


class TestCCIIndicator(unittest.TestCase):
    """Test suite for CCI indicator"""

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

        # Create CCI instance with default parameters
        self.cci = CCIIndicator()

        # Create CCI with custom parameters
        self.cci_custom = CCIIndicator(
            period=10, constant=0.02, overbought=120, oversold=-120
        )

    def test_initialization(self):
        """Test CCI initialization with various parameters"""
        # Test default initialization
        cci_default = CCIIndicator()
        self.assertEqual(cci_default.period, 20)
        self.assertEqual(cci_default.constant, 0.015)
        self.assertEqual(cci_default.overbought, 100.0)
        self.assertEqual(cci_default.oversold, -100.0)

        # Test custom initialization
        cci_custom = CCIIndicator(
            period=14, constant=0.02, overbought=120, oversold=-120
        )
        self.assertEqual(cci_custom.period, 14)
        self.assertEqual(cci_custom.constant, 0.02)
        self.assertEqual(cci_custom.overbought, 120)
        self.assertEqual(cci_custom.oversold, -120)

    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test valid parameters
        self.assertTrue(self.cci.validate_parameters())

        # Test invalid period
        with self.assertRaises(Exception):
            CCIIndicator(period=0)

        with self.assertRaises(Exception):
            CCIIndicator(period=-5)

        with self.assertRaises(Exception):
            CCIIndicator(period=1001)

        # Test invalid constant
        with self.assertRaises(Exception):
            CCIIndicator(constant=0)

        with self.assertRaises(Exception):
            CCIIndicator(constant=-0.01)

        # Test invalid thresholds
        with self.assertRaises(Exception):
            CCIIndicator(overbought=-50, oversold=50)  # overbought <= oversold

    def test_data_validation(self):
        """Test input data validation"""
        # Test valid DataFrame data
        self.assertTrue(self.cci.validate_input_data(self.test_data))

        # Test empty DataFrame
        empty_df = pd.DataFrame()
        with self.assertRaises(Exception):
            self.cci.calculate(empty_df)

        # Test DataFrame without required columns (should work with close)
        close_only_df = pd.DataFrame({"close": [1, 2, 3, 4, 5] * 5})
        result = self.cci.calculate(close_only_df)
        self.assertIsInstance(result, pd.Series)

        # Test insufficient data
        short_data = pd.DataFrame(
            {"high": [1, 2, 3], "low": [0.8, 1.8, 2.8], "close": [0.9, 1.9, 2.9]}
        )
        with self.assertRaises(Exception):
            self.cci.validate_input_data(short_data)

    def test_basic_calculation_dataframe(self):
        """Test basic CCI calculation with DataFrame"""
        result = self.cci.calculate(self.test_data)

        # Check result type and length
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.test_data))
        self.assertEqual(result.name, "CCI")

        # Check that we have NaN values at the beginning (insufficient data)
        period = self.cci.period
        self.assertTrue(result.iloc[: period - 1].isna().any())

        # Check that we have valid values after the initial period
        valid_values = result.dropna()
        self.assertGreater(len(valid_values), 0)

        # CCI values can range widely, but should be finite
        self.assertTrue(np.isfinite(valid_values).all())

    def test_basic_calculation_series(self):
        """Test basic CCI calculation with Series"""
        result = self.cci.calculate(self.test_series)

        # Check result type and length
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.test_series))

    def test_calculation_edge_cases(self):
        """Test CCI calculation with edge cases"""
        # Test with trending data (should have directional CCI values)
        trending_data = pd.DataFrame(
            {
                "high": pd.Series(range(1, 31)) + 0.1,
                "low": pd.Series(range(1, 31)) - 0.1,
                "close": pd.Series(range(1, 31)),
            }
        )
        cci_trend = CCIIndicator(period=10)
        result_trend = cci_trend.calculate(trending_data)

        # Trending data should eventually have positive CCI (above average)
        final_cci = result_trend.iloc[-1]
        self.assertGreater(final_cci, 0)

        # Test with constant prices (should have CCI near 0 or NaN)
        constant_data = pd.DataFrame(
            {"high": [100.1] * 30, "low": [99.9] * 30, "close": [100] * 30}
        )
        cci_const = CCIIndicator(period=10)
        result_const = cci_const.calculate(constant_data)

        # With constant prices, CCI should be very small or NaN (due to zero deviation)
        final_cci_const = result_const.iloc[-1]
        if not pd.isna(final_cci_const):
            self.assertLess(abs(final_cci_const), 1.0)

    def test_different_periods(self):
        """Test CCI with different periods"""
        periods = [10, 20, 30]

        for period in periods:
            cci = CCIIndicator(period=period)
            # Generate enough data for each period
            test_data = pd.DataFrame(
                {
                    "high": np.random.randn(period + 10) + 100.5,
                    "low": np.random.randn(period + 10) + 99.5,
                    "close": np.random.randn(period + 10) + 100,
                }
            )

            result = cci.calculate(test_data)

            # Check that result length matches input length
            self.assertEqual(len(result), len(test_data))

            # Check that we have NaN values for insufficient data
            self.assertTrue(result.iloc[: period - 1].isna().any())

    def test_different_constants(self):
        """Test CCI with different scaling constants"""
        constants = [0.01, 0.015, 0.02, 0.05]

        base_cci = CCIIndicator(period=20, constant=0.015)
        base_result = base_cci.calculate(self.test_data)

        for constant in constants:
            if constant == 0.015:
                continue  # Skip base case

            cci = CCIIndicator(period=20, constant=constant)
            result = cci.calculate(self.test_data)

            # Results should be inversely proportional to the constant
            # (smaller constant = larger CCI values)
            if constant < 0.015:
                # Should have larger absolute values
                self.assertGreater(result.dropna().std(), base_result.dropna().std())
            else:
                # Should have smaller absolute values
                self.assertLess(result.dropna().std(), base_result.dropna().std())

    def test_metadata(self):
        """Test CCI metadata"""
        metadata = self.cci.get_metadata()

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
        self.assertEqual(metadata["name"], "CCI")
        self.assertEqual(metadata["category"], "trend")
        self.assertEqual(metadata["output_type"], "Series")
        self.assertEqual(metadata["version"], "1.0.0")

        # Check required columns
        required_cols = ["high", "low", "close"]
        for col in required_cols:
            self.assertIn(col, metadata["input_requirements"])

        # Check parameters
        params = metadata["parameters"]
        self.assertEqual(params["period"], 20)
        self.assertEqual(params["constant"], 0.015)
        self.assertEqual(params["overbought"], 100.0)
        self.assertEqual(params["oversold"], -100.0)

    def test_backward_compatibility(self):
        """Test backward compatibility property accessors"""
        cci = CCIIndicator(period=14, constant=0.02, overbought=120, oversold=-120)

        # Test property access
        self.assertEqual(cci.period, 14)
        self.assertEqual(cci.constant, 0.02)
        self.assertEqual(cci.overbought, 120)
        self.assertEqual(cci.oversold, -120)

    def test_export_function(self):
        """Test indicator export function for registry"""
        from cci import get_indicator_class

        indicator_class = get_indicator_class()
        self.assertEqual(indicator_class, CCIIndicator)

        # Test that we can create an instance from the exported class
        instance = indicator_class()
        self.assertIsInstance(instance, CCIIndicator)

    def test_signal_methods(self):
        """Test signal generation methods"""
        # Test basic signal classification
        self.assertEqual(self.cci.get_signal(150), "overbought")
        self.assertEqual(self.cci.get_signal(-150), "oversold")
        self.assertEqual(self.cci.get_signal(50), "bullish")
        self.assertEqual(self.cci.get_signal(-50), "bearish")
        self.assertEqual(self.cci.get_signal(0), "neutral")

        # Test extreme signal classification
        self.assertEqual(self.cci.get_extreme_signal(250), "extremely_overbought")
        self.assertEqual(self.cci.get_extreme_signal(-250), "extremely_oversold")
        self.assertEqual(self.cci.get_extreme_signal(50), "normal")

    def test_required_columns(self):
        """Test required columns specification"""
        required = self.cci._get_required_columns()
        expected = ["high", "low", "close"]
        self.assertEqual(required, expected)

    def test_minimum_data_points(self):
        """Test minimum data points calculation"""
        min_points = self.cci._get_minimum_data_points()
        expected = self.cci.period
        self.assertEqual(min_points, expected)

    def test_mathematical_accuracy(self):
        """Test mathematical accuracy with known values"""
        # Create simple test case where we can verify the calculation
        test_data = pd.DataFrame(
            {
                "high": [11, 12, 13, 14, 15],
                "low": [9, 10, 11, 12, 13],
                "close": [10, 11, 12, 13, 14],
            }
        )

        cci = CCIIndicator(period=5, constant=0.015)
        result = cci.calculate(test_data)

        # Manual calculation for verification
        # Typical Price = (H + L + C) / 3
        tp = (test_data["high"] + test_data["low"] + test_data["close"]) / 3
        # tp = [10, 11, 12, 13, 14]

        # SMA(TP) over 5 periods = mean([10, 11, 12, 13, 14]) = 12
        sma_tp = tp.mean()

        # Mean Deviation = mean(|TP - SMA(TP)|) = mean([2, 1, 0, 1, 2]) = 1.2
        mean_dev = abs(tp - sma_tp).mean()

        # CCI for last value = (14 - 12) / (0.015 * 1.2) = 2 / 0.018 ≈ 111.11
        expected_cci = (14 - sma_tp) / (0.015 * mean_dev)

        # Check the last CCI value
        actual_cci = result.iloc[-1]
        self.assertAlmostEqual(actual_cci, expected_cci, places=2)

    def test_performance(self):
        """Test CCI performance with larger dataset"""
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
        result = self.cci.calculate(large_data)
        end_time = time.time()

        # Should complete quickly (less than 1 second for 1000 points)
        self.assertLess(end_time - start_time, 1.0)

        # Result should have correct length
        self.assertEqual(len(result), len(large_data))

    def test_cci_distribution(self):
        """Test that CCI values roughly follow expected distribution"""
        # Generate larger dataset for statistical testing
        np.random.seed(42)
        n_points = 500
        data = pd.DataFrame(
            {
                "high": np.random.randn(n_points) + 100.5,
                "low": np.random.randn(n_points) + 99.5,
                "close": np.random.randn(n_points) + 100,
            }
        )

        cci = CCIIndicator(period=20)
        result = cci.calculate(data)
        valid_cci = result.dropna()

        # With the standard constant of 0.015, approximately 70-80% of values
        # should fall between -100 and +100
        in_normal_range = ((valid_cci >= -100) & (valid_cci <= 100)).mean()

        # Should be roughly between 60% and 90% (allowing for randomness)
        self.assertGreater(in_normal_range, 0.5)
        self.assertLess(in_normal_range, 0.95)


if __name__ == "__main__":
    unittest.main()
