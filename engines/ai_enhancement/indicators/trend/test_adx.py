"""
Unit tests for ADX (Average Directional Index) Indicator
Following Platform3 testing standards
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add the indicators directory to the path
sys.path.append(os.path.dirname(__file__))
from adx import ADXIndicator


class TestADXIndicator(unittest.TestCase):
    """Test suite for ADX indicator"""

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

        # Create ADX instance with default parameters
        self.adx = ADXIndicator()

        # Create ADX with custom parameters
        self.adx_custom = ADXIndicator(period=10, smoothing_period=10)

    def test_initialization(self):
        """Test ADX initialization with various parameters"""
        # Test default initialization
        adx_default = ADXIndicator()
        self.assertEqual(adx_default.period, 14)
        self.assertEqual(adx_default.smoothing_period, 14)

        # Test custom initialization
        adx_custom = ADXIndicator(period=21, smoothing_period=10)
        self.assertEqual(adx_custom.period, 21)
        self.assertEqual(adx_custom.smoothing_period, 10)

    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test valid parameters
        self.assertTrue(self.adx.validate_parameters())

        # Test invalid period
        with self.assertRaises(Exception):
            ADXIndicator(period=0)

        with self.assertRaises(Exception):
            ADXIndicator(period=-5)

        with self.assertRaises(Exception):
            ADXIndicator(period=1001)

        # Test invalid smoothing period
        with self.assertRaises(Exception):
            ADXIndicator(smoothing_period=0)

        with self.assertRaises(Exception):
            ADXIndicator(smoothing_period=1001)

    def test_data_validation(self):
        """Test input data validation"""
        # Test valid data
        self.assertTrue(self.adx.validate_input_data(self.test_data))

        # Test empty DataFrame
        empty_df = pd.DataFrame()
        with self.assertRaises(Exception):
            self.adx.validate_input_data(empty_df)

        # Test DataFrame without required columns
        invalid_df = pd.DataFrame({"open": [1, 2, 3]})
        with self.assertRaises(Exception):
            self.adx.calculate(invalid_df)

        # Test Series input (should fail)
        test_series = pd.Series([1, 2, 3, 4, 5])
        with self.assertRaises(Exception):
            self.adx.calculate(test_series)

        # Test insufficient data
        short_data = pd.DataFrame(
            {"high": [1, 2, 3], "low": [0.8, 1.8, 2.8], "close": [0.9, 1.9, 2.9]}
        )
        with self.assertRaises(Exception):
            self.adx.validate_input_data(short_data)

    def test_basic_calculation(self):
        """Test basic ADX calculation"""
        result = self.adx.calculate(self.test_data)

        # Check result type and structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.test_data))

        # Check required columns
        expected_columns = ["ADX", "PLUS_DI", "MINUS_DI"]
        for col in expected_columns:
            self.assertIn(col, result.columns)

        # Check ADX range (0-100)
        valid_adx = result["ADX"].dropna()
        self.assertTrue(all(valid_adx >= 0))
        self.assertTrue(all(valid_adx <= 100))

        # Check DI ranges (0-100)
        valid_plus_di = result["PLUS_DI"].dropna()
        valid_minus_di = result["MINUS_DI"].dropna()
        self.assertTrue(all(valid_plus_di >= 0))
        self.assertTrue(all(valid_minus_di >= 0))

        # Check that we have NaN values at the beginning (insufficient data)
        min_points = self.adx._get_minimum_data_points()
        self.assertTrue(result["ADX"].iloc[: min_points - 1].isna().any())

    def test_wilders_smoothing(self):
        """Test Wilder's smoothing function"""
        test_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        period = 5

        smoothed = self.adx._wilders_smoothing(test_series, period)

        # Check that we have NaN values for insufficient data
        self.assertTrue(smoothed.iloc[: period - 1].isna().all())

        # Check that first smoothed value is simple average
        first_smoothed = smoothed.iloc[period - 1]
        expected_first = test_series.iloc[:period].mean()
        self.assertAlmostEqual(first_smoothed, expected_first, places=6)

    def test_calculation_edge_cases(self):
        """Test ADX calculation with edge cases"""
        # Test with trending data (should have higher ADX)
        trending_data = pd.DataFrame(
            {
                "high": pd.Series(range(1, 51)) + 0.1,
                "low": pd.Series(range(1, 51)) - 0.1,
                "close": pd.Series(range(1, 51)),
            }
        )
        adx_trend = ADXIndicator(period=14, smoothing_period=14)
        result_trend = adx_trend.calculate(trending_data)

        # Trending data should eventually have higher ADX
        final_adx = result_trend["ADX"].iloc[-1]
        self.assertGreater(
            final_adx, 20
        )  # Should indicate some trend strength        # Test with sideways data (should have lower ADX)
        sideways_data = pd.DataFrame(
            {"high": [100.1] * 40, "low": [99.9] * 40, "close": [100] * 40}
        )
        adx_sideways = ADXIndicator(period=14, smoothing_period=14)
        result_sideways = adx_sideways.calculate(sideways_data)

        # Sideways data should have low ADX or NaN (due to no movement)
        final_adx_sideways = result_sideways["ADX"].iloc[-1]
        # With constant prices, ADX might be NaN or very low
        if not pd.isna(final_adx_sideways):
            self.assertLess(final_adx_sideways, 30)  # Should indicate weak trend

    def test_different_periods(self):
        """Test ADX with different periods"""
        periods = [7, 14, 21]
        smoothing_periods = [7, 14, 21]

        for period in periods:
            for smooth_period in smoothing_periods:
                adx = ADXIndicator(period=period, smoothing_period=smooth_period)
                # Generate enough data for each period combination
                min_points = period + smooth_period + 10
                test_data = pd.DataFrame(
                    {
                        "high": np.random.randn(min_points) + 100.5,
                        "low": np.random.randn(min_points) + 99.5,
                        "close": np.random.randn(min_points) + 100,
                    }
                )

                result = adx.calculate(test_data)

                # Check that result length matches input length
                self.assertEqual(len(result), len(test_data))

                # Check that we have valid data after sufficient periods
                expected_nan_period = period + smooth_period - 1
                self.assertTrue(result["ADX"].iloc[:expected_nan_period].isna().any())

    def test_metadata(self):
        """Test ADX metadata"""
        metadata = self.adx.get_metadata()

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
        self.assertEqual(metadata["name"], "ADX")
        self.assertEqual(metadata["category"], "trend")
        self.assertEqual(metadata["output_type"], "DataFrame")
        self.assertEqual(metadata["version"], "1.0.0")
        self.assertIn("output_columns", metadata)

        # Check required columns
        required_cols = ["high", "low", "close"]
        for col in required_cols:
            self.assertIn(col, metadata["input_requirements"])

        # Check output columns
        expected_output_cols = ["ADX", "PLUS_DI", "MINUS_DI"]
        for col in expected_output_cols:
            self.assertIn(col, metadata["output_columns"])

        # Check parameters
        params = metadata["parameters"]
        self.assertEqual(params["period"], 14)
        self.assertEqual(params["smoothing_period"], 14)

    def test_backward_compatibility(self):
        """Test backward compatibility property accessors"""
        adx = ADXIndicator(period=21, smoothing_period=10)

        # Test property access
        self.assertEqual(adx.period, 21)
        self.assertEqual(adx.smoothing_period, 10)

    def test_export_function(self):
        """Test indicator export function for registry"""
        from adx import get_indicator_class

        indicator_class = get_indicator_class()
        self.assertEqual(indicator_class, ADXIndicator)

        # Test that we can create an instance from the exported class
        instance = indicator_class()
        self.assertIsInstance(instance, ADXIndicator)

    def test_trend_classification(self):
        """Test trend strength and direction classification"""
        # Test trend strength classification
        self.assertEqual(self.adx.get_trend_strength(10), "weak")
        self.assertEqual(self.adx.get_trend_strength(30), "moderate")
        self.assertEqual(self.adx.get_trend_strength(50), "strong")
        self.assertEqual(self.adx.get_trend_strength(70), "very_strong")

        # Test trend direction classification
        self.assertEqual(self.adx.get_trend_direction(30, 20), "bullish")
        self.assertEqual(self.adx.get_trend_direction(20, 30), "bearish")
        self.assertEqual(self.adx.get_trend_direction(25, 25), "neutral")

    def test_required_columns(self):
        """Test required columns specification"""
        required = self.adx._get_required_columns()
        expected = ["high", "low", "close"]
        self.assertEqual(required, expected)

    def test_minimum_data_points(self):
        """Test minimum data points calculation"""
        min_points = self.adx._get_minimum_data_points()
        expected = self.adx.period + self.adx.smoothing_period + 1
        self.assertEqual(min_points, expected)

    def test_performance(self):
        """Test ADX performance with larger dataset"""
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
        result = self.adx.calculate(large_data)
        end_time = time.time()

        # Should complete quickly (less than 1 second for 1000 points)
        self.assertLess(end_time - start_time, 1.0)

        # Result should have correct length
        self.assertEqual(len(result), len(large_data))


if __name__ == "__main__":
    unittest.main()
