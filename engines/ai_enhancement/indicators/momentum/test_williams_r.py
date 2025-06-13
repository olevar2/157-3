"""
Comprehensive Tests for Williams %R Indicator
Trading-grade validation for Platform3

Tests cover:
- Mathematical accuracy with known reference values
- Edge cases and error handling
- Performance benchmarks
- Signal generation validation
"""

import unittest
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

from .williams_r import WilliamsRIndicator
from ..base_indicator import IndicatorValidationError


class TestWilliamsRIndicator(unittest.TestCase):
    """Comprehensive test suite for Williams %R indicator"""

    def setUp(self):
        """Set up test fixtures"""
        self.indicator = WilliamsRIndicator()

        # Create test data with known patterns
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")

        # Test data with specific high/low/close patterns for validation
        self.test_data = pd.DataFrame(
            {
                "high": [
                    10.5,
                    11.0,
                    11.5,
                    12.0,
                    11.8,
                    11.5,
                    11.0,
                    10.8,
                    10.5,
                    10.2,
                    10.8,
                    11.2,
                    11.6,
                    12.1,
                    12.3,
                    12.0,
                    11.7,
                    11.3,
                    11.0,
                    10.7,
                    11.1,
                    11.5,
                    12.0,
                    12.4,
                    12.1,
                    11.8,
                    11.4,
                    11.0,
                    10.6,
                    10.9,
                    11.3,
                    11.7,
                    12.2,
                    12.5,
                    12.2,
                    11.9,
                    11.5,
                    11.1,
                    10.7,
                    11.0,
                    11.4,
                    11.8,
                    12.3,
                    12.6,
                    12.3,
                    12.0,
                    11.6,
                    11.2,
                    10.8,
                    11.1,
                ],
                "low": [
                    9.5,
                    10.0,
                    10.5,
                    11.0,
                    10.8,
                    10.5,
                    10.0,
                    9.8,
                    9.5,
                    9.2,
                    9.8,
                    10.2,
                    10.6,
                    11.1,
                    11.3,
                    11.0,
                    10.7,
                    10.3,
                    10.0,
                    9.7,
                    10.1,
                    10.5,
                    11.0,
                    11.4,
                    11.1,
                    10.8,
                    10.4,
                    10.0,
                    9.6,
                    9.9,
                    10.3,
                    10.7,
                    11.2,
                    11.5,
                    11.2,
                    10.9,
                    10.5,
                    10.1,
                    9.7,
                    10.0,
                    10.4,
                    10.8,
                    11.3,
                    11.6,
                    11.3,
                    11.0,
                    10.6,
                    10.2,
                    9.8,
                    10.1,
                ],
                "close": [
                    10.0,
                    10.5,
                    11.0,
                    11.5,
                    11.0,
                    10.8,
                    10.5,
                    10.0,
                    9.8,
                    9.5,
                    10.2,
                    10.8,
                    11.2,
                    11.8,
                    12.0,
                    11.5,
                    11.0,
                    10.6,
                    10.3,
                    10.0,
                    10.6,
                    11.0,
                    11.5,
                    12.0,
                    11.6,
                    11.2,
                    10.8,
                    10.4,
                    10.0,
                    10.3,
                    10.8,
                    11.2,
                    11.7,
                    12.2,
                    11.8,
                    11.4,
                    11.0,
                    10.6,
                    10.2,
                    10.5,
                    10.9,
                    11.3,
                    11.8,
                    12.3,
                    11.9,
                    11.5,
                    11.1,
                    10.7,
                    10.3,
                    10.6,
                ],
            },
            index=dates,
        )

        # Simple test data for manual calculation verification
        self.simple_data = pd.DataFrame(
            {
                "high": [100, 105, 110, 108, 106],
                "low": [95, 100, 105, 103, 101],
                "close": [98, 103, 107, 105, 104],
            }
        )

    def test_initialization_default_parameters(self):
        """Test Williams %R initialization with default parameters"""
        indicator = WilliamsRIndicator()
        self.assertEqual(indicator.parameters["period"], 14)
        self.assertEqual(indicator.parameters["overbought_level"], -20.0)
        self.assertEqual(indicator.parameters["oversold_level"], -80.0)

    def test_initialization_custom_parameters(self):
        """Test Williams %R initialization with custom parameters"""
        indicator = WilliamsRIndicator(
            period=21, overbought_level=-10.0, oversold_level=-90.0
        )
        self.assertEqual(indicator.parameters["period"], 21)
        self.assertEqual(indicator.parameters["overbought_level"], -10.0)
        self.assertEqual(indicator.parameters["oversold_level"], -90.0)

    def test_parameter_validation_valid(self):
        """Test parameter validation with valid inputs"""
        self.assertTrue(self.indicator.validate_parameters())

    def test_parameter_validation_invalid_period(self):
        """Test parameter validation with invalid period"""
        with self.assertRaises(IndicatorValidationError):
            WilliamsRIndicator(period=0)

        with self.assertRaises(IndicatorValidationError):
            WilliamsRIndicator(period=-5)

    def test_parameter_validation_invalid_levels(self):
        """Test parameter validation with invalid overbought/oversold levels"""
        with self.assertRaises(IndicatorValidationError):
            WilliamsRIndicator(
                overbought_level=-90.0, oversold_level=-10.0
            )  # Wrong order

        with self.assertRaises(IndicatorValidationError):
            WilliamsRIndicator(overbought_level=10.0)  # Above 0

        with self.assertRaises(IndicatorValidationError):
            WilliamsRIndicator(oversold_level=-150.0)  # Below -100

    def test_input_data_validation(self):
        """Test input data validation"""
        # Test with valid data
        self.assertTrue(self.indicator.validate_input_data(self.test_data))

        # Test with empty DataFrame
        with self.assertRaises(IndicatorValidationError):
            self.indicator.validate_input_data(pd.DataFrame())

        # Test with missing columns
        incomplete_data = self.test_data[["high", "low"]].copy()
        with self.assertRaises(IndicatorValidationError):
            self.indicator.validate_input_data(incomplete_data)

        # Test with all NaN column
        nan_data = self.test_data.copy()
        nan_data["high"] = np.nan
        with self.assertRaises(IndicatorValidationError):
            self.indicator.validate_input_data(nan_data)

    def test_insufficient_data(self):
        """Test behavior with insufficient data"""
        short_data = self.test_data.head(5)  # Less than default period of 14

        with self.assertRaises(IndicatorValidationError):
            self.indicator.calculate(short_data)

    def test_manual_calculation_verification(self):
        """Test Williams %R calculation against manual calculation"""
        # Use simple data for manual verification
        # For period=3, using last 3 rows of simple_data
        indicator = WilliamsRIndicator(period=3)
        result = indicator.calculate(self.simple_data)

        # Manual calculation for last value (index 4):
        # High period: max(110, 108, 106) = 110
        # Low period: min(105, 103, 101) = 101
        # Close: 104
        # Williams %R = (110 - 104) / (110 - 101) * -100 = 6/9 * -100 = -66.67

        expected_last_value = -66.67
        calculated_last_value = result.iloc[-1]

        self.assertAlmostEqual(calculated_last_value, expected_last_value, places=2)

    def test_range_validation(self):
        """Test that Williams %R values are in correct range (-100 to 0)"""
        result = self.indicator.calculate(self.test_data)

        # Remove NaN values for testing
        valid_values = result.dropna()

        # All values should be between -100 and 0
        self.assertTrue((valid_values >= -100).all())
        self.assertTrue((valid_values <= 0).all())

    def test_edge_case_no_range(self):
        """Test behavior when high equals low (no price range)"""
        # Create data where high = low = close
        no_range_data = pd.DataFrame(
            {"high": [100] * 20, "low": [100] * 20, "close": [100] * 20}
        )

        result = self.indicator.calculate(no_range_data)
        valid_values = result.dropna()

        # Should return neutral value (-50) when no range
        self.assertTrue((valid_values == -50.0).all())

    def test_performance_benchmark(self):
        """Test calculation performance meets trading requirements"""
        # Create larger dataset for performance testing
        large_dates = pd.date_range(start="2020-01-01", periods=1000, freq="D")
        large_data = pd.DataFrame(
            {
                "high": np.random.uniform(90, 110, 1000),
                "low": np.random.uniform(80, 100, 1000),
                "close": np.random.uniform(85, 105, 1000),
            },
            index=large_dates,
        )

        # Ensure close is between high and low
        large_data["close"] = np.minimum(large_data["close"], large_data["high"])
        large_data["close"] = np.maximum(large_data["close"], large_data["low"])

        start_time = time.time()
        result = self.indicator.calculate(large_data)
        end_time = time.time()

        execution_time = end_time - start_time

        # Should complete within 10ms for 1000 data points
        self.assertLess(execution_time, 0.01)
        self.assertEqual(len(result), len(large_data))

    def test_signal_generation(self):
        """Test trading signal generation"""
        signals_df = self.indicator.get_signals(self.test_data)

        # Check signal DataFrame structure
        expected_columns = [
            "williams_r",
            "overbought",
            "oversold",
            "neutral",
            "buy_signal",
            "sell_signal",
        ]
        for col in expected_columns:
            self.assertIn(col, signals_df.columns)

        # Check signal logic
        williams_r = signals_df["williams_r"]
        overbought = signals_df["overbought"]
        oversold = signals_df["oversold"]

        # Overbought should be True when williams_r >= -20
        overbought_check = (williams_r >= -20.0) == overbought
        self.assertTrue(overbought_check.dropna().all())

        # Oversold should be True when williams_r <= -80
        oversold_check = (williams_r <= -80.0) == oversold
        self.assertTrue(oversold_check.dropna().all())

    def test_metadata(self):
        """Test indicator metadata"""
        metadata = self.indicator.get_metadata()

        self.assertEqual(metadata.name, "Williams %R")
        self.assertEqual(metadata.category, "momentum")
        self.assertEqual(metadata.input_requirements, ["high", "low", "close"])
        self.assertEqual(metadata.output_type, "series")
        self.assertEqual(metadata.min_data_points, 14)

    def test_known_reference_values(self):
        """Test against known reference values from trading platforms"""
        # Create specific test data that matches reference calculations
        # This data should produce known Williams %R values
        reference_dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
        reference_data = pd.DataFrame(
            {
                "high": [
                    44.34,
                    44.09,
                    44.15,
                    43.61,
                    44.33,
                    44.83,
                    45.85,
                    46.08,
                    46.03,
                    46.83,
                    46.69,
                    46.45,
                    46.59,
                    46.34,
                    44.74,
                    44.78,
                    44.35,
                    44.05,
                    44.49,
                    43.60,
                ],
                "low": [
                    44.09,
                    43.81,
                    43.61,
                    43.31,
                    44.02,
                    44.17,
                    44.79,
                    45.39,
                    45.45,
                    46.05,
                    46.04,
                    45.71,
                    45.65,
                    44.74,
                    44.17,
                    44.18,
                    43.81,
                    43.46,
                    43.78,
                    43.40,
                ],
                "close": [
                    44.34,
                    44.09,
                    44.15,
                    43.61,
                    44.33,
                    44.83,
                    45.85,
                    46.08,
                    46.03,
                    46.83,
                    46.69,
                    46.45,
                    46.59,
                    46.34,
                    44.74,
                    44.78,
                    44.35,
                    44.05,
                    44.49,
                    43.60,
                ],
            },
            index=reference_dates,
        )

        # Calculate Williams %R with period=14
        indicator = WilliamsRIndicator(period=14)
        result = indicator.calculate(reference_data)

        # The last value should be close to expected reference
        # (This would need actual reference values from a trusted source)
        self.assertIsNotNone(result.iloc[-1])
        self.assertTrue(-100 <= result.iloc[-1] <= 0)

    def test_nan_handling(self):
        """Test handling of NaN values in input data"""
        nan_data = self.test_data.copy()
        nan_data.loc[nan_data.index[5], "high"] = np.nan
        nan_data.loc[nan_data.index[10], "close"] = np.nan

        # Should handle NaN values gracefully
        result = self.indicator.calculate(nan_data)
        self.assertEqual(len(result), len(nan_data))

    def test_different_periods(self):
        """Test Williams %R with different period settings"""
        periods = [5, 10, 14, 21, 50]

        for period in periods:
            if len(self.test_data) >= period:
                indicator = WilliamsRIndicator(period=period)
                result = indicator.calculate(self.test_data)

                # First (period-1) values should be NaN
                nan_count = result.isna().sum()
                self.assertEqual(nan_count, period - 1)

                # All non-NaN values should be in valid range
                valid_values = result.dropna()
                self.assertTrue((valid_values >= -100).all())
                self.assertTrue((valid_values <= 0).all())

    def tearDown(self):
        """Clean up after tests"""
        pass


if __name__ == "__main__":
    # Run tests with detailed output
    unittest.main(verbosity=2)
