"""
Test suite for Stochastic Oscillator Indicator

Tests for mathematical accuracy, parameter validation, edge cases,
and Platform3 interface compliance.

Author: Platform3 AI Framework
Created: 2025-06-09
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from stochastic import StochasticIndicator

from engines.ai_enhancement.indicators.base_indicator import IndicatorValidationError


class TestStochasticIndicator(unittest.TestCase):
    """Test suite for Stochastic Oscillator indicator"""

    def setUp(self):
        """Set up test fixtures"""
        self.stochastic = StochasticIndicator()

        # Create standard test data - need at least 16 points for default parameters (14+3-1)
        np.random.seed(42)
        self.test_data = pd.DataFrame(
            {
                "high": [
                    105,
                    106,
                    108,
                    107,
                    109,
                    110,
                    108,
                    107,
                    106,
                    108,
                    109,
                    111,
                    110,
                    109,
                    108,
                    107,
                    109,
                    110,
                ],
                "low": [
                    95,
                    96,
                    98,
                    97,
                    99,
                    100,
                    98,
                    97,
                    96,
                    98,
                    99,
                    101,
                    100,
                    99,
                    98,
                    97,
                    99,
                    100,
                ],
                "close": [
                    100,
                    101,
                    103,
                    102,
                    104,
                    105,
                    103,
                    102,
                    101,
                    103,
                    104,
                    106,
                    105,
                    104,
                    103,
                    102,
                    104,
                    105,
                ],
            }
        )

        # Create larger dataset for performance testing
        n_points = 1000
        large_prices = 100 + np.random.randn(n_points).cumsum() * 0.5
        self.large_data = pd.DataFrame(
            {
                "high": large_prices + np.abs(np.random.randn(n_points) * 2),
                "low": large_prices - np.abs(np.random.randn(n_points) * 2),
                "close": large_prices,
            }
        )

    def test_mathematical_accuracy(self):
        """Test mathematical accuracy against hand-calculated values"""
        # Use a small, controlled dataset for manual calculation verification
        test_data = pd.DataFrame(
            {
                "high": [
                    110,
                    109,
                    108,
                    107,
                    106,
                    105,
                    104,
                    103,
                    102,
                    101,
                    100,
                    99,
                    98,
                    97,
                    96,
                    95,
                ],
                "low": [
                    95,
                    96,
                    97,
                    98,
                    99,
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                ],
                "close": [
                    100,
                    105,
                    102,
                    103,
                    104,
                    102,
                    103,
                    102,
                    101,
                    102,
                    103,
                    105,
                    104,
                    103,
                    102,
                    104,
                ],
            }
        )

        result = self.stochastic.calculate(test_data)

        # Test that we get the expected columns
        self.assertIn("%K", result.columns)
        self.assertIn("%D", result.columns)

        # Test specific calculated values (hand-calculated for verification)
        # For close = 104 at index 15 (last row)
        # Looking at 14-period window (indices 2-15): highest_high=109, lowest_low=96
        # %K = ((104 - 96) / (109 - 96)) * 100 = (8 / 13) * 100 = 61.54
        actual_k_last = result["%K"].iloc[-1]
        expected_k_last = 61.54
        self.assertAlmostEqual(actual_k_last, expected_k_last, places=1)

    def test_stochastic_range(self):
        """Test that Stochastic values stay within 0-100 range"""
        result = self.stochastic.calculate(self.large_data)

        # Remove NaN values for testing
        k_values = result["%K"].dropna()
        d_values = result["%D"].dropna()

        self.assertTrue((k_values >= 0).all())
        self.assertTrue((k_values <= 100).all())
        self.assertTrue((d_values >= 0).all())
        self.assertTrue((d_values <= 100).all())

    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test invalid k_period
        with self.assertRaises(IndicatorValidationError):
            StochasticIndicator(k_period=0)

        with self.assertRaises(IndicatorValidationError):
            StochasticIndicator(k_period=-1)

        with self.assertRaises(IndicatorValidationError):
            StochasticIndicator(k_period=1001)

        # Test invalid d_period
        with self.assertRaises(IndicatorValidationError):
            StochasticIndicator(d_period=0)

        with self.assertRaises(IndicatorValidationError):
            StochasticIndicator(d_period=-1)

        # Test invalid smooth_k
        with self.assertRaises(IndicatorValidationError):
            StochasticIndicator(smooth_k=0)

        # Test invalid thresholds
        with self.assertRaises(IndicatorValidationError):
            StochasticIndicator(overbought=50, oversold=80)  # overbought < oversold

        with self.assertRaises(IndicatorValidationError):
            StochasticIndicator(oversold=-10)  # below 0

        with self.assertRaises(IndicatorValidationError):
            StochasticIndicator(overbought=110)  # above 100

    def test_input_validation(self):
        """Test input data validation"""
        # Test missing columns
        with self.assertRaises(IndicatorValidationError):
            self.stochastic.calculate(pd.DataFrame({"close": [1, 2, 3]}))

        # Test Series input (should fail for Stochastic)
        with self.assertRaises(IndicatorValidationError):
            self.stochastic.calculate(pd.Series([1, 2, 3]))

        # Test invalid data type
        with self.assertRaises(IndicatorValidationError):
            self.stochastic.calculate([1, 2, 3])

    def test_edge_cases(self):
        """Test edge cases"""
        # Test with constant prices (no volatility)
        constant_data = pd.DataFrame(
            {
                "high": [100] * 20,
                "low": [100] * 20,
                "close": [100] * 20,
            }
        )

        result = self.stochastic.calculate(constant_data)

        # When high == low, %K should be NaN (0/0 case)
        # Our implementation handles this by replacing 0 denominator with NaN
        k_values = result["%K"].dropna()
        if len(k_values) > 0:
            # If any values are calculated, they should be valid
            self.assertTrue((k_values >= 0).all())
            self.assertTrue((k_values <= 100).all())

    def test_insufficient_data(self):
        """Test with insufficient data"""
        small_data = pd.DataFrame(
            {"high": [101, 102], "low": [99, 100], "close": [100, 101]}
        )

        # Should raise validation error for insufficient data
        with self.assertRaises(IndicatorValidationError):
            self.stochastic.calculate(small_data)

    def test_smoothing_parameters(self):
        """Test different smoothing parameters"""
        # Test with smoothing
        smooth_stoch = StochasticIndicator(k_period=5, d_period=3, smooth_k=3)
        result_smooth = smooth_stoch.calculate(self.test_data)

        # Test without smoothing
        fast_stoch = StochasticIndicator(k_period=5, d_period=3, smooth_k=1)
        result_fast = fast_stoch.calculate(self.test_data)

        # Both should produce valid results
        self.assertIn("%K", result_smooth.columns)
        self.assertIn("%D", result_smooth.columns)
        self.assertIn("%K", result_fast.columns)
        self.assertIn("%D", result_fast.columns)

    def test_performance(self):
        """Test performance with large datasets"""
        import time

        start_time = time.time()
        result = self.stochastic.calculate(self.large_data)
        execution_time = time.time() - start_time

        # Should complete within reasonable time (5 seconds for 1000 points)
        self.assertLess(execution_time, 5.0)

        # Result should have expected shape
        expected_length = len(self.large_data)
        self.assertEqual(len(result), expected_length)

    def test_interface_compliance(self):
        """Test Platform3 interface compliance"""
        # Test required methods exist
        self.assertTrue(hasattr(self.stochastic, "calculate"))
        self.assertTrue(hasattr(self.stochastic, "get_signals"))

        # Test metadata
        self.assertIsNotNone(self.stochastic.CATEGORY)
        self.assertEqual(self.stochastic.CATEGORY, "momentum")
        self.assertIsNotNone(self.stochastic.VERSION)
        self.assertIsNotNone(self.stochastic.AUTHOR)

        # Test return types
        result = self.stochastic.calculate(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)

    def test_signals(self):
        """Test signal generation"""
        result = self.stochastic.calculate(self.test_data)
        signals = self.stochastic.get_signals(result)

        # Should return a dictionary with signal information
        self.assertIsInstance(signals, dict)

        # Should have standard signal keys
        self.assertIn("signal", signals)
        self.assertIn("strength", signals)

    def test_extreme_values(self):
        """Test with extreme price values"""
        # Test with very large numbers
        extreme_data = pd.DataFrame(
            {
                "high": [1e6, 1e6 + 1, 1e6 + 2] * 10,
                "low": [1e6 - 2, 1e6 - 1, 1e6] * 10,
                "close": [1e6, 1e6 + 0.5, 1e6 + 1] * 10,
            }
        )

        result = self.stochastic.calculate(extreme_data)

        # Should not crash and should produce valid ranges
        k_values = result["%K"].dropna()
        if len(k_values) > 0:
            self.assertTrue((k_values >= 0).all())
            self.assertTrue((k_values <= 100).all())

    def test_nan_values(self):
        """Test handling of NaN values in input data"""
        # Create data with some NaN values
        nan_data = self.test_data.copy()
        nan_data.loc[5, "high"] = np.nan
        nan_data.loc[10, "close"] = np.nan

        # Should handle NaN values gracefully
        result = self.stochastic.calculate(nan_data)

        # Should still produce a result (with some NaN values)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("%K", result.columns)
        self.assertIn("%D", result.columns)

    def test_infinite_values(self):
        """Test handling of infinite values"""
        inf_data = self.test_data.copy()
        inf_data.loc[0, "high"] = np.inf

        # Should handle infinite values
        try:
            result = self.stochastic.calculate(inf_data)
            # If it doesn't raise an exception, check the result is valid
            self.assertIsInstance(result, pd.DataFrame)
        except IndicatorValidationError:
            # It's also acceptable to raise a validation error for infinite values
            pass


if __name__ == "__main__":
    unittest.main()