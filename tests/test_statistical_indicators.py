#!/usr/bin/env python3
"""
Unit tests for statistical indicators
"""

import os
import sys
import unittest

import numpy as np

# Add the project root to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from engines.ai_enhancement.statistical_indicators import (
        AutocorrelationIndicator,
        BetaCoefficientIndicator,
        CointegrationIndicator,
        CorrelationCoefficientIndicator,
        LinearRegressionIndicator,
        RSquaredIndicator,
        SkewnessIndicator,
        StandardDeviationIndicator,
        VarianceRatioIndicator,
        ZScoreIndicator,
    )
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class TestStatisticalIndicators(unittest.TestCase):
    """Test suite for all statistical indicators"""

    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.trend_data = np.linspace(100, 120, 50) + np.random.normal(0, 2, 50)
        self.noisy_data = 100 + np.random.normal(0, 5, 50)
        self.cyclical_data = (
            100
            + 10 * np.sin(np.linspace(0, 4 * np.pi, 50))
            + np.random.normal(0, 1, 50)
        )
        self.market_data = np.linspace(1000, 1100, 50) + np.random.normal(0, 10, 50)

        # Create cointegrated series
        self.series1 = np.cumsum(np.random.normal(0, 1, 50)) + 100
        self.series2 = 0.8 * self.series1 + 20 + np.random.normal(0, 2, 50)

    def test_standard_deviation_indicator(self):
        """Test StandardDeviationIndicator"""
        indicator = StandardDeviationIndicator(period=10)
        result = indicator.calculate(self.trend_data)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(len(result) > 0)
        self.assertTrue(all(val >= 0 for val in result))

    def test_correlation_coefficient_indicator(self):
        """Test CorrelationCoefficientIndicator"""
        indicator = CorrelationCoefficientIndicator(period=10)
        result = indicator.calculate(self.trend_data, self.cyclical_data)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(len(result) > 0)
        self.assertTrue(all(-1 <= val <= 1 for val in result))

    def test_linear_regression_indicator(self):
        """Test LinearRegressionIndicator"""
        indicator = LinearRegressionIndicator(period=10)
        result = indicator.calculate(self.trend_data)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

        # Check that each result has required keys
        for r in result:
            self.assertIsInstance(r, dict)
            self.assertIn("r_squared", r)
            self.assertIn("slope", r)
            self.assertIn("trend", r)

    def test_z_score_indicator(self):
        """Test ZScoreIndicator"""
        indicator = ZScoreIndicator(period=10)
        result = indicator.calculate(self.noisy_data)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

        # Check that each result has required keys
        for r in result:
            self.assertIsInstance(r, dict)
            self.assertIn("z_score", r)
            self.assertIn("interpretation", r)

    def test_autocorrelation_indicator(self):
        """Test AutocorrelationIndicator"""
        indicator = AutocorrelationIndicator(period=20, max_lag=5)
        result = indicator.calculate(self.cyclical_data)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_beta_coefficient_indicator(self):
        """Test BetaCoefficientIndicator"""
        indicator = BetaCoefficientIndicator(period=10)
        result = indicator.calculate(self.trend_data, self.market_data)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

        # Check that each result has required keys
        for r in result:
            self.assertIsInstance(r, dict)
            self.assertIn("beta", r)
            self.assertIn("alpha", r)

    def test_skewness_indicator(self):
        """Test SkewnessIndicator"""
        indicator = SkewnessIndicator(period=10)
        result = indicator.calculate(self.noisy_data)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

        # Check that each result has required keys
        for r in result:
            self.assertIsInstance(r, dict)
            self.assertIn("skewness", r)
            self.assertIn("interpretation", r)

    def test_variance_ratio_indicator(self):
        """Test VarianceRatioIndicator"""
        indicator = VarianceRatioIndicator(period=10)
        result = indicator.calculate(self.trend_data)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_cointegration_indicator(self):
        """Test CointegrationIndicator"""
        indicator = CointegrationIndicator(period=30)
        result = indicator.calculate(self.series1, self.series2)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

        # Check that each result has required keys
        for r in result:
            self.assertIsInstance(r, dict)
            self.assertIn("cointegrated", r)
            self.assertIn("correlation", r)

    def test_r_squared_indicator(self):
        """Test RSquaredIndicator"""
        indicator = RSquaredIndicator(period=10)
        result = indicator.calculate(self.trend_data)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

        # Check that each result has required keys
        for r in result:
            self.assertIsInstance(r, dict)
            self.assertIn("r_squared", r)
            self.assertTrue(0 <= r["r_squared"] <= 1)


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
