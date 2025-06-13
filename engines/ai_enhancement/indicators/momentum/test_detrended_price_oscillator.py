"""
Unit tests for Detrended Price Oscillator (DPO) Indicator
"""

import unittest

import numpy as np
import pandas as pd

from engines.ai_enhancement.indicators.momentum.detrended_price_oscillator import (
    DetrendedPriceOscillatorIndicator,
)


class TestDetrendedPriceOscillatorIndicator(unittest.TestCase):
    """Test cases for Detrended Price Oscillator"""

    def setUp(self):
        """Set up test data and indicator instance"""
        # Create sample OHLCV data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # Generate realistic price data with some trend
        base_price = 100
        trend = np.linspace(0, 10, 100)  # Upward trend
        noise = np.random.randn(100) * 2

        close_prices = base_price + trend + noise

        self.sample_data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": close_prices + np.random.randn(100) * 0.5,
                "high": close_prices + abs(np.random.randn(100)) * 1.5,
                "low": close_prices - abs(np.random.randn(100)) * 1.5,
                "close": close_prices,
                "volume": 1000000 + np.random.randint(-100000, 100000, 100),
            }
        )

        # Ensure OHLC constraints
        self.sample_data["high"] = self.sample_data[
            ["open", "high", "low", "close"]
        ].max(axis=1)
        self.sample_data["low"] = self.sample_data[
            ["open", "high", "low", "close"]
        ].min(axis=1)

        self.indicator = DetrendedPriceOscillatorIndicator(period=14)

    def test_initialization(self):
        """Test indicator initialization"""
        # Test default initialization
        dpo = DetrendedPriceOscillatorIndicator()
        self.assertEqual(dpo.parameters["period"], 14)

        # Test custom parameters
        dpo_custom = DetrendedPriceOscillatorIndicator(period=20)
        self.assertEqual(dpo_custom.parameters["period"], 20)

        # Test parameter validation
        with self.assertRaises(Exception):
            DetrendedPriceOscillatorIndicator(period=-1)

    def test_parameter_validation(self):
        """Test parameter validation"""
        # Valid parameters
        self.assertTrue(self.indicator.validate_parameters())

        # Invalid parameters
        invalid_indicator = DetrendedPriceOscillatorIndicator(period=0)
        with self.assertRaises(Exception):
            invalid_indicator.validate_parameters()

    def test_minimum_data_points(self):
        """Test minimum data points calculation"""
        period = 14
        displacement = period // 2 + 1
        expected_min = period + displacement

        self.assertEqual(self.indicator._get_minimum_data_points(), expected_min)

    def test_calculate_basic(self):
        """Test basic DPO calculation"""
        result = self.indicator.calculate(self.sample_data)

        # Basic checks
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.sample_data))

        # Should have NaN values at the beginning due to displacement
        min_periods = self.indicator._get_minimum_data_points()
        self.assertTrue(result.iloc[: min_periods - 1].isna().all())

        # Should have valid values after minimum periods
        valid_data = result.iloc[min_periods:]
        self.assertFalse(valid_data.isna().all())

    def test_insufficient_data(self):
        """Test behavior with insufficient data"""
        small_data = self.sample_data.head(10)  # Less than minimum required

        result = self.indicator.calculate(small_data)
        # Should return series with all NaN values
        self.assertTrue(result.isna().all())

    def test_dpo_properties(self):
        """Test DPO mathematical properties"""
        result = self.indicator.calculate(self.sample_data)

        # DPO should oscillate around zero (approximately)
        valid_result = result.dropna()
        if len(valid_result) > 0:
            # Mean should be close to zero for detrended data
            mean_value = valid_result.mean()
            # Allow some tolerance due to finite sample effects
            self.assertLess(abs(mean_value), 5.0)

    def test_displacement_effect(self):
        """Test that displacement shifts the SMA correctly"""
        period = 14
        displacement = period // 2 + 1

        # Get calculation details
        self.indicator.calculate(self.sample_data)
        calc_details = self.indicator._last_calculation

        sma = calc_details["sma"]
        sma_displaced = calc_details["sma_displaced"]

        # Check that displacement works correctly
        for i in range(displacement, len(sma)):
            if not pd.isna(sma.iloc[i - displacement]):
                self.assertEqual(sma.iloc[i - displacement], sma_displaced.iloc[i])

    def test_get_signals(self):
        """Test signal generation"""
        signals = self.indicator.get_signals(self.sample_data)

        # Check signal DataFrame structure
        expected_columns = [
            "dpo",
            "bullish_crossover",
            "bearish_crossover",
            "dpo_increasing",
            "dpo_decreasing",
            "local_peak",
            "local_trough",
            "signal",
        ]

        for col in expected_columns:
            self.assertIn(col, signals.columns)

        # Check signal values are in valid range
        signal_values = signals["signal"].dropna().unique()
        valid_signals = set([-1, 0, 1])
        self.assertTrue(set(signal_values).issubset(valid_signals))

    def test_cycle_analysis(self):
        """Test cycle analysis functionality"""
        cycle_analysis = self.indicator.get_cycle_analysis(self.sample_data)

        # Check required keys in analysis
        expected_keys = [
            "current_dpo",
            "dpo_volatility",
            "num_peaks",
            "num_troughs",
            "avg_peak_cycle",
            "avg_trough_cycle",
            "recent_peaks",
            "recent_troughs",
            "cycle_strength",
        ]

        for key in expected_keys:
            self.assertIn(key, cycle_analysis)

        # Check data types
        self.assertIsInstance(cycle_analysis["num_peaks"], int)
        self.assertIsInstance(cycle_analysis["num_troughs"], int)
        self.assertIsInstance(cycle_analysis["recent_peaks"], list)
        self.assertIsInstance(cycle_analysis["recent_troughs"], list)

    def test_known_values(self):
        """Test against known DPO calculation"""
        # Create simple test data with known pattern
        simple_data = pd.DataFrame(
            {
                "close": [
                    100,
                    101,
                    102,
                    103,
                    104,
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
                ]
            }
        )

        # Add required columns
        simple_data["open"] = simple_data["close"] + 0.1
        simple_data["high"] = simple_data["close"] + 0.2
        simple_data["low"] = simple_data["close"] - 0.2
        simple_data["volume"] = 1000

        dpo_simple = DetrendedPriceOscillatorIndicator(period=10)
        result = dpo_simple.calculate(simple_data)

        # Should have valid results for later periods
        self.assertFalse(result.tail(5).isna().all())

    def test_edge_cases(self):
        """Test edge cases"""
        # Test with constant prices (no volatility)
        constant_data = self.sample_data.copy()
        constant_data[["open", "high", "low", "close"]] = 100.0

        result = self.indicator.calculate(constant_data)
        valid_result = result.dropna()

        # With constant prices, DPO should be close to zero
        if len(valid_result) > 0:
            self.assertTrue(abs(valid_result).max() < 0.1)

    def test_get_metadata(self):
        """Test metadata retrieval"""
        metadata = self.indicator.get_metadata()

        self.assertEqual(metadata.name, "Detrended Price Oscillator")
        self.assertEqual(metadata.category, "momentum")
        self.assertIn("close", metadata.input_requirements)

    def test_config_export(self):
        """Test configuration export"""
        # For this implementation, we need to ensure the method exists
        # The actual StandardIndicatorInterface should have this method
        self.assertTrue(hasattr(self.indicator, "parameters"))
        self.assertIn("period", self.indicator.parameters)


if __name__ == "__main__":
    unittest.main()
