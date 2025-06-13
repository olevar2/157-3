"""
Comprehensive tests for Awesome Oscillator Indicator
Trading-grade validation and performance tests
"""

import pytest
import pandas as pd
import numpy as np
import time
from unittest.mock import patch

from engines.ai_enhancement.indicators.momentum.awesome_oscillator import (
    AwesomeOscillatorIndicator,
)
from engines.ai_enhancement.indicators.base_indicator import IndicatorValidationError


class TestAwesomeOscillatorIndicator:
    """Test suite for Awesome Oscillator indicator"""

    def setup_method(self):
        """Setup test fixtures"""
        self.ao = AwesomeOscillatorIndicator()

        # Create sample data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # Generate realistic price data
        base_price = 100
        price_changes = np.random.normal(0, 1, 100)
        prices = [base_price]

        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change / 100)
            prices.append(max(new_price, 1))  # Prevent negative prices

        # Create OHLC data with realistic relationships
        closes = np.array(prices)
        highs = closes * (1 + np.abs(np.random.normal(0, 0.01, 100)))
        lows = closes * (1 - np.abs(np.random.normal(0, 0.01, 100)))
        opens = closes * (1 + np.random.normal(0, 0.005, 100))

        self.sample_data = pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": np.random.randint(1000, 10000, 100),
            },
            index=dates,
        )

    def test_initialization_default_parameters(self):
        """Test default parameter initialization"""
        ao = AwesomeOscillatorIndicator()
        assert ao.parameters["fast_period"] == 5
        assert ao.parameters["slow_period"] == 34
        assert ao.CATEGORY == "momentum"

    def test_initialization_custom_parameters(self):
        """Test custom parameter initialization"""
        ao = AwesomeOscillatorIndicator(fast_period=3, slow_period=21)
        assert ao.parameters["fast_period"] == 3
        assert ao.parameters["slow_period"] == 21

    def test_parameter_validation_valid(self):
        """Test valid parameter validation"""
        ao = AwesomeOscillatorIndicator(fast_period=5, slow_period=34)
        assert ao.validate_parameters() is True

    def test_parameter_validation_invalid_fast_period(self):
        """Test invalid fast period validation"""
        with pytest.raises(IndicatorValidationError):
            AwesomeOscillatorIndicator(fast_period=0, slow_period=34)

        with pytest.raises(IndicatorValidationError):
            AwesomeOscillatorIndicator(fast_period=-1, slow_period=34)

    def test_parameter_validation_invalid_slow_period(self):
        """Test invalid slow period validation"""
        with pytest.raises(IndicatorValidationError):
            AwesomeOscillatorIndicator(fast_period=5, slow_period=0)

    def test_parameter_validation_periods_relationship(self):
        """Test that fast period must be less than slow period"""
        with pytest.raises(IndicatorValidationError):
            AwesomeOscillatorIndicator(fast_period=34, slow_period=5)

        with pytest.raises(IndicatorValidationError):
            AwesomeOscillatorIndicator(fast_period=21, slow_period=21)

    def test_required_columns(self):
        """Test required data columns"""
        ao = AwesomeOscillatorIndicator()
        required = ao._get_required_columns()
        assert "high" in required
        assert "low" in required

    def test_minimum_data_points(self):
        """Test minimum data points requirement"""
        ao = AwesomeOscillatorIndicator(slow_period=34)
        assert ao._get_minimum_data_points() == 34

    def test_input_validation_missing_columns(self):
        """Test input validation with missing columns"""
        ao = AwesomeOscillatorIndicator()

        # Missing high column
        bad_data = pd.DataFrame({"low": [1, 2, 3], "close": [1, 2, 3]})
        with pytest.raises(IndicatorValidationError):
            ao.calculate(bad_data)

        # Missing low column
        bad_data = pd.DataFrame({"high": [1, 2, 3], "close": [1, 2, 3]})
        with pytest.raises(IndicatorValidationError):
            ao.calculate(bad_data)

    def test_input_validation_insufficient_data(self):
        """Test input validation with insufficient data"""
        ao = AwesomeOscillatorIndicator(slow_period=34)

        # Only 10 data points when 34 needed
        small_data = self.sample_data.head(10)
        with pytest.raises(IndicatorValidationError):
            ao.calculate(small_data)

    def test_input_validation_empty_data(self):
        """Test input validation with empty data"""
        ao = AwesomeOscillatorIndicator()

        empty_data = pd.DataFrame()
        with pytest.raises(IndicatorValidationError):
            ao.calculate(empty_data)

    def test_calculation_basic(self):
        """Test basic AO calculation"""
        ao = AwesomeOscillatorIndicator()
        result = ao.calculate(self.sample_data)

        # Check result properties
        assert isinstance(result, pd.Series)
        assert len(result) == len(self.sample_data)
        assert result.index.equals(self.sample_data.index)

        # Check that we have valid values after the slow period
        slow_period = ao.parameters["slow_period"]
        assert not pd.isna(result.iloc[slow_period - 1 :]).any()

        # Check that early values are NaN
        assert pd.isna(result.iloc[: slow_period - 1]).all()

    def test_calculation_known_values(self):
        """Test AO calculation against known values"""
        # Create simple test data with known expected result
        test_data = pd.DataFrame(
            {
                "high": [10, 11, 12, 13, 14] * 20,  # 100 points
                "low": [9, 10, 11, 12, 13] * 20,
            }
        )

        ao = AwesomeOscillatorIndicator(fast_period=2, slow_period=4)
        result = ao.calculate(test_data)

        # With constant pattern, median price is [9.5, 10.5, 11.5, 12.5, 13.5] repeated
        # After initial period, AO should stabilize to approximately 0
        # (as the fast and slow SMAs will be very close)
        stable_values = result.iloc[10:].dropna()
        assert abs(stable_values.mean()) < 0.1  # Should be close to 0

    def test_calculation_manual_verification(self):
        """Test AO calculation with manual verification"""
        # Simple data for manual calculation
        simple_data = pd.DataFrame(
            {
                "high": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                "low": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            }
        )

        ao = AwesomeOscillatorIndicator(fast_period=2, slow_period=4)
        result = ao.calculate(simple_data)

        # Manual calculation for the 4th point (index 3):
        # Median prices: [11, 12, 13, 14]
        # Fast SMA (2-period): (13 + 14) / 2 = 13.5
        # Slow SMA (4-period): (11 + 12 + 13 + 14) / 4 = 12.5
        # AO = 13.5 - 12.5 = 1.0
        expected_ao_at_index_3 = 1.0

        # Allow small floating point tolerance
        assert abs(result.iloc[3] - expected_ao_at_index_3) < 1e-10

    def test_signals_generation(self):
        """Test signal generation"""
        ao = AwesomeOscillatorIndicator()
        signals = ao.get_signals(self.sample_data)

        # Check signal columns exist
        expected_columns = [
            "ao",
            "bullish_crossover",
            "bearish_crossover",
            "ao_increasing",
            "ao_decreasing",
            "signal",
        ]
        for col in expected_columns:
            assert col in signals.columns

        # Check signal values are valid
        assert signals["signal"].isin([-1, 0, 1]).all()

        # Check crossover signals are boolean
        assert signals["bullish_crossover"].dtype == bool
        assert signals["bearish_crossover"].dtype == bool

    def test_get_metadata(self):
        """Test metadata retrieval"""
        ao = AwesomeOscillatorIndicator()
        metadata = ao.get_metadata()

        assert metadata.name == "Awesome Oscillator"
        assert metadata.category == "momentum"
        assert "5-period" in metadata.description
        assert "34-period" in metadata.description
        assert metadata.output_type == "series"
        assert metadata.min_data_points == 34

    def test_performance_benchmark(self):
        """Test calculation performance"""
        ao = AwesomeOscillatorIndicator()

        # Create larger dataset for performance testing
        large_data = pd.concat([self.sample_data] * 10, ignore_index=True)
        large_data.index = pd.date_range(
            "2023-01-01", periods=len(large_data), freq="D"
        )

        # Benchmark calculation time
        start_time = time.time()
        result = ao.calculate(large_data)
        end_time = time.time()

        calculation_time = end_time - start_time

        # Should complete in under 1ms per 1000 data points (very generous)
        points_per_ms = len(large_data) / (calculation_time * 1000)
        assert points_per_ms > 100, f"Performance too slow: {points_per_ms} points/ms"

        # Verify result is correct
        assert isinstance(result, pd.Series)
        assert len(result) == len(large_data)

    def test_edge_case_identical_high_low(self):
        """Test edge case where high equals low"""
        edge_data = pd.DataFrame(
            {
                "high": [100] * 50,
                "low": [100] * 50,
            }
        )

        ao = AwesomeOscillatorIndicator()
        result = ao.calculate(edge_data)

        # With identical high/low, median price is constant
        # So AO should be 0 after initial period
        stable_values = result.dropna()
        assert (stable_values == 0).all()

    def test_edge_case_trending_data(self):
        """Test with strongly trending data"""
        # Create strongly uptrending data
        trend_data = pd.DataFrame(
            {
                "high": range(1, 101),  # 1 to 100
                "low": range(0, 100),  # 0 to 99
            }
        )

        ao = AwesomeOscillatorIndicator()
        result = ao.calculate(trend_data)

        # In uptrend, fast SMA should be above slow SMA, so AO > 0
        stable_values = result.dropna()
        assert (stable_values > 0).all()

    def test_numerical_precision(self):
        """Test numerical precision and stability"""
        ao = AwesomeOscillatorIndicator()
        result = ao.calculate(self.sample_data)

        # Check for NaN values (should only be in initial period)
        slow_period = ao.parameters["slow_period"]
        valid_results = result.iloc[slow_period - 1 :]
        assert not valid_results.isna().any()

        # Check for infinite values
        assert not np.isinf(valid_results).any()

        # Check precision (values should be reasonable for price data)
        assert valid_results.abs().max() < 1000  # Reasonable for typical price ranges

    def test_consistency_multiple_calculations(self):
        """Test calculation consistency"""
        ao = AwesomeOscillatorIndicator()

        # Calculate multiple times
        result1 = ao.calculate(self.sample_data)
        result2 = ao.calculate(self.sample_data)
        result3 = ao.calculate(self.sample_data)

        # Results should be identical
        pd.testing.assert_series_equal(result1, result2)
        pd.testing.assert_series_equal(result2, result3)

    @pytest.mark.parametrize(
        "fast_period,slow_period", [(3, 21), (5, 34), (8, 55), (10, 50)]
    )
    def test_different_parameter_combinations(self, fast_period, slow_period):
        """Test various parameter combinations"""
        ao = AwesomeOscillatorIndicator(
            fast_period=fast_period, slow_period=slow_period
        )
        result = ao.calculate(self.sample_data)

        # Basic validation
        assert isinstance(result, pd.Series)
        assert len(result) == len(self.sample_data)

        # Should have valid values after slow period
        valid_results = result.iloc[slow_period - 1 :]
        if len(valid_results) > 0:
            assert not valid_results.isna().any()


if __name__ == "__main__":
    pytest.main([__file__])
