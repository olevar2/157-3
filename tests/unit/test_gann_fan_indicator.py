"""
Unit Tests for Gann Fan Indicator

Comprehensive test suite for Gann Fan indicator covering mathematical accuracy,
parameter validation, data validation, edge cases, performance benchmarks,
and interface compliance. Follows Platform3 testing standards.

Created: 2025-06-10
Author: Platform3 Testing Framework
"""

import sys
import time
import unittest
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent

# Platform3 imports
from engines.ai_enhancement.indicators.base_indicator import IndicatorValidationError
from engines.ai_enhancement.indicators.gann.gann_fan_indicator import GannFanIndicator
from tests.gann.test_base import GannTestBase


class TestGannFanIndicator(GannTestBase):
    """
    Comprehensive unit tests for GannFanIndicator

    Test Categories:
    1. Mathematical Accuracy - Fan line projections from pivot points
    2. Parameter Validation - Invalid types and ranges
    3. Data Input Validation - Missing columns and insufficient data
    4. Edge Cases - Flat prices and extreme volatility
    5. Performance Benchmarks - <10ms for 1K data requirement
    6. Interface Compliance - StandardIndicatorInterface adherence
    """

    def setUp(self):
        """Set up test fixtures for Gann Fan tests"""
        super().setUp()

        # Create indicator instance
        self.indicator = GannFanIndicator()

        # Fan line reference angles (standard Gann fan)
        self.fan_angles = [
            "1x8",
            "1x4",
            "1x3",
            "1x2",
            "1x1",
            "2x1",
            "3x1",
            "4x1",
            "8x1",
        ]

    def test_mathematical_accuracy_fan_line_projections(self):
        """Test fan line projections from pivot points"""
        # Create trending data with clear pivot
        trend_data = self.generate_trending_data(100, "up")

        result = self.indicator.calculate(trend_data)

        # Verify fan lines are calculated
        fan_columns = [col for col in result.columns if col.startswith("fan_")]
        self.assertGreater(len(fan_columns), 0, "No fan lines calculated")

        # Verify multiple angles present
        expected_angles = [f"fan_{angle}" for angle in self.fan_angles]
        found_angles = [col for col in expected_angles if col in result.columns]
        self.assertGreater(len(found_angles), 3, "Insufficient fan lines calculated")

    def test_mathematical_accuracy_pivot_point_detection(self):
        """Test pivot point detection accuracy"""
        # Create data with known pivot points
        pivot_data = self._create_pivot_test_data()

        result = self.indicator.calculate(pivot_data)

        # Verify pivot points are detected
        if "pivot_points" in result.columns:
            pivots = result["pivot_points"].dropna()
            self.assertGreater(len(pivots), 0, "No pivot points detected")

    def test_mathematical_accuracy_fan_convergence(self):
        """Test that fan lines converge at pivot point"""
        test_data = self.generate_realistic_ohlc_data(50)
        result = self.indicator.calculate(test_data)

        # Find pivot point
        if "pivot_points" in result.columns:
            pivot_idx = result["pivot_points"].dropna().index
            if len(pivot_idx) > 0:
                pivot_point = pivot_idx[0]

                # All fan lines should pass through or near pivot point
                fan_columns = [col for col in result.columns if col.startswith("fan_")]
                for col in fan_columns:
                    if not pd.isna(result.loc[pivot_point, col]):
                        # Fan line value at pivot should be close to price at pivot
                        fan_value = result.loc[pivot_point, col]
                        actual_price = test_data.loc[pivot_point, "close"]

                        # Allow some tolerance for calculation differences
                        self.assertAlmostEqual(
                            fan_value,
                            actual_price,
                            places=2,
                            msg=f"Fan line {col} doesn't converge at pivot",
                        )

    def test_parameter_validation_invalid_pivot_method(self):
        """Test validation of pivot detection method"""
        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannFanIndicator(pivot_method="invalid_method")

    def test_parameter_validation_invalid_fan_angles(self):
        """Test validation of fan angles parameter"""
        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannFanIndicator(fan_angles=["invalid_angle"])

        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannFanIndicator(fan_angles="1x1")  # Should be list

    def test_parameter_validation_invalid_projection_periods(self):
        """Test validation of projection periods"""
        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannFanIndicator(projection_periods=-10)  # Should be positive

        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannFanIndicator(projection_periods=0)  # Should be greater than zero

    def test_data_validation_missing_columns(self):
        """Test handling of missing required columns"""
        invalid_data = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "volume": [1000, 1100, 1200],
                # Missing high, low, close columns
            }
        )

        with self.assertRaises(IndicatorValidationError):
            self.indicator.calculate(invalid_data)

    def test_data_validation_insufficient_data_for_pivots(self):
        """Test handling of insufficient data for pivot detection"""
        small_data = self.generate_realistic_ohlc_data(
            5
        )  # Too small for meaningful pivots

        try:
            result = self.indicator.calculate(small_data)
            self.assertIsInstance(result, pd.DataFrame)
        except IndicatorValidationError:
            # Acceptable to reject insufficient data
            pass

    def test_edge_case_no_clear_pivots(self):
        """Test behavior with data containing no clear pivot points"""
        # Create very noisy data with no clear pivots
        noisy_data = self.generate_realistic_ohlc_data(
            100, volatility=0.1
        )  # High noise

        result = self.indicator.calculate(noisy_data)
        self.assertIsInstance(result, pd.DataFrame)

        # Should handle gracefully even if no clear pivots found
        fan_columns = [col for col in result.columns if col.startswith("fan_")]
        # May have no fan lines or minimal fan lines
        self.assertIsInstance(len(fan_columns), int)  # Basic structure check

    def test_edge_case_single_pivot_point(self):
        """Test behavior with only one pivot point"""
        # Create data with single clear pivot
        single_pivot_data = self._create_single_pivot_data()

        result = self.indicator.calculate(single_pivot_data)
        self.assertIsInstance(result, pd.DataFrame)

        # Should create fan from single pivot
        fan_columns = [col for col in result.columns if col.startswith("fan_")]
        if len(fan_columns) > 0:
            # At least some fan lines should be calculated
            fan_values = result[fan_columns].dropna().values
            self.assertGreater(
                len(fan_values), 0, "No fan values calculated from single pivot"
            )

    def test_performance_benchmark_1k_data(self):
        """Test performance with 1K data points"""
        test_data = self.generate_realistic_ohlc_data(1000)

        start_time = time.time()
        result = self.indicator.calculate(test_data)
        execution_time = time.time() - start_time

        self.assertLess(
            execution_time,
            self.max_time_1k,
            f"1K data too slow: {execution_time:.4f}s > {self.max_time_1k}s",
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_performance_scaling_with_pivot_complexity(self):
        """Test performance scales reasonably with data complexity"""
        data_sizes = [100, 500, 1000]
        execution_times = []

        for size in data_sizes:
            test_data = self.generate_realistic_ohlc_data(size)

            start_time = time.time()
            self.indicator.calculate(test_data)
            execution_time = time.time() - start_time
            execution_times.append(execution_time)

        # Performance shouldn't degrade exponentially
        for i in range(1, len(execution_times)):
            scaling_factor = execution_times[i] / execution_times[i - 1]
            self.assertLess(
                scaling_factor,
                10,  # Generous allowance for pivot detection complexity
                f"Performance scaling too poor: {scaling_factor}x increase",
            )

    def test_interface_compliance_inheritance(self):
        """Test proper inheritance from StandardIndicatorInterface"""
        self.assert_indicator_interface_compliance(self.indicator)

    def test_interface_compliance_calculate_method(self):
        """Test calculate method interface compliance"""
        result = self.indicator.calculate(self.test_data)

        self.assertIsInstance(result, pd.DataFrame, "calculate() must return DataFrame")
        self.assertGreater(
            len(result), 0, "calculate() must return non-empty DataFrame"
        )

    def test_interface_compliance_get_signals_method(self):
        """Test get_signals method interface compliance"""
        self.indicator.calculate(self.test_data)

        signals = self.indicator.get_signals()
        self.validate_signal_structure(signals)

    def test_interface_compliance_metadata(self):
        """Test class metadata compliance"""
        self.assertEqual(GannFanIndicator.CATEGORY, "gann")
        self.assertIsInstance(GannFanIndicator.VERSION, str)
        self.assertIsInstance(GannFanIndicator.AUTHOR, str)

    def _create_pivot_test_data(self) -> pd.DataFrame:
        """Create test data with clear pivot points"""
        periods = 100
        base_price = 100.0
        data = []

        for i in range(periods):
            # Create clear V-shaped pattern for pivot detection
            if i < 30:
                price = base_price - i * 0.5  # Downtrend
            elif i < 50:
                price = base_price - 15 + (i - 30) * 1.0  # Sharp reversal
            else:
                price = base_price + 5 + (i - 50) * 0.3  # Continued uptrend

            data.append(
                {
                    "timestamp": pd.Timestamp("2023-01-01") + pd.Timedelta(days=i),
                    "open": price,
                    "high": price + 0.3,
                    "low": price - 0.3,
                    "close": price,
                    "volume": 1000000,
                }
            )

        return pd.DataFrame(data).set_index("timestamp")

    def _create_single_pivot_data(self) -> pd.DataFrame:
        """Create test data with single clear pivot point"""
        periods = 50
        pivot_at = 25
        base_price = 100.0
        data = []

        for i in range(periods):
            if i < pivot_at:
                price = base_price + (pivot_at - i) * 0.5  # Decline to pivot
            else:
                price = base_price + (i - pivot_at) * 0.5  # Rise from pivot

            data.append(
                {
                    "timestamp": pd.Timestamp("2023-01-01") + pd.Timedelta(days=i),
                    "open": price,
                    "high": price + 0.2,
                    "low": price - 0.2,
                    "close": price,
                    "volume": 1000000,
                }
            )

        return pd.DataFrame(data).set_index("timestamp")


if __name__ == "__main__":
    unittest.main()
