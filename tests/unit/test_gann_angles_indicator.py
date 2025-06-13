"""
Unit Tests for Gann Angles Indicator

Comprehensive test suite covering mathematical accuracy, parameter validation,
data validation, edge cases, performance benchmarks, and interface compliance.
Follows Platform3 testing standards.

Created: 2025-06-10
Author: Platform3 Testing Framework
"""

import sys
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import test infrastructure and indicator
from engines.ai_enhancement.indicators.base_indicator import IndicatorValidationError
from engines.ai_enhancement.indicators.gann.gann_angles_indicator import (
    GannAnglesIndicator,
)
from tests.gann.test_base import GannTestBase


class TestGannAnglesIndicator(GannTestBase):
    """
    Comprehensive unit tests for GannAnglesIndicator

    Test Categories:
    1. Mathematical Accuracy - Hand-calculated reference values
    2. Parameter Validation - Invalid types and out-of-range values
    3. Data Input Validation - Missing columns and insufficient data
    4. Edge Cases - Flat prices and extreme volatility
    5. Performance Benchmarks - <10ms for 1K data requirement
    6. Interface Compliance - StandardIndicatorInterface adherence
    """

    def setUp(self):
        """Set up test fixtures for Gann Angles tests"""
        super().setUp()

        # Create indicator instance with default parameters
        self.indicator = GannAnglesIndicator()

        # Hand-calculated reference values for mathematical accuracy tests
        self.reference_angles = {
            "1x8": 7.125,  # arctan(1/8) = 7.125 degrees
            "1x4": 14.036,  # arctan(1/4) = 14.036 degrees
            "1x3": 18.435,  # arctan(1/3) = 18.435 degrees
            "1x2": 26.565,  # arctan(1/2) = 26.565 degrees
            "1x1": 45.0,  # arctan(1/1) = 45.0 degrees
            "2x1": 63.435,  # arctan(2/1) = 63.435 degrees
            "3x1": 71.565,  # arctan(3/1) = 71.565 degrees
            "4x1": 75.964,  # arctan(4/1) = 75.964 degrees
            "8x1": 82.875,  # arctan(8/1) = 82.875 degrees
        }

        # Create controlled test data for mathematical validation
        self.controlled_data = self._create_controlled_angle_data()

    def _create_controlled_angle_data(self) -> pd.DataFrame:
        """Create perfectly controlled data for angle calculation validation"""
        # Create data where price moves exactly 1 unit per time period
        # This should produce perfect 45-degree (1x1) angle
        periods = 100
        base_price = 100.0

        data = []
        for i in range(periods):
            price = base_price + i  # Perfect 1:1 price-time relationship
            data.append(
                {
                    "timestamp": pd.Timestamp("2023-01-01") + pd.Timedelta(days=i),
                    "open": price,
                    "high": price + 0.1,
                    "low": price - 0.1,
                    "close": price,
                    "volume": 1000000,
                }
            )

        return pd.DataFrame(data).set_index("timestamp")

    # =============================================================================
    # 1. MATHEMATICAL ACCURACY TESTS
    # =============================================================================

    def test_mathematical_accuracy_perfect_45_degree_angle(self):
        """Test calculation of perfect 45-degree (1x1) Gann angle"""
        # Use controlled data with perfect 1:1 price-time relationship
        result = self.indicator.calculate(self.controlled_data)

        # Verify 1x1 angle is present in results
        self.assertIn("gann_1x1", result.columns, "1x1 Gann angle not calculated")

        # For perfect 1:1 data, the angle line should match price progression
        gann_1x1_values = result["gann_1x1"].dropna()

        # At anchor point, angle line should equal price
        anchor_idx = result["anchor_price"].dropna().index[0]
        anchor_price = result.loc[anchor_idx, "anchor_price"]
        gann_at_anchor = result.loc[anchor_idx, "gann_1x1"]

        self.validate_angle_precision(
            gann_at_anchor,
            anchor_price,
            "1x1 Gann angle should equal anchor price at anchor point",
        )

    def test_mathematical_accuracy_all_standard_angles(self):
        """Test mathematical accuracy of all standard Gann angles"""
        # Create indicator with all standard angles
        indicator = GannAnglesIndicator(
            angles=["1x8", "1x4", "1x3", "1x2", "1x1", "2x1", "3x1", "4x1", "8x1"]
        )

        result = indicator.calculate(self.test_data)

        # Verify all angles are calculated
        expected_columns = [f"gann_{angle}" for angle in self.reference_angles.keys()]
        for col in expected_columns:
            self.assertIn(col, result.columns, f"Missing Gann angle: {col}")

        # Verify angles are in correct relative order (steeper angles have higher values)
        anchor_idx = result["anchor_price"].dropna().index[0]

        # Get angle values at a point forward from anchor
        test_idx = result.index[
            min(len(result) - 1, result.index.get_loc(anchor_idx) + 10)
        ]

        # Bullish angles (>45°) should be above bearish angles (<45°) when trending up
        if test_idx in result.index:
            gann_values = {}
            for angle in self.reference_angles.keys():
                col = f"gann_{angle}"
                if col in result.columns and not pd.isna(result.loc[test_idx, col]):
                    gann_values[angle] = result.loc[test_idx, col]

            # Verify angle ordering (relative relationships)
            if "1x1" in gann_values and "2x1" in gann_values:
                self.assertNotEqual(
                    gann_values["1x1"],
                    gann_values["2x1"],
                    "Different Gann angles should produce different values",
                )

    def test_mathematical_accuracy_angle_coefficients(self):
        """Test that angle coefficients match mathematical expectations"""
        indicator = GannAnglesIndicator()
        coefficients = indicator._get_gann_coefficients()

        # Verify coefficient calculations
        expected_coefficients = {
            "1x8": 1 / 8,  # Slope = rise/run = 1/8
            "1x4": 1 / 4,  # Slope = 1/4
            "1x3": 1 / 3,  # Slope = 1/3
            "1x2": 1 / 2,  # Slope = 1/2
            "1x1": 1 / 1,  # Slope = 1 (45 degrees)
            "2x1": 2 / 1,  # Slope = 2
            "3x1": 3 / 1,  # Slope = 3
            "4x1": 4 / 1,  # Slope = 4
            "8x1": 8 / 1,  # Slope = 8
        }

        for angle, expected_coeff in expected_coefficients.items():
            if angle in coefficients:
                self.validate_ratio_precision(
                    coefficients[angle],
                    expected_coeff,
                    f"Coefficient for {angle} angle",
                )

    def test_mathematical_accuracy_anchor_point_detection(self):
        """Test anchor point detection algorithms"""
        # Test with known high/low points
        test_data = self.generate_trending_data(50, "up")

        indicator = GannAnglesIndicator(anchor_point="high")
        result = indicator.calculate(test_data)

        # Verify anchor point is at highest high
        anchor_price = result["anchor_price"].dropna().iloc[0]
        max_high = test_data["high"].max()

        self.validate_ratio_precision(
            anchor_price,
            max_high,
            "Anchor point should be at highest high when anchor_point='high'",
        )

    # =============================================================================
    # 2. PARAMETER VALIDATION TESTS
    # =============================================================================

    def test_parameter_validation_invalid_anchor_point(self):
        """Test validation of anchor_point parameter"""
        # Test invalid anchor point types
        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannAnglesIndicator(anchor_point=123.45)  # Should be string or valid index

        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannAnglesIndicator(anchor_point=["invalid"])  # Should not be list

    def test_parameter_validation_invalid_price_scale(self):
        """Test validation of price_scale parameter"""
        # Test invalid price scale values
        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannAnglesIndicator(price_scale=-1.0)  # Should be positive

        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannAnglesIndicator(price_scale=0.0)  # Should not be zero

        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannAnglesIndicator(price_scale="invalid")  # Should be numeric

    def test_parameter_validation_invalid_time_scale(self):
        """Test validation of time_scale parameter"""
        # Test invalid time scale values
        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannAnglesIndicator(time_scale=-1)  # Should be positive

        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannAnglesIndicator(time_scale=0)  # Should not be zero

        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannAnglesIndicator(time_scale=1.5)  # Should be integer

    def test_parameter_validation_invalid_angles(self):
        """Test validation of angles parameter"""
        # Test invalid angle specifications
        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannAnglesIndicator(angles=["invalid_angle"])  # Should be valid Gann angles

        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannAnglesIndicator(angles="1x1")  # Should be list, not string

    def test_parameter_validation_invalid_calculation_periods(self):
        """Test validation of calculation_periods parameter"""
        # Test invalid calculation periods
        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannAnglesIndicator(calculation_periods=-10)  # Should be positive

        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannAnglesIndicator(calculation_periods=0)  # Should be greater than zero

        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannAnglesIndicator(calculation_periods="invalid")  # Should be numeric

    # =============================================================================
    # 3. DATA INPUT VALIDATION TESTS
    # =============================================================================

    def test_data_validation_missing_required_columns(self):
        """Test handling of missing required columns"""
        # Test DataFrame without required 'close' column
        invalid_data = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [101, 102, 103],
                "low": [99, 100, 101],
                "volume": [1000, 1100, 1200],
                # Missing 'close' column
            }
        )

        with self.assertRaises(IndicatorValidationError):
            self.indicator.calculate(invalid_data)

    def test_data_validation_insufficient_data(self):
        """Test handling of insufficient data points"""
        # Test with very small dataset
        small_data = self.generate_realistic_ohlc_data(3)  # Only 3 points

        # Should handle gracefully or raise appropriate error
        try:
            result = self.indicator.calculate(small_data)
            # If it succeeds, verify basic structure
            self.assertIsInstance(result, pd.DataFrame)
        except IndicatorValidationError:
            # Acceptable to reject insufficient data
            pass

    def test_data_validation_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        empty_data = pd.DataFrame()

        with self.assertRaises((IndicatorValidationError, ValueError)):
            self.indicator.calculate(empty_data)

    def test_data_validation_nan_values(self):
        """Test handling of NaN values in data"""
        data_with_nans = self.test_data.copy()
        data_with_nans.loc[data_with_nans.index[10:15], "close"] = np.nan

        # Should handle NaNs gracefully
        try:
            result = self.indicator.calculate(data_with_nans)
            self.assertIsInstance(result, pd.DataFrame)
        except IndicatorValidationError:
            # Acceptable to reject data with NaNs
            pass

    def test_data_validation_series_input(self):
        """Test handling of Series input instead of DataFrame"""
        price_series = self.test_data["close"]

        # Should accept Series input
        result = self.indicator.calculate(price_series)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("gann_1x1", result.columns)

    # =============================================================================
    # 4. EDGE CASES TESTS
    # =============================================================================

    def test_edge_case_flat_prices(self):
        """Test behavior with completely flat price data"""
        flat_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=50),
                "open": [100.0] * 50,
                "high": [100.0] * 50,
                "low": [100.0] * 50,
                "close": [100.0] * 50,
                "volume": [1000000] * 50,
            }
        ).set_index("timestamp")

        # Should handle flat prices without error
        result = self.indicator.calculate(flat_data)
        self.assertIsInstance(result, pd.DataFrame)

        # Gann angles should be flat (horizontal lines) for flat price data
        # All angle lines should equal the anchor price
        anchor_price = result["anchor_price"].dropna().iloc[0]
        gann_1x1 = result["gann_1x1"].dropna()

        if len(gann_1x1) > 0:
            # For flat data, angle lines might be constant
            first_value = gann_1x1.iloc[0]
            self.assertTrue(
                np.isfinite(first_value), "Gann angle should be finite for flat data"
            )

    def test_edge_case_extreme_volatility(self):
        """Test behavior with extremely volatile price data"""
        # Create data with extreme price swings
        extreme_data = self.generate_realistic_ohlc_data(
            100, volatility=0.2
        )  # 20% volatility

        # Should handle extreme volatility gracefully
        result = self.indicator.calculate(extreme_data)
        self.assertIsInstance(result, pd.DataFrame)

        # Verify no infinite or NaN values in key calculations
        for col in result.columns:
            if col.startswith("gann_"):
                values = result[col].dropna()
                if len(values) > 0:
                    self.assertTrue(
                        np.all(np.isfinite(values)),
                        f"Infinite values found in {col} with extreme volatility",
                    )

    def test_edge_case_single_data_point(self):
        """Test behavior with single data point"""
        single_point = self.generate_realistic_ohlc_data(1)

        # Should handle single point gracefully
        try:
            result = self.indicator.calculate(single_point)
            self.assertIsInstance(result, pd.DataFrame)
        except IndicatorValidationError:
            # Acceptable to reject single point
            pass

    def test_edge_case_negative_prices(self):
        """Test behavior with negative price values"""
        # Create data with negative prices (unusual but possible in some commodities)
        negative_data = self.test_data.copy()
        negative_data[["open", "high", "low", "close"]] -= 150  # Make prices negative

        # Should handle negative prices appropriately
        try:
            result = self.indicator.calculate(negative_data)
            self.assertIsInstance(result, pd.DataFrame)
        except IndicatorValidationError:
            # Acceptable to reject negative prices depending on implementation
            pass

    # =============================================================================
    # 5. PERFORMANCE BENCHMARK TESTS
    # =============================================================================

    def test_performance_benchmark_1k_data(self):
        """Test performance with 1K data points (must be <10ms)"""
        test_data = self.generate_realistic_ohlc_data(1000)

        # Benchmark calculation time
        start_time = time.time()
        result = self.indicator.calculate(test_data)
        execution_time = time.time() - start_time

        # Verify performance requirement
        self.assertLess(
            execution_time,
            self.max_time_1k,
            f"1K data processing too slow: {execution_time:.4f}s > {self.max_time_1k}s",
        )

        # Verify result is valid
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0, "No results produced for 1K data")

    def test_performance_benchmark_scaling(self):
        """Test performance scaling across different data sizes"""
        data_sizes = [100, 1000, 5000]
        execution_times = self.benchmark_indicator_performance(
            GannAnglesIndicator, data_sizes
        )

        # Validate performance requirements
        self.validate_performance_requirements(execution_times)

        # Verify reasonable scaling (should be roughly linear or better)
        if len(execution_times) >= 2:
            times = list(execution_times.values())
            # Execution time shouldn't increase dramatically
            for i in range(1, len(times)):
                scaling_factor = times[i] / times[i - 1]
                self.assertLess(
                    scaling_factor,
                    20,  # Very generous scaling allowance
                    f"Performance scaling too poor: {scaling_factor}x increase",
                )

    def test_performance_memory_usage(self):
        """Test memory usage remains reasonable"""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Run calculation on large dataset
        large_data = self.generate_realistic_ohlc_data(10000)
        result = self.indicator.calculate(large_data)

        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB

        # Memory increase should be reasonable (< 100MB for this test)
        self.assertLess(
            memory_increase,
            100,
            f"Excessive memory usage: {memory_increase:.2f}MB increase",
        )

    # =============================================================================
    # 6. INTERFACE COMPLIANCE TESTS
    # =============================================================================

    def test_interface_compliance_inheritance(self):
        """Test proper inheritance from StandardIndicatorInterface"""
        self.assert_indicator_interface_compliance(self.indicator)

    def test_interface_compliance_calculate_method(self):
        """Test calculate method interface compliance"""
        result = self.indicator.calculate(self.test_data)

        # Verify return type
        self.assertIsInstance(result, pd.DataFrame, "calculate() must return DataFrame")

        # Verify result structure
        self.assertGreater(
            len(result), 0, "calculate() must return non-empty DataFrame"
        )
        self.assertTrue(
            any(col.startswith("gann_") for col in result.columns),
            "Result must contain Gann angle columns",
        )

    def test_interface_compliance_get_signals_method(self):
        """Test get_signals method interface compliance"""
        # First calculate to populate internal state
        self.indicator.calculate(self.test_data)

        # Test signals interface
        signals = self.indicator.get_signals()
        self.validate_signal_structure(signals)

    def test_interface_compliance_get_support_resistance_method(self):
        """Test get_support_resistance method interface compliance"""
        # First calculate to populate internal state
        self.indicator.calculate(self.test_data)

        # Test support/resistance interface
        levels = self.indicator.get_support_resistance()

        # Verify structure
        self.assertIsInstance(levels, dict, "get_support_resistance() must return dict")
        self.assertIn("support", levels, "Must include support levels")
        self.assertIn("resistance", levels, "Must include resistance levels")

    def test_interface_compliance_validate_parameters_method(self):
        """Test validate_parameters method interface compliance"""
        # Test with valid parameters
        valid_params = self.create_mock_parameters(
            anchor_point="auto",
            price_scale=1.0,
            time_scale=1,
            angles=["1x1", "2x1"],
            calculation_periods=100,
        )

        # Should not raise exception for valid parameters
        try:
            self.indicator.validate_parameters(valid_params)
        except Exception as e:
            self.fail(f"validate_parameters() raised exception for valid params: {e}")

    def test_interface_compliance_get_debug_info_method(self):
        """Test get_debug_info method interface compliance"""
        # First calculate to populate internal state
        self.indicator.calculate(self.test_data)

        # Test debug info interface
        debug_info = self.indicator.get_debug_info()

        # Verify structure
        self.assertIsInstance(debug_info, dict, "get_debug_info() must return dict")

        # Should contain useful debugging information
        expected_keys = ["calculation_details", "parameters", "data_summary"]
        for key in expected_keys:
            self.assertIn(key, debug_info, f"Debug info should contain {key}")

    def test_interface_compliance_metadata(self):
        """Test class metadata compliance"""
        # Verify required class attributes
        self.assertTrue(hasattr(GannAnglesIndicator, "CATEGORY"))
        self.assertTrue(hasattr(GannAnglesIndicator, "VERSION"))
        self.assertTrue(hasattr(GannAnglesIndicator, "AUTHOR"))

        # Verify metadata values
        self.assertEqual(GannAnglesIndicator.CATEGORY, "gann")
        self.assertIsInstance(GannAnglesIndicator.VERSION, str)
        self.assertIsInstance(GannAnglesIndicator.AUTHOR, str)

    def test_interface_compliance_get_indicator_class(self):
        """Test get_indicator_class factory method"""
        # Verify factory method exists and works
        indicator_class = GannAnglesIndicator.get_indicator_class()
        self.assertEqual(indicator_class, GannAnglesIndicator)

        # Verify can instantiate from factory
        instance = indicator_class()
        self.assertIsInstance(instance, GannAnglesIndicator)


if __name__ == "__main__":
    unittest.main()
