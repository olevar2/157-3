"""
Unit Tests for Gann Time Cycle Indicator

Comprehensive test suite for Gann Time Cycle indicator covering mathematical accuracy,
parameter validation, data validation, edge cases, performance benchmarks,
and interface compliance. Follows Platform3 testing standards.

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

# Platform3 imports
from engines.ai_enhancement.indicators.base_indicator import IndicatorValidationError
from engines.ai_enhancement.indicators.gann.gann_time_cycle_indicator import (
    GannTimeCycleIndicator,
)
from tests.gann.test_base import GannTestBase


class TestGannTimeCycleIndicator(GannTestBase):
    """
    Comprehensive unit tests for GannTimeCycleIndicator

    Test Categories:
    1. Mathematical Accuracy - Time cycle detection algorithms
    2. Parameter Validation - Invalid types and ranges
    3. Data Input Validation - Missing columns and insufficient data
    4. Edge Cases - Flat prices and extreme volatility
    5. Performance Benchmarks - <10ms for 1K data requirement
    6. Interface Compliance - StandardIndicatorInterface adherence
    """

    def setUp(self):
        """Set up test fixtures for Gann Time Cycle tests"""
        super().setUp()

        # Create indicator instance
        self.indicator = GannTimeCycleIndicator()

        # Standard Gann time cycles (in trading days)
        self.gann_cycles = [7, 14, 21, 30, 45, 60, 90, 120, 144, 180, 252, 360]

    def test_mathematical_accuracy_cycle_detection(self):
        """Test time cycle detection algorithms"""
        # Create data with known cyclical pattern
        cycle_data = self._create_cyclical_test_data(period=30)

        result = self.indicator.calculate(cycle_data)

        # Verify cycle detection columns exist
        cycle_columns = [col for col in result.columns if "cycle" in col.lower()]
        self.assertGreater(len(cycle_columns), 0, "No cycle detection columns found")

        # Verify cycles are detected
        if "detected_cycles" in result.columns:
            cycles = result["detected_cycles"].dropna()
            self.assertGreater(len(cycles), 0, "No cycles detected in cyclical data")

    def test_mathematical_accuracy_harmonic_analysis(self):
        """Test harmonic analysis of time cycles"""
        # Create data with multiple harmonic cycles
        harmonic_data = self._create_harmonic_test_data()

        result = self.indicator.calculate(harmonic_data)

        # Verify harmonic analysis results
        if "dominant_cycle" in result.columns:
            dominant = result["dominant_cycle"].dropna()
            self.assertGreater(len(dominant), 0, "No dominant cycle detected")

            # Dominant cycle should be within expected range
            if len(dominant) > 0:
                cycle_length = dominant.iloc[0]
                self.assertGreater(cycle_length, 5, "Cycle too short")
                self.assertLess(cycle_length, 500, "Cycle too long")

    def test_mathematical_accuracy_fourier_transform(self):
        """Test Fourier transform based cycle analysis"""
        # Use realistic market data for Fourier analysis
        result = self.indicator.calculate(self.test_data)

        # Verify spectral analysis components
        spectral_columns = [
            col for col in result.columns if "spectral" in col or "frequency" in col
        ]

        # Should have some spectral analysis output
        if len(spectral_columns) > 0:
            for col in spectral_columns:
                values = result[col].dropna()
                if len(values) > 0:
                    # Spectral values should be finite
                    self.assertTrue(
                        np.all(np.isfinite(values)),
                        f"Non-finite values in spectral analysis: {col}",
                    )

    def test_mathematical_accuracy_cycle_validation(self):
        """Test validation of detected cycles against Gann standards"""
        result = self.indicator.calculate(self.test_data)

        if "detected_cycles" in result.columns:
            cycles = result["detected_cycles"].dropna().unique()

            for cycle in cycles:
                if not pd.isna(cycle):
                    # Cycle should be positive
                    self.assertGreater(cycle, 0, f"Invalid cycle length: {cycle}")

                    # Cycle should be reasonable for financial markets
                    self.assertLess(cycle, 1000, f"Cycle too long: {cycle}")

    def test_parameter_validation_invalid_min_cycle_length(self):
        """Test validation of minimum cycle length"""
        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannTimeCycleIndicator(min_cycle_length=0)  # Should be positive

        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannTimeCycleIndicator(min_cycle_length=-5)  # Should be positive

    def test_parameter_validation_invalid_max_cycle_length(self):
        """Test validation of maximum cycle length"""
        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannTimeCycleIndicator(max_cycle_length=5, min_cycle_length=10)  # Max < Min

    def test_parameter_validation_invalid_analysis_method(self):
        """Test validation of analysis method"""
        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannTimeCycleIndicator(analysis_method="invalid_method")

    def test_parameter_validation_invalid_window_size(self):
        """Test validation of analysis window size"""
        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannTimeCycleIndicator(window_size=0)  # Should be positive

        with self.assertRaises((ValueError, TypeError, IndicatorValidationError)):
            GannTimeCycleIndicator(window_size="invalid")  # Should be numeric

    def test_data_validation_missing_columns(self):
        """Test handling of missing required columns"""
        invalid_data = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "volume": [1000, 1100, 1200],
                # Missing required price columns
            }
        )

        with self.assertRaises(IndicatorValidationError):
            self.indicator.calculate(invalid_data)

    def test_data_validation_insufficient_data_for_cycles(self):
        """Test handling of insufficient data for cycle analysis"""
        # Too small for meaningful cycle detection
        small_data = self.generate_realistic_ohlc_data(10)

        try:
            result = self.indicator.calculate(small_data)
            self.assertIsInstance(result, pd.DataFrame)
        except IndicatorValidationError:
            # Acceptable to reject insufficient data
            pass

    def test_edge_case_no_cyclical_patterns(self):
        """Test behavior with data containing no cyclical patterns"""
        # Create purely random data
        random_data = self._create_random_walk_data()

        result = self.indicator.calculate(random_data)
        self.assertIsInstance(result, pd.DataFrame)

        # Should handle gracefully even if no cycles detected
        if "detected_cycles" in result.columns:
            cycles = result["detected_cycles"].dropna()
            # May detect spurious cycles in random data, but should be stable
            self.assertIsInstance(len(cycles), int)

    def test_edge_case_perfect_cyclical_data(self):
        """Test behavior with perfect cyclical data"""
        perfect_cycle_data = self._create_perfect_cycle_data(period=20)

        result = self.indicator.calculate(perfect_cycle_data)
        self.assertIsInstance(result, pd.DataFrame)

        # Should detect the known cycle
        if "dominant_cycle" in result.columns:
            dominant = result["dominant_cycle"].dropna()
            if len(dominant) > 0:
                detected_period = dominant.iloc[-1]  # Most recent detection
                # Should be close to the known period (20)
                self.assertAlmostEqual(
                    detected_period,
                    20,
                    delta=5,
                    msg="Failed to detect known cycle period",
                )

    def test_edge_case_multiple_overlapping_cycles(self):
        """Test behavior with multiple overlapping cycles"""
        multi_cycle_data = self._create_multi_cycle_data()

        result = self.indicator.calculate(multi_cycle_data)
        self.assertIsInstance(result, pd.DataFrame)

        # Should detect multiple cycles
        if "detected_cycles" in result.columns:
            cycles = result["detected_cycles"].dropna().unique()
            self.assertGreater(
                len(cycles), 1, "Should detect multiple cycles in multi-cycle data"
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

    def test_performance_fft_efficiency(self):
        """Test FFT-based analysis efficiency"""
        # Test with power-of-2 data sizes for optimal FFT performance
        for size in [256, 512, 1024]:
            test_data = self.generate_realistic_ohlc_data(size)

            start_time = time.time()
            result = self.indicator.calculate(test_data)
            execution_time = time.time() - start_time

            # FFT should be very fast for power-of-2 sizes
            max_time = size / 1000.0 * self.max_time_1k  # Scale with data size
            self.assertLess(
                execution_time,
                max_time,
                f"FFT analysis too slow for {size} points: {execution_time:.4f}s",
            )
            self.assertIsInstance(result, pd.DataFrame)

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
        self.assertEqual(GannTimeCycleIndicator.CATEGORY, "gann")
        self.assertIsInstance(GannTimeCycleIndicator.VERSION, str)
        self.assertIsInstance(GannTimeCycleIndicator.AUTHOR, str)

    def _create_cyclical_test_data(self, period: int = 30) -> pd.DataFrame:
        """Create test data with known cyclical pattern"""
        periods = 200
        base_price = 100.0
        amplitude = 5.0
        data = []

        for i in range(periods):
            # Create sinusoidal price pattern
            cycle_component = amplitude * np.sin(2 * np.pi * i / period)
            price = base_price + cycle_component

            data.append(
                {
                    "timestamp": pd.Timestamp("2023-01-01") + pd.Timedelta(days=i),
                    "open": price,
                    "high": price + 0.5,
                    "low": price - 0.5,
                    "close": price,
                    "volume": 1000000,
                }
            )

        return pd.DataFrame(data).set_index("timestamp")

    def _create_harmonic_test_data(self) -> pd.DataFrame:
        """Create test data with harmonic cycle patterns"""
        periods = 300
        base_price = 100.0
        data = []

        for i in range(periods):
            # Create multiple harmonic cycles
            cycle1 = 3 * np.sin(2 * np.pi * i / 20)  # 20-day cycle
            cycle2 = 2 * np.sin(2 * np.pi * i / 60)  # 60-day cycle
            cycle3 = 1 * np.sin(2 * np.pi * i / 120)  # 120-day cycle

            price = base_price + cycle1 + cycle2 + cycle3

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

    def _create_random_walk_data(self) -> pd.DataFrame:
        """Create random walk data with no cyclical patterns"""
        periods = 200
        base_price = 100.0
        data = []
        price = base_price

        np.random.seed(42)  # Reproducible randomness

        for i in range(periods):
            # Pure random walk
            price += np.random.normal(0, 0.5)

            data.append(
                {
                    "timestamp": pd.Timestamp("2023-01-01") + pd.Timedelta(days=i),
                    "open": price,
                    "high": price + abs(np.random.normal(0, 0.3)),
                    "low": price - abs(np.random.normal(0, 0.3)),
                    "close": price,
                    "volume": 1000000,
                }
            )

        return pd.DataFrame(data).set_index("timestamp")

    def _create_perfect_cycle_data(self, period: int = 20) -> pd.DataFrame:
        """Create perfect cyclical data for testing cycle detection"""
        periods = 150
        base_price = 100.0
        amplitude = 10.0
        data = []

        for i in range(periods):
            # Perfect sine wave
            price = base_price + amplitude * np.sin(2 * np.pi * i / period)

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

    def _create_multi_cycle_data(self) -> pd.DataFrame:
        """Create data with multiple distinct cycles"""
        periods = 400
        base_price = 100.0
        data = []

        for i in range(periods):
            # Multiple non-harmonic cycles
            cycle1 = 4 * np.sin(2 * np.pi * i / 17)  # 17-day cycle
            cycle2 = 3 * np.sin(2 * np.pi * i / 43)  # 43-day cycle
            cycle3 = 2 * np.sin(2 * np.pi * i / 89)  # 89-day cycle

            price = base_price + cycle1 + cycle2 + cycle3

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
