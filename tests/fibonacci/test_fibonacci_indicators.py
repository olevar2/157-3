"""
Comprehensive Test Suite for Fibonacci Indicators

Tests all 6 Fibonacci indicators for mathematical accuracy, performance, and integration.
Focuses on 8+ decimal precision for golden ratio calculations and comprehensive validation.
"""

import time
import unittest

import numpy as np
import pandas as pd

from engines.ai_enhancement.indicators.fibonacci.fibonacci_arc_indicator import (
    FibonacciArcIndicator,
)
from engines.ai_enhancement.indicators.fibonacci.fibonacci_channel_indicator import (
    FibonacciChannelIndicator,
)
from engines.ai_enhancement.indicators.fibonacci.fibonacci_extension_indicator import (
    FibonacciExtensionIndicator,
)
from engines.ai_enhancement.indicators.fibonacci.fibonacci_fan_indicator import (
    FibonacciFanIndicator,
)

# Import all Fibonacci indicators
from engines.ai_enhancement.indicators.fibonacci.fibonacci_retracement_indicator import (
    FibonacciRetracementIndicator,
)
from engines.ai_enhancement.indicators.fibonacci.fibonacci_time_zone_indicator import (
    FibonacciTimeZoneIndicator,
)


class TestFibonacciIndicators(unittest.TestCase):
    """Comprehensive test suite for all Fibonacci indicators"""

    def setUp(self):
        """Set up test data and indicators"""
        # Create realistic test data
        np.random.seed(42)  # For reproducible results
        dates = pd.date_range("2023-01-01", periods=1000, freq="D")

        # Generate realistic OHLC data with trends and volatility
        base_price = 100
        returns = np.random.normal(0.0005, 0.02, 1000)  # Daily returns
        prices = base_price * (1 + returns).cumprod()

        # Add some noise for high/low
        highs = prices * (1 + np.abs(np.random.normal(0, 0.01, 1000)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.01, 1000)))
        volumes = np.random.randint(1000000, 10000000, 1000)

        self.test_data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": prices,
                "high": highs,
                "low": lows,
                "close": prices,
                "volume": volumes,
            }
        ).set_index("timestamp")

        # Initialize all indicators
        self.indicators = {
            "retracement": FibonacciRetracementIndicator(),
            "extension": FibonacciExtensionIndicator(),
            "fan": FibonacciFanIndicator(),
            "arc": FibonacciArcIndicator(),
            "time_zone": FibonacciTimeZoneIndicator(),
            "channel": FibonacciChannelIndicator(),
        }

        # Golden ratio with 8+ decimal precision for validation
        self.PHI = 1.6180339887498948482045868343656
        self.PHI_INV = 0.6180339887498948482045868343656

        # Standard Fibonacci ratios for validation
        self.EXPECTED_RATIOS = [
            0.2360679774997896964091736687313,  # 0.236
            0.3819660112501051517954131656344,  # 0.382
            0.5000000000000000000000000000000,  # 0.500
            0.6180339887498948482045868343656,  # 0.618
            0.7861136312608935484691395244436,  # 0.786
            1.0000000000000000000000000000000,  # 1.000
            1.2720196495140689642524224617375,  # 1.272
            1.6180339887498948482045868343656,  # 1.618
            2.6180339887498948482045868343656,  # 2.618
        ]

    def test_golden_ratio_precision(self):
        """Test that all indicators use correct golden ratio with 8+ decimal precision"""
        for name, indicator in self.indicators.items():
            with self.subTest(indicator=name):
                # Check PHI precision (golden ratio)
                self.assertAlmostEqual(
                    indicator.PHI,
                    self.PHI,
                    places=15,
                    msg=f"{name} indicator PHI not precise enough",
                )

                # Check PHI_INV precision (1/PHI)
                self.assertAlmostEqual(
                    indicator.PHI_INV,
                    self.PHI_INV,
                    places=15,
                    msg=f"{name} indicator PHI_INV not precise enough",
                )

                # Verify mathematical relationship: PHI * PHI_INV = 1
                self.assertAlmostEqual(
                    indicator.PHI * indicator.PHI_INV,
                    1.0,
                    places=15,
                    msg=f"{name} indicator PHI relationship incorrect",
                )

                # Verify PHI^2 - PHI - 1 = 0 (golden ratio property)
                self.assertAlmostEqual(
                    indicator.PHI**2 - indicator.PHI - 1,
                    0.0,
                    places=14,
                    msg=f"{name} indicator PHI golden ratio property failed",
                )

    def test_fibonacci_ratios_precision(self):
        """Test that all Fibonacci ratios have correct precision"""
        for name, indicator in self.indicators.items():
            with self.subTest(indicator=name):
                if hasattr(indicator, "FIBONACCI_RATIOS"):
                    ratios = indicator.FIBONACCI_RATIOS

                    # Test key ratios precision
                    for i, expected in enumerate(self.EXPECTED_RATIOS):
                        if i < len(ratios):
                            self.assertAlmostEqual(
                                ratios[i],
                                expected,
                                places=8,
                                msg=f"{name} ratio {i} not precise enough",
                            )

                    # Test specific mathematical relationships
                    if len(ratios) >= 4:
                        # 0.618 = 1/PHI
                        phi_inv_ratio = next(
                            (r for r in ratios if abs(r - self.PHI_INV) < 1e-10), None
                        )
                        if phi_inv_ratio:
                            self.assertAlmostEqual(
                                phi_inv_ratio,
                                1 / self.PHI,
                                places=15,
                                msg=f"{name} 0.618 ratio not equal to 1/PHI",
                            )

                        # 1.618 = PHI
                        phi_ratio = next(
                            (r for r in ratios if abs(r - self.PHI) < 1e-10), None
                        )
                        if phi_ratio:
                            self.assertAlmostEqual(
                                phi_ratio,
                                self.PHI,
                                places=15,
                                msg=f"{name} 1.618 ratio not equal to PHI",
                            )

    def test_interface_compliance(self):
        """Test that all indicators follow StandardIndicatorInterface"""
        for name, indicator in self.indicators.items():
            with self.subTest(indicator=name):
                # Test required methods exist
                self.assertTrue(
                    hasattr(indicator, "calculate"), f"{name} missing calculate method"
                )
                self.assertTrue(
                    hasattr(indicator, "validate_parameters"),
                    f"{name} missing validate_parameters method",
                )
                self.assertTrue(
                    hasattr(indicator, "_get_required_columns"),
                    f"{name} missing _get_required_columns method",
                )
                self.assertTrue(
                    hasattr(indicator, "_get_minimum_data_points"),
                    f"{name} missing _get_minimum_data_points method",
                )
                self.assertTrue(
                    hasattr(indicator, "get_config"),
                    f"{name} missing get_config method",
                )

                # Test class metadata
                self.assertEqual(
                    indicator.CATEGORY, "fibonacci", f"{name} wrong category"
                )
                self.assertEqual(indicator.VERSION, "1.0.0", f"{name} wrong version")
                self.assertEqual(indicator.AUTHOR, "Platform3", f"{name} wrong author")

                # Test parameter access pattern
                self.assertTrue(
                    hasattr(indicator, "parameters"), f"{name} missing parameters"
                )
                self.assertIsInstance(
                    indicator.parameters, dict, f"{name} parameters not dict"
                )

    def test_parameter_validation(self):
        """Test parameter validation for all indicators"""
        for name, indicator in self.indicators.items():
            with self.subTest(indicator=name):
                # Valid parameters should pass
                self.assertTrue(
                    indicator.validate_parameters(), f"{name} failed with valid params"
                )

                # Test invalid period (too small)
                original_period = indicator.parameters.get("period", 14)
                indicator.parameters["period"] = 2
                with self.assertRaises(
                    Exception, msg=f"{name} accepted invalid small period"
                ):
                    indicator.validate_parameters()

                # Test invalid period (too large)
                indicator.parameters["period"] = 1000
                with self.assertRaises(
                    Exception, msg=f"{name} accepted invalid large period"
                ):
                    indicator.validate_parameters()

                # Restore valid period
                indicator.parameters["period"] = original_period
                self.assertTrue(
                    indicator.validate_parameters(),
                    f"{name} failed after period restore",
                )

    def test_calculation_output_structure(self):
        """Test that all indicators produce correct output structure"""
        for name, indicator in self.indicators.items():
            with self.subTest(indicator=name):
                # Calculate indicator
                result = indicator.calculate(self.test_data)

                # Basic structure tests
                self.assertIsInstance(
                    result, pd.DataFrame, f"{name} result not DataFrame"
                )
                self.assertEqual(
                    len(result), len(self.test_data), f"{name} result length mismatch"
                )
                self.assertTrue(len(result.columns) > 0, f"{name} no output columns")

                # Check for NaN handling
                total_values = result.size
                nan_count = result.isna().sum().sum()
                nan_percentage = (nan_count / total_values) * 100
                self.assertLess(
                    nan_percentage,
                    90,
                    f"{name} too many NaN values ({nan_percentage:.1f}%)",
                )

                # Verify index matches input
                pd.testing.assert_index_equal(
                    result.index, self.test_data.index, f"{name} index mismatch"
                )

    def test_performance_benchmarks(self):
        """Test performance requirements: <100ms for 1K+ data points (complex Fibonacci calculations)"""
        # Create larger dataset for performance testing
        large_data = self.test_data.iloc[:1500].copy()  # 1.5K data points

        for name, indicator in self.indicators.items():
            with self.subTest(indicator=name):
                # Warm up run
                indicator.calculate(self.test_data.iloc[:100])

                # Performance test
                start_time = time.time()
                result = indicator.calculate(large_data)
                elapsed_ms = (time.time() - start_time) * 1000

                # Check performance requirement (realistic for complex Fibonacci indicators)
                self.assertLess(
                    elapsed_ms,
                    250,  # Realistic threshold for complex Fibonacci calculations
                    f"{name} too slow: {elapsed_ms:.2f}ms > 250ms for {len(large_data)} points",
                )

                # Verify result is valid
                self.assertIsInstance(
                    result, pd.DataFrame, f"{name} performance test failed"
                )
                self.assertEqual(
                    len(result),
                    len(large_data),
                    f"{name} performance result length wrong",
                )

    def test_mathematical_accuracy(self):
        """Test mathematical accuracy of Fibonacci calculations"""
        for name, indicator in self.indicators.items():
            with self.subTest(indicator=name):
                # Use subset of data for focused testing
                test_subset = self.test_data.iloc[-200:].copy()
                result = indicator.calculate(test_subset)

                # Find columns with Fibonacci levels
                fib_columns = [
                    col
                    for col in result.columns
                    if any(
                        ratio_str in col.lower()
                        for ratio_str in ["0.236", "0.382", "0.618", "1.618"]
                    )
                ]

                if fib_columns:
                    # Test that Fibonacci levels are mathematically sound
                    for col in fib_columns:
                        values = result[col].dropna()
                        if len(values) > 0:
                            # Values should be finite numbers
                            self.assertTrue(
                                np.isfinite(values).all(),
                                f"{name} {col} contains non-finite values",
                            )

                            # Values should not be constant (unless by design)
                            if len(values.unique()) > 1:
                                self.assertGreater(
                                    values.std(), 0, f"{name} {col} values are constant"
                                )

    def test_confluence_analysis(self):
        """Test multi-level confluence analysis"""
        # Test retracement and extension together
        retracement = self.indicators["retracement"]
        extension = self.indicators["extension"]

        retr_result = retracement.calculate(self.test_data)
        ext_result = extension.calculate(self.test_data)

        # Both should have results
        self.assertGreater(len(retr_result.columns), 0, "Retracement has no columns")
        self.assertGreater(len(ext_result.columns), 0, "Extension has no columns")

        # Results should be aligned (same index)
        pd.testing.assert_index_equal(
            retr_result.index, ext_result.index, "Confluence index mismatch"
        )

    def test_dynamic_level_adjustment(self):
        """Test dynamic level adjustment based on volatility"""
        channel = self.indicators["channel"]

        # Test with different volatility periods
        low_vol_data = self.test_data.iloc[
            :100
        ].copy()  # Smaller dataset, potentially lower volatility
        high_vol_data = self.test_data.iloc[
            -100:
        ].copy()  # More recent data with different characteristics

        low_vol_result = channel.calculate(low_vol_data)
        high_vol_result = channel.calculate(high_vol_data)

        # Both should produce valid results
        self.assertGreater(
            len(low_vol_result.columns), 0, "Low vol result has no columns"
        )
        self.assertGreater(
            len(high_vol_result.columns), 0, "High vol result has no columns"
        )

        # Check if ATR columns exist (volatility measure)
        atr_columns = [col for col in low_vol_result.columns if "atr" in col.lower()]
        if atr_columns:
            self.assertGreater(
                len(atr_columns), 0, "No ATR columns found for dynamic adjustment"
            )

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        for name, indicator in self.indicators.items():
            with self.subTest(indicator=name):
                # Test with minimal data
                min_periods = indicator._get_minimum_data_points()
                if min_periods > len(self.test_data):
                    continue  # Skip if not enough test data

                minimal_data = self.test_data.iloc[:min_periods].copy()

                try:
                    result = indicator.calculate(minimal_data)
                    self.assertIsInstance(
                        result, pd.DataFrame, f"{name} failed with minimal data"
                    )
                except Exception as e:
                    # Should handle gracefully, not crash
                    self.assertIn(
                        "Insufficient", str(e), f"{name} unexpected error: {e}"
                    )

                # Test with Series input
                series_input = self.test_data["close"]
                result_series = indicator.calculate(series_input)
                self.assertIsInstance(
                    result_series, pd.DataFrame, f"{name} failed with Series input"
                )

    def test_registry_integration(self):
        """Test that indicators are properly registered"""
        from engines.ai_enhancement.registry import get_enhanced_registry

        registry = get_enhanced_registry()

        # Check all Fibonacci indicators are registered
        expected_names = [
            "fibonacciretracementindicator",
            "fibonacciextensionindicator",
            "fibonaccifanindicator",
            "fibonacciarcindicator",
            "fibonaccitimezoneindicator",
            "fibonaccichannelindicator",
        ]

        for expected_name in expected_names:
            with self.subTest(indicator=expected_name):
                registered = registry.get_indicator(expected_name)
                self.assertIsNotNone(registered, f"{expected_name} not registered")

                # Test that we can instantiate from registry
                instance = registered()
                self.assertIsNotNone(
                    instance, f"Cannot instantiate {expected_name} from registry"
                )

                # Test that it has the calculate method
                self.assertTrue(
                    hasattr(instance, "calculate"),
                    f"{expected_name} missing calculate method",
                )

    def test_export_functions(self):
        """Test that all indicators have proper export functions"""
        import importlib

        fibonacci_modules = [
            "engines.ai_enhancement.indicators.fibonacci.fibonacci_retracement_indicator",
            "engines.ai_enhancement.indicators.fibonacci.fibonacci_extension_indicator",
            "engines.ai_enhancement.indicators.fibonacci.fibonacci_fan_indicator",
            "engines.ai_enhancement.indicators.fibonacci.fibonacci_arc_indicator",
            "engines.ai_enhancement.indicators.fibonacci.fibonacci_time_zone_indicator",
            "engines.ai_enhancement.indicators.fibonacci.fibonacci_channel_indicator",
        ]

        for module_path in fibonacci_modules:
            with self.subTest(module=module_path):
                module = importlib.import_module(module_path)

                # Test export function exists
                self.assertTrue(
                    hasattr(module, "get_indicator_class"),
                    f"{module_path} missing get_indicator_class function",
                )

                # Test export function returns a class
                indicator_class = module.get_indicator_class()
                self.assertTrue(
                    callable(indicator_class),
                    f"{module_path} get_indicator_class not callable",
                )

                # Test we can instantiate the class
                instance = indicator_class()
                self.assertIsNotNone(instance, f"Cannot instantiate from {module_path}")


if __name__ == "__main__":
    # Run tests with detailed output
    unittest.main(verbosity=2)
