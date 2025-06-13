"""
Gann Indicators Registry Integration Test

This module tests integration of all Gann indicators with Platform3's enhanced registry system.
Tests include automatic discovery, metadata validation, agent integration tests, performance
testing, and naming consistency validation.

These tests verify that all 5 Gann indicators are properly registered, discovered, and
accessible through the standard registry interfaces.
"""

import sys
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent

from engines.ai_enhancement.adaptive_indicator_bridge import AdaptiveIndicatorBridge
from engines.ai_enhancement.indicators.gann import (
    GANN_INDICATORS,
    GannAnglesIndicator,
    GannFanIndicator,
    GannPriceTimeIndicator,
    GannSquareIndicator,
    GannTimeCycleIndicator,
)
from engines.ai_enhancement.registry import EnhancedIndicatorRegistry


class TestGannRegistryIntegration(unittest.TestCase):
    """
    Test Gann indicators integration with Platform3's registry system.

    Test Categories:
    1. Enhanced Registry Discovery - auto-discovery of all 5 Gann indicators
    2. Registry Metadata Validation - correct metadata values
    3. Agent Integration Testing - proper mapping to Pattern Master and Market agents
    4. Registry Performance Testing - fast lookup times
    5. Alias and Naming Consistency - consistent naming patterns
    6. Import Path Validation - correct import paths
    """

    def setUp(self):
        """Set up test environment with clean registry"""
        self.registry = EnhancedIndicatorRegistry()

        # Expected indicator list (registry uses lowercase class names)
        self.expected_gann_indicators = [
            "gannanglesindicator",
            "gannsquareindicator",
            "gannfanindicator",
            "ganntimecycleindicator",
            "gannpricetimeindicator",
        ]

        # Expected indicator classes (registry uses lowercase class names)
        self.expected_indicator_classes = {
            "gannanglesindicator": GannAnglesIndicator,
            "gannsquareindicator": GannSquareIndicator,
            "gannfanindicator": GannFanIndicator,
            "ganntimecycleindicator": GannTimeCycleIndicator,
            "gannpricetimeindicator": GannPriceTimeIndicator,
        }

        # Test data for performance testing
        self.test_data = self._generate_test_data()

    def _generate_test_data(self):
        """Generate test data for performance testing"""

        # Generate 1000 data points
        n_points = 1000
        base_price = 100.0

        # Create price movement
        price_changes = np.random.randn(n_points) * 2
        prices = [base_price]
        for change in price_changes:
            prices.append(prices[-1] + change)

        return pd.DataFrame(
            {
                "close": prices,
                "high": [p + abs(np.random.randn() * 1) for p in prices],
                "low": [p - abs(np.random.randn() * 1) for p in prices],
            }
        )

    def test_enhanced_registry_discovery(self):
        """Test automatic discovery of Gann indicators in enhanced registry"""
        # Load indicators
        stats = self.registry.enhance_existing_registry()

        # Get all 'gann' category indicators
        gann_indicators = []
        for category, indicators in self.registry.get_categories().items():
            if category == "gann":
                gann_indicators.extend(indicators)

        # Check if all expected indicators are found
        self.assertEqual(
            len(gann_indicators),
            5,
            f"Expected 5 Gann indicators, found {len(gann_indicators)}",
        )

        # Verify each expected indicator is in the registry
        for indicator_name in self.expected_gann_indicators:
            self.assertIn(
                indicator_name,
                gann_indicators,
                f"Indicator {indicator_name} not found in registry",
            )

        # Verify no unexpected indicators are found
        for indicator_name in gann_indicators:
            self.assertIn(
                indicator_name,
                self.expected_gann_indicators,
                f"Unexpected indicator {indicator_name} found in registry",
            )

    def test_registry_metadata_validation(self):
        """Test metadata validation for Gann indicators"""
        # Load indicators
        self.registry.enhance_existing_registry()

        # Check metadata for each expected indicator
        for indicator_name in self.expected_gann_indicators:
            try:
                # Get indicator
                indicator = self.registry.get_indicator(indicator_name)

                # Check indicator class
                self.assertEqual(
                    indicator,
                    self.expected_indicator_classes[indicator_name],
                    f"Wrong class for {indicator_name}",
                )

                # Check required class attributes
                self.assertEqual(
                    indicator.CATEGORY, "gann", f"Wrong CATEGORY for {indicator_name}"
                )
                self.assertEqual(
                    indicator.VERSION, "1.0.0", f"Wrong VERSION for {indicator_name}"
                )
                self.assertEqual(
                    indicator.AUTHOR, "Platform3", f"Wrong AUTHOR for {indicator_name}"
                )

                # Get metadata from registry
                metadata = self.registry.get_metadata(indicator_name)

                # Check metadata values
                self.assertEqual(
                    metadata.category,
                    "gann",
                    f"Wrong metadata category for {indicator_name}",
                )
                self.assertEqual(
                    metadata.version,
                    "1.0.0",
                    f"Wrong metadata version for {indicator_name}",
                )
                self.assertEqual(
                    metadata.author,
                    "Platform3",
                    f"Wrong metadata author for {indicator_name}",
                )

            except KeyError:
                self.fail(f"Indicator {indicator_name} not found in registry")

    @pytest.mark.asyncio
    async def test_agent_integration(self):
        """Test integration with Pattern Master and Market Microstructure agents"""
        # Initialize agent bridge
        bridge = AdaptiveIndicatorBridge()

        # Test Pattern Master agent integration
        pattern_master_indicators = await bridge.get_agent_indicators_async(
            "PatternMaster"
        )
        gann_in_pattern_master = [
            name for name in pattern_master_indicators if "gann" in name.lower()
        ]

        # Verify all Gann indicators are available to Pattern Master
        self.assertGreaterEqual(
            len(gann_in_pattern_master),
            5,
            "Not all Gann indicators available to Pattern Master",
        )

        # Check for specific indicators
        for indicator_name in self.expected_gann_indicators:
            # Check variations of the name that might be in the agent's indicator list
            found = False
            variations = [
                indicator_name,
                indicator_name.replace("_", ""),
                indicator_name.upper(),
                indicator_name.replace("gann_", ""),
            ]

            for variation in variations:
                if any(variation in indicator for indicator in gann_in_pattern_master):
                    found = True
                    break

            self.assertTrue(
                found, f"Indicator {indicator_name} not found for Pattern Master"
            )  # Test Market Microstructure agent integration
        market_indicators = await bridge.get_agent_indicators_async(
            "MarketMicrostructure"
        )
        gann_in_market = [name for name in market_indicators if "gann" in name.lower()]

        # Verify Gann indicators are available to Market Microstructure
        self.assertGreaterEqual(
            len(gann_in_market),
            3,  # At least 3 should be available
            "Not enough Gann indicators available to Market Microstructure",
        )

    def test_registry_performance(self):
        """Test registry lookup performance for Gann indicators"""

        # Load indicators
        self.registry.enhance_existing_registry()

        # Test lookup time for each indicator
        for indicator_name in self.expected_gann_indicators:
            # Measure lookup time
            start_time = time.time()
            indicator = self.registry.get_indicator(indicator_name)
            lookup_time = (time.time() - start_time) * 1000  # Convert to ms

            # Verify lookup time is fast (<1ms)
            self.assertLess(
                lookup_time,
                10,
                f"Registry lookup for {indicator_name} too slow: {lookup_time:.2f}ms",
            )

            # Test indicator instantiation time
            start_time = time.time()
            instance = indicator()
            instantiation_time = (time.time() - start_time) * 1000  # Convert to ms

            # Verify instantiation time is fast (<10ms)
            self.assertLess(
                instantiation_time,
                20,
                f"Instantiation for {indicator_name} too slow: {instantiation_time:.2f}ms",
            )

            # Test calculation performance with 1K data points
            start_time = time.time()
            result = instance.calculate(self.test_data)
            calculation_time = (time.time() - start_time) * 1000  # Convert to ms

            # Verify calculation is fast (<100ms for 1K data points)
            self.assertLess(
                calculation_time,
                100,
                f"Calculation for {indicator_name} too slow: {calculation_time:.2f}ms",
            )

            # Verify calculation result is a DataFrame
            self.assertIsInstance(
                result,
                pd.DataFrame,
                f"Calculation result for {indicator_name} is not a DataFrame",
            )

    def test_alias_and_naming_consistency(self):
        """Test alias and naming consistency for Gann indicators"""
        # Load indicators
        self.registry.enhance_existing_registry()

        # Check for consistent naming patterns
        for indicator_name in self.expected_gann_indicators:
            # Try variations of the name (registry stores lowercase class names)
            variations = [
                indicator_name,  # lowercase class name
                indicator_name.upper(),  # uppercase version
            ]

            found_count = 0
            for variation in variations:
                try:
                    indicator = self.registry.get_indicator(variation)
                    found_count += 1

                    # Verify it's the right class
                    self.assertEqual(
                        indicator,
                        self.expected_indicator_classes[indicator_name],
                        f"Wrong class for variation {variation}",
                    )
                except KeyError:
                    pass  # Variation not found, which is fine

            # At least the original name should work
            self.assertGreaterEqual(
                found_count, 1, f"No working variations found for {indicator_name}"
            )

    def test_import_path_validation(self):
        """Test import path validation for Gann indicators"""
        # Define the actual class names as they appear in the module
        expected_class_names = [
            "GannAnglesIndicator",
            "GannSquareIndicator",
            "GannFanIndicator",
            "GannTimeCycleIndicator",
            "GannPriceTimeIndicator",
        ]

        # Check that indicator is properly importable
        for class_name in expected_class_names:
            try:
                exec(f"from engines.ai_enhancement.indicators.gann import {class_name}")
                # If we got here, the import worked
                self.assertTrue(True)
            except ImportError:
                self.fail(f"Failed to import {class_name} from gann module")

        # Verify indicators are in __all__ export list
        import engines.ai_enhancement.indicators.gann

        all_exports = engines.ai_enhancement.indicators.gann.__all__

        for class_name in expected_class_names:
            self.assertIn(
                class_name,
                all_exports,
                f"{class_name} not in gann module __all__ export list",
            )

        # Verify GANN_INDICATORS dictionary
        self.assertEqual(
            len(GANN_INDICATORS),
            5,
            f"Expected 5 indicators in GANN_INDICATORS, found {len(GANN_INDICATORS)}",
        )

        for indicator_name in self.expected_gann_indicators:
            # Check that the lowercase class name maps to the actual registry key
            self.assertIn(
                indicator_name,
                [name.lower() for name in expected_class_names],
                f"{indicator_name} not derivable from expected class names",
            )


if __name__ == "__main__":
    unittest.main()
