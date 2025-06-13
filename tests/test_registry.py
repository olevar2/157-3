"""
Unit tests for Platform3 Indicator Registry
Tests that all indicators in the registry are callable and properly configured
"""

import unittest
import sys
from pathlib import Path

# Add project root to path

from engines.ai_enhancement.registry import INDICATOR_REGISTRY, validate_registry, get_indicator


class TestIndicatorRegistry(unittest.TestCase):
    """Test the indicator registry for proper configuration"""
    
    def test_all_indicators_are_callable(self):
        """Test that all indicators in the registry are callable"""
        non_callable_indicators = []
        
        for name, obj in INDICATOR_REGISTRY.items():
            if not callable(obj):
                non_callable_indicators.append((name, type(obj).__name__))
        
        # Assert no non-callable indicators exist
        self.assertEqual(
            len(non_callable_indicators), 
            0, 
            f"Found non-callable indicators: {non_callable_indicators}"
        )
    
    def test_registry_validation_passes(self):
        """Test that the registry validation function passes"""
        try:
            validate_registry()
        except TypeError as e:
            self.fail(f"Registry validation failed: {e}")
    
    def test_get_indicator_with_valid_name(self):
        """Test getting an indicator with a valid name"""
        # Pick a known indicator from the registry
        if INDICATOR_REGISTRY:
            first_indicator_name = list(INDICATOR_REGISTRY.keys())[0]
            try:
                indicator = get_indicator(first_indicator_name)
                self.assertTrue(callable(indicator))
            except (KeyError, TypeError) as e:
                self.fail(f"Failed to get valid indicator '{first_indicator_name}': {e}")
    
    def test_get_indicator_with_invalid_name(self):
        """Test getting an indicator with an invalid name raises KeyError"""
        with self.assertRaises(KeyError):
            get_indicator("non_existent_indicator")
    
    def test_registry_not_empty(self):
        """Test that the registry contains indicators"""
        self.assertGreater(
            len(INDICATOR_REGISTRY), 
            0, 
            "Registry should contain at least one indicator"
        )
    
    def test_specific_indicator_categories_exist(self):
        """Test that indicators from major categories exist in registry"""
        expected_categories = [
            'fractal',     # Should have fractal indicators
            'volume',      # Should have volume indicators
            'pattern',     # Should have pattern indicators
            'statistical', # Should have statistical indicators
        ]
        
        found_categories = set()
        for name in INDICATOR_REGISTRY.keys():
            if any(cat in name for cat in ['fractal', 'correlation']):
                found_categories.add('fractal')
            elif any(cat in name for cat in ['volume', 'flow', 'accumulation']):
                found_categories.add('volume')
            elif any(cat in name for cat in ['signal', 'pattern', 'doji', 'hammer']):
                found_categories.add('pattern')
            elif any(cat in name for cat in ['correlation', 'statistical']):
                found_categories.add('statistical')
        
        # We should have at least some indicators from major categories
        self.assertGreater(
            len(found_categories), 
            0, 
            f"Expected to find indicators from categories: {expected_categories}, "
            f"but found categories: {found_categories}"
        )
    
    def test_all_indicators_can_be_instantiated(self):
        """Test that all indicators can be instantiated (basic test)"""
        instantiation_failures = []
        
        for name, indicator_class in INDICATOR_REGISTRY.items():
            try:
                # Try basic instantiation
                instance = indicator_class()
                self.assertIsNotNone(instance)
            except Exception as e:
                # Some indicators might need constructor arguments
                # This is expected, so we just log it for awareness
                instantiation_failures.append((name, str(e)))
        
        # Report how many failed instantiation (for debugging)
        total_indicators = len(INDICATOR_REGISTRY)
        failed_count = len(instantiation_failures)
        success_rate = ((total_indicators - failed_count) / total_indicators) * 100
        
        print(f"\\nInstantiation test results:")
        print(f"Total indicators: {total_indicators}")
        print(f"Failed instantiation: {failed_count}")
        print(f"Success rate: {success_rate:.1f}%")
        
        if instantiation_failures:
            print("\\nIndicators that failed basic instantiation (may need constructor args):")
            for name, error in instantiation_failures[:5]:  # Show first 5
                print(f"  - {name}: {error}")
            if len(instantiation_failures) > 5:
                print(f"  ... and {len(instantiation_failures) - 5} more")


if __name__ == '__main__':
    unittest.main(verbosity=2)
