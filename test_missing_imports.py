#!/usr/bin/env python3
"""
Test importing missing indicators to see why they're not loaded
"""

import sys
import os

def test_import_missing_indicators():
    """Test importing the missing indicators to see import errors"""
    
    missing_indicators = [
        ("PriceVolumeTrend", "engines.ai_enhancement.indicators.volume.price_volume_trend"),
        ("SelfSimilaritySignal", "engines.ai_enhancement.indicators.statistical.self_similarity_signal"),
        ("ThreeInsideSignal", "engines.ai_enhancement.indicators.pattern.three_inside_signal"),
        ("ThreeLineStrikeSignal", "engines.ai_enhancement.indicators.pattern.three_line_strike_signal"),
        ("ThreeOutsideSignal", "engines.ai_enhancement.indicators.pattern.three_outside_signal"),
        ("TrixIndicator", "engines.ai_enhancement.indicators.momentum.trix"),
        ("TrueStrengthIndexIndicator", "engines.ai_enhancement.indicators.momentum.true_strength_index"),
    ]
    
    print("=== TESTING IMPORTS ===\n")
    
    successful_imports = []
    for name, module_path in missing_indicators:
        try:
            module = __import__(module_path, fromlist=[name])
            indicator_class = getattr(module, name)
            print(f"[OK] {name} imported successfully from {module_path}")
            successful_imports.append((name, module_path, indicator_class))
        except Exception as e:
            print(f"[ERROR] {name} failed to import: {e}")
    
    print(f"\n=== RESULTS ===")
    print(f"Successfully imported: {len(successful_imports)}")
    print(f"Failed imports: {len(missing_indicators) - len(successful_imports)}")
    
    if successful_imports:
        print(f"\nReady to add to registry:")
        for name, module_path, cls in successful_imports:
            print(f"  - {name}")

if __name__ == "__main__":
    test_import_missing_indicators()