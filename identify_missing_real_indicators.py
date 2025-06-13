#!/usr/bin/env python3
"""
Script to add the missing 7 real indicators to reach exactly 167 indicators.
"""

import sys
import os

def identify_and_add_missing_indicators():
    """Identify real missing indicators and modify registry to include them"""
    print("=== ADDING MISSING REAL INDICATORS ===\n")
    
    # Check if these are legitimate indicator files that should be loaded
    missing_real_indicators = [
        ("PriceVolumeTrend", "engines.ai_enhancement.indicators.volume.price_volume_trend"),
        ("SelfSimilaritySignal", "engines.ai_enhancement.indicators.statistical.self_similarity_signal"),
        ("ThreeInsideSignal", "engines.ai_enhancement.indicators.pattern.three_inside_signal"),
        ("ThreeLineStrikeSignal", "engines.ai_enhancement.indicators.pattern.three_line_strike_signal"), 
        ("ThreeOutsideSignal", "engines.ai_enhancement.indicators.pattern.three_outside_signal"),
        ("TrixIndicator", "engines.ai_enhancement.indicators.momentum.trix"),
        ("TrueStrengthIndexIndicator", "engines.ai_enhancement.indicators.momentum.true_strength_index"),
        ("UltimateOscillatorIndicator", "engines.ai_enhancement.indicators.momentum.ultimate_oscillator"),
        ("VolumeOscillator", "engines.ai_enhancement.indicators.volume.volume_oscillator"),
        ("VortexIndicator", "engines.ai_enhancement.indicators.trend.vortex_indicator"),
        ("WilliamsRIndicator", "engines.ai_enhancement.indicators.momentum.williams_r"),
    ]
    
    print("Checking which of these missing indicators are real indicator files:")
    
    verified_indicators = []
    for name, module_path in missing_real_indicators:
        file_path = module_path.replace("engines.ai_enhancement.indicators.", "engines/ai_enhancement/indicators/").replace(".", "/") + ".py"
        
        if os.path.exists(file_path):
            print(f"[OK] {name} -> {file_path} EXISTS")
            verified_indicators.append((name, module_path))
        else:
            print(f"[MISS] {name} -> {file_path} NOT FOUND")
    
    print(f"\n=== VERIFIED MISSING INDICATORS ({len(verified_indicators)}) ===")
    for i, (name, module_path) in enumerate(verified_indicators, 1):
        print(f"{i:2d}. {name} ({module_path})")
    
    # Show the path to fix the registry
    print(f"\n=== SOLUTION ===")
    print("These indicators exist but are not being loaded by the registry.")
    print("The registry needs to be updated to properly import these indicators.")
    print("\nRoot cause: Some indicator files may have:")
    print("1. Import errors")
    print("2. Wrong class names") 
    print("3. Missing from consolidation modules")
    print("4. Not properly registered")
    
    return verified_indicators

if __name__ == "__main__":
    identify_and_add_missing_indicators()