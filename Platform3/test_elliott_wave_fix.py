#!/usr/bin/env python3
"""
Test script to verify Elliott Wave import fix
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_elliott_wave_import():
    """Test Elliott Wave import"""
    try:
        # Test direct file import
        import sys
        sys.path.append('services/analytics-service/src/engines/swingtrading')
        from ShortTermElliottWaves import (
            ShortTermElliottWaves,
            WaveType,
            WaveDirection,
            WavePoint,
            ElliottWavePattern,
            WaveAnalysisResult
        )
        print("✅ Elliott Wave direct import successful")

        # Test initialization
        elliott_engine = ShortTermElliottWaves()
        print("✅ Elliott Wave engine initialization successful")

        return True

    except Exception as e:
        print(f"❌ Elliott Wave import failed: {e}")
        return False

def test_advanced_indicators_import():
    """Test Advanced Indicators import"""
    try:
        import sys
        sys.path.append('services/analytics-service/src/engines/indicators/advanced')
        from __init__ import AdvancedIndicatorSuite
        print("✅ Advanced indicators import successful")

        # Test initialization
        suite = AdvancedIndicatorSuite()
        print("✅ Advanced indicators initialization successful")

        return True

    except Exception as e:
        print(f"❌ Advanced indicators import failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Elliott Wave and Advanced Indicators imports...")
    print("=" * 60)

    elliott_success = test_elliott_wave_import()
    print()

    advanced_success = test_advanced_indicators_import()
    print()

    if elliott_success and advanced_success:
        print("🎉 All imports successful! The merge conflict has been resolved.")
    else:
        print("⚠️  Some imports still failing. Further investigation needed.")
