# -*- coding: utf-8 -*-
"""
Platform3 Simple Indicator Mapper
=================================

Quick mapping system to resolve import path issues.
"""

import os
import ast
import sys
from pathlib import Path

# Add the engines directory to the path
engines_path = Path(__file__).parent.parent
sys.path.insert(0, str(engines_path))

def scan_momentum_indicators():
    """Test known working momentum indicators first."""
    print("Testing momentum indicators...")
    momentum_path = engines_path / "momentum"
    
    # Test the known working ones
    working_indicators = ['cci', 'mfi', 'roc', 'williams_r', 'ultimate_oscillator']
    
    for indicator in working_indicators:
        try:
            exec(f"from engines.momentum.{indicator} import *")
            print(f"  SUCCESS: {indicator}")
        except Exception as e:
            print(f"  FAILED: {indicator} - {e}")

def scan_trend_indicators():
    """Scan trend indicators to find working ones."""
    print("\nTesting trend indicators...")
    trend_path = engines_path / "trend"
    
    # Get actual files
    if trend_path.exists():
        py_files = [f.stem for f in trend_path.glob("*.py") 
                   if not f.name.startswith('__') 
                   and not 'backup' in f.name.lower()]
        
        for filename in py_files:
            try:
                # Try basic import
                exec(f"import engines.trend.{filename}")
                print(f"  SUCCESS: {filename}")
            except Exception as e:
                print(f"  FAILED: {filename} - {str(e)[:100]}")

def scan_volume_indicators():
    """Scan volume indicators to identify TickVolumeIndicators issues."""
    print("\nTesting volume indicators...")
    volume_path = engines_path / "volume"
    
    if volume_path.exists():
        py_files = [f.stem for f in volume_path.glob("*.py") 
                   if not f.name.startswith('__') 
                   and not 'backup' in f.name.lower()]
        
        for filename in py_files:
            try:
                exec(f"import engines.volume.{filename}")
                print(f"  SUCCESS: {filename}")
            except Exception as e:
                print(f"  FAILED: {filename} - {str(e)[:100]}")

def main():
    print("Platform3 Quick Indicator Mapping")
    print("=" * 40)
    
    scan_momentum_indicators()
    scan_trend_indicators() 
    scan_volume_indicators()
    
    print("\n" + "=" * 40)
    print("Quick scan complete!")

if __name__ == "__main__":
    main()