#!/usr/bin/env python3
"""
Debug script to test ComprehensiveIndicatorAdapter_67 step by step
"""

import sys
import os
import traceback

def test_imports():
    """Test individual imports"""
    print("Testing imports...")
    
    try:
        print("1. Testing basic imports...")
        import numpy as np
        from typing import Dict, List, Optional, Union, Any, Tuple
        from dataclasses import dataclass
        from enum import Enum
        import logging
        import time
        print("   ✓ Basic imports successful")
        
        print("2. Testing swingtrading import...")
        import swingtrading
        print("   ✓ swingtrading import successful")
        
        print("3. Testing adapter classes import...")
        from ComprehensiveIndicatorAdapter_67 import IndicatorCategory, MarketData
        print("   ✓ Adapter classes import successful")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_adapter_creation():
    """Test adapter creation step by step"""
    print("\nTesting adapter creation...")
    
    try:
        print("1. Importing adapter class...")
        from ComprehensiveIndicatorAdapter_67 import ComprehensiveIndicatorAdapter_67
        print("   ✓ Adapter class imported")
        
        print("2. Creating adapter instance...")
        # This is where it might hang
        adapter = ComprehensiveIndicatorAdapter_67()
        print("   ✓ Adapter instance created")
        
        print("3. Testing basic methods...")
        count = len(adapter.get_all_indicator_names())
        print(f"   ✓ Found {count} indicators")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Adapter creation failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=== ComprehensiveIndicatorAdapter_67 Debug Test ===")
    
    # Test imports first
    if not test_imports():
        print("Import test failed, stopping.")
        return
    
    # Test adapter creation
    if not test_adapter_creation():
        print("Adapter creation test failed, stopping.")
        return
    
    print("\n=== All tests passed! ===")

if __name__ == "__main__":
    main()
