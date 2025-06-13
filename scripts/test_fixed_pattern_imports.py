#!/usr/bin/env python3
# Fixed Pattern Indicators Import Test Script

import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent

import traceback

def test_indicator_import(module_path):
    """Test importing a specific indicator module and report any issues"""
    try:
        module_name = module_path.replace('/', '.')
        exec(f"import {module_name}")
        print(f"SUCCESS: {module_path}")
        return True, None
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"FAILED: {module_path}")
        print(f"ERROR: {str(e)}")
        print(f"DETAILS: {error_msg}")
        return False, error_msg

# Test the original pattern indicator imports
original_pattern_files = [
    "engines.pattern.doji_recognition",
    "engines.pattern.engulfing_pattern",
    "engines.pattern.hammer_hanging_man",
    "engines.pattern.harami_pattern"
]

# Test the fixed pattern indicator imports
fixed_pattern_files = [
    "engines.pattern.doji_recognition_fixed",
    "engines.pattern.engulfing_pattern_fixed",
    "engines.pattern.hammer_hanging_man_fixed",
    "engines.pattern.harami_pattern_fixed"
]

results = {}

print("\n===== TESTING ORIGINAL FILES =====")
for pattern_file in original_pattern_files:
    success, error = test_indicator_import(pattern_file)
    results[pattern_file] = {
        "success": success,
        "error": error
    }

print("\n===== TESTING FIXED FILES =====")
for pattern_file in fixed_pattern_files:
    success, error = test_indicator_import(pattern_file)
    results[pattern_file] = {
        "success": success,
        "error": error
    }

# Print summary
print("\n===== IMPORT TEST SUMMARY =====")
print("Original files:")
for pattern_file in original_pattern_files:
    status = "SUCCESS" if results[pattern_file]["success"] else "FAILED"
    print(f"{pattern_file}: {status}")

print("\nFixed files:")
for pattern_file in fixed_pattern_files:
    status = "SUCCESS" if results[pattern_file]["success"] else "FAILED"
    print(f"{pattern_file}: {status}")

# Print detailed errors
print("\n===== DETAILED ERRORS =====")
for pattern_file, result in results.items():
    if not result["success"]:
        print(f"\n{pattern_file}:")
        print(f"{result['error']}")