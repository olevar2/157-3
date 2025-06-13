#!/usr/bin/env python3
"""
Fix Indicator Registry Mapping

This script identifies all existing indicator files and ensures they are properly mapped
in the registry with the correct naming conventions. It will not create any new files,
only fix the registry loading logic.

Author: Platform3 AI Enhancement Engine
Date: 2025-06-11
"""

import os
import re
import importlib.util
from pathlib import Path
from typing import Dict, List, Set

def find_all_indicator_files():
    """Find all indicator Python files in the indicators directory"""
    indicators_dir = Path("engines/ai_enhancement/indicators")
    if not indicators_dir.exists():
        print(f"Error: Indicators directory not found: {indicators_dir}")
        return []
    
    py_files = []
    for root, dirs, files in os.walk(indicators_dir):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for file in files:
            if file.endswith('.py') and not file.startswith('__') and not file.startswith('test_'):
                full_path = os.path.join(root, file)
                py_files.append(full_path)
    
    return py_files

def extract_class_names_from_file(file_path: str) -> List[str]:
    """Extract class names from a Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find class definitions
        class_pattern = r'class\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?:'
        matches = re.findall(class_pattern, content)
        
        # Filter out common base classes and utility classes
        filtered_classes = []
        exclude_patterns = [
            'BaseIndicator', 'StandardIndicatorInterface', 'IndicatorConfig',
            'TestCase', 'Test', 'Mock', 'Stub'
        ]
        
        for class_name in matches:
            if not any(pattern in class_name for pattern in exclude_patterns):
                filtered_classes.append(class_name)
        
        return filtered_classes
    
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

def create_mapping_dict():
    """Create a comprehensive mapping of all indicator files and their classes"""
    files = find_all_indicator_files()
    mapping = {}
    
    print(f"Found {len(files)} indicator files to analyze...")
    
    for file_path in files:
        relative_path = file_path.replace('\\', '/').replace('engines/ai_enhancement/indicators/', '')
        classes = extract_class_names_from_file(file_path)
        
        if classes:
            mapping[relative_path] = {
                'file_path': file_path,
                'classes': classes,
                'module_path': file_path.replace('\\', '.').replace('/', '.').replace('.py', '').replace('engines.', '')
            }
    
    return mapping

def identify_missing_vs_existing():
    """Identify which indicators are missing vs which exist but aren't loaded"""
    
    # Expected indicators from the documentation
    expected_indicators = [
        'movingaverageconvergencedivergenceindicator',
        'pricevolumetrend', 
        'relativestrengthindexindicator',
        'selfsimilaritysignal',
        'stochasticindicator',
        'threeInsidesignal',
        'threelineStrikesignal', 
        'threeoutsidesignal',
        'trixindicator',
        'truestrengthindexindicator',
        'ultimateoscillatorindicator',
        'volumeoscillator',
        'volumerateofchange',
        'vortexindicator',
        'williamsrindicator'
    ]
    
    # Get all existing files and classes
    mapping = create_mapping_dict()
    
    print("\n=== EXISTING FILES AND CLASSES ===")
    existing_classes = set()
    for file_info in mapping.values():
        for class_name in file_info['classes']:
            existing_classes.add(class_name.lower())
            print(f"Class: {class_name} -> {class_name.lower()}")
    
    print(f"\nTotal existing classes: {len(existing_classes)}")
    
    print("\n=== CHECKING EXPECTED INDICATORS ===")
    found_count = 0
    not_found = []
    
    for expected in expected_indicators:
        # Try various name patterns
        patterns_to_check = [
            expected,
            expected.replace('indicator', ''),
            expected.replace('signal', ''),
            expected + 'indicator',
            expected + 'signal'
        ]
        
        found = False
        for pattern in patterns_to_check:
            if pattern in existing_classes:
                print(f"[FOUND] {expected} matches existing class pattern: {pattern}")
                found_count += 1
                found = True
                break
        
        if not found:
            not_found.append(expected)
            print(f"[NOT FOUND] {expected}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Expected indicators: {len(expected_indicators)}")
    print(f"Found: {found_count}")
    print(f"Not found: {len(not_found)}")
    
    if not_found:
        print(f"\nIndicators that may need aliases or file creation:")
        for indicator in not_found:
            print(f"  - {indicator}")
    
    return mapping, not_found

def suggest_fixes():
    """Suggest fixes for the registry mapping"""
    mapping, not_found = identify_missing_vs_existing()
    
    print(f"\n=== SUGGESTED REGISTRY FIXES ===")
    
    # Suggest alias mappings for common patterns
    alias_suggestions = {
        'movingaverageconvergencedivergenceindicator': 'movingaverageconvergencedivergence',
        'relativestrengthindexindicator': 'relativestrengthindex', 
        'stochasticindicator': 'stochasticoscillator',
        'threeInsidesignal': 'standardindicatorinterface',  # This might be loaded as generic interface
        'threelineStrikesignal': 'standardindicatorinterface',
    }
    
    for missing, suggested_alias in alias_suggestions.items():
        print(f"ADD ALIAS: {missing} -> {suggested_alias}")
    
    print(f"\nThe registry loading mechanism should be updated to handle these naming variations.")

if __name__ == "__main__":
    print("=== Platform3 Indicator Registry Mapping Analysis ===")
    suggest_fixes()