#!/usr/bin/env python3
import os
import re
from collections import Counter

print("=== Registry Duplicate Check ===")
print("Checking registry file directly without importing modules...")

try:
    # Read the registry file directly
    with open('engines/ai_enhancement/registry.py', 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Look for indicator registrations
    # Pattern to match: register_indicator('name', ...)
    pattern = r'register_indicator\([\'"]([^\'"]+)[\'"]'
    matches = re.findall(pattern, content)
    
    if matches:
        print(f"\nFound {len(matches)} indicator registrations:")
        for i, match in enumerate(matches, 1):
            print(f"{i:3d}. {match}")
            
        # Check for duplicates
        unique_indicators = set(matches)
        print(f"\nTotal registrations: {len(matches)}")
        print(f"Unique indicators: {len(unique_indicators)}")
        
        if len(unique_indicators) < len(matches):
            print(f"\n=== DUPLICATES FOUND! ===")
            counts = Counter(matches)
            duplicates = [name for name, count in counts.items() if count > 1]
            print("Duplicate indicators:")
            for dup in duplicates:
                print(f"  - '{dup}' appears {counts[dup]} times")
        else:
            print("\n✓ No duplicates found in registrations.")
    else:
        print("No indicator registrations found in registry file.")
        
    # Also look for INDICATOR_REGISTRY dictionary entries
    dict_pattern = r'[\'"]([^\'"]+)[\'"]\s*:\s*\w+'
    dict_matches = re.findall(dict_pattern, content)
    
    if dict_matches:
        print(f"\nFound {len(dict_matches)} dictionary entries:")
        dict_counts = Counter(dict_matches)
        dict_duplicates = [name for name, count in dict_counts.items() if count > 1]
        
        if dict_duplicates:
            print("Duplicate dictionary entries:")
            for dup in dict_duplicates:
                print(f"  - '{dup}' appears {dict_counts[dup]} times")
        else:
            print("✓ No duplicates found in dictionary entries.")
            
except FileNotFoundError:
    print("Registry file not found at: engines/ai_enhancement/registry.py")
except Exception as e:
    print(f"Error reading registry file: {e}")

print("\nDone!")
