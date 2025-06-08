#!/usr/bin/env python3
import os
import re
from collections import Counter

print("=== Platform3 Indicator Registry Duplicate Check ===")

try:
    # Read the registry file directly
    with open('engines/ai_enhancement/registry.py', 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Find all INDICATOR_REGISTRY.update() calls and extract the keys
    registry_updates = []
    
    # Pattern to match INDICATOR_REGISTRY.update({...})
    update_pattern = r'INDICATOR_REGISTRY\.update\(\{([^}]+)\}\)'
    update_matches = re.findall(update_pattern, content, re.DOTALL)
    
    all_indicators = []
    
    for match in update_matches:
        # Extract key-value pairs from each update block
        # Pattern to match 'key': Value, 
        key_pattern = r'[\'"]([^\'"]+)[\'"]\s*:\s*\w+'
        keys = re.findall(key_pattern, match)
        all_indicators.extend(keys)
        registry_updates.extend(keys)
    
    # Also look for direct assignments to INDICATOR_REGISTRY
    direct_pattern = r'INDICATOR_REGISTRY\[[\'"]([^\'"]+)[\'"]\]\s*='
    direct_matches = re.findall(direct_pattern, content)
    all_indicators.extend(direct_matches)
    
    if all_indicators:
        print(f"\nFound {len(all_indicators)} indicator registrations:")
        for i, indicator in enumerate(sorted(all_indicators), 1):
            print(f"{i:3d}. {indicator}")
            
        # Check for duplicates
        unique_indicators = set(all_indicators)
        print(f"\nSummary:")
        print(f"Total registrations: {len(all_indicators)}")
        print(f"Unique indicators: {len(unique_indicators)}")
        
        if len(unique_indicators) < len(all_indicators):
            print(f"\n❌ DUPLICATES FOUND!")
            counts = Counter(all_indicators)
            duplicates = [name for name, count in counts.items() if count > 1]
            print("\nDuplicate indicators:")
            for dup in duplicates:
                print(f"  - '{dup}' appears {counts[dup]} times")
                
            print(f"\nTotal duplicate entries: {len(all_indicators) - len(unique_indicators)}")
        else:
            print("\n✅ No duplicates found! All indicator names are unique.")
    else:
        print("❌ No indicator registrations found in registry file.")
        
except FileNotFoundError:
    print("❌ Registry file not found at: engines/ai_enhancement/registry.py")
except Exception as e:
    print(f"❌ Error reading registry file: {e}")

print("\n" + "="*50)
