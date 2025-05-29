#!/usr/bin/env python3
"""
Comprehensive Indicator Counter for Platform3
Counts all indicators across all engine categories
"""

import os
from pathlib import Path

def count_indicators():
    engines_path = Path("D:/MD/Platform3/engines")
    
    # Categories to check
    categories = {}
    grand_total = 0
    
    print("PLATFORM3 COMPREHENSIVE INDICATOR COUNT")
    print("=" * 50)
    
    # Walk through all directories in engines
    for category_dir in engines_path.iterdir():
        if category_dir.is_dir() and category_dir.name != "__pycache__":
            category_name = category_dir.name
            indicator_count = 0
            indicators = []
            
            # Count Python files (excluding __init__.py and indicator_base.py)
            for py_file in category_dir.rglob("*.py"):
                if py_file.name not in ["__init__.py", "indicator_base.py"]:
                    indicator_count += 1
                    indicators.append(py_file.stem)
            
            if indicator_count > 0:
                categories[category_name] = {
                    'count': indicator_count,
                    'indicators': indicators
                }
                grand_total += indicator_count
                
                print(f"\n{category_name.upper()} ({indicator_count} indicators):")
                print("-" * 40)
                for indicator in sorted(indicators):
                    print(f"  • {indicator}")
    
    print("\n" + "=" * 50)
    print("CATEGORY SUMMARY:")
    print("=" * 30)
    
    for category, data in sorted(categories.items()):
        print(f"{category}: {data['count']} indicators")
    
    print(f"\nGRAND TOTAL: {grand_total} indicators")
    print("=" * 50)
    
    return grand_total, categories

if __name__ == "__main__":
    total, cats = count_indicators()
