#!/usr/bin/env python3
"""
ACCURATE INDICATOR COUNTER - ENGINES FOLDER ONLY
=================================================
Count only actual trading indicators in the engines folder
"""

import os
from pathlib import Path
from collections import defaultdict

def is_actual_indicator(filepath):
    """Determine if a Python file is actually a trading indicator"""
    filename = os.path.basename(filepath).lower()
    
    # Skip these files that are NOT indicators
    skip_files = [
        '__init__.py', 'indicator_base.py', 'base_indicator.py',
        '.pyc'
    ]
    
    for skip in skip_files:
        if skip in filename:
            return False
    
    return filename.endswith('.py')

def count_engines_indicators():
    """Count only indicators in the engines folder"""
    engines_path = Path('D:/MD/Platform3/engines')
    
    if not engines_path.exists():
        print("❌ Engines folder not found!")
        return 0, {}
    
    print("🔍 COUNTING ACTUAL INDICATORS IN ENGINES FOLDER")
    print("=" * 55)
    
    category_counts = defaultdict(list)
    total_count = 0
    
    # Walk through engines folder only
    for root, dirs, files in os.walk(engines_path):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for file in files:
            filepath = os.path.join(root, file)
            if is_actual_indicator(filepath):
                # Get category from directory structure
                rel_path = os.path.relpath(root, engines_path)
                category = rel_path.split(os.sep)[0] if rel_path != '.' else 'root'
                
                indicator_name = os.path.splitext(file)[0]
                category_counts[category].append(indicator_name)
                total_count += 1
    
    # Display results
    print(f"\n📊 INDICATOR COUNT BY CATEGORY:")
    print("-" * 40)
    
    for category, indicators in sorted(category_counts.items()):
        print(f"\n🔹 {category.upper()} ({len(indicators)} indicators):")
        for indicator in sorted(indicators):
            print(f"   • {indicator}")
    
    print(f"\n🎯 TOTAL INDICATORS IN ENGINES FOLDER: {total_count}")
    
    return total_count, category_counts

def verify_core_indicators():
    """Verify we have the core indicators we expect"""
    engines_path = Path('D:/MD/Platform3/engines')
    
    print(f"\n🔍 VERIFYING CORE INDICATORS:")
    print("-" * 35)
    
    core_categories = [
        'momentum', 'trend', 'volume', 'volatility', 
        'core_momentum', 'core_trend', 'pattern',
        'gann', 'fibonacci', 'elliott_wave', 'fractal',
        'cycle', 'divergence', 'statistical', 'ai_enhancement'
    ]
    
    found_categories = []
    missing_categories = []
    
    for category in core_categories:
        category_path = engines_path / category
        if category_path.exists():
            indicator_count = len([f for f in category_path.glob('*.py') 
                                 if f.name != '__init__.py' and not f.name.startswith('.')])
            found_categories.append((category, indicator_count))
            print(f"✅ {category}: {indicator_count} indicators")
        else:
            missing_categories.append(category)
            print(f"❌ {category}: NOT FOUND")
    
    if missing_categories:
        print(f"\n⚠️  Missing categories: {', '.join(missing_categories)}")
    
    return found_categories, missing_categories

def main():
    print("🚀 ACCURATE PLATFORM3 INDICATOR COUNT")
    print("=" * 45)
    print("Counting ONLY actual trading indicators in engines folder...")
    
    # Count indicators
    total, categories = count_engines_indicators()
    
    # Verify core categories
    found, missing = verify_core_indicators()
    
    print(f"\n📈 FINAL SUMMARY")
    print("=" * 25)
    print(f"📊 Total Categories: {len(categories)}")
    print(f"🎯 Total Indicators: {total}")
    
    if total > 67:
        percentage = (total / 67) * 100
        print(f"🎉 Achievement: {percentage:.1f}% of original target!")
        print(f"🏆 Bonus indicators: {total - 67}")
    
    return total, categories

if __name__ == "__main__":
    total, categories = main()
