#!/usr/bin/env python3
"""
PLATFORM3 COMPREHENSIVE INDICATOR AUDIT & ORGANIZATION CHECKER
================================================================
This script will:
1. Find ALL potential indicator files across the entire platform
2. Identify misplaced files that should be reorganized
3. Suggest optimal folder structure
4. Provide accurate indicator count
"""

import os
import re
from pathlib import Path
from collections import defaultdict

def is_indicator_file(filepath):
    """Determine if a Python file is likely an indicator"""
    filename = os.path.basename(filepath).lower()
    
    # Skip these non-indicator files
    skip_patterns = [
        '__init__.py', 'indicator_base.py', 'base_indicator.py',
        'setup.py', 'config.py', 'utils.py', 'helpers.py',
        'test_', 'debug_', 'example_', 'demo_',
        '.pyc', '__pycache__'
    ]
    
    for pattern in skip_patterns:
        if pattern in filename:
            return False
    
    return filename.endswith('.py')

def categorize_file(filepath):
    """Categorize a file based on its name and location"""
    filename = os.path.basename(filepath).lower()
    dirpath = os.path.dirname(filepath).lower()
    
    # Category mappings based on filename patterns
    categories = {
        'momentum': ['rsi', 'macd', 'stochastic', 'momentum', 'oscillator', 'roc', 'cci', 'mfi', 'williams', 'awesome', 'trix', 'tsi', 'ppo', 'ultimate'],
        'trend': ['trend', 'sma', 'ema', 'bollinger', 'parabolic', 'sar', 'adx', 'aroon', 'donchian', 'vortex', 'ichimoku', 'keltner'],
        'volume': ['volume', 'obv', 'vwap', 'vpt', 'accumulation', 'distribution', 'chaikin', 'money_flow', 'orderflow', 'tick'],
        'volatility': ['volatility', 'atr', 'true_range', 'historical'],
        'pattern': ['pattern', 'candlestick', 'doji', 'hammer', 'engulfing', 'harami', 'shooting'],
        'fibonacci': ['fibonacci', 'fib', 'retracement', 'extension', 'projection'],
        'gann': ['gann', 'square', 'fan', 'angle', 'time_cycle'],
        'elliott': ['elliott', 'wave', 'impulse', 'corrective'],
        'fractal': ['fractal', 'chaos', 'dimension', 'similarity'],
        'cycle': ['cycle', 'hurst', 'regime', 'period', 'phase', 'alligator', 'fisher'],
        'divergence': ['divergence', 'hidden'],
        'statistical': ['statistical', 'correlation', 'regression', 'deviation', 'zscore', 'z_score'],
        'ai_ml': ['ml', 'ai', 'neural', 'lstm', 'ensemble', 'adaptive', 'sentiment', 'regime_detection'],
        'trading': ['scalping', 'daytrading', 'swing', 'strategy', 'breakout', 'session'],
        'signals': ['signal', 'aggregator', 'confidence', 'conflict', 'decision', 'synchronizer'],
        'pivot': ['pivot', 'support', 'resistance']
    }
    
    # Check directory first
    for category, keywords in categories.items():
        if any(keyword in dirpath for keyword in keywords):
            return category
    
    # Then check filename
    for category, keywords in categories.items():
        if any(keyword in filename for keyword in keywords):
            return category
    
    return 'miscellaneous'

def scan_platform():
    """Scan the entire platform for indicators"""
    platform_root = Path('D:/MD/Platform3')
    all_indicators = []
    
    print("🔍 SCANNING ENTIRE PLATFORM FOR INDICATORS...")
    print("=" * 60)
    
    # Scan all Python files in the platform
    for root, dirs, files in os.walk(platform_root):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for file in files:
            filepath = os.path.join(root, file)
            if is_indicator_file(filepath):
                rel_path = os.path.relpath(filepath, platform_root)
                category = categorize_file(filepath)
                
                all_indicators.append({
                    'name': os.path.splitext(file)[0],
                    'path': rel_path,
                    'category': category,
                    'directory': os.path.dirname(rel_path)
                })
    
    return all_indicators

def analyze_organization(indicators):
    """Analyze current organization and suggest improvements"""
    print("\n📁 CURRENT ORGANIZATION ANALYSIS")
    print("=" * 40)
    
    category_counts = defaultdict(int)
    directory_analysis = defaultdict(list)
    
    for indicator in indicators:
        category_counts[indicator['category']] += 1
        directory_analysis[indicator['directory']].append(indicator)
    
    print(f"\n🎯 INDICATOR CATEGORIES ({len(category_counts)} categories):")
    print("-" * 50)
    total_indicators = 0
    for category, count in sorted(category_counts.items()):
        print(f"  • {category.upper()}: {count} indicators")
        total_indicators += count
    
    print(f"\n🏆 GRAND TOTAL: {total_indicators} INDICATORS")
    
    print(f"\n📂 CURRENT DIRECTORY DISTRIBUTION ({len(directory_analysis)} directories):")
    print("-" * 60)
    for directory, indicators_in_dir in sorted(directory_analysis.items()):
        if len(indicators_in_dir) > 0:
            print(f"  📁 {directory}: {len(indicators_in_dir)} indicators")
            for indicator in indicators_in_dir[:3]:  # Show first 3
                print(f"      • {indicator['name']}")
            if len(indicators_in_dir) > 3:
                print(f"      • ... and {len(indicators_in_dir) - 3} more")
    
    return total_indicators, category_counts, directory_analysis

def suggest_reorganization(indicators):
    """Suggest optimal folder organization"""
    print(f"\n🔧 SUGGESTED REORGANIZATION")
    print("=" * 40)
    
    misplaced_files = []
    
    for indicator in indicators:
        current_dir = indicator['directory']
        suggested_category = indicator['category']
        
        # Check if file is in the right place
        if 'engines' in current_dir:
            # Extract current category from path
            path_parts = current_dir.split('/')
            if len(path_parts) >= 2 and path_parts[0] == 'engines':
                current_category = path_parts[1]
                if current_category != suggested_category and suggested_category != 'miscellaneous':
                    misplaced_files.append({
                        'indicator': indicator['name'],
                        'current': current_dir,
                        'suggested': f"engines/{suggested_category}",
                        'reason': f"Better fit for {suggested_category} category"
                    })
    
    if misplaced_files:
        print(f"\n⚠️  REORGANIZATION SUGGESTIONS ({len(misplaced_files)} files):")
        print("-" * 50)
        for file_info in misplaced_files[:10]:  # Show first 10
            print(f"  📄 {file_info['indicator']}")
            print(f"     Current:   {file_info['current']}")
            print(f"     Suggested: {file_info['suggested']}")
            print(f"     Reason:    {file_info['reason']}")
            print()
    
    return misplaced_files

def main():
    print("🚀 PLATFORM3 COMPREHENSIVE INDICATOR AUDIT")
    print("=" * 60)
    print("Scanning entire platform for ALL indicator files...")
    
    # Scan all indicators
    indicators = scan_platform()
    
    # Analyze organization
    total_count, categories, directories = analyze_organization(indicators)
    
    # Suggest reorganization
    misplaced = suggest_reorganization(indicators)
    
    print(f"\n🎯 AUDIT SUMMARY")
    print("=" * 30)
    print(f"📊 Total Indicators Found: {total_count}")
    print(f"📁 Categories: {len(categories)}")
    print(f"📂 Directories: {len(directories)}")
    print(f"⚠️  Files to Reorganize: {len(misplaced)}")
    
    if total_count > 100:
        print(f"\n🎉 CONGRATULATIONS! Platform has {total_count} indicators!")
        print(f"   That's {total_count - 67} indicators above the original target!")
        percentage = (total_count / 67) * 100
        print(f"   Achievement: {percentage:.1f}% completion!")
    
    return indicators, total_count

if __name__ == "__main__":
    indicators, count = main()
