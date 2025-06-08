import os
from pathlib import Path

def count_indicator_files():
    indicator_count = 0
    categories = [
        'momentum', 'trend', 'volume', 'volatility', 'pattern',
        'statistical', 'fractal', 'elliott_wave', 'gann',
        'fibonacci', 'cycle', 'divergence'
    ]
    
    engines_dir = Path("D:/MD/Platform3/engines")
    
    # Count indicators in each category
    for category in categories:
        category_dir = engines_dir / category
        if category_dir.exists():
            indicator_files = [f for f in category_dir.glob("*.py") 
                              if not f.name.startswith("__") and not "backup" in f.name]
            print(f"{category}: {len(indicator_files)} indicators")
            indicator_count += len(indicator_files)
    
    # Also check for standalone indicators in engines root
    if engines_dir.exists():
        root_indicators = [f for f in engines_dir.glob("*.py")
                          if not f.name.startswith("__") and 
                          "indicator" in f.name.lower() and
                          not "base" in f.name.lower() and
                          not "backup" in f.name]
        print(f"root: {len(root_indicators)} indicators")
        indicator_count += len(root_indicators)
    
    return indicator_count

if __name__ == "__main__":
    count = count_indicator_files()
    print(f"\nTotal number of indicator files in engines directory: {count}")