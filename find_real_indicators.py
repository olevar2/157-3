"""
Find REAL indicator implementation files (not __init__.py, not test files, not utilities)
"""
import os
from pathlib import Path

def find_real_indicator_files():
    """Find actual indicator implementation files"""
    base_dir = Path(__file__).parent
    indicator_dirs = [
        "engines/ai_enhancement/indicators",
        "engines/pattern", 
        "engines/trend",
        "engines/volume",
        "engines/momentum"
    ]
    
    real_indicators = []
    
    for indicator_dir in indicator_dirs:
        full_path = base_dir / indicator_dir
        if full_path.exists():
            print(f"\n=== Checking {indicator_dir} ===")
            for file in full_path.rglob("*.py"):
                filename = file.name
                
                # Skip non-indicator files
                if (filename == "__init__.py" or 
                    filename.startswith("test_") or
                    filename.startswith("_") or
                    "test" in filename.lower() or
                    "util" in filename.lower() or
                    "helper" in filename.lower() or
                    "base" in filename.lower()):
                    continue
                
                # Check if it's actually an indicator by looking for key patterns
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Look for indicator patterns
                    if ("class " in content and 
                        ("calculate" in content or "compute" in content) and
                        ("def " in content)):
                        
                        indicator_name = filename.replace(".py", "")
                        relative_path = str(file.relative_to(base_dir))
                        real_indicators.append({
                            'name': indicator_name,
                            'file': relative_path,
                            'directory': indicator_dir
                        })
                        print(f"  [OK] {indicator_name}")
                        
                except Exception as e:
                    print(f"  [ERROR] Error reading {filename}: {e}")
    
    return real_indicators

if __name__ == "__main__":
    print("FINDING REAL INDICATOR IMPLEMENTATION FILES")
    print("=" * 60)
    
    indicators = find_real_indicator_files()
    
    print(f"\n" + "=" * 60)
    print(f"SUMMARY:")
    print(f"Found {len(indicators)} real indicator implementation files")
    
    # Group by directory
    by_dir = {}
    for ind in indicators:
        dir_name = ind['directory']
        if dir_name not in by_dir:
            by_dir[dir_name] = []
        by_dir[dir_name].append(ind['name'])
    
    print("\nBy directory:")
    for dir_name, ind_list in by_dir.items():
        print(f"  {dir_name}: {len(ind_list)} indicators")
    
    print(f"\nThis is much more reasonable than 1045!")