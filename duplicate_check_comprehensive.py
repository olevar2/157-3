"""
Duplicate Indicator Checker
Systematically checks for duplicates between implemented indicators and the 157-indicator target list.
"""

import os
import re
from pathlib import Path

def get_implemented_indicators():
    """Get all implemented indicators from the codebase."""
    indicators_path = Path(__file__).parent / "engines" / "ai_enhancement" / "indicators"
    implemented = []
    
    # Recursively find all Python files
    for py_file in indicators_path.rglob("*.py"):
        if py_file.name.startswith("test_") or py_file.name == "__init__.py" or py_file.name == "base_indicator.py":
            continue
        
        # Get relative path and extract indicator name
        rel_path = py_file.relative_to(indicators_path)
        category = rel_path.parent.name if rel_path.parent.name != "." else "root"
        filename = py_file.stem
        
        implemented.append({
            'file': str(py_file),
            'category': category,
            'name': filename,
            'class_name': filename_to_class_name(filename)
        })
    
    return implemented

def filename_to_class_name(filename):
    """Convert filename to likely class name."""
    # Convert snake_case to PascalCase
    parts = filename.split('_')
    return ''.join(word.capitalize() for word in parts)

def load_target_list():
    """Load the 157 indicator target list."""
    target_file = Path(__file__).parent / "all_157_indicators.txt"
    if not target_file.exists():
        print(f"Target file not found: {target_file}")
        return []
    
    with open(target_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def find_potential_matches(implemented, target_list):
    """Find potential matches between implemented and target indicators."""
    matches = []
    missing = []
    
    # Create lookup sets for fuzzy matching
    implemented_names = set()
    implemented_class_names = set()
    
    for impl in implemented:
        implemented_names.add(impl['name'].lower())
        implemented_class_names.add(impl['class_name'].lower())
    
    for target in target_list:
        target_lower = target.lower()
        
        # Direct match
        if target_lower in implemented_class_names:
            matches.append(f"EXACT MATCH: {target}")
            continue
        
        # Enhanced fuzzy matching for common naming patterns
        found_match = False
        
        # Check for partial matches with improved logic
        for impl in implemented:
            impl_name = impl['name'].lower()
            impl_class = impl['class_name'].lower()
            
            # Enhanced matching patterns
            match_found = False
            match_type = ""
            
            # Pattern 1: Direct substring matches
            if target_lower in impl_name or impl_name in target_lower:
                match_found = True
                match_type = "name-substring"
            elif target_lower in impl_class or impl_class in target_lower:
                match_found = True
                match_type = "class-substring"
            
            # Pattern 2: Handle common naming variations
            # Remove common suffixes/prefixes for better matching
            target_clean = target_lower.replace('signal', '').replace('type', '').replace('indicator', '')
            impl_clean = impl_name.replace('_signal', '').replace('_pattern', '').replace('_indicator', '')
            
            if target_clean in impl_clean or impl_clean in target_clean:
                match_found = True
                match_type = "cleaned-match"
            
            # Pattern 3: Handle compound words
            # Split camelCase and check parts
            target_parts = re.findall(r'[A-Z][a-z]*', target)
            target_parts_lower = [p.lower() for p in target_parts]
            
            impl_parts = impl_name.split('_')
            
            # Check if most parts match
            if len(target_parts_lower) > 1 and len(impl_parts) > 1:
                common_parts = set(target_parts_lower) & set(impl_parts)
                if len(common_parts) >= min(2, max(1, len(target_parts_lower) - 1)):
                    match_found = True
                    match_type = "parts-match"
            
            # Pattern 4: Special cases for known variations
            special_cases = {
                'darkcloudtype': 'dark_cloud',
                'engulfingtype': 'engulfing',
                'hammertype': 'hammer',
                'dojitype': 'doji',
                'haramitype': 'harami',
                'beltholdtype': 'belt_hold',
                'marubozu': 'marubozu',
                'kickingsignal': 'kicking',
                'threeinsignal': 'three_inside'
            }
            
            target_key = target_lower.replace('_', '')
            impl_key = impl_name.replace('_', '')
            
            if target_key in special_cases and special_cases[target_key] in impl_name:
                match_found = True
                match_type = "special-case"
            
            if match_found:
                matches.append(f"MATCH ({match_type}): {target} <-> {impl['name']} ({impl['category']})")
                found_match = True
                break
        
        if not found_match:
            missing.append(target)
    
    return matches, missing

def check_for_duplicates():
    """Main function to check for duplicates."""
    print("Duplicate Indicator Checker")
    print("=" * 50)
    
    # Get implemented indicators
    implemented = get_implemented_indicators()
    print(f"Found {len(implemented)} implemented indicators")
    
    # Load target list
    target_list = load_target_list()
    print(f"Target list has {len(target_list)} indicators")
    print()
    
    # Find matches and missing
    matches, missing = find_potential_matches(implemented, target_list)
    
    print("MATCHES FOUND:")
    print("-" * 30)
    for match in matches[:20]:  # Show first 20 matches
        print(match)
    if len(matches) > 20:
        print(f"... and {len(matches) - 20} more matches")
    print()
    
    print("MISSING INDICATORS:")
    print("-" * 30)
    for missing_indicator in missing[:20]:  # Show first 20 missing
        print(f"MISSING: {missing_indicator}")
    if len(missing) > 20:
        print(f"... and {len(missing) - 20} more missing indicators")
    print()
    
    print("SUMMARY:")
    print(f"Total implemented: {len(implemented)}")
    print(f"Total target: {len(target_list)}")
    print(f"Matches found: {len(matches)}")
    print(f"Still missing: {len(missing)}")
    print(f"Coverage: {len(matches)/len(target_list)*100:.1f}%")
    
    # Show recently implemented indicators (today's work)
    print("\nRECENTLY IMPLEMENTED (Today's Work):")
    print("-" * 40)
    recent_patterns = ['kicking_signal', 'three_inside_signal', 'engulfing_pattern', 
                      'hammer_pattern', 'belt_hold_type', 'harami_type', 'abandoned_baby_signal']
    for impl in implemented:
        if impl['name'] in recent_patterns:
            print(f"RECENT: {impl['name']} ({impl['category']})")

if __name__ == "__main__":
    check_for_duplicates()