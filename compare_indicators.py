"""
Compare available indicator files vs registered indicators
"""
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def get_registered_indicators():
    """Get indicators currently registered in the registry"""
    try:
        from engines.ai_enhancement.registry import INDICATOR_REGISTRY
        return set(INDICATOR_REGISTRY.keys())
    except Exception as e:
        print(f"Error importing registry: {e}")
        return set()

def get_available_indicator_files():
    """Get all available indicator files"""
    base_dir = Path(__file__).parent
    indicator_dirs = [
        "engines/ai_enhancement/indicators",
        "engines/pattern", 
        "engines/trend",
        "engines/volume",
        "engines/momentum"
    ]
    
    available = set()
    
    for indicator_dir in indicator_dirs:
        full_path = base_dir / indicator_dir
        if full_path.exists():
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
                
                # Check if it's actually an indicator
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    if ("class " in content and 
                        ("calculate" in content or "compute" in content) and
                        ("def " in content)):
                        
                        indicator_name = filename.replace(".py", "").lower()
                        available.add(indicator_name)
                        
                except Exception:
                    continue
    
    return available

def main():
    print("COMPARING AVAILABLE VS REGISTERED INDICATORS")
    print("=" * 60)
    
    registered = get_registered_indicators()
    available = get_available_indicator_files()
    
    print(f"Available indicator files: {len(available)}")
    print(f"Registered indicators: {len(registered)}")
    
    # Find missing indicators (available but not registered)
    missing = available - registered
    print(f"Missing indicators (available but not registered): {len(missing)}")
    
    # Find registered but not available
    orphaned = registered - available  
    print(f"Registered but no file found: {len(orphaned)}")
    
    print("\n" + "=" * 60)
    print("MISSING INDICATORS (Available files but not registered):")
    print("=" * 60)
    for i, indicator in enumerate(sorted(missing), 1):
        print(f"{i:3d}. {indicator}")
    
    print("\n" + "=" * 60)
    print("REGISTERED BUT NO FILE FOUND:")
    print("=" * 60)
    for i, indicator in enumerate(sorted(orphaned), 1):
        print(f"{i:3d}. {indicator}")
    
    print(f"\n" + "=" * 60)
    print("SUMMARY:")
    print(f"We need to register {len(missing)} additional indicators")
    print(f"We should remove {len(orphaned)} orphaned registrations")
    print(f"Target: 167 indicators")
    print(f"Current registered: {len(registered)}")
    print(f"After fixes: {len(registered) - len(orphaned) + min(len(missing), 167 - len(registered) + len(orphaned))}")

if __name__ == "__main__":
    main()