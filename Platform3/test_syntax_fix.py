#!/usr/bin/env python3
"""
Test script to verify the Elliott Wave syntax fix
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_syntax_compilation():
    """Test that the files compile without syntax errors"""
    import py_compile
    
    try:
        # Test Elliott Wave file compilation
        elliott_file = 'services/analytics-service/src/engines/swingtrading/ShortTermElliottWaves.py'
        py_compile.compile(elliott_file, doraise=True)
        print("✅ Elliott Wave file compiles successfully")
        
        # Test Advanced Indicators file compilation
        advanced_file = 'services/analytics-service/src/engines/indicators/advanced/__init__.py'
        py_compile.compile(advanced_file, doraise=True)
        print("✅ Advanced Indicators file compiles successfully")
        
        return True
        
    except py_compile.PyCompileError as e:
        print(f"❌ Compilation failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_elliott_wave_import():
    """Test Elliott Wave import"""
    try:
        sys.path.append('services/analytics-service/src/engines/swingtrading')
        from ShortTermElliottWaves import ShortTermElliottWaves
        print("✅ Elliott Wave import successful")
        
        # Test basic functionality
        engine = ShortTermElliottWaves()
        print("✅ Elliott Wave engine creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Elliott Wave import failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing syntax fixes for Elliott Wave merge conflict...")
    print("=" * 60)
    
    syntax_success = test_syntax_compilation()
    print()
    
    import_success = test_elliott_wave_import()
    print()
    
    if syntax_success and import_success:
        print("🎉 SUCCESS! The merge conflict syntax error has been resolved!")
        print("   - Elliott Wave file compiles without syntax errors")
        print("   - Elliott Wave imports and initializes successfully")
        print("   - The unterminated triple-quoted string issue is fixed")
    else:
        print("⚠️  Some issues remain. Check the error messages above.")
