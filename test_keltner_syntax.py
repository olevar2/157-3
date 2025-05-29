"""
Test script to check if Keltner Channels has syntax errors
"""

import ast
import sys

def check_syntax(file_path):
    """Check if a Python file has syntax errors"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Parse the code
        ast.parse(source_code)
        print(f"✅ SYNTAX CHECK PASSED: {file_path}")
        return True
        
    except SyntaxError as e:
        print(f"❌ SYNTAX ERROR in {file_path}:")
        print(f"   Line {e.lineno}: {e.text.strip() if e.text else 'Unknown'}")
        print(f"   Error: {e.msg}")
        return False
        
    except Exception as e:
        print(f"❌ OTHER ERROR in {file_path}: {str(e)}")
        return False

if __name__ == "__main__":
    files_to_check = [
        "engines/volatility/keltner_channels.py"
    ]
    
    all_passed = True
    for file_path in files_to_check:
        if not check_syntax(file_path):
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL FILES PASSED SYNTAX CHECK!")
    else:
        print("\n💥 SOME FILES HAVE SYNTAX ERRORS!")
        sys.exit(1)
