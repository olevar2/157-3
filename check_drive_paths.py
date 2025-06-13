#!/usr/bin/env python3
"""
Check for remaining hardcoded drive paths after migration from D: to E: drive
"""

import os
import re
from pathlib import Path

def scan_for_hardcoded_paths(root_dir):
    """Scan for hardcoded D: drive paths"""
    patterns = [
        r'[dD]:\\MD\\Platform3',
        r'[dD]:/MD/Platform3',
        r'"[dD]\\MD\\Platform3"',
        r"'[dD]\\MD\\Platform3'",
        r'"[dD]/MD/Platform3"',
        r"'[dD]/MD/Platform3'"
    ]
    
    issues = []
    exclude_dirs = {'.git', '__pycache__', '.venv', 'node_modules', '.pytest_cache'}
    exclude_files = {'.pyc', '.pyo', '.git'}
    
    for root, dirs, files in os.walk(root_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            # Skip excluded file types
            if any(file.endswith(ext) for ext in exclude_files):
                continue
                
            file_path = os.path.join(root, file)
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        issues.append({
                            'file': file_path,
                            'line': line_num,
                            'match': match.group(),
                            'pattern': pattern
                        })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    return issues

def main():
    root_dir = Path(__file__).parent
    print(f"Scanning {root_dir} for hardcoded D: drive paths...")
    
    issues = scan_for_hardcoded_paths(root_dir)
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} remaining hardcoded path(s):")
        for issue in issues:
            rel_path = os.path.relpath(issue['file'], root_dir)
            print(f"  📁 {rel_path}:{issue['line']} - {issue['match']}")
        
        print("\n🔧 These need to be updated to use relative paths or current drive location.")
        return False
    else:
        print("\n✅ No hardcoded D: drive paths found! Migration appears successful.")
        return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
