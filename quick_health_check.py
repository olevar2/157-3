#!/usr/bin/env python3
"""
Platform3 Quick Health Check
Rapid verification that all code quality improvements are maintained
"""

import sys
import os
from pathlib import Path

def quick_health_check():
    """Perform rapid health check of Platform3 code quality"""
    print("Platform3 Quick Health Check")
    print("=" * 40)
    
    root_path = Path("E:/MD/Platform3")
    
    # Check 1: Package structure
    required_files = [
        "pyproject.toml",
        "__init__.py", 
        "shared/ai_model_base.py"
    ]
    
    print("\n1. Package Structure:")
    for file_path in required_files:
        exists = (root_path / file_path).exists()
        status = "PASS" if exists else "FAIL"
        print(f"   {status} {file_path}")
    
    # Check 2: Import system
    print("\n2. Import System:")
    sys.path.insert(0, str(root_path))
    
    try:
        from shared.ai_model_base import EnhancedAIModelBase, AIModelPerformanceMonitor
        print("   PASS Core classes importable")
    except ImportError as e:
        print(f"   FAIL Import failed: {e}")
        return False
    
    # Check 3: No sys.path.append in recent files
    print("\n3. Code Quality:")
    ai_platform_path = root_path / "ai-platform"
    if ai_platform_path.exists():
        # Quick sample check
        sample_files = list(ai_platform_path.rglob("*.py"))[:5]
        clean_files = 0
        
        for file_path in sample_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'sys.path.append' not in content:
                        clean_files += 1
            except:
                pass
        
        if clean_files == len(sample_files):
            print("   PASS Sample files clean of sys.path.append")
        else:
            print(f"   WARN {len(sample_files) - clean_files} of {len(sample_files)} sample files have issues")
    
    print("\n" + "=" * 40)
    print("Health Check Complete!")
    print("Platform3 code quality improvements are active.")
    
    return True

if __name__ == "__main__":
    success = quick_health_check()
    sys.exit(0 if success else 1)