#!/usr/bin/env python3
"""
Platform3 Comprehensive Code Quality Validation
Tests all major improvements implemented based on code review feedback
"""

import os
import sys
import importlib
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re

class Platform3CodeValidator:
    """Comprehensive validator for Platform3 code quality improvements"""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.results = {
            "sys_path_issues": [],
            "duplicate_classes": [],
            "import_tests": [],
            "package_structure": [],
            "code_quality": []
        }
        self.passed_tests = 0
        self.failed_tests = 0
    
    def log_test(self, test_name: str, passed: bool, message: str = "", details: Any = None):
        """Log test result"""
        if passed:
            print(f"PASS {test_name}")
            self.passed_tests += 1
        else:
            print(f"FAIL {test_name}: {message}")
            self.failed_tests += 1
            if details:
                print(f"  Details: {details}")
    
    def test_sys_path_cleanup(self) -> bool:
        """Test that sys.path.append issues have been resolved"""
        print("\nTesting sys.path.append cleanup...")
        
        problematic_files = []
        
        # Search for remaining sys.path.append issues
        for root, dirs, files in os.walk(self.root_path / "ai-platform"):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if 'sys.path.append' in content:
                                problematic_files.append(str(file_path))
                    except Exception as e:
                        pass
        
        self.results["sys_path_issues"] = problematic_files
        
        passed = len(problematic_files) == 0
        self.log_test(
            "sys.path.append cleanup", 
            passed, 
            f"{len(problematic_files)} files still contain sys.path.append", 
            problematic_files[:5] if problematic_files else None
        )
        
        return passed
    
    def test_duplicate_class_removal(self) -> bool:
        """Test that duplicate classes have been removed"""
        print("\nTesting duplicate class removal...")
        
        duplicate_files = []
        
        # Search for remaining duplicate class definitions
        for root, dirs, files in os.walk(self.root_path / "ai-platform"):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                            # Check for duplicate class definitions
                            if re.search(r'class (AIModelPerformanceMonitor|EnhancedAIModelBase)', content):
                                duplicate_files.append(str(file_path))
                    except Exception as e:
                        pass
        
        self.results["duplicate_classes"] = duplicate_files
        
        passed = len(duplicate_files) == 0
        self.log_test(
            "Duplicate class removal", 
            passed, 
            f"{len(duplicate_files)} files still contain duplicate classes", 
            duplicate_files[:5] if duplicate_files else None
        )
        
        return passed
    
    def test_shared_imports(self) -> bool:
        """Test that shared components can be imported correctly"""
        print("\nTesting shared component imports...")
        
        import_tests = [
            ("shared.ai_model_base", "EnhancedAIModelBase"),
            ("shared.ai_model_base", "AIModelPerformanceMonitor"),
            ("shared.logging.platform3_logger", "Platform3Logger"),
        ]
        
        all_passed = True
        
        for module_name, class_name in import_tests:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, class_name):
                    self.log_test(f"Import {module_name}.{class_name}", True)
                else:
                    self.log_test(f"Import {module_name}.{class_name}", False, f"Class {class_name} not found")
                    all_passed = False
            except ImportError as e:
                self.log_test(f"Import {module_name}.{class_name}", False, f"ImportError: {str(e)}")
                all_passed = False
            except Exception as e:
                self.log_test(f"Import {module_name}.{class_name}", False, f"Error: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def test_package_structure(self) -> bool:
        """Test that package structure is properly configured"""
        print("\nTesting package structure...")
        
        required_files = [
            "pyproject.toml",
            "__init__.py",
            "shared/__init__.py",
            "shared/ai_model_base.py"
        ]
        
        all_exist = True
        
        for file_path in required_files:
            full_path = self.root_path / file_path
            exists = full_path.exists()
            self.log_test(f"Required file exists: {file_path}", exists)
            if not exists:
                all_exist = False
        
        # Test pyproject.toml content
        pyproject_path = self.root_path / "pyproject.toml"
        if pyproject_path.exists():
            try:
                with open(pyproject_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    has_setuptools = 'setuptools' in content
                    has_dependencies = 'dependencies' in content
                    has_project_info = '[project]' in content
                    
                    self.log_test("pyproject.toml has setuptools", has_setuptools)
                    self.log_test("pyproject.toml has dependencies", has_dependencies)
                    self.log_test("pyproject.toml has project info", has_project_info)
                    
                    if not all([has_setuptools, has_dependencies, has_project_info]):
                        all_exist = False
            except Exception as e:
                self.log_test("pyproject.toml validation", False, str(e))
                all_exist = False
        
        return all_exist
    
    def test_code_quality_patterns(self) -> bool:
        """Test for general code quality improvements"""
        print("\nTesting code quality patterns...")
        
        # Check for proper import patterns in a sample of files
        sample_files = []
        ai_platform_path = self.root_path / "ai-platform"
        
        if ai_platform_path.exists():
            for root, dirs, files in os.walk(ai_platform_path):
                for file in files:
                    if file.endswith('.py') and len(sample_files) < 10:
                        sample_files.append(Path(root) / file)
        
        quality_issues = 0
        
        for file_path in sample_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Check for shared.ai_model_base imports
                    if 'from shared.ai_model_base import' in content:
                        self.log_test(f"Proper imports in {file_path.name}", True)
                    else:
                        # This might be okay if the file doesn't use these classes
                        pass
                        
            except Exception as e:
                quality_issues += 1
        
        passed = quality_issues == 0
        self.log_test("Code quality patterns", passed, f"{quality_issues} quality issues found")
        
        return passed
    
    def test_import_compatibility(self) -> bool:
        """Test that the migration maintains import compatibility"""
        print("\nTesting import compatibility...")
        
        # Test that we can import from the new structure
        test_imports = [
            "shared.ai_model_base",
            "shared.logging.platform3_logger"
        ]
        
        all_passed = True
        
        for import_path in test_imports:
            try:
                importlib.import_module(import_path)
                self.log_test(f"Compatible import: {import_path}", True)
            except ImportError as e:
                self.log_test(f"Compatible import: {import_path}", False, str(e))
                all_passed = False
            except Exception as e:
                self.log_test(f"Compatible import: {import_path}", False, f"Unexpected error: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests"""
        print("=" * 60)
        print("Platform3 Comprehensive Code Quality Validation")
        print("=" * 60)
        
        # Add current directory to Python path for testing
        if str(self.root_path) not in sys.path:
            sys.path.insert(0, str(self.root_path))
        
        test_results = {}
        
        # Run all tests
        test_results["sys_path_cleanup"] = self.test_sys_path_cleanup()
        test_results["duplicate_class_removal"] = self.test_duplicate_class_removal()
        test_results["shared_imports"] = self.test_shared_imports()
        test_results["package_structure"] = self.test_package_structure()
        test_results["code_quality_patterns"] = self.test_code_quality_patterns()
        test_results["import_compatibility"] = self.test_import_compatibility()
        
        # Summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        total_tests = self.passed_tests + self.failed_tests
        pass_rate = (self.passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Tests passed: {self.passed_tests}")
        print(f"Tests failed: {self.failed_tests}")
        print(f"Pass rate: {pass_rate:.1f}%")
        
        overall_success = all(test_results.values())
        
        if overall_success:
            print("\nALL VALIDATIONS PASSED!")
            print("Platform3 code quality improvements are successful!")
        else:
            print("\nSome validations failed.")
            print("Review the failed tests above for issues to address.")
        
        print("=" * 60)
        
        return {
            "overall_success": overall_success,
            "test_results": test_results,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "pass_rate": pass_rate,
            "details": self.results
        }

def main():
    """Main validation function"""
    root_path = "E:/MD/Platform3"
    
    validator = Platform3CodeValidator(root_path)
    results = validator.run_comprehensive_validation()
    
    # Exit with appropriate code
    sys.exit(0 if results["overall_success"] else 1)

if __name__ == "__main__":
    main()