#!/usr/bin/env python3
"""
Complete E: Drive Migration Verification Script
Tests all fixed paths and validates the project setup
"""

import os
import sys
import json
from pathlib import Path

class MigrationVerifier:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.passed_tests = 0
        self.failed_tests = 0
        self.issues = []
        
    def log_test(self, test_name, passed, message=""):
        """Log test result"""
        if passed:
            print(f"  ✅ {test_name}")
            self.passed_tests += 1
        else:
            print(f"  ❌ {test_name}: {message}")
            self.failed_tests += 1
            self.issues.append(f"{test_name}: {message}")
    
    def test_project_structure(self):
        """Test essential project directories exist"""
        print("\n🏗️  Testing Project Structure...")
        
        essential_dirs = [
            "engines",
            "shared", 
            "tests",
            "ai-platform",
            "scripts",
            "database",
            "logs"
        ]
        
        for dir_name in essential_dirs:
            dir_path = self.project_root / dir_name
            self.log_test(f"Directory {dir_name}/", 
                         dir_path.exists(),
                         f"Missing directory: {dir_path}")
    
    def test_python_imports(self):
        """Test critical Python imports work"""
        print("\n🐍 Testing Python Imports...")
        
        # Add project root to Python path
        sys.path.insert(0, str(self.project_root))
        
        critical_modules = [
            ("MCP Context Recovery", "mcp_context_recovery"),
            ("Microservices Integration", "shared.communication.microservices_integration"),
            ("Simple Correlation System", "shared.communication.simple_correlation_system"),
        ]
        
        for name, module_path in critical_modules:
            try:
                __import__(module_path)
                self.log_test(f"Import {name}", True)
            except ImportError as e:
                self.log_test(f"Import {name}", False, str(e))
            except Exception as e:
                self.log_test(f"Import {name}", False, f"Error: {str(e)}")
    
    def test_script_paths(self):
        """Test PowerShell scripts have correct paths"""
        print("\n📜 Testing Script Paths...")
        
        scripts_to_check = [
            "scripts/start-communication-bridge.ps1",
            "database/setup_database.ps1"
        ]
        
        for script_path in scripts_to_check:
            full_path = self.project_root / script_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for hardcoded D: paths
                    if "D:\\MD\\Platform3" in content or "D:/MD/Platform3" in content:
                        self.log_test(f"Script {script_path} paths", 
                                    False, 
                                    "Still contains hardcoded D: drive paths")
                    else:
                        self.log_test(f"Script {script_path} paths", True)
                        
                except Exception as e:
                    self.log_test(f"Script {script_path} readable", False, str(e))
            else:
                self.log_test(f"Script {script_path} exists", False, "File not found")
    
    def test_venv_setup(self):
        """Test virtual environment setup"""
        print("\n🔧 Testing Virtual Environment...")
        
        venv_path = self.project_root / ".venv"
        
        if venv_path.exists():
            self.log_test("Virtual environment exists", True)
            
            # Check for Windows vs Linux structure
            if sys.platform == "win32":
                python_exe = venv_path / "Scripts" / "python.exe"
                activate_script = venv_path / "Scripts" / "activate.bat"
            else:
                python_exe = venv_path / "bin" / "python"
                activate_script = venv_path / "bin" / "activate"
            
            self.log_test("Python executable", 
                         python_exe.exists(),
                         f"Missing: {python_exe}")
            
            self.log_test("Activation script", 
                         activate_script.exists(),
                         f"Missing: {activate_script}")
        else:
            self.log_test("Virtual environment exists", 
                         False, 
                         f"No .venv found at {venv_path}")
    
    def test_config_files(self):
        """Test configuration files"""
        print("\n⚙️  Testing Configuration Files...")
        
        config_files = [
            ("package.json", True),
            ("docker-compose.yml", True), 
            (".env.template", True),
            (".gitignore", True),
            ("requirements.txt", False)  # Optional
        ]
        
        for filename, required in config_files:
            file_path = self.project_root / filename
            if file_path.exists():
                self.log_test(f"Config {filename}", True)
            elif required:
                self.log_test(f"Config {filename}", False, "Required file missing")
            else:
                print(f"  ℹ️  Optional file {filename} not found")
    
    def scan_remaining_hardcoded_paths(self):
        """Scan for any remaining hardcoded D: drive paths"""
        print("\n🔍 Scanning for Remaining Hardcoded Paths...")
        
        patterns = [
            "D:\\MD\\Platform3",
            "D:/MD/Platform3",
            "d:\\MD\\Platform3", 
            "d:/MD/Platform3"
        ]
        
        exclude_dirs = {'.git', '__pycache__', '.venv', 'node_modules', '.pytest_cache', 'dist'}
        exclude_files = {'.pyc', '.pyo', '.git', '.png', '.jpg', '.jpeg', '.exe', '.dll'}
        
        issues_found = []
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                # Skip excluded file types
                if any(file.endswith(ext) for ext in exclude_files):
                    continue
                    
                file_path = Path(root) / file
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    for pattern in patterns:
                        if pattern in content:
                            rel_path = file_path.relative_to(self.project_root)
                            issues_found.append(f"{rel_path}: contains '{pattern}'")
                            
                except Exception:
                    continue  # Skip files that can't be read
        
        if issues_found:
            self.log_test("No hardcoded paths remaining", 
                         False, 
                         f"Found {len(issues_found)} files with hardcoded paths")
            for issue in issues_found[:5]:  # Show first 5
                print(f"    📁 {issue}")
            if len(issues_found) > 5:
                print(f"    ... and {len(issues_found) - 5} more")
        else:
            self.log_test("No hardcoded paths remaining", True)
    
    def generate_recommendations(self):
        """Generate recommendations for next steps"""
        print("\n📋 Recommendations for Next Steps:")
        
        if self.failed_tests == 0:
            print("  🎉 All tests passed! Your migration is complete.")
            print("  🚀 You can now run your Platform3 tests safely.")
        else:
            print(f"  ⚠️  {self.failed_tests} issues found that need attention:")
            for issue in self.issues:
                print(f"    • {issue}")
        
        print("\n  📝 Suggested next actions:")
        print("    1. Activate your virtual environment:")
        print("       • Windows: .venv\\Scripts\\activate")
        print("       • Linux/Mac: source .venv/bin/activate")
        print("    2. Install/update dependencies: pip install -r requirements.txt")
        print("    3. Test your Platform3 functionality")
        print("    4. Run your indicator tests")
        
    def run_complete_verification(self):
        """Run all verification tests"""
        print("🔍 Platform3 E: Drive Migration Verification")
        print("=" * 60)
        print(f"📁 Project Root: {self.project_root}")
        
        # Run all tests
        self.test_project_structure()
        self.test_python_imports()
        self.test_script_paths()
        self.test_venv_setup()
        self.test_config_files()
        self.scan_remaining_hardcoded_paths()
        
        # Summary
        print("\n" + "=" * 60)
        print(f"📊 Test Results: {self.passed_tests} passed, {self.failed_tests} failed")
        
        self.generate_recommendations()
        
        return self.failed_tests == 0

if __name__ == "__main__":
    verifier = MigrationVerifier()
    success = verifier.run_complete_verification()
    
    exit(0 if success else 1)