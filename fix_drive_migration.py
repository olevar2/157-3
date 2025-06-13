#!/usr/bin/env python3
"""
Complete Drive Migration Fixer - E: Drive Setup
Fixes remaining hardcoded paths and validates the migration
"""

import os
import sys
from pathlib import Path
import re
import json

class DriveMigrationFixer:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.issues_found = []
        self.issues_fixed = []
        
    def scan_and_fix_patterns(self):
        """Scan for and fix hardcoded D: drive patterns"""
        patterns_to_fix = [
            (r'[dD]:\\MD\\Platform3', lambda m: str(self.project_root).replace('/', '\\')),
            (r'[dD]:/MD/Platform3', lambda m: str(self.project_root).replace('\\', '/')),
            (r'"[dD]\\MD\\Platform3"', lambda m: f'"{self.project_root}"'),
            (r"'[dD]\\MD\\Platform3'", lambda m: f"'{self.project_root}'"),
            (r'"[dD]/MD/Platform3"', lambda m: f'"{self.project_root}"'),
            (r"'[dD]/MD/Platform3'", lambda m: f"'{self.project_root}'"),
        ]
        
        # File types to check
        extensions = {'.py', '.ps1', '.sh', '.md', '.json', '.yaml', '.yml', '.txt'}
        exclude_dirs = {'.git', '__pycache__', '.venv', 'node_modules', '.pytest_cache', 'dist'}
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = Path(root) / file
                    self._fix_file_patterns(file_path, patterns_to_fix)
    
    def _fix_file_patterns(self, file_path, patterns):
        """Fix patterns in a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            original_content = content
            
            for pattern, replacement_func in patterns:
                matches = list(re.finditer(pattern, content, re.IGNORECASE))
                if matches:
                    for match in reversed(matches):  # Reverse to maintain positions
                        old_text = match.group()
                        new_text = replacement_func(match)
                        content = content[:match.start()] + new_text + content[match.end():]
                        
                        self.issues_fixed.append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'old': old_text,
                            'new': new_text
                        })
            
            # Write back if changed
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    def check_critical_configs(self):
        """Check critical configuration files"""
        critical_files = [
            '.env.local',
            '.env.template', 
            'package.json',
            'docker-compose.yml',
            'docker-compose.platform3.yml'
        ]
        
        print("\\n🔍 Checking critical configuration files:")
        for file_name in critical_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                print(f"  ✅ {file_name} exists")
            else:
                print(f"  ❌ {file_name} missing")
    
    def verify_python_environment(self):
        """Check Python environment setup"""
        print("\\n🐍 Python Environment Check:")
        
        venv_path = self.project_root / '.venv'
        if venv_path.exists():
            print(f"  ✅ Virtual environment found: {venv_path}")
            
            # Check if activate script exists
            if sys.platform == "win32":
                activate_script = venv_path / 'Scripts' / 'activate.bat'
                python_exe = venv_path / 'Scripts' / 'python.exe'
            else:
                activate_script = venv_path / 'bin' / 'activate'
                python_exe = venv_path / 'bin' / 'python'
                
            if activate_script.exists():
                print(f"  ✅ Activation script found")
            else:
                print(f"  ❌ Activation script missing")
                
            if python_exe.exists():
                print(f"  ✅ Python executable found")
            else:
                print(f"  ❌ Python executable missing")
        else:
            print(f"  ❌ Virtual environment not found at {venv_path}")
            print(f"  💡 Run: python -m venv .venv")
    
    def create_quick_test_script(self):
        """Create a quick test script to verify everything works"""
        test_script = f'''#!/usr/bin/env python3
"""
Quick test to verify Platform3 setup after drive migration
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test critical imports"""
    try:
        # Add project root to path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        print("Testing critical imports...")
        
        # Test some key imports
        test_modules = [
            "engines.ai_enhancement.indicators.trend.sma_indicator",
            "shared.communication.microservices_integration", 
            "mcp_context_recovery"
        ]
        
        for module in test_modules:
            try:
                __import__(module)
                print(f"  ✅ {module}")
            except ImportError as e:
                print(f"  ❌ {module}: {e}")
        
        print("\\n✅ Import test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_paths():
    """Test path resolution"""
    project_root = Path(__file__).parent
    
    critical_paths = [
        "engines",
        "shared", 
        "tests",
        "logs"
    ]
    
    print("\\nTesting path access...")
    for path_name in critical_paths:
        path = project_root / path_name
        if path.exists():
            print(f"  ✅ {path_name}/")
        else:
            print(f"  ❌ {path_name}/ - missing")
            
    return True

if __name__ == "__main__":
    print("🚀 Platform3 E: Drive Migration Test")
    print("=" * 50)
    
    test_paths()
    test_imports()
    
    print("\\n🎉 Migration test completed!")
'''
        
        test_file = self.project_root / "test_e_drive_setup.py"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_script)
            
        print(f"\\n📝 Created test script: {test_file.name}")
    
    def run_complete_fix(self):
        """Run complete migration fix"""
        print("🔧 Platform3 Drive Migration Fixer")
        print("=" * 50)
        
        # Step 1: Scan and fix patterns
        print("\\n1. Scanning for hardcoded D: drive paths...")
        self.scan_and_fix_patterns()
        
        if self.issues_fixed:
            print(f"   ✅ Fixed {len(self.issues_fixed)} hardcoded path(s)")
            for fix in self.issues_fixed[:5]:  # Show first 5
                print(f"     📁 {fix['file']}: {fix['old']} → {fix['new']}")
            if len(self.issues_fixed) > 5:
                print(f"     ... and {len(self.issues_fixed) - 5} more")
        else:
            print("   ✅ No hardcoded paths found!")
        
        # Step 2: Check configs
        self.check_critical_configs()
        
        # Step 3: Check Python environment  
        self.verify_python_environment()
        
        # Step 4: Create test script
        self.create_quick_test_script()
        
        print("\\n🎉 Migration fix completed!")
        print("\\n📋 Next Steps:")
        print("   1. Run: python test_e_drive_setup.py")
        print("   2. Activate venv: .venv\\Scripts\\activate (Windows)")
        print("   3. Install dependencies: pip install -r requirements.txt")
        print("   4. Run your tests!")

if __name__ == "__main__":
    fixer = DriveMigrationFixer()
    fixer.run_complete_fix()
