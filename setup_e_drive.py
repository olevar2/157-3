#!/usr/bin/env python3
"""
Platform3 E: Drive Setup Helper
Quick setup script for the migrated Platform3 project
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and show the result"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ Success: {description}")
            return True
        else:
            print(f"  ❌ Failed: {description}")
            print(f"     Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ❌ Error running command: {e}")
        return False

def check_python_environment():
    """Check if Python environment is properly set up"""
    print("\n🐍 Checking Python Environment...")
    
    project_root = Path(__file__).parent
    venv_path = project_root / ".venv"
    
    if not venv_path.exists():
        print("  ❌ Virtual environment not found")
        print("  💡 Creating virtual environment...")
        
        if run_command("python -m venv .venv", "Creating virtual environment"):
            print("  ✅ Virtual environment created")
        else:
            print("  ❌ Failed to create virtual environment")
            return False
    else:
        print("  ✅ Virtual environment found")
    
    return True

def activate_instructions():
    """Show activation instructions"""
    print("\n📝 To activate your environment:")
    if sys.platform == "win32":
        print("   .venv\\Scripts\\activate")
    else:
        print("   source .venv/bin/activate")

def check_requirements():
    """Check for requirements.txt"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        print("\n📦 Requirements file found")
        print("   After activating the environment, run:")
        print("   pip install -r requirements.txt")
    else:
        print("\n📦 No requirements.txt found")
        print("   You may need to install packages manually")

def main():
    print("🚀 Platform3 E: Drive Setup Helper")
    print("=" * 50)
    
    # Check if we're in the right directory
    project_root = Path(__file__).parent
    if not (project_root / "engines").exists():
        print("❌ Error: This doesn't appear to be the Platform3 root directory")
        print("   Make sure you're running this from E:\\MD\\Platform3")
        return False
    
    print(f"📁 Project Location: {project_root}")
    
    # Check Python environment
    if not check_python_environment():
        return False
    
    # Show next steps
    activate_instructions()
    check_requirements()
    
    print("\n🎉 Setup helper completed!")
    print("\n📋 Next steps:")
    print("   1. Activate the virtual environment (see above)")
    print("   2. Install dependencies")
    print("   3. Run: python verify_e_drive_migration.py")
    print("   4. Test your Platform3 functionality")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)