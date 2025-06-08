#!/usr/bin/env python3
"""
Platform3 Python AI Engine API Server Startup Script
"""

import os
import sys
import subprocess
from pathlib import Path

def install_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        sys.exit(1)

def setup_environment():
    """Setup environment variables"""
    print("Setting up environment...")
    
    # Set default environment variables if not already set
    os.environ.setdefault("PYTHON_ENGINE_PORT", "8000")
    os.environ.setdefault("PYTHON_ENGINE_HOST", "0.0.0.0")
    os.environ.setdefault("LOG_LEVEL", "info")
    os.environ.setdefault("API_VERSION", "1.0.0")
    
    print("Environment configured")

def check_python_version():
    """Check Python version compatibility"""
    if sys.version_info < (3, 8):
        print("Python 3.8+ required")
        sys.exit(1)
    print(f"Python {sys.version_info.major}.{sys.version_info.minor} is compatible")

def main():
    """Main startup function"""
    print("Starting Platform3 Python AI Engine API Server...")
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Checks and setup
    check_python_version()
    setup_environment()
    
    # Install dependencies if needed
    if not Path("requirements.txt").exists():
        print("requirements.txt not found, skipping dependency installation")
    else:
        install_dependencies()
    
    # Start the server
    print("Starting FastAPI server...")
    try:
        import uvicorn
        uvicorn.run(
            "server:app",
            host=os.environ.get("PYTHON_ENGINE_HOST", "0.0.0.0"),
            port=int(os.environ.get("PYTHON_ENGINE_PORT", "8000")),
            reload=False,
            workers=1,
            log_level=os.environ.get("LOG_LEVEL", "info"),
            access_log=True
        )
    except ImportError:
        print("uvicorn not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "uvicorn[standard]"], check=True)
        import uvicorn
        uvicorn.run(
            "server:app",
            host=os.environ.get("PYTHON_ENGINE_HOST", "0.0.0.0"),
            port=int(os.environ.get("PYTHON_ENGINE_PORT", "8000")),
            reload=False,
            workers=1,
            log_level=os.environ.get("LOG_LEVEL", "info"),
            access_log=True
        )
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()