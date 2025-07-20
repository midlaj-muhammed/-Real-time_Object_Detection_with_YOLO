#!/usr/bin/env python3
"""
Setup script for YOLO Object Detection application.
This script helps set up the environment and install dependencies.
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🔍 Checking Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("   Required: Python 3.8 or higher")
        return False

def check_virtual_environment():
    """Check if virtual environment exists."""
    print("🔍 Checking virtual environment...")
    
    if os.path.exists("venv"):
        print("✅ Virtual environment found")
        return True
    else:
        print("⚠️ Virtual environment not found")
        return False

def create_virtual_environment():
    """Create virtual environment."""
    return run_command(
        f"{sys.executable} -m venv venv",
        "Creating virtual environment"
    )

def get_activation_command():
    """Get the appropriate activation command for the platform."""
    if platform.system() == "Windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"

def install_dependencies():
    """Install required dependencies."""
    activation_cmd = get_activation_command()
    
    # Install basic dependencies first
    basic_deps = [
        "pip --upgrade",
        "opencv-python",
        "streamlit",
        "ultralytics",
        "pillow",
        "numpy",
        "pandas"
    ]
    
    for dep in basic_deps:
        if not run_command(
            f"{activation_cmd} && pip install {dep}",
            f"Installing {dep}"
        ):
            return False
    
    return True

def generate_requirements():
    """Generate requirements.txt file."""
    activation_cmd = get_activation_command()
    return run_command(
        f"{activation_cmd} && pip freeze > requirements.txt",
        "Generating requirements.txt"
    )

def test_installation():
    """Test the installation."""
    activation_cmd = get_activation_command()
    return run_command(
        f"{activation_cmd} && python test_app.py",
        "Testing installation"
    )

def main():
    """Main setup function."""
    print("🎯 YOLO Object Detection - Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check/create virtual environment
    if not check_virtual_environment():
        print("📦 Creating virtual environment...")
        if not create_virtual_environment():
            return 1
    
    # Install dependencies
    print("📥 Installing dependencies...")
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return 1
    
    # Generate requirements.txt
    print("📝 Generating requirements.txt...")
    generate_requirements()  # Don't fail if this doesn't work
    
    # Test installation
    print("🧪 Testing installation...")
    if test_installation():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Activate the virtual environment:")
        print(f"   {get_activation_command()}")
        print("2. Run the application:")
        print("   streamlit run app.py")
        print("3. Open your browser to the URL shown in the terminal")
    else:
        print("\n⚠️ Setup completed but tests failed")
        print("You can still try running the application manually")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
