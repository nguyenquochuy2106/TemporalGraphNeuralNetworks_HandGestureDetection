# install_dependencies.py
import subprocess
import sys

# List of required libraries
REQUIRED_LIBS = [
    "torch",
    "torchvision",
    "torch-geometric",
    "opencv-python",
    "opencv-python-headless",
    "streamlit",
    "Pillow",
    "labelImg",  # Make sure to check if this is necessary
    "mediapipe"
]

def install_packages():
    """Install required packages."""
    for package in REQUIRED_LIBS:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    install_packages()
