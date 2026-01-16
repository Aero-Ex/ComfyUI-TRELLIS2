#!/usr/bin/env python3
"""
Direct installation script for TRELLIS2 dependencies.

This installs all dependencies INTO THE HOST PYTHON ENVIRONMENT,
bypassing the isolated venv. Use this when you want to run with direct_mode=True.

Downloads pre-built CUDA wheels for Python 3.10 from JeffreyXiang's repository.

WARNING: This may cause conflicts with other ComfyUI nodes. Use at your own risk.

Usage:
    python install_direct.py
    
After installation, set direct_mode=True in the Load TRELLIS.2 Models node.
"""

import sys
import os
import subprocess
from pathlib import Path


# Wheel download URLs for cp310 Linux x86_64
WHEEL_BASE_URL = "https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210"

CUDA_WHEELS = [
    f"{WHEEL_BASE_URL}/cumesh-0.0.1-cp310-cp310-linux_x86_64.whl",
    f"{WHEEL_BASE_URL}/flex_gemm-0.0.1-cp310-cp310-linux_x86_64.whl",
    f"{WHEEL_BASE_URL}/o_voxel-0.0.1-cp310-cp310-linux_x86_64.whl",
    f"{WHEEL_BASE_URL}/nvdiffrast-0.3.5-cp310-cp310-linux_x86_64.whl",
    f"{WHEEL_BASE_URL}/nvdiffrec_render-0.0.0-cp310-cp310-linux_x86_64.whl",
]

# HuggingFace mirror (alternative)
HF_WHEEL_BASE = "https://huggingface.co/spaces/JeffreyXiang/TRELLIS.2/resolve/main/wheels"

HF_CUDA_WHEELS = [
    f"{HF_WHEEL_BASE}/cumesh-0.0.1-cp310-cp310-linux_x86_64.whl",
    f"{HF_WHEEL_BASE}/flex_gemm-0.0.1-cp310-cp310-linux_x86_64.whl", 
    f"{HF_WHEEL_BASE}/o_voxel-0.0.1-cp310-cp310-linux_x86_64.whl",
    f"{HF_WHEEL_BASE}/nvdiffrast-0.3.5-cp310-cp310-linux_x86_64.whl",
    f"{HF_WHEEL_BASE}/nvdiffrec_render-0.0.0-cp310-cp310-linux_x86_64.whl",
]


def main():
    print("\n" + "=" * 70)
    print("TRELLIS2 Direct Installation (Host Environment)")
    print("=" * 70)
    print()
    print("WARNING: This installs dependencies into your main Python environment.")
    print("         This may conflict with other packages. Use at your own risk.")
    print()
    
    # Check Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Python version: {py_version}")
    
    if sys.version_info[:2] != (3, 10):
        print()
        print("WARNING: Pre-built wheels are for Python 3.10 (cp310).")
        print(f"         Your Python is {py_version}. Wheels may not work!")
        print()
    
    # PyPI packages (platform-independent)
    pypi_packages = [
        "huggingface_hub>=0.20.0,<1.0.0",
        "hf_transfer",
        "transformers>=4.56.0",
        "safetensors",
        "pillow>=9.0.0",
        "opencv-python-headless",
        "kornia",
        "imageio>=2.0.0",
        "imageio-ffmpeg>=0.4.0",
        "trimesh>=3.0.0",
        "plyfile",
        "timm",
        "lpips",
        "einops",
        "numpy>=1.21.0",
        "easydict",
        "tqdm",
        "zstandard",
        "setuptools",
    ]
    
    # Install PyPI packages
    print("=" * 70)
    print("Step 1: Installing PyPI packages...")
    print("=" * 70)
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade"
        ] + pypi_packages)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install PyPI packages: {e}")
        return 1
    
    # Install CUDA extensions from GitHub
    print()
    print("=" * 70)
    print("Step 2: Installing CUDA extensions (cp310 Linux x86_64)...")
    print("=" * 70)
    
    # Try GitHub first, then HuggingFace
    for wheel_url in CUDA_WHEELS:
        wheel_name = wheel_url.split("/")[-1]
        print(f"\nInstalling {wheel_name}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--no-deps", wheel_url
            ])
        except subprocess.CalledProcessError:
            # Try HuggingFace mirror
            hf_url = wheel_url.replace(WHEEL_BASE_URL, HF_WHEEL_BASE)
            print(f"         Trying HuggingFace mirror...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--no-deps", hf_url
                ])
            except subprocess.CalledProcessError as e:
                print(f"WARNING: Failed to install {wheel_name}: {e}")
    
    print()
    print("=" * 70)
    print("Installation Complete!")
    print("=" * 70)
    print()
    print("You can now use direct_mode=True in Load TRELLIS.2 Models node.")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
