#!/usr/bin/env python3
"""
Direct installation script for TRELLIS2 dependencies.

This installs all dependencies INTO THE HOST PYTHON ENVIRONMENT,
bypassing the isolated venv. Use this when you want to run with direct_mode=True.

Builds CUDA extensions from source since public pre-built wheels require authentication.

WARNING: This may cause conflicts with other ComfyUI nodes. Use at your own risk.
WARNING: Building from source requires CUDA toolkit and may take several minutes.

Usage:
    python install_direct.py
    
After installation, set direct_mode=True in the Load TRELLIS.2 Models node.
"""

import sys
import os
import subprocess
from pathlib import Path


# GitHub repos to clone and build
CUDA_REPOS = [
    ("cumesh", "https://github.com/JeffreyXiang/CuMesh.git"),
    ("flex_gemm", "https://github.com/JeffreyXiang/FlexGEMM.git"),
    ("o_voxel", "https://github.com/JeffreyXiang/o-voxel.git"),
]


def main():
    print("\n" + "=" * 70)
    print("TRELLIS2 Direct Installation (Host Environment)")
    print("=" * 70)
    print()
    print("WARNING: This installs dependencies into your main Python environment.")
    print("         This may conflict with other packages. Use at your own risk.")
    print()
    print("NOTE: Building CUDA extensions from source requires CUDA toolkit.")
    print("      This may take several minutes.")
    print()
    
    # Check Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Python version: {py_version}")
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
        "ninja",  # For faster builds
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
    
    # Install nvdiffrast from GitHub
    print()
    print("=" * 70)
    print("Step 2: Installing nvdiffrast from GitHub...")
    print("=" * 70)
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "git+https://github.com/NVlabs/nvdiffrast.git",
            "--no-build-isolation"
        ])
    except subprocess.CalledProcessError as e:
        print(f"WARNING: Failed to install nvdiffrast: {e}")
    
    # Build CUDA extensions from source
    print()
    print("=" * 70)
    print("Step 3: Building CUDA extensions from source...")
    print("=" * 70)
    
    build_dir = Path(__file__).parent / "_build_temp"
    build_dir.mkdir(exist_ok=True)
    
    for pkg_name, repo_url in CUDA_REPOS:
        print(f"\nBuilding {pkg_name}...")
        pkg_dir = build_dir / pkg_name
        
        # Clone if not exists
        if not pkg_dir.exists():
            try:
                subprocess.check_call(["git", "clone", repo_url, str(pkg_dir)])
            except subprocess.CalledProcessError as e:
                print(f"WARNING: Failed to clone {pkg_name}: {e}")
                continue
        
        # Build and install
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                str(pkg_dir), "--no-build-isolation"
            ])
        except subprocess.CalledProcessError as e:
            print(f"WARNING: Failed to build {pkg_name}: {e}")
    
    print()
    print("=" * 70)
    print("Installation Complete!")
    print("=" * 70)
    print()
    print("You can now use direct_mode=True in Load TRELLIS.2 Models node.")
    print()
    print("Note: If any CUDA extensions failed to build, you may need to:")
    print("  1. Install CUDA toolkit (nvcc)")
    print("  2. Check that your PyTorch CUDA version matches")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
