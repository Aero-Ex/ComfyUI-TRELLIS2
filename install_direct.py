#!/usr/bin/env python3
"""
Direct installation script for TRELLIS2 dependencies.

This installs all dependencies INTO THE HOST PYTHON ENVIRONMENT,
bypassing the isolated venv. Use this when you want to run with direct_mode=True.

Uses pre-built wheels from pozzettiandrea/cumesh-wheels for CUDA 12.8 + PyTorch 2.8.0.

WARNING: This may cause conflicts with other ComfyUI nodes. Use at your own risk.

Usage:
    python install_direct.py
    
After installation, set direct_mode=True in the Load TRELLIS.2 Models node.
"""

import sys
import os
import subprocess
from pathlib import Path


# Pre-built wheel sources for CUDA 12.8 + PyTorch 2.8.0
CUMESH_WHEEL_INDEX = "https://pozzettiandrea.github.io/cumesh-wheels/cu128-torch280/"


def main():
    print("\n" + "=" * 70)
    print("TRELLIS2 Direct Installation (Host Environment)")
    print("=" * 70)
    print()
    print("WARNING: This installs dependencies into your main Python environment.")
    print("         This may conflict with other packages. Use at your own risk.")
    print()
    print("Using pre-built wheels for CUDA 12.8 + PyTorch 2.8.0")
    print()
    
    # PyPI packages
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
    
    # Install cumesh from pre-built wheels (CUDA 12.8 + PyTorch 2.8.0)
    print()
    print("=" * 70)
    print("Step 2: Installing CuMesh (CUDA 12.8 + PyTorch 2.8.0 wheels)...")
    print("=" * 70)
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "cumesh", "--find-links", CUMESH_WHEEL_INDEX
        ])
        print("         cumesh installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"WARNING: Failed to install cumesh: {e}")
    
    # Install nvdiffrast from source
    print()
    print("=" * 70)
    print("Step 3: Installing nvdiffrast from source...")
    print("=" * 70)
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "git+https://github.com/NVlabs/nvdiffrast.git",
            "--no-build-isolation"
        ])
        print("         nvdiffrast installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"WARNING: Failed to install nvdiffrast: {e}")
    
    # Build flex_gemm and o_voxel from source
    print()
    print("=" * 70)
    print("Step 4: Building flex_gemm and o_voxel from source...")
    print("=" * 70)
    
    build_dir = Path(__file__).parent / "_build_temp"
    build_dir.mkdir(exist_ok=True)
    
    source_repos = [
        ("flex_gemm", "https://github.com/JeffreyXiang/FlexGEMM.git"),
        ("o_voxel", "https://github.com/JeffreyXiang/o-voxel.git"),
    ]
    
    for pkg_name, repo_url in source_repos:
        print(f"\nBuilding {pkg_name} from source...")
        pkg_dir = build_dir / pkg_name
        
        if not pkg_dir.exists():
            try:
                subprocess.check_call(["git", "clone", "--depth", "1", repo_url, str(pkg_dir)])
            except subprocess.CalledProcessError as e:
                print(f"WARNING: Failed to clone {pkg_name}: {e}")
                continue
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                str(pkg_dir), "--no-build-isolation"
            ])
            print(f"         {pkg_name} built successfully!")
        except subprocess.CalledProcessError as e:
            print(f"WARNING: Failed to build {pkg_name}: {e}")
    
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
