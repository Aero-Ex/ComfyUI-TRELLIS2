#!/usr/bin/env python3
"""
Direct installation script for TRELLIS2 dependencies.

This installs all dependencies INTO THE HOST PYTHON ENVIRONMENT,
bypassing the isolated venv. Use this when you want to run with direct_mode=True.

Uses pre-built wheels from PozzettiAndrea for CUDA 12.8 + PyTorch 2.8.0.
Supports Python 3.10, 3.11, 3.12, 3.13 on Linux and Windows.

WARNING: This may cause conflicts with other ComfyUI nodes. Use at your own risk.

Usage:
    python install_direct.py
    
After installation, set direct_mode=True in the Load TRELLIS.2 Models node.
"""

import sys
import subprocess


# Pre-built wheel sources for CUDA 12.8 + PyTorch 2.8.0
WHEEL_INDEXES = {
    "cumesh": "https://pozzettiandrea.github.io/cumesh-wheels/cu128-torch280/",
    "flex_gemm": "https://pozzettiandrea.github.io/flexgemm-wheels/cu128-torch280/",
    "o_voxel": "https://pozzettiandrea.github.io/ovoxel-wheels/cu128-torch280/",
}


def main():
    print("\n" + "=" * 70)
    print("TRELLIS2 Direct Installation (Host Environment)")
    print("=" * 70)
    print()
    print("WARNING: This installs dependencies into your main Python environment.")
    print("         This may conflict with other packages. Use at your own risk.")
    print()
    print("Using pre-built wheels for CUDA 12.8 + PyTorch 2.8.0")
    print(f"Python: {sys.version_info.major}.{sys.version_info.minor}")
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
    
    # Install CUDA extensions from pre-built wheels
    print()
    print("=" * 70)
    print("Step 2: Installing CUDA extensions (CUDA 12.8 + PyTorch 2.8.0)...")
    print("=" * 70)
    
    for pkg_name, wheel_index in WHEEL_INDEXES.items():
        print(f"\nInstalling {pkg_name}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                pkg_name, "--find-links", wheel_index
            ])
            print(f"         {pkg_name} installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"WARNING: Failed to install {pkg_name}: {e}")
    
    # Install nvdiffrast from source
    print()
    print("=" * 70)
    print("Step 3: Installing nvdiffrast from GitHub...")
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
