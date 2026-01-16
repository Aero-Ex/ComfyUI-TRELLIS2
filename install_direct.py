#!/usr/bin/env python3
"""
Direct installation script for TRELLIS2 dependencies.

This installs all dependencies INTO THE HOST PYTHON ENVIRONMENT,
bypassing the isolated venv. Use this when you want to run with direct_mode=True.

WARNING: This may cause conflicts with other ComfyUI nodes. Use at your own risk.

Usage:
    python install_direct.py
    
After installation, set direct_mode=True in the Load TRELLIS.2 Models node.
"""

import sys
import os
import platform
import subprocess
from pathlib import Path


def get_cuda_version():
    """Detect CUDA version from nvcc or torch."""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            if cuda_version:
                major, minor = cuda_version.split(".")[:2]
                return f"cu{major}{minor}"
    except ImportError:
        pass
    
    # Check nvcc
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            import re
            match = re.search(r"release (\d+)\.(\d+)", result.stdout)
            if match:
                return f"cu{match.group(1)}{match.group(2)}"
    except:
        pass
    
    return "cu124"  # Default fallback


def get_python_version():
    """Get Python major.minor version string."""
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def main():
    print("\n" + "=" * 70)
    print("TRELLIS2 Direct Installation (Host Environment)")
    print("=" * 70)
    print()
    print("WARNING: This installs dependencies into your main Python environment.")
    print("         This may conflict with other packages. Use at your own risk.")
    print()
    
    cuda_version = get_cuda_version()
    python_version = get_python_version()
    
    print(f"Detected CUDA: {cuda_version}")
    print(f"Python: {python_version}")
    print()
    
    # PyPI packages (platform-independent)
    pypi_packages = [
        "huggingface_hub>=0.20.0,<1.0.0",
        "hf_transfer",
        "hf_xet",
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
    
    # CUDA extensions from custom index
    cuda_index = "https://storage.googleapis.com/comfy-env-wheels/cuda/"
    cuda_packages = [
        f"nvdiffrast==0.4.0+{cuda_version}",
        f"flex_gemm==0.0.1+{cuda_version}",
        f"cumesh==0.0.1+{cuda_version}",
        f"o_voxel==0.0.1+{cuda_version}",
        f"nvdiffrec_render==0.0.1+{cuda_version}",
    ]
    
    # Optional: sageattention and flash-attn (may fail on some systems)
    optional_cuda = [
        f"sageattention==2.2.0+{cuda_version}",
        f"flash-attn==2.8.3+{cuda_version}",
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
    
    # Install CUDA extensions
    print()
    print("=" * 70)
    print("Step 2: Installing CUDA extensions...")
    print("=" * 70)
    
    for pkg in cuda_packages:
        print(f"\nInstalling {pkg}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "--no-deps", "--index-url", cuda_index, pkg
            ])
        except subprocess.CalledProcessError as e:
            print(f"WARNING: Failed to install {pkg}: {e}")
            print("         You may need to build from source.")
    
    # Install optional CUDA packages
    print()
    print("=" * 70)
    print("Step 3: Installing optional CUDA packages (may fail)...")
    print("=" * 70)
    
    for pkg in optional_cuda:
        print(f"\nInstalling {pkg}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "--no-deps", "--index-url", cuda_index, pkg
            ])
        except subprocess.CalledProcessError:
            print(f"         Skipped (optional)")
    
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
