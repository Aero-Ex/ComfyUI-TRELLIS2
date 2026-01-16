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
import platform as platform_module


def get_wheel_url(pkg_name: str, py_version: str, platform_tag: str) -> str:
    """Get direct wheel URL for a CUDA extension package."""
    # Base URLs for each package
    base_urls = {
        "cumesh": "https://github.com/PozzettiAndrea/cumesh-wheels/releases/download/cu128-torch280",
        "flex_gemm": "https://github.com/PozzettiAndrea/flexgemm-wheels/releases/download/cu128-torch280",
        "o_voxel": "https://github.com/PozzettiAndrea/ovoxel-wheels/releases/download/cu128-torch280",
    }
    
    # Wheel filename format: {pkg}-0.0.1+cu128torch28-{cpXXX}-{cpXXX}-{platform}.whl
    wheel_name = f"{pkg_name}-0.0.1+cu128torch28-{py_version}-{py_version}-{platform_tag}.whl"
    return f"{base_urls[pkg_name]}/{wheel_name}"


CUDA_EXTENSION_PACKAGES = ["cumesh", "flex_gemm", "o_voxel"]


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
    
    # Determine platform and Python version for wheel selection
    py_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if platform_module.system() == "Windows":
        platform_tag = "win_amd64"
    else:
        platform_tag = "linux_x86_64"
    
    print(f"Platform: {platform_tag}, Python: {py_version}")
    print()
    
    # Install CUDA extensions directly from wheel URLs
    # Using direct URL + --no-deps bypasses pip's version string validation
    # which fails due to metadata mismatch (filename has +cu128torch28 but metadata says 0.0.1)
    for pkg_name in CUDA_EXTENSION_PACKAGES:
        wheel_url = get_wheel_url(pkg_name, py_version, platform_tag)
        print(f"\nInstalling {pkg_name} from {wheel_url}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                wheel_url, "--no-deps", "--force-reinstall"
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
