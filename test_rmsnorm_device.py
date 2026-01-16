"""
Test RMSNorm device handling with GGUF tensors
"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trellis2.utils import gguf_utils
from trellis2.modules.attention.modules import MultiHeadRMSNorm

GGUF_PATH = r"D:\@home\aero\comfy\ComfyUI\models\trellis2\ckpts\ss_flow_img_dit_1_3B_64_bf16_Q8_0.gguf"

def test_rmsnorm_device():
    print("="*70)
    print("Testing RMSNorm Device Handling with GGUF")
    print("="*70)
    
    if not os.path.exists(GGUF_PATH):
        print(f"[SKIP] GGUF file not found: {GGUF_PATH}")
        return False
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Target device: {device}")
    
    # Load GGUF checkpoint
    print("\n[1] Loading GGUF checkpoint...")
    gguf_sd, _ = gguf_utils.load_gguf_checkpoint(GGUF_PATH)
    
    # Find a gamma parameter (for RMSNorm)
    gamma_key = None
    for key in gguf_sd.keys():
        if "rms_norm" in key and "gamma" in key:
            gamma_key = key
            break
    
    if not gamma_key:
        print("[WARN] No RMSNorm gamma found in GGUF, using first F32 tensor")
        for key, tensor in gguf_sd.items():
            if hasattr(tensor, 'tensor_type') and tensor.tensor_type == 0:
                gamma_key = key
                break
    
    if not gamma_key:
        print("[SKIP] No suitable gamma tensor found")
        return False
    
    gamma_tensor = gguf_sd[gamma_key]
    print(f"    Found gamma: {gamma_key}")
    print(f"    Tensor type: {getattr(gamma_tensor, 'tensor_type', 'N/A')}")
    print(f"    Tensor device: {gamma_tensor.device}")
    print(f"    Tensor shape: {gamma_tensor.shape}")
    
    # Create MultiHeadRMSNorm
    print("\n[2] Creating MultiHeadRMSNorm...")
    # Get shape info to create the module
    if gamma_tensor.dim() == 2:
        heads, dim = gamma_tensor.shape
    else:
        heads, dim = 24, 64  # Default for TRELLIS
    
    rms_norm = MultiHeadRMSNorm(dim, heads)
    
    # Load state dict manually
    print("\n[3] Loading gamma into RMSNorm via state_dict...")
    state_dict = {'gamma': gamma_tensor}
    rms_norm._load_from_state_dict(state_dict, "", {}, False, [], [], [])
    
    print(f"    rms_norm.gamma type: {type(rms_norm.gamma)}")
    print(f"    rms_norm.gamma device: {rms_norm.gamma.device}")
    print(f"    Has tensor_type attr: {hasattr(rms_norm.gamma, 'tensor_type')}")
    
    # NOTE: We do NOT call model.to(device) - this simulates the bug where
    # model.to() doesn't properly move nn.Parameter wrapped GGUF tensors
    # Instead, let's test the forward pass directly with CUDA input
    
    print("\n[4] Testing forward pass with CUDA input (gamma stays on CPU)...")
    x = torch.randn(1, 64, heads, dim, dtype=torch.float16, device=device)
    print(f"    Input shape: {x.shape}")
    print(f"    Input device: {x.device}")
    print(f"    Gamma device: {rms_norm.gamma.device}")
    
    try:
        output = rms_norm(x)
        print(f"    Output shape: {output.shape}")
        print(f"    Output device: {output.device}")
        print(f"    [PASS] RMSNorm forward pass succeeded!")
        return True
    except Exception as e:
        print(f"    [FAIL] RMSNorm forward pass failed: {e}")
        return False

def test_rmsnorm_with_model_to():
    print("\n" + "="*70)
    print("Testing RMSNorm After model.to(device)")
    print("="*70)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create a simple model with RMSNorm
    rms_norm = MultiHeadRMSNorm(64, 24)
    print(f"Before .to(device): gamma device = {rms_norm.gamma.device}")
    
    # Move to device
    rms_norm = rms_norm.to(device)
    print(f"After .to(device): gamma device = {rms_norm.gamma.device}")
    
    # Test forward
    x = torch.randn(1, 64, 24, 64, dtype=torch.float16, device=device)
    try:
        output = rms_norm(x)
        print(f"[PASS] Forward pass works after .to(device)")
        return True
    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")
        return False

if __name__ == "__main__":
    test1 = test_rmsnorm_device()
    test2 = test_rmsnorm_with_model_to()
    
    print("\n" + "="*70)
    if test1 and test2:
        print("ALL TESTS PASSED!")
    else:
        print(f"Test results: Device handling = {test1}, Model.to() = {test2}")
    print("="*70)
