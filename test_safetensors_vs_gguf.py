"""
E2E Comparison Test: Safetensors vs GGUF
Compares forward pass outputs between safetensors and GGUF versions of the same model.
"""
import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trellis2.utils import gguf_utils
from trellis2.modules.sparse.linear import SparseLinear
from safetensors import safe_open

GGUF_PATH = r"D:\@home\aero\comfy\ComfyUI\models\trellis2\ckpts\ss_flow_img_dit_1_3B_64_bf16_Q8_0.gguf"
SAFETENSORS_PATH = r"D:\@home\aero\comfy\ComfyUI\models\trellis2\ckpts\ss_flow_img_dit_1_3B_64_bf16.safetensors"

def load_safetensors_weight(path, key):
    """Load a single weight from safetensors file."""
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(key)

def compare_forward_pass():
    print("="*70)
    print("E2E Comparison: Safetensors vs GGUF")
    print("="*70)
    
    if not os.path.exists(GGUF_PATH):
        print(f"[SKIP] GGUF file not found: {GGUF_PATH}")
        return
    
    if not os.path.exists(SAFETENSORS_PATH):
        print(f"[SKIP] Safetensors file not found: {SAFETENSORS_PATH}")
        return
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Test case: blocks.0.self_attn.to_qkv
    test_key = "blocks.0.self_attn.to_qkv"
    in_features, out_features = 1536, 4608
    
    print(f"\n[1] Loading weights for: {test_key}")
    
    # Load GGUF
    print("    Loading GGUF checkpoint...")
    gguf_sd, _ = gguf_utils.load_gguf_checkpoint(GGUF_PATH)
    gguf_weight = gguf_sd[f"{test_key}.weight"]
    gguf_bias = gguf_sd.get(f"{test_key}.bias")
    print(f"    GGUF weight type: {getattr(gguf_weight, 'tensor_type', 'N/A')}")
    print(f"    GGUF weight shape: {getattr(gguf_weight, 'tensor_shape', gguf_weight.shape)}")
    
    # Load Safetensors
    print("    Loading Safetensors...")
    st_weight = load_safetensors_weight(SAFETENSORS_PATH, f"{test_key}.weight")
    st_bias = load_safetensors_weight(SAFETENSORS_PATH, f"{test_key}.bias") if gguf_bias is not None else None
    print(f"    Safetensors weight shape: {st_weight.shape}")
    print(f"    Safetensors weight dtype: {st_weight.dtype}")
    
    # Create layers
    print("\n[2] Creating layers...")
    
    # GGUF Layer
    gguf_layer = SparseLinear(in_features, out_features, bias=gguf_bias is not None)
    gguf_layer._load_from_state_dict(
        {"weight": gguf_weight, "bias": gguf_bias} if gguf_bias is not None else {"weight": gguf_weight},
        "", {}, False, [], [], []
    )
    gguf_layer.to(device)
    
    # Safetensors Layer
    st_layer = SparseLinear(in_features, out_features, bias=st_bias is not None)
    st_layer.weight = nn.Parameter(st_weight.to(device))
    if st_bias is not None:
        st_layer.bias = nn.Parameter(st_bias.to(device))
    
    # Create test input (same for both)
    print("\n[3] Running forward pass...")
    torch.manual_seed(42)
    test_input = torch.randn(1, 256, in_features, dtype=torch.float16, device=device)
    
    with torch.no_grad():
        gguf_output = gguf_layer(test_input)
        st_output = st_layer(test_input.to(st_weight.dtype))
    
    print(f"    GGUF output shape: {gguf_output.shape}, dtype: {gguf_output.dtype}")
    print(f"    Safetensors output shape: {st_output.shape}, dtype: {st_output.dtype}")
    
    # Compare outputs
    print("\n[4] Comparing outputs...")
    
    # Convert to same dtype for comparison
    gguf_fp32 = gguf_output.float()
    st_fp32 = st_output.float()
    
    # Calculate differences
    abs_diff = (gguf_fp32 - st_fp32).abs()
    rel_diff = abs_diff / (st_fp32.abs() + 1e-8)
    
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    
    print(f"    Max absolute difference: {max_abs_diff:.6f}")
    print(f"    Mean absolute difference: {mean_abs_diff:.6f}")
    print(f"    Max relative difference: {max_rel_diff:.6f}")
    print(f"    Mean relative difference: {mean_rel_diff:.6f}")
    
    # Check correlation
    correlation = torch.corrcoef(torch.stack([gguf_fp32.flatten(), st_fp32.flatten()]))[0, 1].item()
    print(f"    Correlation: {correlation:.6f}")
    
    # Pass/Fail criteria
    print("\n[5] Test Results...")
    
    # Q8_0 quantization typically has ~1% error
    if correlation > 0.99:
        print(f"    [PASS] High correlation ({correlation:.4f} > 0.99)")
    else:
        print(f"    [WARN] Lower correlation ({correlation:.4f})")
    
    if max_abs_diff < 1.0:
        print(f"    [PASS] Max abs diff acceptable ({max_abs_diff:.4f} < 1.0)")
    else:
        print(f"    [WARN] Max abs diff high ({max_abs_diff:.4f})")
    
    # Show sample values
    print("\n[6] Sample values (first 5 elements of output[0,0,:]):")
    print(f"    GGUF:        {gguf_fp32[0, 0, :5].tolist()}")
    print(f"    Safetensors: {st_fp32[0, 0, :5].tolist()}")
    
    print("\n" + "="*70)
    print("Comparison Complete!")
    print("="*70)

if __name__ == "__main__":
    compare_forward_pass()
