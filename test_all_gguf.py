"""
E2E Comparison Test: All GGUF models vs their Safetensors counterparts
"""
import sys
import os
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trellis2.utils import gguf_utils
from trellis2.modules.sparse.linear import SparseLinear
from safetensors import safe_open

MODELS_DIR = Path(r"D:\@home\aero\comfy\ComfyUI\models\trellis2")

# GGUF -> Safetensors mapping
GGUF_TO_SAFETENSORS = {
    "ckpts/ss_flow_img_dit_1_3B_64_bf16_Q8_0.gguf": "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
    "ckpts/ss_flow_img_dit_1_3B_64_bf16.gguf": "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",  # F32/BF16 GGUF
    "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16_Q8_0.gguf": "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
    "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.gguf": "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
    "slat_flow_img2shape_dit_1_3B_1024_bf16_Q8_0.gguf": "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
}

def load_safetensors_weight(path, key):
    """Load a single weight from safetensors file."""
    with safe_open(path, framework="pt", device="cpu") as f:
        if key in f.keys():
            return f.get_tensor(key)
    return None

def find_test_key(gguf_sd):
    """Find a suitable test key (preferring to_qkv.weight for attention layers)."""
    # Try common layer patterns
    for key in gguf_sd.keys():
        if "to_qkv.weight" in key and "blocks.0" in key:
            return key.replace(".weight", "")
    for key in gguf_sd.keys():
        if ".weight" in key and "blocks.0" in key:
            return key.replace(".weight", "")
    return None

def compare_single_model(gguf_path, st_path):
    """Compare a single GGUF model against its safetensors counterpart."""
    gguf_name = gguf_path.name
    
    print(f"\n{'='*70}")
    print(f"Testing: {gguf_name}")
    print(f"{'='*70}")
    
    if not gguf_path.exists():
        print(f"  [SKIP] GGUF not found: {gguf_path}")
        return None
    
    if not st_path.exists():
        print(f"  [SKIP] Safetensors not found: {st_path}")
        return None
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load GGUF
        print(f"  Loading GGUF...")
        gguf_sd, metadata = gguf_utils.load_gguf_checkpoint(str(gguf_path))
        
        # Find test key
        test_key = find_test_key(gguf_sd)
        if not test_key:
            print(f"  [SKIP] No suitable test key found")
            return None
        
        print(f"  Test layer: {test_key}")
        
        gguf_weight = gguf_sd[f"{test_key}.weight"]
        gguf_bias = gguf_sd.get(f"{test_key}.bias")
        
        gguf_shape = getattr(gguf_weight, "tensor_shape", gguf_weight.shape)
        in_features = gguf_shape[-1]
        out_features = gguf_shape[0]
        
        print(f"  GGUF type: {getattr(gguf_weight, 'tensor_type', 'N/A')}, shape: {gguf_shape}")
        
        # Load Safetensors
        st_weight = load_safetensors_weight(str(st_path), f"{test_key}.weight")
        st_bias = load_safetensors_weight(str(st_path), f"{test_key}.bias")
        
        if st_weight is None:
            print(f"  [SKIP] Key not found in safetensors: {test_key}.weight")
            return None
        
        print(f"  Safetensors dtype: {st_weight.dtype}, shape: {st_weight.shape}")
        
        # Create GGUF layer
        gguf_layer = SparseLinear(in_features, out_features, bias=gguf_bias is not None)
        layer_sd = {"weight": gguf_weight}
        if gguf_bias is not None:
            layer_sd["bias"] = gguf_bias
        gguf_layer._load_from_state_dict(layer_sd, "", {}, False, [], [], [])
        gguf_layer.to(device)
        
        # Create Safetensors layer
        st_layer = SparseLinear(in_features, out_features, bias=st_bias is not None)
        st_layer.weight = nn.Parameter(st_weight.to(device))
        if st_bias is not None:
            st_layer.bias = nn.Parameter(st_bias.to(device))
        
        # Forward pass
        torch.manual_seed(42)
        test_input = torch.randn(1, 64, in_features, dtype=torch.float16, device=device)
        
        with torch.no_grad():
            gguf_output = gguf_layer(test_input)
            st_output = st_layer(test_input.to(st_weight.dtype))
        
        # Compare
        gguf_fp32 = gguf_output.float()
        st_fp32 = st_output.float()
        
        abs_diff = (gguf_fp32 - st_fp32).abs()
        correlation = torch.corrcoef(torch.stack([gguf_fp32.flatten(), st_fp32.flatten()]))[0, 1].item()
        
        max_abs = abs_diff.max().item()
        mean_abs = abs_diff.mean().item()
        
        result = {
            "model": gguf_name,
            "qtype": str(getattr(gguf_weight, 'tensor_type', 'N/A')),
            "correlation": correlation,
            "max_diff": max_abs,
            "mean_diff": mean_abs,
            "pass": correlation > 0.99 and max_abs < 1.0
        }
        
        status = "PASS" if result["pass"] else "FAIL"
        print(f"  Correlation: {correlation:.6f}")
        print(f"  Max/Mean Diff: {max_abs:.4f} / {mean_abs:.6f}")
        print(f"  [{status}]")
        
        return result
        
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {"model": gguf_name, "error": str(e), "pass": False}

def main():
    print("="*70)
    print("GGUF vs Safetensors Comparison - All Models")
    print("="*70)
    
    results = []
    
    for gguf_rel, st_rel in GGUF_TO_SAFETENSORS.items():
        gguf_path = MODELS_DIR / gguf_rel
        st_path = MODELS_DIR / st_rel
        
        result = compare_single_model(gguf_path, st_path)
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Model':<55} {'Corr':>8} {'Status':>8}")
    print("-"*70)
    
    passed = 0
    failed = 0
    for r in results:
        if "error" in r:
            print(f"{r['model']:<55} {'ERROR':>8} {'FAIL':>8}")
            failed += 1
        else:
            status = "PASS" if r["pass"] else "FAIL"
            print(f"{r['model']:<55} {r['correlation']:>8.4f} {status:>8}")
            if r["pass"]:
                passed += 1
            else:
                failed += 1
    
    print("-"*70)
    print(f"Total: {passed} passed, {failed} failed out of {len(results)} tested")
    print("="*70)

if __name__ == "__main__":
    main()
