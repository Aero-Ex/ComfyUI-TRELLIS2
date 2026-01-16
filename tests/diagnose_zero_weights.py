"""
GGUF Zero-Weight Diagnostic Script for TRELLIS2

Identifies which weights are all-zero after GGUF loading.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from trellis2 import models

def diagnose_zero_weights(model_path):
    print(f"\n{'='*60}")
    print(f"Diagnosing Zero Weights: {os.path.basename(model_path)}")
    print("="*60)
    
    print(f"\nLoading model from: {model_path}")
    model = models.from_pretrained(
        model_path,
        device="cpu",
        enable_gguf=True,
        gguf_quant="Q8_0"
    )
    
    # Find all-zero params
    zero_params = []
    non_zero_count = 0
    for name, param in model.named_parameters():
        if (param == 0).all():
            zero_params.append((name, param.shape, param.numel()))
        else:
            non_zero_count += 1
    
    print(f"\nTotal parameters: {non_zero_count + len(zero_params)}")
    print(f"Non-zero parameters: {non_zero_count}")
    print(f"All-zero parameters: {len(zero_params)}")
    
    if zero_params:
        print(f"\n{'='*60}")
        print("ALL-ZERO PARAMETERS:")
        print("="*60)
        for name, shape, numel in zero_params:
            print(f"  - {name}")
            print(f"      Shape: {shape}, Elements: {numel:,}")
    else:
        print("\n✓ All parameters have valid (non-zero) values!")
    
    # Also check state dict keys vs model keys
    print(f"\n{'='*60}")
    print("Checking for missing/unexpected keys:")
    print("="*60)
    
    # Reload raw state_dict
    from trellis2.utils import gguf_utils
    gguf_path = f"{model_path}_Q8_0.gguf"
    if not os.path.exists(gguf_path):
        gguf_path = f"{model_path}.gguf"
    
    sd, metadata = gguf_utils.load_gguf_checkpoint(gguf_path)
    
    model_keys = set(name for name, _ in model.named_parameters())
    sd_keys = set(sd.keys())
    
    missing_in_sd = model_keys - sd_keys
    extra_in_sd = sd_keys - model_keys
    
    if missing_in_sd:
        print(f"\nKeys in model but NOT in GGUF state_dict ({len(missing_in_sd)}):")
        for k in sorted(missing_in_sd)[:20]:
            print(f"  - {k}")
        if len(missing_in_sd) > 20:
            print(f"  ... and {len(missing_in_sd) - 20} more")
    
    if extra_in_sd:
        print(f"\nKeys in GGUF state_dict but NOT in model ({len(extra_in_sd)}):")
        for k in sorted(extra_in_sd)[:20]:
            print(f"  - {k}")
        if len(extra_in_sd) > 20:
            print(f"  ... and {len(extra_in_sd) - 20} more")
    
    if not missing_in_sd and not extra_in_sd:
        print("  ✓ All keys match perfectly!")
    
    return len(zero_params), missing_in_sd, extra_in_sd

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to model (without extension)")
    args = parser.parse_args()
    
    diagnose_zero_weights(args.model_path)
