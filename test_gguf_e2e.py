"""
Comprehensive end-to-end test for GGUF TRELLIS2 inference.
This test simulates the actual forward pass with quantized weights.
"""
import sys
import os
import torch
import torch.nn as nn

# Add the custom node directory to sys.path
node_path = os.path.dirname(os.path.abspath(__file__))
if node_path not in sys.path:
    sys.path.append(node_path)

from trellis2.utils import gguf_utils
from trellis2.modules.sparse.linear import SparseLinear

GGUF_PATH = r"D:\@home\aero\comfy\ComfyUI\models\trellis2\ckpts\ss_flow_img_dit_1_3B_64_bf16_Q8_0.gguf"

def test_gguf_forward_pass():
    """Test end-to-end forward pass with GGUF quantized weights."""
    print("="*60)
    print("GGUF Forward Pass Test")
    print("="*60)
    
    if not os.path.exists(GGUF_PATH):
        print(f"[SKIP] GGUF file not found: {GGUF_PATH}")
        return False
    
    # Load GGUF checkpoint
    print(f"\n[1] Loading GGUF checkpoint...")
    state_dict, metadata = gguf_utils.load_gguf_checkpoint(GGUF_PATH)
    print(f"    Loaded {len(state_dict)} tensors")
    
    # Test cases with different tensor types
    test_cases = [
        {
            "name": "Q8_0 Quantized Linear (to_qkv)",
            "weight_key": "blocks.0.self_attn.to_qkv.weight",
            "bias_key": "blocks.0.self_attn.to_qkv.bias",
            "in_features": 1536,
            "out_features": 4608,
            "input_shape": (1, 4096, 1536),
        },
        {
            "name": "F32 Linear (input_layer)",
            "weight_key": "input_layer.weight",
            "bias_key": "input_layer.bias",
            "in_features": 8,
            "out_features": 1536,
            "input_shape": (1, 4096, 8),
        },
    ]
    
    all_passed = True
    for case in test_cases:
        print(f"\n[2] Testing: {case['name']}")
        
        weight_key = case["weight_key"]
        bias_key = case["bias_key"]
        
        if weight_key not in state_dict:
            print(f"    [SKIP] Weight key '{weight_key}' not found")
            continue
            
        weight = state_dict[weight_key]
        bias = state_dict.get(bias_key)
        
        print(f"    Weight Type: {getattr(weight, 'tensor_type', 'N/A')}")
        print(f"    Weight Logical Shape: {getattr(weight, 'tensor_shape', weight.shape)}")
        print(f"    Weight Underlying Shape: {weight.shape}")
        
        # Create a SparseLinear layer
        layer = SparseLinear(case["in_features"], case["out_features"], bias=bias is not None)
        
        # Load the weights using _load_from_state_dict
        layer_state = {
            "weight": weight,
        }
        if bias is not None:
            layer_state["bias"] = bias
            
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        layer._load_from_state_dict(layer_state, "", {}, False, missing_keys, unexpected_keys, error_msgs)
        
        if error_msgs:
            print(f"    [FAIL] Errors during load: {error_msgs}")
            all_passed = False
            continue
            
        # Move to GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        layer.to(device)
        
        # Create input tensor
        input_tensor = torch.randn(case["input_shape"], dtype=torch.float16, device=device)
        
        # Forward pass
        print(f"    Input Shape: {input_tensor.shape}")
        try:
            with torch.no_grad():
                output = layer(input_tensor)
            print(f"    Output Shape: {output.shape}")
            
            expected_output_shape = (*case["input_shape"][:-1], case["out_features"])
            if output.shape == torch.Size(expected_output_shape):
                print(f"    [PASS] Output shape matches expected: {expected_output_shape}")
            else:
                print(f"    [FAIL] Output shape mismatch! Expected {expected_output_shape}, got {output.shape}")
                all_passed = False
                
            # Verify output is not NaN or Inf
            if torch.isnan(output).any():
                print(f"    [FAIL] Output contains NaN values!")
                all_passed = False
            elif torch.isinf(output).any():
                print(f"    [FAIL] Output contains Inf values!")
                all_passed = False
            else:
                print(f"    [PASS] Output values are valid (no NaN/Inf)")
                
        except Exception as e:
            print(f"    [FAIL] Forward pass failed: {e}")
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("="*60)
    
    return all_passed

def test_dequantize_preserves_shape():
    """Test that dequantize_tensor preserves shape after Parameter wrapping and .to()."""
    print("\n" + "="*60)
    print("Dequantize Shape Preservation Test")
    print("="*60)
    
    if not os.path.exists(GGUF_PATH):
        print(f"[SKIP] GGUF file not found: {GGUF_PATH}")
        return False
    
    state_dict, _ = gguf_utils.load_gguf_checkpoint(GGUF_PATH)
    
    # Test Q8_0 tensor
    q8_key = "blocks.0.self_attn.to_qkv.weight"
    if q8_key in state_dict:
        tensor = state_dict[q8_key]
        expected_shape = getattr(tensor, "tensor_shape", tensor.shape)
        
        print(f"\n[1] Testing: {q8_key}")
        print(f"    Original tensor type: {type(tensor)}")
        print(f"    Original tensor_shape attr: {getattr(tensor, 'tensor_shape', 'MISSING!')}")
        print(f"    Original tensor.shape property: {tensor.shape}")
        
        # Wrap in Parameter (simulating model creation)
        param = nn.Parameter(tensor, requires_grad=False)
        print(f"    After Parameter(): param.data type: {type(param.data)}")
        print(f"    After Parameter(): param.data.tensor_shape: {getattr(param.data, 'tensor_shape', 'MISSING!')}")
        
        # Extract data and move to device (simulating cast_bias_weight fix)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data = param.data
        print(f"    Extracted data type: {type(data)}")
        print(f"    Extracted data tensor_shape: {getattr(data, 'tensor_shape', 'MISSING!')}")
        
        data_on_device = data.to(device)
        print(f"    After .to(device) type: {type(data_on_device)}")
        print(f"    After .to(device) tensor_shape: {getattr(data_on_device, 'tensor_shape', 'MISSING!')}")
        
        # Dequantize
        dequantized = gguf_utils.dequantize_tensor(data_on_device, dtype=torch.float16)
        print(f"    Dequantized shape: {dequantized.shape}")
        
        if dequantized.shape == expected_shape:
            print(f"    [PASS] Shape preserved correctly!")
            return True
        else:
            print(f"    [FAIL] Shape mismatch! Expected {expected_shape}, got {dequantized.shape}")
            return False
    
    return True

if __name__ == "__main__":
    result1 = test_dequantize_preserves_shape()
    result2 = test_gguf_forward_pass()
    
    sys.exit(0 if (result1 and result2) else 1)
