import sys
import os
import torch

# Add the custom node directory to sys.path
node_path = os.path.dirname(os.path.abspath(__file__))
if node_path not in sys.path:
    sys.path.append(node_path)

from trellis2.utils import gguf_utils

def test_loading():
    gguf_path = r"D:\@home\aero\comfy\ComfyUI\models\trellis2\ckpts\ss_flow_img_dit_1_3B_64_bf16_Q8_0.gguf"
    if not os.path.exists(gguf_path):
        print(f"Error: GGUF file not found at {gguf_path}")
        return

    print(f"Loading GGUF: {gguf_path}")
    state_dict, metadata = gguf_utils.load_gguf_checkpoint(gguf_path)
    
    print(f"\nMetadata: {metadata}")
    
    # Test cases:
    # 1. BF16 tensor (should be viewed as bfloat16 and have correct shape)
    bf16_key = "blocks.0.cross_attn.k_rms_norm.gamma"
    if bf16_key in state_dict:
        tensor = state_dict[bf16_key]
        print(f"\nTensor: {bf16_key}")
        print(f"  Type: {getattr(tensor, 'tensor_type', 'N/A')}")
        print(f"  Logical Shape: {getattr(tensor, 'tensor_shape', 'N/A')}")
        print(f"  Underlying Shape: {tensor.shape}")
        
        # Dequantize
        dequantized = gguf_utils.dequantize_tensor(tensor)
        print(f"  Dequantized Shape: {dequantized.shape}")
        print(f"  Dequantized Dtype: {dequantized.dtype}")
        
        expected_shape = torch.Size([12, 128]) # Based on error log shape '[12, 128]'
        if dequantized.shape == expected_shape:
            print("  [PASS] Shape matches expected.")
        else:
            print(f"  [FAIL] Shape mismatch! Expected {expected_shape}, got {dequantized.shape}")
            
    # 2. Q8_0 tensor (should be flat and dequantize to logical shape)
    q8_key = "blocks.0.self_attn.to_qkv.weight"
    if q8_key in state_dict:
        tensor = state_dict[q8_key]
        print(f"\nTensor: {q8_key}")
        print(f"  Type: {getattr(tensor, 'tensor_type', 'N/A')}")
        print(f"  Logical Shape: {getattr(tensor, 'tensor_shape', 'N/A')}")
        print(f"  Underlying Shape: {tensor.shape}")
        
        # Dequantize
        dequantized = gguf_utils.dequantize_tensor(tensor)
        print(f"  Dequantized Shape: {dequantized.shape}")
        print(f"  Dequantized Dtype: {dequantized.dtype}")
        
    # 3. F32 tensor (should have correct 2D shape and NOT be flat)
    f32_key = "input_layer.weight"
    if f32_key in state_dict:
        tensor = state_dict[f32_key]
        print(f"\nTensor: {f32_key}")
        print(f"  Type: {getattr(tensor, 'tensor_type', 'N/A')}")
        print(f"  Underlying Shape: {tensor.shape}")
        
        expected_shape = torch.Size([1536, 8])
        if tensor.shape == expected_shape:
            print("  [PASS] Underlying shape matches expected 2D shape.")
        else:
            print(f"  [FAIL] Underlying shape is {tensor.shape}, expected {expected_shape}")
            
        # Verify it's not a GGMLTensor after dequantize_tensor (though it's F32)
        dequantized = gguf_utils.dequantize_tensor(tensor)
        print(f"  Dequantized Shape: {dequantized.shape}")
        if not isinstance(dequantized, gguf_utils.GGMLTensor):
            print("  [PASS] Dequantized result is a regular Tensor.")
        else:
            print("  [FAIL] Dequantized result is still a GGMLTensor!")

if __name__ == "__main__":
    test_loading()
