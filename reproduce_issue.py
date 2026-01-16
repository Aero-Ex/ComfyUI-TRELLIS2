import torch
import torch.nn as nn
import sys
import os

# Add the custom node directory to path
sys.path.append('/home/aero/comfy/ComfyUI/custom_nodes/ComfyUI-TRELLIS2')

try:
    from trellis2.modules.sparse.attention.modules import SparseMultiHeadAttention
    from trellis2.modules.sparse import VarLenTensor, SparseTensor
    print("Imports successful")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def verify_empty_forward():
    print("Initializing SparseMultiHeadAttention...")
    channels = 192
    num_heads = 12
    attn = SparseMultiHeadAttention(
        channels=channels,
        num_heads=num_heads,
        type="self",
        attn_mode="full"
    )
    
    # Mock an empty SparseTensor
    # feats shape [L, C] -> [0, 192]
    # layout is empty list? Or list of empty ranges?
    # Usually empty input means L=0
    print("Creating empty SparseTensor...")
    feats = torch.randn(0, channels)
    coords = torch.zeros(0, 4, dtype=torch.int)
    layout = []
    
    x = SparseTensor(
        feats=feats,
        coords=coords,
        layout=layout,
        spatial_shape=(1, 32, 32, 32) # Arbitrary
    )
    
    print(f"Running forward pass with {x.feats.shape} elements...")
    try:
        # Test 1: Forward pass (should hit the short-circuit I added)
        out = attn(x)
        print(f"Forward pass successful! Output shape: {out.feats.shape}")
        
        # Test 2: Standard deviation (triggered the new RuntimeError)
        print("Testing .std() on empty SparseTensor...")
        s = x.std(dim=1, keepdim=True)
        print(f"Std successful! Shape: {s.shape}")
        
        # Test 3: Dtype mismatch (Float input, Half model)
        print("Testing dtype mismatch (Float input, Half model)...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        attn_half = attn.half().to(device)
        # Correct SparseTensor initialization: feats, coords, shape as positional args
        coords = torch.zeros((10, 3), dtype=torch.int32).to(device)
        spatial_shape = torch.Size([10, 10, 10])
        x_float = SparseTensor(torch.randn(10, 192).to(device), coords, spatial_shape, layout=[slice(0, 10)])
        try:
            out_half = attn_half(x_float)
            print(f"Dtype mismatch test successful! Output dtype: {out_half.feats.dtype}")
        except Exception as e:
            print(f"Dtype mismatch test FAILED: {e}")
            import traceback
            traceback.print_exc()
            raise e

        return True
    except Exception as e:
        print(f"Forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if verify_empty_forward():
        print("VERIFICATION PASSED")
        sys.exit(0)
    else:
        print("VERIFICATION FAILED")
        sys.exit(1)
