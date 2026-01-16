import torch
import sys
import os

# Add custom nodes path to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trellis2.modules.sparse.basic import SparseTensor
from trellis2.modules.sparse import config

def test_empty_sparse():
    print("Testing empty SparseTensor...")
    config.set_conv_backend('none')
    print(f"Current conv backend: {config.get_conv_backend()}")

    feats = torch.randn(0, 16)
    coords = torch.zeros(0, 4, dtype=torch.int32)
    
    print("Initializing SparseTensor with empty coords...")
    try:
        sp = SparseTensor(feats, coords)
        print("Successfully created SparseTensor")
        
        print("Accessing shape...")
        print(f"Shape: {sp.shape}")
        
        print("Accessing layout...")
        print(f"Layout: {sp.layout}")
        
        print("Accessing spatial_shape...")
        print(f"Spatial shape: {sp.spatial_shape}")
        
    except Exception as e:
        print(f"Failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_empty_sparse()
