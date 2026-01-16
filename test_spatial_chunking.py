import torch
import numpy as np
import sys
import os
from fractions import Fraction

# Add the custom node directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trellis2.modules.sparse import SparseTensor, sparse_cat, sparse_unbind
from trellis2.representations import Mesh

def test_spatial_chunking_logic():
    print("Testing Spatial Chunking Logic...")
    
    # 1. Create a dummy SparseTensor (8x8x8 grid)
    # Coords: [B, X, Y, Z]
    coords = []
    for x in range(8):
        for y in range(8):
            for z in range(8):
                coords.append([0, x, y, z])
    coords = torch.tensor(coords, dtype=torch.int32)
    feats = torch.arange(coords.shape[0], dtype=torch.float32).unsqueeze(1)
    
    # We need to mock the spatial_shape property if it's not there
    # In Trellis2, SparseTensor has spatial_shape
    x = SparseTensor(feats=feats, coords=coords)
    print(f"Input SparseTensor: {x.feats.shape[0]} voxels, spatial_shape={x.spatial_shape}")

    # Mock the pipeline class to test _spatial_chunked_call
    class MockPipeline:
        def __init__(self):
            self.device = torch.device('cpu')
            
        def _spatial_chunked_call(self, func, x, padding=1, **kwargs):
            # Copy-paste the implementation from trellis2_image_to_3d.py for testing
            coords = x.coords
            spatial_coords = coords[:, 1:]
            res = x.spatial_shape
            mid = [r // 2 for r in res]
            chunk_results = []
            
            for i in range(8):
                bx = (i >> 2) & 1
                by = (i >> 1) & 1
                bz = i & 1
                
                x_min, x_max = (0, mid[0]) if bx == 0 else (mid[0], res[0])
                y_min, y_max = (0, mid[1]) if by == 0 else (mid[1], res[1])
                z_min, z_max = (0, mid[2]) if bz == 0 else (mid[2], res[2])
                
                mask_padded = (
                    (spatial_coords[:, 0] >= x_min - padding) & (spatial_coords[:, 0] < x_max + padding) &
                    (spatial_coords[:, 1] >= y_min - padding) & (spatial_coords[:, 1] < y_max + padding) &
                    (spatial_coords[:, 2] >= z_min - padding) & (spatial_coords[:, 2] < z_max + padding)
                )
                
                if not mask_padded.any():
                    chunk_results.append(None)
                    continue
                    
                chunk_x = SparseTensor(
                    feats=x.feats[mask_padded],
                    coords=coords[mask_padded],
                    shape=x.shape,
                    scale=x._scale
                )
                
                res_chunk = func(chunk_x, **kwargs)
                
                # Crop
                if isinstance(res_chunk, SparseTensor):
                    c = res_chunk.coords[:, 1:]
                    mask_crop = (
                        (c[:, 0] >= x_min) & (c[:, 0] < x_max) &
                        (c[:, 1] >= y_min) & (c[:, 1] < y_max) &
                        (c[:, 2] >= z_min) & (c[:, 2] < z_max)
                    )
                    res_chunk = SparseTensor(
                        feats=res_chunk.feats[mask_crop],
                        coords=res_chunk.coords[mask_crop],
                        shape=res_chunk.shape,
                        scale=res_chunk._scale
                    )
                chunk_results.append(res_chunk)
                
            valid_results = [r for r in chunk_results if r is not None]
            if not valid_results: return None
            
            if isinstance(valid_results[0], SparseTensor):
                # Custom merge for spatial chunks (preserve batch indices)
                merged_feats = torch.cat([r.feats for r in valid_results], dim=0)
                merged_coords = torch.cat([r.coords for r in valid_results], dim=0)
                return SparseTensor(
                    feats=merged_feats, 
                    coords=merged_coords, 
                    shape=valid_results[0].shape, 
                    scale=valid_results[0]._scale
                )
            return valid_results

    pipeline = MockPipeline()

    # Test 1: Identity function (should return same tensor)
    def identity(x): return x
    
    print("Running Identity test...")
    out = pipeline._spatial_chunked_call(identity, x, padding=1)
    
    # Sort both by coords to compare
    def sort_sparse(st):
        # Sort by coords (B, X, Y, Z)
        c = st.coords.cpu().numpy()
        idx = np.lexsort((c[:, 3], c[:, 2], c[:, 1], c[:, 0]))
        return st.feats[idx], st.coords[idx]

    feats_in, coords_in = sort_sparse(x)
    feats_out, coords_out = sort_sparse(out)
    
    print(f"Input voxels: {feats_in.shape[0]}")
    print(f"Output voxels: {feats_out.shape[0]}")
    
    if feats_in.shape[0] != feats_out.shape[0]:
        print(f"❌ Voxel count mismatch! In: {feats_in.shape[0]}, Out: {feats_out.shape[0]}")
    
    assert torch.allclose(feats_in, feats_out), "Feats mismatch in identity test"
    assert torch.all(coords_in == coords_out), "Coords mismatch in identity test"
    print("✅ Identity test passed!")

    # Test 2: Function that uses neighbors (to verify padding)
    # We'll mock a "convolution" that sums neighbors within distance 1
    def mock_conv(st):
        # For each voxel, return sum of feats of itself and neighbors in the chunk
        # This will fail if padding is 0 and we are at the boundary
        new_feats = []
        for i in range(st.coords.shape[0]):
            c = st.coords[i, 1:]
            dist = torch.abs(st.coords[:, 1:] - c).sum(dim=1)
            mask = dist <= 1
            new_feats.append(st.feats[mask].sum().unsqueeze(0))
        return SparseTensor(feats=torch.stack(new_feats), coords=st.coords)

    print("Running Neighbor Sum test (verifying padding)...")
    # Reference: run on full tensor
    ref = mock_conv(x)
    # Chunked: run with padding
    chunked = pipeline._spatial_chunked_call(mock_conv, x, padding=1)
    
    feats_ref, _ = sort_sparse(ref)
    feats_chunked, _ = sort_sparse(chunked)
    
    assert torch.allclose(feats_ref, feats_chunked), "Padding test failed! Boundary artifacts detected."
    print("✅ Padding/Overlap test passed! No boundary artifacts.")

if __name__ == "__main__":
    try:
        test_spatial_chunking_logic()
        print("\n✨ ALL SPATIAL CHUNKING LOGIC TESTS PASSED!")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
