import sys
from unittest.mock import MagicMock
sys.modules['easydict'] = MagicMock()

import torch
import numpy as np
from unittest.mock import patch
from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

def test_inject_sampler_multi_image():
    # Mock pipeline and sampler
    pipeline = MagicMock(spec=Trellis2ImageTo3DPipeline)
    pipeline.default_pipeline_type = '1024_cascade'
    pipeline.device = 'cpu'
    
    sampler = MagicMock()
    sampler._inference_model = MagicMock(return_value="original_result")
    pipeline.sparse_structure_sampler = sampler
    
    # Mock cond
    cond = [torch.randn(1, 10, 1024) for _ in range(3)]
    neg_cond = torch.randn(1, 10, 1024)
    
    # Test Stochastic Mode
    print("\n--- Testing Stochastic Mode ---")
    with Trellis2ImageTo3DPipeline.inject_sampler_multi_image(pipeline, 'sparse_structure_sampler', num_images=3, num_steps=6, mode='stochastic'):
        # Step 1: t=1.0 (should use cond[0])
        pipeline.sparse_structure_sampler._inference_model(None, torch.randn(1, 3, 32, 32, 32), 1.0, cond, neg_cond=neg_cond, guidance_strength=7.5)
        # Step 1: t=1.0 again (should still use cond[0])
        pipeline.sparse_structure_sampler._inference_model(None, torch.randn(1, 3, 32, 32, 32), 1.0, cond, neg_cond=neg_cond, guidance_strength=7.5)
        # Step 2: t=0.8 (should use cond[1])
        pipeline.sparse_structure_sampler._inference_model(None, torch.randn(1, 3, 32, 32, 32), 0.8, cond, neg_cond=neg_cond, guidance_strength=7.5)
        
    # Verify calls
    calls = sampler._inference_model.call_args_list
    print(f"Total calls: {len(calls)}")
    # Check cond_i in calls
    # Note: cond_i is passed as cond=cond_i
    assert torch.equal(calls[0].kwargs['cond'], cond[0])
    assert torch.equal(calls[1].kwargs['cond'], cond[0])
    assert torch.equal(calls[2].kwargs['cond'], cond[1])
    print("Stochastic mode consistency verified!")

    # Reset mock
    sampler._inference_model.reset_mock()
    sampler._inference_model.return_value = torch.ones(1, 3, 32, 32, 32)

    # Test Multidiffusion Mode
    print("\n--- Testing Multidiffusion Mode ---")
    with Trellis2ImageTo3DPipeline.inject_sampler_multi_image(pipeline, 'sparse_structure_sampler', num_images=3, num_steps=6, mode='multidiffusion'):
        pipeline.sparse_structure_sampler._inference_model(None, torch.randn(1, 3, 32, 32, 32), 1.0, cond, neg_cond=neg_cond, guidance_strength=7.5)

    # Verify calls
    calls = sampler._inference_model.call_args_list
    print(f"Total calls: {len(calls)}")
    # Should be 3 pos calls (guidance_strength=1) + 1 neg call (guidance_strength=0) = 4 calls
    assert len(calls) == 4
    
    # Check guidance_strength in calls
    gs_calls = [c.kwargs['guidance_strength'] for c in calls]
    assert gs_calls == [1, 1, 1, 0]
    
    # Check cond in pos calls
    for i in range(3):
        assert torch.equal(calls[i].kwargs['cond'], cond[i])
    
    print("Multidiffusion mode CFG logic verified!")

if __name__ == "__main__":
    test_inject_sampler_multi_image()
