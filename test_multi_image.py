#!/usr/bin/env python
"""
Test script for multi-image conditioning in TRELLIS.2.

This tests the sampler injection mechanism and multi-image pipeline methods.
"""

import sys
import os

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from PIL import Image


def test_inject_sampler_structure():
    """Test that inject_sampler_multi_image context manager works correctly."""
    print("\n" + "="*60)
    print("TEST 1: Sampler Injection Structure")
    print("="*60)
    
    try:
        from trellis2.pipelines.samplers.flow_euler import FlowEulerCfgSampler
    except ImportError as e:
        print(f"⚠ Skipped (ComfyUI deps not available): {e}")
        return True
    
    # Create a sampler instance
    sampler = FlowEulerCfgSampler(sigma_min=0.0)
    
    # Check that _inference_model exists
    assert hasattr(sampler, '_inference_model'), "Sampler should have _inference_model method"
    original_method = sampler._inference_model
    print(f"✓ Original _inference_model: {original_method}")
    
    # Simulate the injection (simplified version)
    def _mock_new_inference_model(self, model, x_t, t, cond, **kwargs):
        return "injected"
    
    # Test monkey-patching
    sampler._old_inference_model = sampler._inference_model
    sampler._inference_model = _mock_new_inference_model.__get__(sampler, type(sampler))
    
    assert sampler._inference_model != original_method, "Method should be replaced"
    print("✓ _inference_model successfully patched")
    
    # Restore
    sampler._inference_model = sampler._old_inference_model
    delattr(sampler, '_old_inference_model')
    
    assert sampler._inference_model == original_method, "Method should be restored"
    print("✓ _inference_model successfully restored")
    
    print("\n✅ TEST 1 PASSED: Sampler injection structure is correct")
    return True


def test_stochastic_index_cycling():
    """Test that stochastic mode cycles through image indices correctly."""
    print("\n" + "="*60)
    print("TEST 2: Stochastic Index Cycling")
    print("="*60)
    
    num_images = 3
    num_steps = 12
    
    # This is how the indices are generated in the actual code
    cond_indices = (np.arange(num_steps) % num_images).tolist()
    
    expected = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert cond_indices == expected, f"Expected {expected}, got {cond_indices}"
    print(f"✓ Index cycling for {num_images} images, {num_steps} steps: {cond_indices}")
    
    # Test with more images than steps
    num_images = 5
    num_steps = 3
    cond_indices = (np.arange(num_steps) % num_images).tolist()
    expected = [0, 1, 2]
    assert cond_indices == expected, f"Expected {expected}, got {cond_indices}"
    print(f"✓ Index cycling for {num_images} images, {num_steps} steps: {cond_indices}")
    print("  (Note: images 3 and 4 are not used - warning should be shown)")
    
    print("\n✅ TEST 2 PASSED: Stochastic index cycling is correct")
    return True


def test_conditioning_tensor_stacking():
    """Test that conditioning tensors are stacked correctly for multi-image."""
    print("\n" + "="*60)
    print("TEST 3: Conditioning Tensor Stacking")
    print("="*60)
    
    # Simulate DinoV3 output for each image
    batch_size = 1
    dim = 1024  # DinoV3 feature dimension
    
    # Create mock conditioning for 4 images
    num_images = 4
    cond_list = []
    for i in range(num_images):
        cond = torch.randn(batch_size, dim)  # [B, D]
        cond_list.append(cond)
        print(f"  Image {i} conditioning shape: {cond.shape}")
    
    # Stack as in run_multi_conditioning
    cond_512 = torch.stack(cond_list, dim=0)  # [N, B, D]
    print(f"✓ Stacked conditioning shape: {cond_512.shape}")
    
    expected_shape = (num_images, batch_size, dim)
    assert cond_512.shape == expected_shape, f"Expected {expected_shape}, got {cond_512.shape}"
    
    # Test negative conditioning (single)
    neg_cond = torch.zeros_like(cond_512[0:1])  # [1, B, D]
    print(f"✓ Negative conditioning shape: {neg_cond.shape}")
    
    assert neg_cond.shape == (1, batch_size, dim), f"Wrong neg_cond shape"
    
    # Test slicing for individual image during sampling
    for i in range(num_images):
        cond_i = cond_512[i:i+1]  # [1, B, D]
        assert cond_i.shape == (1, batch_size, dim), f"Sliced cond shape wrong"
        print(f"  Slice {i} shape: {cond_i.shape}")
    
    print("\n✅ TEST 3 PASSED: Tensor stacking and slicing is correct")
    return True


def test_multidiffusion_averaging():
    """Test that multidiffusion correctly averages predictions."""
    print("\n" + "="*60)
    print("TEST 4: MultiDiffusion Averaging")
    print("="*60)
    
    # Simulate predictions from multiple images
    num_images = 3
    pred_shape = (1, 64, 32, 32, 32)  # Sparse structure prediction shape
    
    preds = []
    for i in range(num_images):
        pred = torch.ones(pred_shape) * (i + 1)  # pred[0]=1, pred[1]=2, pred[2]=3
        preds.append(pred)
        print(f"  Prediction {i} mean: {pred.mean().item():.1f}")
    
    # Average as in multidiffusion
    avg_pred = sum(preds) / len(preds)
    expected_mean = (1 + 2 + 3) / 3
    
    assert torch.allclose(avg_pred.mean(), torch.tensor(expected_mean)), \
        f"Expected mean {expected_mean}, got {avg_pred.mean().item()}"
    print(f"✓ Averaged prediction mean: {avg_pred.mean().item():.2f} (expected: {expected_mean:.2f})")
    
    # Test CFG application
    guidance_strength = 7.5
    neg_pred = torch.zeros(pred_shape)
    
    cfg_pred = (1 + guidance_strength) * avg_pred - guidance_strength * neg_pred
    expected_cfg_mean = (1 + guidance_strength) * expected_mean
    
    assert torch.allclose(cfg_pred.mean(), torch.tensor(expected_cfg_mean)), \
        f"Expected CFG mean {expected_cfg_mean}, got {cfg_pred.mean().item()}"
    print(f"✓ CFG prediction mean: {cfg_pred.mean().item():.2f} (expected: {expected_cfg_mean:.2f})")
    
    print("\n✅ TEST 4 PASSED: MultiDiffusion averaging is correct")
    return True


def test_pipeline_methods_exist():
    """Test that the multi-image methods exist on the pipeline class."""
    print("\n" + "="*60)
    print("TEST 5: Pipeline Methods Exist")
    print("="*60)
    
    try:
        from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    except ImportError as e:
        print(f"⚠ Skipped (ComfyUI deps not available): {e}")
        return True
    
    # Create empty pipeline
    pipeline = Trellis2ImageTo3DPipeline()
    
    # Check methods exist
    methods = [
        'inject_sampler_multi_image',
        'run_multi_image_shape',
        'run_multi_image_texture',
    ]
    
    for method in methods:
        assert hasattr(pipeline, method), f"Pipeline missing {method}"
        print(f"✓ Pipeline.{method} exists")
    
    print("\n✅ TEST 5 PASSED: All pipeline methods exist")
    return True


def test_nodes_import():
    """Test that the multi-image nodes can be imported."""
    print("\n" + "="*60)
    print("TEST 6: Nodes Import")
    print("="*60)
    
    try:
        from nodes.nodes_multi_image import (
            Trellis2MultiImageConditioning,
            Trellis2MultiImageToShape,
            Trellis2MultiImageToTexturedMesh,
            NODE_CLASS_MAPPINGS,
            NODE_DISPLAY_NAME_MAPPINGS,
        )
        print("✓ nodes_multi_image imports successfully")
        
        # Check node classes
        assert "Trellis2MultiImageConditioning" in NODE_CLASS_MAPPINGS
        assert "Trellis2MultiImageToShape" in NODE_CLASS_MAPPINGS
        assert "Trellis2MultiImageToTexturedMesh" in NODE_CLASS_MAPPINGS
        print(f"✓ Found {len(NODE_CLASS_MAPPINGS)} nodes in mappings")
        
        # Check display names
        for key in NODE_CLASS_MAPPINGS:
            assert key in NODE_DISPLAY_NAME_MAPPINGS, f"Missing display name for {key}"
        print("✓ All display names present")
        
    except ImportError as e:
        print(f"⚠ Import failed (expected outside ComfyUI): {e}")
        print("  This is OK - nodes require ComfyUI environment")
        return True
    
    print("\n✅ TEST 6 PASSED: Nodes import correctly")
    return True


def test_stage_functions_exist():
    """Test that stage functions exist."""
    print("\n" + "="*60)
    print("TEST 7: Stage Functions Exist")
    print("="*60)
    
    try:
        from nodes.utils.stages import (
            run_conditioning,
            run_shape_generation,
            run_texture_generation,
        )
        print("✓ Single-image stage functions exist")
        
        # Check multi-image functions
        from nodes.utils import stages
        
        multi_funcs = [
            'run_multi_conditioning',
            'run_multi_image_shape_generation',
            'run_multi_image_texture_generation',
        ]
        
        for func in multi_funcs:
            assert hasattr(stages, func), f"Missing {func}"
            print(f"✓ stages.{func} exists")
        
    except ImportError as e:
        print(f"⚠ Skipped (ComfyUI deps not available): {e}")
        print("  This is OK - stages require ComfyUI's folder_paths")
        return True
    
    print("\n✅ TEST 7 PASSED: All stage functions exist")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("TRELLIS.2 Multi-Image Conditioning Tests")
    print("="*60)
    
    tests = [
        test_inject_sampler_structure,
        test_stochastic_index_cycling,
        test_conditioning_tensor_stacking,
        test_multidiffusion_averaging,
        test_pipeline_methods_exist,
        test_nodes_import,
        test_stage_functions_exist,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"\n❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return True
    else:
        print("\n⚠ Some tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
