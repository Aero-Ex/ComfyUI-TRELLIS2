"""
GGUF Robustness Test Suite for TRELLIS2

This script tests the GGUF loading pipeline at multiple levels:
1. Basic GGUF file reading and metadata extraction
2. GGMLTensor creation and dequantization
3. Model loading via from_pretrained
4. Forward pass with dummy inputs
5. Comparison with safetensors output (if available)

Usage:
    python tests/test_gguf_robustness.py [model_path]
    
Example:
    python tests/test_gguf_robustness.py D:/models/trellis2/ckpts/ss_flow_img_dit_1_3B_64_bf16
"""

import sys
import os
import argparse
import traceback

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

def print_header(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_result(name, passed, details=""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {name}")
    if details:
        for line in details.split("\n"):
            print(f"           {line}")

def test_gguf_file_reading(gguf_path):
    """Test 1: Basic GGUF file reading"""
    print_header("Test 1: GGUF File Reading")
    
    try:
        import gguf
        reader = gguf.GGUFReader(gguf_path)
        
        # Check tensors
        tensor_count = len(reader.tensors)
        print_result("File readable", True, f"Found {tensor_count} tensors")
        
        # Check metadata
        metadata = {}
        for field_name in reader.fields:
            try:
                field = reader.get_field(field_name)
                if len(field.types) == 1:
                    if field.types[0] == gguf.GGUFValueType.STRING:
                        metadata[field_name] = str(field.parts[field.data[-1]], "utf-8")
                    elif field.types[0] == gguf.GGUFValueType.INT32:
                        metadata[field_name] = int(field.parts[field.data[-1]])
            except:
                pass
        
        print_result("Metadata extracted", len(metadata) > 0, 
                     f"Found {len(metadata)} metadata fields")
        
        # Show some key metadata
        arch = metadata.get("general.architecture", "unknown")
        print_result("Architecture detected", True, f"Architecture: {arch}")
        
        # Count tensor types
        type_counts = {}
        for tensor in reader.tensors:
            qtype = tensor.tensor_type.name
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
        
        type_str = ", ".join(f"{k}:{v}" for k, v in type_counts.items())
        print_result("Tensor types", True, type_str)
        
        return True, reader, metadata
        
    except Exception as e:
        print_result("File readable", False, str(e))
        traceback.print_exc()
        return False, None, None

def test_gguf_tensor_loading(reader):
    """Test 2: GGMLTensor creation"""
    print_header("Test 2: GGMLTensor Creation")
    
    try:
        from trellis2.utils.gguf_utils import GGMLTensor, get_orig_shape
        import warnings
        
        sample_tensors = []
        for i, tensor in enumerate(reader.tensors[:5]):  # Test first 5
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
                torch_tensor = torch.from_numpy(tensor.data)
            
            shape = get_orig_shape(reader, tensor.name)
            if shape is None:
                shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            
            ggml_tensor = GGMLTensor(
                torch_tensor, 
                tensor_type=tensor.tensor_type, 
                tensor_shape=shape
            )
            sample_tensors.append((tensor.name, ggml_tensor))
            
        print_result("GGMLTensor creation", True, 
                     f"Created {len(sample_tensors)} sample tensors")
        
        # Check tensor properties
        for name, t in sample_tensors[:3]:
            props = f"{name}: shape={t.shape}, type={t.tensor_type.name}"
            print_result("Tensor properties", True, props)
        
        return True, sample_tensors
        
    except Exception as e:
        print_result("GGMLTensor creation", False, str(e))
        traceback.print_exc()
        return False, None

def test_dequantization(sample_tensors):
    """Test 3: Dequantization"""
    print_header("Test 3: Dequantization")
    
    try:
        from trellis2.utils.gguf_utils import dequantize_tensor, is_quantized
        
        for name, tensor in sample_tensors:
            is_quant = is_quantized(tensor)
            
            # Dequantize to float32
            dequant = dequantize_tensor(tensor, dtype=torch.float32)
            
            # Check for NaN/Inf
            has_nan = torch.isnan(dequant).any().item()
            has_inf = torch.isinf(dequant).any().item()
            
            # Check for all zeros
            all_zeros = (dequant == 0).all().item()
            
            # Check stats
            mean = dequant.mean().item()
            std = dequant.std().item()
            min_val = dequant.min().item()
            max_val = dequant.max().item()
            
            status = "GOOD"
            if has_nan:
                status = "HAS NaN!"
            elif has_inf:
                status = "HAS Inf!"
            elif all_zeros:
                status = "ALL ZEROS!"
            elif std == 0:
                status = "NO VARIANCE!"
            
            details = f"mean={mean:.4f}, std={std:.4f}, min={min_val:.4f}, max={max_val:.4f}"
            print_result(f"{name[:40]}", status == "GOOD", f"{status} | {details}")
            
            if status != "GOOD":
                return False, status
        
        return True, None
        
    except Exception as e:
        print_result("Dequantization", False, str(e))
        traceback.print_exc()
        return False, str(e)

def test_model_loading(model_path):
    """Test 4: Full model loading via from_pretrained"""
    print_header("Test 4: Model Loading")
    
    try:
        from trellis2 import models
        
        # Load with GGUF enabled
        print(f"  Loading model from: {model_path}")
        model = models.from_pretrained(
            model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            enable_gguf=True,
            gguf_quant="Q8_0"
        )
        
        all_zeros = (output_tensor == 0).all().item()
        
        mean = output_tensor.float().mean().item()
        std = output_tensor.float().std().item()
        
        print_result("Output computed", True, f"shape={output_tensor.shape}")
        print_result("No NaN in output", not has_nan)
        print_result("No Inf in output", not has_inf)
        print_result("Non-zero output", not all_zeros, f"mean={mean:.6f}, std={std:.6f}")
        
        if has_nan or has_inf or all_zeros:
            return False
        
        return True
        
    except Exception as e:
        print_result("Forward pass", False, str(e))
        traceback.print_exc()
        return False

def test_compare_with_safetensors(model_path, gguf_model):
    """Test 6: Compare GGUF output with Safetensors (if available)"""
    print_header("Test 6: GGUF vs Safetensors Comparison")
    
    safetensors_path = f"{model_path}.safetensors"
    if not os.path.exists(safetensors_path):
        print_result("Safetensors comparison", True, "Skipped - no safetensors file")
        return True
    
    try:
        from trellis2 import models
        
        print(f"  Loading safetensors model for comparison...")
        device = next(gguf_model.parameters()).device
        
        sf_model = models.from_pretrained(
            model_path,
            device=str(device),
            enable_gguf=False
        )
        
        # Create identical inputs
        torch.manual_seed(42)
        if hasattr(gguf_model, 'resolution'):
            reso = gguf_model.resolution
            in_ch = gguf_model.in_channels
            x = torch.randn(1, in_ch, reso, reso, reso, device=device, dtype=torch.float16)
            t = torch.tensor([500.0], device=device, dtype=torch.float32)
            cond = torch.randn(1, 512, 1536, device=device, dtype=torch.float16)
            
            with torch.no_grad():
                gguf_out = gguf_model(x, t, cond)
                sf_out = sf_model(x, t, cond)
            
            gguf_tensor = gguf_out if isinstance(gguf_out, torch.Tensor) else gguf_out.feats
            sf_tensor = sf_out if isinstance(sf_out, torch.Tensor) else sf_out.feats
        else:
            print_result("Comparison", True, "Skipped - sparse model comparison not implemented")
            return True
        
        # Compare
        mse = ((gguf_tensor.float() - sf_tensor.float()) ** 2).mean().item()
        cosine = torch.nn.functional.cosine_similarity(
            gguf_tensor.float().flatten().unsqueeze(0),
            sf_tensor.float().flatten().unsqueeze(0)
        ).item()
        
        print_result("MSE", mse < 0.01, f"MSE = {mse:.6f}")
        print_result("Cosine Similarity", cosine > 0.99, f"Cosine = {cosine:.6f}")
        
        return mse < 0.1 and cosine > 0.9
        
    except Exception as e:
        print_result("Comparison", False, str(e))
        traceback.print_exc()
        return False

def find_test_models():
    """Find available GGUF models for testing"""
    try:
        import folder_paths
        models_dir = os.path.join(folder_paths.models_dir, "trellis2")
    except:
        models_dir = None
    
    if models_dir is None or not os.path.exists(models_dir):
        # Fallback paths
        possible_dirs = [
            "D:/models/trellis2",
            "D:/@home/aero/comfy/ComfyUI/models/trellis2",
            os.path.expanduser("~/models/trellis2"),
        ]
        for d in possible_dirs:
            if os.path.exists(d):
                models_dir = d
                break
    
    if not models_dir or not os.path.exists(models_dir):
        return []
    
    gguf_files = []
    for root, dirs, files in os.walk(models_dir):
        for f in files:
            if f.endswith(".gguf"):
                # Return path without extension
                base = os.path.join(root, f.rsplit(".", 1)[0])
                # Remove quantization suffix if present
                for qsuffix in ["_Q8_0", "_Q6_K", "_Q5_K_M", "_Q4_K_M"]:
                    if base.endswith(qsuffix):
                        base = base[:-len(qsuffix)]
                        break
                if base not in gguf_files:
                    gguf_files.append(base)
    
    return gguf_files

def main():
    parser = argparse.ArgumentParser(description="GGUF Robustness Test Suite")
    parser.add_argument("model_path", nargs="?", help="Path to model (without extension)")
    parser.add_argument("--all", action="store_true", help="Test all found models")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  TRELLIS2 GGUF Robustness Test Suite")
    print("="*60)
    
    if args.all or args.model_path is None:
        models = find_test_models()
        if not models:
            print("\nNo GGUF models found! Please specify a model path.")
            print("Usage: python test_gguf_robustness.py /path/to/model")
            return 1
        print(f"\nFound {len(models)} models to test")
    else:
        models = [args.model_path]
    
    all_passed = True
    for model_path in models:
        print(f"\n{'='*60}")
        print(f"Testing: {os.path.basename(model_path)}")
        print("="*60)
        
        # Find GGUF file
        gguf_path = None
        for suffix in ["_Q8_0.gguf", ".gguf"]:
            if os.path.exists(model_path + suffix):
                gguf_path = model_path + suffix
                break
        
        if not gguf_path:
            print(f"  ERROR: No GGUF file found for {model_path}")
            all_passed = False
            continue
        
        print(f"  GGUF file: {os.path.basename(gguf_path)}")
        
        # Run tests
        passed, reader, metadata = test_gguf_file_reading(gguf_path)
        if not passed:
            all_passed = False
            continue
        
        passed, tensors = test_gguf_tensor_loading(reader)
        if not passed:
            all_passed = False
            continue
        
        passed, error = test_dequantization(tensors)
        if not passed:
            all_passed = False
            continue
        
        passed, model = test_model_loading(model_path)
        if not passed:
            all_passed = False
            continue
        
        passed = test_forward_pass(model)
        if not passed:
            all_passed = False
            continue
        
        passed = test_compare_with_safetensors(model_path, model)
        if not passed:
            all_passed = False
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    if all_passed:
        print("  ALL TESTS PASSED!")
    else:
        print("  SOME TESTS FAILED - See above for details")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
