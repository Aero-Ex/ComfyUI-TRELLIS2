# Adapted from ComfyUI-GGUF (c) City96 || Apache-2.0
import sys
import os
import gguf
import torch
import sys
import logging
import warnings
import re
import importlib.util
from typing import Dict, Any, Tuple, Optional, List

# Try to find ComfyUI-GGUF for native support
def _setup_native_gguf():
    global GGMLTensor, GGMLLayer, dequantize_tensor, is_quantized, HAS_GGUF_OPS, get_orig_shape
    
    # Try local import first (in case it's in sys.path)
    try:
        from . import dequant as local_dequant
    except:
        local_dequant = None

    # Check for ComfyUI-GGUF in custom_nodes
    # We go up 3 levels from trellis2/utils/ to reach custom_nodes/
    custom_nodes_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    gguf_path = os.path.join(custom_nodes_path, "ComfyUI-GGUF")
    
    if not os.path.isdir(gguf_path):
        return False
        
    try:
        # To avoid "attempted relative import with no known parent package",
        # we need to load these modules as part of a package.
        # Since "ComfyUI-GGUF" has a dash, we use importlib.
        
        def load_module(name, path):
            spec = importlib.util.spec_from_file_location(f"gguf_native.{name}", path)
            module = importlib.util.module_from_spec(spec)
            # Set a dummy parent package to allow relative imports within the GGUF node
            module.__package__ = "gguf_native"
            sys.modules[f"gguf_native.{name}"] = module
            spec.loader.exec_module(module)
            return module

        # Create a dummy parent package in sys.modules
        import types
        gguf_native = types.ModuleType("gguf_native")
        gguf_native.__path__ = [gguf_path]
        sys.modules["gguf_native"] = gguf_native

        # Load modules in order
        gguf_dequant = load_module("dequant", os.path.join(gguf_path, "dequant.py"))
        gguf_ops = load_module("ops", os.path.join(gguf_path, "ops.py"))
        gguf_loader = load_module("loader", os.path.join(gguf_path, "loader.py"))
        
        GGMLTensor = gguf_ops.GGMLTensor
        GGMLLayer = gguf_ops.GGMLLayer
        dequantize_tensor = gguf_dequant.dequantize_tensor
        is_quantized = gguf_dequant.is_quantized
        get_orig_shape = gguf_loader.get_orig_shape
        HAS_GGUF_OPS = True
        print("[TRELLIS2-DEBUG] Using native ComfyUI-GGUF support (ops/dequant/loader)", file=sys.stderr)
        return True
    except Exception as e:
        # This is expected in isolated subprocess envs where 'comfy' module isn't available
        # The internal fallback works correctly, so just note it silently in debug mode
        if "comfy" in str(e):
            pass  # Expected - using internal fallback silently
        else:
            print(f"[TRELLIS2-DEBUG] ComfyUI-GGUF import failed: {e}. Using internal GGUF implementation.", file=sys.stderr)
        # Clean up partial imports on failure
        for k in list(sys.modules.keys()):
            if k.startswith("gguf_native"):
                del sys.modules[k]
        return False

# --- Fallback Implementations (if ComfyUI-GGUF is missing) ---

TORCH_COMPATIBLE_QTYPES = (None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16)

QTYPES_TO_DTYPE = {
    gguf.GGMLQuantizationType.F32: torch.float32,
    gguf.GGMLQuantizationType.F16: torch.float16,
    gguf.GGMLQuantizationType.BF16: torch.bfloat16,
}

# FP8 Scaled Quantization Support
# FP8 uses native PyTorch dtypes with per-tensor scales (unlike GGML block quantization)
FP8_DTYPES = set()
try:
    FP8_DTYPES.add(torch.float8_e4m3fn)
    FP8_DTYPES.add(torch.float8_e5m2)
except AttributeError:
    # PyTorch version doesn't support FP8
    pass

def is_fp8_tensor(tensor):
    """Check if a tensor is FP8 quantized (native PyTorch FP8 dtype)."""
    if tensor is None:
        return False
    return tensor.dtype in FP8_DTYPES

def dequantize_fp8(tensor, scale=None, target_dtype=torch.bfloat16):
    """Dequantize FP8 tensor using per-tensor scale.
    
    Args:
        tensor: FP8 tensor (float8_e4m3fn or float8_e5m2)
        scale: Per-tensor scale factor. If None, uses 1.0.
        target_dtype: Output dtype (default: bfloat16)
        
    Returns:
        Dequantized tensor: tensor.to(float32) * scale
    """
    if scale is None:
        scale = torch.ones((), device=tensor.device, dtype=torch.float32)
    
    # FP8 dequantization: weight_fp32 = weight_fp8 * scale
    result = tensor.to(torch.float32) * scale.to(tensor.device).float()
    return result.to(target_dtype)

def is_torch_compatible(tensor):
    """Check if tensor can be used directly without dequantization."""
    return tensor is None or getattr(tensor, "tensor_type", None) in TORCH_COMPATIBLE_QTYPES

def is_quantized(tensor):
    """Check if tensor requires dequantization (GGML or FP8)."""
    if is_fp8_tensor(tensor):
        return True
    return not is_torch_compatible(tensor)

def to_uint32(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)

def to_uint16(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8).unsqueeze(1)

def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)

def dequantize_blocks_BF16(blocks, block_size, type_size, dtype=None):
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)

def dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    d, x = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return (d * x)

def dequantize_blocks_Q5_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, m, qh, qs = split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = to_uint32(qh)
    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))
    qs = (ql | (qh << 4))
    return (d * qs) + m

def dequantize_blocks_Q5_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qh, qs = split_block_dims(blocks, 2, 4)
    d  = d.view(torch.float16).to(dtype)
    qh = to_uint32(qh)
    qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)
    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return (d * qs)

def dequantize_blocks_Q4_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, m, qs = split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)
    return (d * qs) + m

def dequantize_blocks_Q4_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qs = split_block_dims(blocks, 2)
    d  = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int8) - 8
    return (d * qs)

# K Quants #
QK_K = 256
K_SCALE_SIZE = 12

def get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))
    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    min = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))

def dequantize_blocks_Q6_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    ql, qh, scales, d, = split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)
    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))
    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 4, 1, 1)).to(torch.uint8)
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))
    return (d * q).reshape((n_blocks, QK_K))

def dequantize_blocks_Q5_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qh, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4))
    return (d * q - dm).reshape((n_blocks, QK_K))

def dequantize_blocks_Q4_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))
    return (d * qs - dm).reshape((n_blocks, QK_K))

def dequantize_blocks_Q3_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    hmask, qs, scales, d = split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)
    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = (scales.to(torch.int8) - 32)
    dl = (d * scales).reshape((n_blocks, 16, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = (ql.to(torch.int8) - (qh << 2).to(torch.int8))
    return (dl * q).reshape((n_blocks, QK_K))

def dequantize_blocks_Q2_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    scales, qs, d, dmin = split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))
    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml
    return qs.reshape((n_blocks, -1))

# IQ quants
KVALUES = torch.tensor([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=torch.int8)

def dequantize_blocks_IQ4_NL(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size//2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 1)).to(torch.int64)
    kvalues = KVALUES.to(qs.device).expand(*qs.shape[:-1], 16)
    qs = torch.gather(kvalues, dim=-1, index=qs).reshape((n_blocks, -1))
    return (d * qs)

dequantize_functions = {
    gguf.GGMLQuantizationType.BF16: dequantize_blocks_BF16,
    gguf.GGMLQuantizationType.Q8_0: dequantize_blocks_Q8_0,
    gguf.GGMLQuantizationType.Q5_1: dequantize_blocks_Q5_1,
    gguf.GGMLQuantizationType.Q5_0: dequantize_blocks_Q5_0,
    gguf.GGMLQuantizationType.Q4_1: dequantize_blocks_Q4_1,
    gguf.GGMLQuantizationType.Q4_0: dequantize_blocks_Q4_0,
    gguf.GGMLQuantizationType.Q6_K: dequantize_blocks_Q6_K,
    gguf.GGMLQuantizationType.Q5_K: dequantize_blocks_Q5_K,
    gguf.GGMLQuantizationType.Q4_K: dequantize_blocks_Q4_K,
    gguf.GGMLQuantizationType.Q3_K: dequantize_blocks_Q3_K,
    gguf.GGMLQuantizationType.Q2_K: dequantize_blocks_Q2_K,
    gguf.GGMLQuantizationType.IQ4_NL: dequantize_blocks_IQ4_NL,
}

def dequantize(data, qtype, oshape, dtype=None):
    """
    Dequantize tensor back to usable shape/dtype (matches ComfyUI-GGUF)
    """
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    dequantize_blocks = dequantize_functions[qtype]
    
    # Reshape to 2D and view as uint8 for block processing (ComfyUI-GGUF approach)
    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)
    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    blocks = dequantize_blocks(blocks, block_size, type_size, dtype)
    return blocks.reshape(oshape)

def dequantize_tensor(tensor, dtype=None, dequant_dtype=None, device=None, scale=None):
    """Dequantize a GGML or FP8 tensor to a regular torch.Tensor.
    
    IMPORTANT: This function MUST return a regular torch.Tensor, not GGMLTensor,
    to ensure compatibility with standard PyTorch operations like attention.
    
    Args:
        tensor: GGML quantized tensor or FP8 tensor
        dtype: Target dtype for the result
        dequant_dtype: Intermediate dtype for GGML dequantization
        device: Target device
        scale: Per-tensor scale for FP8 dequantization (optional)
    """
    # Handle nn.Parameter by extracting data
    if isinstance(tensor, torch.nn.Parameter):
        tensor = tensor.data
    
    # Handle FP8 tensors (native PyTorch FP8 dtypes)
    if is_fp8_tensor(tensor):
        target_dtype = dtype if dtype is not None else torch.bfloat16
        result = dequantize_fp8(tensor, scale=scale, target_dtype=target_dtype)
        if device is not None:
            result = result.to(device)
        return result

    qtype = getattr(tensor, "tensor_type", None)
    oshape = getattr(tensor, "tensor_shape", None)
    
    # Handle logical shape for Parameters or if attribute is missing
    if oshape is None:
        if isinstance(tensor, torch.nn.Parameter):
            oshape = getattr(tensor.data, "tensor_shape", tensor.shape)
        else:
            oshape = tensor.shape
    
    if qtype in TORCH_COMPATIBLE_QTYPES:
        # For compatible types (F32, F16, BF16), reshape and cast
        # Use as_subclass(torch.Tensor) to ensure we strip GGMLTensor
        result = tensor.as_subclass(torch.Tensor)
        if qtype in QTYPES_TO_DTYPE:
            # Ensure we view as the correct dtype before reshaping, 
            # especially for BF16 which might be loaded as uint8
            if result.dtype != QTYPES_TO_DTYPE[qtype]:
                result = result.view(QTYPES_TO_DTYPE[qtype])
        
        # Use oshape if available, otherwise trust the tensor's own shape
        if oshape is not None:
            result = result.reshape(oshape)
        result = result.to(dtype)
        # Don't return early - fall through to device and subclass handling below

    elif qtype in dequantize_functions:
        dequant_dtype = dtype if dequant_dtype == "target" else dequant_dtype
        # dequantize() returns a regular tensor from its internal logic
        result = dequantize(tensor.as_subclass(torch.Tensor), qtype, oshape, dtype=dequant_dtype).to(dtype)
    else:
        new = gguf.quants.dequantize(tensor.cpu().numpy(), qtype)
        result = torch.from_numpy(new).to(tensor.device, dtype=dtype)
        # Handle padding for numpy dequant fallback
        target_numel = torch.Size(oshape).numel()
        if result.numel() != target_numel:
            result = result.reshape(-1)[:target_numel].reshape(oshape)
    
    # Move to specified device if provided
    if device is not None:
        result = result.to(device)
    
    # CRITICAL: Ensure result is a regular torch.Tensor, not GGMLTensor
    # GGMLTensor.to() preserves the subclass, which breaks standard operations
    if not type(result) is torch.Tensor:
        result = result.as_subclass(torch.Tensor)
    
    return result

class GGMLTensor(torch.Tensor):
    def __init__(self, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        self.patches = patches

    def __new__(cls, data, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        return super().__new__(cls, data, **kwargs)

    def to(self, *args, **kwargs):
        # Save attributes BEFORE calling super().to() which might create a new tensor
        tensor_type = getattr(self, "tensor_type", None)
        tensor_shape = getattr(self, "tensor_shape", None)
        patches = getattr(self, "patches", []).copy()
        
        new = super().to(*args, **kwargs)
        new.tensor_type = tensor_type
        new.tensor_shape = tensor_shape
        new.patches = patches
        return new

    def clone(self, *args, **kwargs): return self
    def detach(self, *args, **kwargs): return self

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape") or self.tensor_shape is None:
            # Use super().size() to avoid recursion with self.size()
            self.tensor_shape = torch.Size(super().size())
        return self.tensor_shape

    def size(self, dim=None):
        if dim is not None:
            return self.shape[dim]
        return self.shape

    def dim(self):
        return len(self.shape)

    @property
    def ndim(self):
        return self.dim()

class GGMLLayer(torch.nn.Module):
    comfy_cast_weights = True
    dequant_dtype = None
    
    def is_ggml_quantized(self, *, weight=None, bias=None):
        if weight is None: weight = getattr(self, "weight", None)
        if bias is None: bias = getattr(self, "bias", None)
        return is_quantized(weight) or is_quantized(bias)

    def get_weight(self, tensor, dtype):
        if tensor is None: return None
        weight = dequantize_tensor(tensor, dtype, self.dequant_dtype)
        # Extra safety check to ensure we have a regular tensor
        if not type(weight) is torch.Tensor:
            weight = weight.as_subclass(torch.Tensor)
        return weight

    def cast_bias_weight(self, input=None, dtype=None, device=None):
        if input is not None:
            if dtype is None: dtype = getattr(input, "dtype", torch.float32)
            if device is None: device = input.device
        bias = None
        if hasattr(self, 'bias') and self.bias is not None:
            # Extract data from Parameter to ensure GGMLTensor.to() is called
            bias_data = self.bias.data if isinstance(self.bias, torch.nn.Parameter) else self.bias
            bias = self.get_weight(bias_data.to(device), dtype)
        # Extract data from Parameter to ensure GGMLTensor.to() is called
        weight_data = self.weight.data if isinstance(self.weight, torch.nn.Parameter) else self.weight
        weight_data = weight_data.to(device)
        
        # Use stored shape and type if available (nn.Parameter strips tensor_* attributes)
        stored_shape = getattr(self, '_ggml_weight_shape', None)
        stored_type = getattr(self, '_ggml_weight_type', None)
        if stored_shape is not None and getattr(weight_data, 'tensor_shape', None) is None:
            weight_data.tensor_shape = stored_shape
        if stored_type is not None and getattr(weight_data, 'tensor_type', None) is None:
            weight_data.tensor_type = stored_type
        
        weight = self.get_weight(weight_data, dtype)
        return weight, bias

def get_orig_shape(reader, tensor_name):
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None: return None
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))

# Initialize native/fallback
if not _setup_native_gguf():
    HAS_GGUF_OPS = False

def remap_gguf_state_dict(sd: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remaps GGUF state dict keys from 'Flux' style names back to Trellis2 names.
    This implements the 'Flux-Wrapper Hack' where a model is tagged as 'flux' 
    but contains Trellis2 architecture.
    """
    arch = metadata.get("general.architecture", "flux")
    if arch != "flux":
        return sd
        
    print(f"[TRELLIS2-DEBUG] Applying Flux-Wrapper remapping...", file=sys.stderr)
    
    new_sd = {}
    remap_count = 0
    
    # Common mappings - Flux -> TRELLIS2
    mappings = {
        # Embedders
        "time_in.": "t_embedder.",               # Time embedder
        "guidance_in.": "adaLN_modulation.1.",   # Guidance -> AdaLN final layer
        # IO layers
        "img_in.": "input_layer.",
        "final_layer.": "out_layer.",
        # Blocks
        "blk.": "blocks.",
    }
    
    # Internal mappings for blocks
    block_mappings = {
        ".attn.q.": ".self_attn.to_q.",
        ".attn.k.": ".self_attn.to_k.",
        ".attn.v.": ".self_attn.to_v.",
        ".attn.qkv.": ".self_attn.to_qkv.",
        ".attn.out.": ".self_attn.to_out.",
        ".attn.proj.": ".self_attn.to_proj.",
        
        ".cross_attn.q.": ".cross_attn.to_q.",
        ".cross_attn.k.": ".cross_attn.to_k.",
        ".cross_attn.v.": ".cross_attn.to_v.",
        ".cross_attn.kv.": ".cross_attn.to_kv.",
        ".cross_attn.out.": ".cross_attn.to_out.",
        ".cross_attn.proj.": ".cross_attn.to_proj.",
        
        # Norms
        ".ln1.": ".norm1.",
        ".ln2.": ".norm2.",
        ".ln3.": ".norm3.",
    }

    for k, v in sd.items():
        original_k = k
        # Stage 1: Basic remapping
        for old, new in mappings.items():
            if k.startswith(old):
                k = k.replace(old, new, 1)
                break
        
        # Stage 2: Block internal remapping
        if "blocks." in k:
            for old, new in block_mappings.items():
                if old in k:
                    k = k.replace(old, new)
        
        if k != original_k:
            remap_count += 1
        new_sd[k] = v
        
    print(f"[TRELLIS2-DEBUG]   Remapped {remap_count} tensors", file=sys.stderr)
    return new_sd


def load_gguf_checkpoint(path):
    print(f"[TRELLIS2-DEBUG] Loading GGUF checkpoint: {os.path.basename(path)}", file=sys.stderr)
    reader = gguf.GGUFReader(path)
    state_dict = {}
    metadata = {}
    
    for field_name in reader.fields:
        try:
            field = reader.get_field(field_name)
            if len(field.types) == 1:
                if field.types[0] == gguf.GGUFValueType.STRING:
                    metadata[field_name] = str(field.parts[field.data[-1]], "utf-8")
                elif field.types[0] == gguf.GGUFValueType.INT32:
                    metadata[field_name] = int(field.parts[field.data[-1]])
                elif field.types[0] == gguf.GGUFValueType.F32:
                    metadata[field_name] = float(field.parts[field.data[-1]])
                elif field.types[0] == gguf.GGUFValueType.BOOL:
                    metadata[field_name] = bool(field.parts[field.data[-1]])
        except: continue

    if metadata:
        print(f"[TRELLIS2-DEBUG]   GGUF Metadata: {len(metadata)} fields found", file=sys.stderr)
        for k, v in metadata.items():
            if 'version' in k or 'architecture' in k or 'quant' in k:
                print(f"[TRELLIS2-DEBUG]     - {k}: {v}", file=sys.stderr)

    tensor_counts = {}
    for tensor in reader.tensors:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            torch_tensor = torch.from_numpy(tensor.data)
        
        # Use native get_orig_shape or fallback
        shape = get_orig_shape(reader, tensor.name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
        
        qtype = tensor.tensor_type.name
        tensor_counts[qtype] = tensor_counts.get(qtype, 0) + 1

        if tensor.tensor_type in QTYPES_TO_DTYPE:
            target_dtype = QTYPES_TO_DTYPE[tensor.tensor_type]
            if torch_tensor.dtype != target_dtype:
                torch_tensor = torch_tensor.view(target_dtype)
            torch_tensor = torch_tensor.view(*shape)
        # For quantized types, keep the raw shape from numpy - dequantize() expects 2D data
        # Do NOT flatten - the shape (e.g., [4608, 1632]) represents the packed byte layout
        
        state_dict[tensor.name] = GGMLTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)
        if "to_qkv" in tensor.name:
            print(f"[TRELLIS2-DEBUG]   Loaded {tensor.name}: shape={shape}, type={tensor.tensor_type.name}", file=sys.stderr)
        
    print(f"[TRELLIS2-DEBUG]   GGUF Tensors: {len(reader.tensors)} loaded", file=sys.stderr)
    for qtype, count in tensor_counts.items():
        print(f"[TRELLIS2-DEBUG]     - {qtype}: {count}", file=sys.stderr)
    
    # Apply remapping if it's a Flux-wrapped model
    state_dict = remap_gguf_state_dict(state_dict, metadata)
        
    return state_dict, metadata
