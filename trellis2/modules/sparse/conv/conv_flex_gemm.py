import math
import torch
import torch.nn as nn
from .. import SparseTensor
from . import config
import flex_gemm
from flex_gemm.ops.spconv import sparse_submanifold_conv3d


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    assert stride == 1 and (padding is None), 'Currently flex_gemm implementation only support submanifold sparse convolution (stride=1, padding=None)'
    
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size, ) * 3
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, ) * 3
    self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation, ) * 3

    self.weight = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
    else:
        self.register_parameter("bias", None)

    # initialize parameters
    torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    # Permute weight (Co, Ci, Kd, Kh, Kw) -> (Co, Kd, Kh, Kw, Ci)
    self.weight = nn.Parameter(self.weight.permute(0, 2, 3, 4, 1).contiguous())


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    flex_gemm.ops.spconv.set_algorithm(config.FLEX_GEMM_ALGO)
    flex_gemm.ops.spconv.set_hashmap_ratio(config.FLEX_GEMM_HASHMAP_RATIO)

    # Handle GGUF weights
    if self.is_ggml_quantized():
        weight, bias = self.cast_bias_weight(x)
        # GGUF weights are typically loaded in their original layout (Co, Ci, Kd, Kh, Kw).
        # We need to permute them to (Co, Kd, Kh, Kw, Ci) for flex_gemm.
        if weight.shape != self.weight.shape:
             weight = weight.permute(0, 2, 3, 4, 1).contiguous()
    else:
        weight, bias = self.weight, self.bias
        if x.feats.dtype != weight.dtype:
            x = x.replace(x.feats.to(weight.dtype))

    # check if neighbor map is already computed
    Co, Kd, Kh, Kw, Ci = weight.shape
    neighbor_cache_key = f'SubMConv3d_neighbor_cache_{Kw}x{Kh}x{Kd}_dilation{self.dilation}'
    neighbor_cache = x.get_spatial_cache(neighbor_cache_key)

    if getattr(self, 'low_vram', False):
        # Manual chunked explicit GEMM to save memory
        chunk_size = getattr(self, 'chunk_size', 65536)
        N = x.feats.shape[0]
        V = Kd * Kh * Kw
        
        # Ensure neighbor cache is computed
        if neighbor_cache is None:
            from flex_gemm.ops.spconv.submanifold_conv3d import SubMConv3dFunction
            neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(
                x.coords, torch.Size([*x.shape, *x.spatial_shape]), (Kw, Kh, Kd), self.dilation
            )
            x.register_spatial_cache(neighbor_cache_key, neighbor_cache)
        
        neighbor_map = neighbor_cache['neighbor_map']
        
        # VRAM Monitoring
        import sys
        print(f"[TRELLIS2-DEBUG] SparseConv3d Allocation: N={N}, Co={Co}, V={V}, Ci={Ci}, dtype={x.feats.dtype}", file=sys.stderr)
        
        out = torch.empty((N, Co), device=x.feats.device, dtype=x.feats.dtype)
        weight_flat = weight.reshape(Co, V * Ci).t()
        
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            curr_chunk_size = end - i
            
            # im2col for this chunk
            chunk_neighbor_map = neighbor_map[i:end].view(-1)
            chunk_im2col = torch.zeros((curr_chunk_size * V, Ci), device=x.feats.device, dtype=x.feats.dtype)
            mask = chunk_neighbor_map != 0xffffffff
            if mask.any():
                # Cast to long before indexing with mask to avoid "index_cuda" not implemented for 'UInt32'
                chunk_im2col[mask] = x.feats[chunk_neighbor_map.long()[mask]]
            chunk_im2col = chunk_im2col.view(curr_chunk_size, V * Ci)
            
            # GEMM
            if bias is not None:
                torch.addmm(bias, chunk_im2col, weight_flat, out=out[i:end])
            else:
                torch.mm(chunk_im2col, weight_flat, out=out[i:end])
            
            del chunk_im2col, mask, chunk_neighbor_map
        
        out = x.replace(out)
        return out

    out, neighbor_cache_ = sparse_submanifold_conv3d(
        x.feats,
        x.coords,
        torch.Size([*x.shape, *x.spatial_shape]),
        weight,
        bias,
        neighbor_cache,
        self.dilation
    )
    
    if neighbor_cache is None:
        x.register_spatial_cache(neighbor_cache_key, neighbor_cache_)
    
    out = x.replace(out)
    return out


def sparse_inverse_conv3d_init(self, *args, **kwargs):
    raise NotImplementedError('SparseInverseConv3d with flex_gemm is not implemented yet')


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    raise NotImplementedError('SparseInverseConv3d with flex_gemm is not implemented yet')
