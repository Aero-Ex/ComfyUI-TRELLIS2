import torch
import sys
from typing import *
import torch
import torch.nn as nn
from . import VarLenTensor
from ...utils.gguf_utils import GGMLLayer

__all__ = [
    'SparseLinear'
]


class SparseLinear(GGMLLayer, nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__(in_features, out_features, bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        weight_key = f"{prefix}weight"
        bias_key = f"{prefix}bias"
        
        if weight_key in state_dict:
            weight = state_dict[weight_key]
            if hasattr(weight, "tensor_type") and weight.tensor_type not in {None, 0, 1}:
                # Store tensor_shape before wrapping in Parameter (which strips custom attributes)
                self._ggml_weight_shape = getattr(weight, "tensor_shape", weight.shape)
                self._ggml_weight_type = getattr(weight, "tensor_type", None)
                # Direct assignment to bypass copy_ shape mismatch (byte-width vs element-count)
                self.weight = nn.Parameter(weight, requires_grad=False)
                if bias_key in state_dict and state_dict[bias_key] is not None:
                    self.bias = nn.Parameter(state_dict[bias_key], requires_grad=False)
                return
        
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, input: Union[VarLenTensor, torch.Tensor]) -> Union[VarLenTensor, torch.Tensor]:
        is_varlen = isinstance(input, VarLenTensor)
        feats = input.feats if is_varlen else input
        
        from ..utils import _apply_in_chunks
        low_vram = getattr(self, 'low_vram', False)
        chunk_size = getattr(self, 'chunk_size', 65536)

        def _forward_dense(x):
            if self.is_ggml_quantized():
                weight, bias = self.cast_bias_weight(x)
                return torch.nn.functional.linear(x, weight, bias)
            else:
                if x.dtype != self.weight.dtype:
                    x = x.to(self.weight.dtype)
                return super(SparseLinear, self).forward(x)

        if low_vram:
            out = _apply_in_chunks(_forward_dense, feats, chunk_size)
        else:
            try:
                out = _forward_dense(feats)
            except torch.OutOfMemoryError:
                if feats.is_cuda:
                    torch.cuda.empty_cache()
                out = _apply_in_chunks(_forward_dense, feats, chunk_size)
            
        return input.replace(out) if is_varlen else out
