import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import manual_cast, _apply_in_chunks
from ..utils.gguf_utils import GGMLLayer, dequantize_tensor


class LayerNorm32(GGMLLayer, nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.low_vram = False
        self.chunk_size = 65536

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for key in ["weight", "bias"]:
            k = f"{prefix}{key}"
            if k in state_dict:
                v = state_dict[k]
                if hasattr(v, "tensor_type") and v.tensor_type not in {None, 0, 1}:
                    setattr(self, key, nn.Parameter(v, requires_grad=False))
                    state_dict.pop(k)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _forward(self, x: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        x = manual_cast(x, torch.float32)
        if self.is_ggml_quantized():
            weight, bias = self.cast_bias_weight(x)
            o = torch.nn.functional.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
        else:
            o = super().forward(x)
        return manual_cast(o, out_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        
        if self.low_vram:
            o = _apply_in_chunks(lambda t: self._forward(t, x_dtype), x, self.chunk_size)
        else:
            try:
                o = self._forward(x, x_dtype)
            except torch.OutOfMemoryError:
                if x.is_cuda:
                    torch.cuda.empty_cache()
                o = _apply_in_chunks(lambda t: self._forward(t, x_dtype), x, self.chunk_size)
                
        return o
    

class GroupNorm32(GGMLLayer, nn.GroupNorm):
    """
    A GroupNorm layer that converts to float32 before the forward pass.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.low_vram = False
        self.chunk_size = 65536

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for key in ["weight", "bias"]:
            k = f"{prefix}{key}"
            if k in state_dict:
                v = state_dict[k]
                if hasattr(v, "tensor_type") and v.tensor_type not in {None, 0, 1}:
                    setattr(self, key, nn.Parameter(v, requires_grad=False))
                    state_dict.pop(k)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _forward(self, x: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        x = manual_cast(x, torch.float32)
        if self.is_ggml_quantized():
            weight, bias = self.cast_bias_weight(x)
            o = torch.nn.functional.group_norm(x, self.num_groups, weight, bias, self.eps)
        else:
            o = super().forward(x)
        return manual_cast(o, out_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        
        if self.low_vram:
            o = _apply_in_chunks(lambda t: self._forward(t, x_dtype), x, self.chunk_size)
        else:
            try:
                o = self._forward(x, x_dtype)
            except torch.OutOfMemoryError:
                if x.is_cuda:
                    torch.cuda.empty_cache()
                o = _apply_in_chunks(lambda t: self._forward(t, x_dtype), x, self.chunk_size)
                
        return o
    
    
class ChannelLayerNorm32(LayerNorm32):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()
        x = x.permute(0, *range(2, DIM), 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, DIM-1, *range(1, DIM-1)).contiguous()
        return x
    