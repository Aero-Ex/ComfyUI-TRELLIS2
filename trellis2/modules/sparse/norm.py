import torch
import torch.nn as nn
from ..utils import manual_cast
from . import VarLenTensor
from . import config
from ...utils.gguf_utils import GGMLLayer, dequantize_tensor

__all__ = [
    'SparseGroupNorm',
    'SparseLayerNorm',
    'SparseGroupNorm32',
    'SparseLayerNorm32',
]


class SparseGroupNorm(GGMLLayer, nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(SparseGroupNorm, self).__init__(num_groups, num_channels, eps, affine)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for key in ["weight", "bias"]:
            k = f"{prefix}{key}"
            if k in state_dict:
                v = state_dict[k]
                if hasattr(v, "tensor_type") and v.tensor_type not in {None, 0, 1}:
                    setattr(self, key, nn.Parameter(v, requires_grad=False))
                    state_dict.pop(k)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, input: VarLenTensor) -> VarLenTensor:
        if self.is_ggml_quantized():
            weight, bias = self.cast_bias_weight(input)
        else:
            weight, bias = self.weight, self.bias
            if weight is not None and weight.dtype != input.dtype:
                weight = weight.to(input.dtype)
            if bias is not None and bias.dtype != input.dtype:
                bias = bias.to(input.dtype)

        nfeats = torch.zeros_like(input.feats)
        for k in range(input.shape[0]):
            bfeats = input.feats[input.layout[k]]
            bfeats = bfeats.permute(1, 0).reshape(1, input.shape[1], -1)
            bfeats = torch.nn.functional.group_norm(bfeats, self.num_groups, weight, bias, self.eps)
            bfeats = bfeats.reshape(input.shape[1], -1).permute(1, 0)
            nfeats[input.layout[k]] = bfeats
        return input.replace(nfeats)


class SparseLayerNorm(GGMLLayer, nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(SparseLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for key in ["weight", "bias"]:
            k = f"{prefix}{key}"
            if k in state_dict:
                v = state_dict[k]
                if hasattr(v, "tensor_type") and v.tensor_type not in {None, 0, 1}:
                    setattr(self, key, nn.Parameter(v, requires_grad=False))
                    state_dict.pop(k)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, input: VarLenTensor) -> VarLenTensor:
        if self.is_ggml_quantized():
            weight, bias = self.cast_bias_weight(input)
        else:
            weight, bias = self.weight, self.bias
            if weight is not None and weight.dtype != input.dtype:
                weight = weight.to(input.dtype)
            if bias is not None and bias.dtype != input.dtype:
                bias = bias.to(input.dtype)

        nfeats = torch.zeros_like(input.feats)
        for k in range(input.shape[0]):
            bfeats = input.feats[input.layout[k]]
            bfeats = bfeats.permute(1, 0).reshape(1, input.shape[1], -1)
            bfeats = torch.nn.functional.layer_norm(bfeats, self.normalized_shape, weight, bias, self.eps)
            bfeats = bfeats.reshape(input.shape[1], -1).permute(1, 0)
            nfeats[input.layout[k]] = bfeats
        return input.replace(nfeats)


class SparseGroupNorm32(SparseGroupNorm):
    """
    A GroupNorm layer that converts to float32 before the forward pass.
    """
    def forward(self, x: VarLenTensor) -> VarLenTensor:
        x_dtype = x.dtype
        x = manual_cast(x, torch.float32)
        o = super().forward(x)
        return manual_cast(o, x_dtype)


class SparseLayerNorm32(SparseLayerNorm):
    """
    A LayerNorm layer that converts to float32 before the forward pass.
    """
    def forward(self, x: VarLenTensor) -> VarLenTensor:
        x_dtype = x.dtype
        x = manual_cast(x, torch.float32)
        o = super().forward(x)
        return manual_cast(o, x_dtype)
