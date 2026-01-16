from .. import config
import importlib
import torch
import torch.nn as nn
from .. import SparseTensor
from ....utils.gguf_utils import GGMLLayer


_backends = {}


class SparseConv3d(GGMLLayer, nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
        super(SparseConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        backend = config.get_conv_backend()
        if backend not in _backends:
            _backends[backend] = importlib.import_module(f'..conv_{backend}', __name__)
        self.low_vram = False
        self.chunk_size = 65536
        _backends[backend].sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride, dilation, padding, bias, indice_key)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for key in ["weight", "bias"]:
            k = f"{prefix}{key}"
            if k in state_dict:
                v = state_dict[k]
                if hasattr(v, "tensor_type") and v.tensor_type not in {None, 0, 1}:
                    setattr(self, key, nn.Parameter(v, requires_grad=False))
                    state_dict.pop(k)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: SparseTensor) -> SparseTensor:
        if x.feats.shape[0] == 0:
            # Short-circuit for empty sparse tensors
            out_feats = torch.zeros((0, self.out_channels), device=x.feats.device, dtype=x.feats.dtype)
            return x.replace(out_feats)
        backend = config.get_conv_backend()
        return _backends[backend].sparse_conv3d_forward(self, x)


class SparseInverseConv3d(GGMLLayer, nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
        super(SparseInverseConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        backend = config.get_conv_backend()
        if backend not in _backends:
            _backends[backend] = importlib.import_module(f'..conv_{backend}', __name__)
        self.low_vram = False
        self.chunk_size = 65536
        _backends[backend].sparse_inverse_conv3d_init(self, in_channels, out_channels, kernel_size, stride, dilation, bias, indice_key)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for key in ["weight", "bias"]:
            k = f"{prefix}{key}"
            if k in state_dict:
                v = state_dict[k]
                if hasattr(v, "tensor_type") and v.tensor_type not in {None, 0, 1}:
                    setattr(self, key, nn.Parameter(v, requires_grad=False))
                    state_dict.pop(k)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: SparseTensor) -> SparseTensor:
        if x.feats.shape[0] == 0:
            # Short-circuit for empty sparse tensors
            out_feats = torch.zeros((0, self.out_channels), device=x.feats.device, dtype=x.feats.dtype)
            return x.replace(out_feats)
        backend = config.get_conv_backend()
        return _backends[backend].sparse_inverse_conv3d_forward(self, x)
