from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .full_attn import scaled_dot_product_attention
from .rope import RotaryPositionEmbedder
from ..sparse.linear import SparseLinear
from ...utils.gguf_utils import GGMLLayer, dequantize_tensor


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        gamma_key = f"{prefix}gamma"
        if gamma_key in state_dict:
            gamma = state_dict[gamma_key]
            if hasattr(gamma, "tensor_type") and gamma.tensor_type not in {None, 0, 1}:
                self.gamma = nn.Parameter(gamma, requires_grad=False)
                state_dict.pop(gamma_key)
        nn.Module._load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma
        # Always ensure gamma is on the correct device and correct dtype
        # For quantized GGUF tensors, dequantize first
        if hasattr(gamma, "tensor_type") and gamma.tensor_type not in {None, 0, 1}:
            gamma = dequantize_tensor(gamma, x.dtype, device=x.device)
        else:
            # Always move to ensure correct device (nn.Parameter.device may be unreliable)
            if isinstance(gamma, torch.nn.Parameter):
                gamma = gamma.data.to(x.device)
            else:
                gamma = gamma.to(x.device)
        return (F.normalize(x.float(), dim = -1) * gamma * self.scale).to(x.dtype)

    

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int]=None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        
        if attn_mode == "windowed":
            raise NotImplementedError("Windowed attention is not yet implemented")
        
        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = SparseLinear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = SparseLinear(channels, channels, bias=qkv_bias)
            self.to_kv = SparseLinear(self.ctx_channels, channels * 2, bias=qkv_bias)
            
        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            
        self.to_out = SparseLinear(channels, channels)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, phases: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        if self._type == "self":
            qkv = self.to_qkv(x)
            qkv = qkv.reshape(B, L, 3, self.num_heads, -1)
            
            if self.attn_mode == "full":
                if self.qk_rms_norm or self.use_rope:
                    q, k, v = qkv.unbind(dim=2)
                    if self.qk_rms_norm:
                        q = self.q_rms_norm(q)
                        k = self.k_rms_norm(k)
                    if self.use_rope:
                        assert phases is not None, "Phases must be provided for RoPE"
                        q = RotaryPositionEmbedder.apply_rotary_embedding(q, phases)
                        k = RotaryPositionEmbedder.apply_rotary_embedding(k, phases)
                    h = scaled_dot_product_attention(q, k, v)
                else:
                    h = scaled_dot_product_attention(qkv)
            elif self.attn_mode == "windowed":
                raise NotImplementedError("Windowed attention is not yet implemented")
        else:
            Lkv = context.shape[1]
            q = self.to_q(x)
            kv = self.to_kv(context)
            q = q.reshape(B, L, self.num_heads, -1)
            kv = kv.reshape(B, Lkv, 2, self.num_heads, -1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=2)
                k = self.k_rms_norm(k)
                h = scaled_dot_product_attention(q, k, v)
            else:
                h = scaled_dot_product_attention(q, kv)
        h = h.reshape(B, L, -1)
        h = self.to_out(h)
        return h
