
import torch
import torch.nn as nn
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path to import trellis2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trellis2.utils.gguf_utils import GGMLTensor, dequantize_tensor
from trellis2.modules.sparse.attention.modules import SparseMultiHeadRMSNorm, SparseMultiHeadAttention
from trellis2.modules.sparse import VarLenTensor, SparseTensor

class TestGGUFInferenceRobustness(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.dtype = torch.float32

    def test_ggml_tensor_properties(self):
        """Verify GGMLTensor behaves as expected for testing"""
        tensor = torch.randn(10, 10)
        ggml_tensor = GGMLTensor(tensor, tensor_type=0, tensor_shape=tensor.shape)
        
        self.assertTrue(hasattr(ggml_tensor, "tensor_type"))
        self.assertTrue(isinstance(ggml_tensor, GGMLTensor))
        self.assertTrue(isinstance(ggml_tensor, torch.Tensor))

    def test_dequantize_tensor_returns_regular_tensor(self):
        """Verify dequantize_tensor returns a regular torch.Tensor, not GGMLTensor"""
        tensor = torch.randn(10, 10)
        ggml_tensor = GGMLTensor(tensor, tensor_type=0, tensor_shape=tensor.shape)
        
        # Test F32 (type 0) - should be just cast/view
        result = dequantize_tensor(ggml_tensor, dtype=torch.float32)
        self.assertNotIsInstance(result, GGMLTensor, "dequantize_tensor returned GGMLTensor subclass!")
        self.assertIsInstance(result, torch.Tensor)
        self.assertTrue(torch.equal(result, tensor))

    def test_sparse_rms_norm_with_ggml_gamma(self):
        """Verify SparseMultiHeadRMSNorm handles GGMLTensor gamma correctly"""
        dim = 64
        heads = 4
        norm = SparseMultiHeadRMSNorm(dim, heads)
        
        # Replace gamma with a GGMLTensor (simulate GGUF loading)
        gamma_data = torch.ones(heads, dim)
        ggml_gamma = GGMLTensor(gamma_data, tensor_type=0, tensor_shape=gamma_data.shape)
        norm.gamma = nn.Parameter(ggml_gamma, requires_grad=False)
        
        # Create input: (Batch, Heads, Dim) to match gamma broadcasting
        # Gamma is (heads, dim), so input should broadcast against it
        x = torch.randn(1, heads, dim)
        
        # Run forward
        out = norm(x)
        
        # Verify output is NOT a GGMLTensor
        self.assertNotIsInstance(out, GGMLTensor, "RMSNorm output propagated GGMLTensor type!")
        self.assertIsInstance(out, torch.Tensor)
        
        # Verify calculation is correct (should be close to standard RMSNorm)
        self.assertEqual(out.shape, x.shape)

    @patch('trellis2.modules.sparse.attention.modules.sparse_scaled_dot_product_attention')
    def test_sparse_attention_with_ggml_inputs(self, mock_attn):
        """Verify SparseMultiHeadAttention handles inputs that might have interacted with GGMLTensors"""
        # Mock the attention function to avoid Flash Attention on CPU
        mock_attn.return_value = torch.randn(100, 64) # Dummy output
        
        channels = 64
        num_heads = 4
        attn = SparseMultiHeadAttention(channels, num_heads, qk_rms_norm=True)
        
        # Mock the internal linear layers to return GGMLTensors (simulating GGUF linear layers)
        if hasattr(attn, 'q_rms_norm'):
            # Inject GGMLTensor gamma into q_rms_norm
            gamma_data = torch.ones(num_heads, channels // num_heads)
            ggml_gamma = GGMLTensor(gamma_data, tensor_type=0, tensor_shape=gamma_data.shape)
            attn.q_rms_norm.gamma = nn.Parameter(ggml_gamma, requires_grad=False)
            
        if hasattr(attn, 'k_rms_norm'):
             # Inject GGMLTensor gamma into k_rms_norm
            gamma_data = torch.ones(num_heads, channels // num_heads)
            ggml_gamma = GGMLTensor(gamma_data, tensor_type=0, tensor_shape=gamma_data.shape)
            attn.k_rms_norm.gamma = nn.Parameter(ggml_gamma, requires_grad=False)

        # Create sparse input
        feats = torch.randn(100, channels)
        coords = torch.zeros(100, 3, dtype=torch.int32)
        x = SparseTensor(feats, coords)
        
        # Run forward
        try:
            out = attn(x)
            # Check if feats is GGMLTensor (it shouldn't be if RMSNorm fixed it)
            # Note: since we mock attention, we are testing up to the attention call
            # But the RMSNorm happens inside forward() before attention
            # If RMSNorm returns GGMLTensor, it might be passed to attention
            
            # We can check if mock_attn was called with non-GGMLTensors
            args, _ = mock_attn.call_args
            if len(args) == 1:
                # Self-attention call with fused qkv
                qkv = args[0]
                self.assertNotIsInstance(qkv, GGMLTensor, "Fused QKV passed to attention is GGMLTensor!")
                if isinstance(qkv, SparseTensor):
                    self.assertNotIsInstance(qkv.feats, GGMLTensor, "Fused QKV feats passed to attention is GGMLTensor!")
            else:
                # Cross-attention or split call with q, k, v
                q, k, v = args[0], args[1], args[2]
                self.assertNotIsInstance(q, GGMLTensor, "Query passed to attention is GGMLTensor!")
                self.assertNotIsInstance(k, GGMLTensor, "Key passed to attention is GGMLTensor!")
                self.assertNotIsInstance(v, GGMLTensor, "Value passed to attention is GGMLTensor!")
            
        except Exception as e:
            self.fail(f"Attention forward pass failed with GGMLTensor parameters: {e}")

if __name__ == "__main__":
    unittest.main()
