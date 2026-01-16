"""Model loading nodes for TRELLIS.2.

This returns a lightweight config object - actual model loading
happens inside @isolated subprocess methods.
"""

try:
    from .trellis2_config import Trellis2ModelConfig
except ImportError:
    from trellis2_config import Trellis2ModelConfig

# Resolution modes (matching original TRELLIS.2)
RESOLUTION_MODES = ['512', '1024_cascade', '1536_cascade']

# Attention backend options
ATTN_BACKENDS = ['flash_attn', 'xformers', 'sdpa', 'sageattn']

# VRAM usage modes
VRAM_MODES = ['keep_loaded', 'cpu_offload', 'disk_offload']

# GGUF Quantization levels
GGUF_QUANTS = ['Q4_K_S', 'Q4_K_M', 'Q5_K_S', 'Q5_K_M', 'Q6_K', 'Q8_0']


class LoadTrellis2Models:
    """Load TRELLIS.2 models for 3D generation."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "resolution": (RESOLUTION_MODES, {"default": '1024_cascade'}),
                "enable_gguf": ("BOOLEAN", {"default": False}),
                "gguf_quant": (GGUF_QUANTS, {"default": "Q8_0"}),
                "enable_fp8": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "attn_backend": (ATTN_BACKENDS, {"default": "flash_attn"}),
                "vram_mode": (VRAM_MODES, {"default": "keep_loaded"}),
                "direct_mode": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("TRELLIS2_MODEL_CONFIG",)
    RETURN_NAMES = ("model_config",)
    FUNCTION = "load_models"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Load TRELLIS.2 model configuration for image-to-3D generation.

This node creates a configuration object that inference nodes use
to load models on-demand.

Resolution modes:
- 512: Fast, lower quality
- 1024_cascade: Best quality, uses 512->1024 cascade
- 1536_cascade: Highest resolution output

Attention backend:
- flash_attn: FlashAttention (default, requires flash_attn package)
- xformers: Memory-efficient attention (requires xformers package)
- sdpa: PyTorch native scaled_dot_product_attention (PyTorch >= 2.0)
- sageattn: SageAttention (not yet implemented)

VRAM mode:
- keep_loaded: Keep all models in VRAM (fastest, ~12GB VRAM)
- cpu_offload: Offload unused models to CPU RAM (~3-4GB VRAM, ~15-25% slower)
- disk_offload: Delete unused models, reload from disk (~3GB VRAM & CPU RAM, ~2-3x slower)

Direct mode:
- If enabled, runs inference in the main ComfyUI process instead of an isolated venv.
  Use this if you are on a cloud service that doesn't allow virtual environments.
  Note: You must have all dependencies installed in your main environment.
"""

    def load_models(self, resolution='1024_cascade', attn_backend="flash_attn", vram_mode="keep_loaded", enable_gguf=False, gguf_quant="Q8_0", enable_fp8=False, direct_mode=False):
        print(f"[TRELLIS2-DEBUG] LoadTrellis2Models.load_models: resolution={resolution}, attn_backend={attn_backend}, vram_mode={vram_mode}, enable_gguf={enable_gguf}, gguf_quant={gguf_quant}, enable_fp8={enable_fp8}, direct_mode={direct_mode}")
        # Create lightweight config object
        # Actual model loading happens in @isolated subprocess methods (unless direct_mode is True)
        config = Trellis2ModelConfig(
            model_name="microsoft/TRELLIS.2-4B",
            resolution=resolution,
            attn_backend=attn_backend,
            vram_mode=vram_mode,
            enable_gguf=enable_gguf,
            gguf_quant=gguf_quant,
            enable_fp8=enable_fp8,
            direct_mode=direct_mode,
        )
        return (config,)


NODE_CLASS_MAPPINGS = {
    "LoadTrellis2Models": LoadTrellis2Models,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadTrellis2Models": "Load TRELLIS.2 Models",
}
