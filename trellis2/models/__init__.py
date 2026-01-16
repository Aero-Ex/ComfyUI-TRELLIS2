import importlib
import sys

__attributes = {
    # Sparse Structure
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    # SLat Generation
    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',
    
    # SC-VAEs
    'SparseUnetVaeEncoder': 'sc_vaes.sparse_unet_vae',
    'SparseUnetVaeDecoder': 'sc_vaes.sparse_unet_vae',
    'FlexiDualGridVaeEncoder': 'sc_vaes.fdg_vae',
    'FlexiDualGridVaeDecoder': 'sc_vaes.fdg_vae'
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def _get_trellis2_models_dir():
    """Get the ComfyUI/models/trellis2 directory."""
    import os
    try:
        import folder_paths
        models_dir = os.path.join(folder_paths.models_dir, "trellis2")
    except ImportError:
        # Fallback if folder_paths not available
        models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "models", "trellis2")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def from_pretrained(path: str, disk_offload_manager=None, model_key: str = None, device=None, 
                    enable_gguf=False, gguf_quant="Q8_0", enable_fp8=False, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        disk_offload_manager: Optional DiskOffloadManager for RAM-efficient loading.
                              When provided, the model's safetensors path will be registered
                              for later disk-to-GPU direct loading.
        model_key: Optional key to identify this model in the disk_offload_manager.
                   Required if disk_offload_manager is provided.
        enable_gguf: Enable GGUF loading (GGML quantization).
        gguf_quant: GGUF quantization type (e.g., "Q8_0", "Q4_K").
        enable_fp8: Enable FP8 scaled safetensors loading.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    import shutil
    from safetensors.torch import load_file
    from ..utils import gguf_utils

    # Check for GGUF and FP8 model files
    model_file_quant = f"{path}_{gguf_quant}.gguf"
    model_file_gguf = f"{path}.gguf"
    model_file_fp8 = f"{path}_fp8.safetensors"
    
    print(f"[TRELLIS2-DEBUG] Searching for model at: {path} (enable_gguf={enable_gguf}, quant={gguf_quant}, enable_fp8={enable_fp8})", file=sys.stderr)
    
    is_gguf = False
    is_fp8 = False
    
    # Priority: GGUF > FP8 > Safetensors (based on flags)
    if enable_gguf and os.path.exists(model_file_quant):
        print(f"[TRELLIS2-DEBUG]   Found quantized GGUF: {model_file_quant}", file=sys.stderr)
        model_file = model_file_quant
        config_file = f"{path}.json"
        is_gguf = True
    elif enable_gguf and os.path.exists(model_file_gguf):
        print(f"[TRELLIS2-DEBUG]   Found generic GGUF: {model_file_gguf}", file=sys.stderr)
        model_file = model_file_gguf
        config_file = f"{path}.json"
        is_gguf = True
    elif enable_fp8 and os.path.exists(model_file_fp8):
        print(f"[TRELLIS2-DEBUG]   Found FP8 Safetensors: {model_file_fp8}", file=sys.stderr)
        config_file = f"{path}.json"
        model_file = model_file_fp8
        is_fp8 = True
    elif os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors"):
        print(f"[TRELLIS2-DEBUG]   Found local Safetensors: {path}.safetensors", file=sys.stderr)
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
        is_gguf = False
    elif os.path.exists(model_file_fp8):  # Auto-detect FP8 if it exists
        print(f"[TRELLIS2-DEBUG]   Auto-detected FP8 Safetensors: {model_file_fp8}", file=sys.stderr)
        config_file = f"{path}.json"
        model_file = model_file_fp8
        is_fp8 = True
    elif os.path.exists(model_file_quant): # Auto-detect quant if it exists
        print(f"[TRELLIS2-DEBUG]   Auto-detected quantized GGUF: {model_file_quant}", file=sys.stderr)
        model_file = model_file_quant
        config_file = f"{path}.json"
        is_gguf = True
    elif os.path.exists(model_file_gguf): # Auto-detect plain gguf
        print(f"[TRELLIS2-DEBUG]   Auto-detected generic GGUF: {model_file_gguf}", file=sys.stderr)
        model_file = model_file_gguf
        config_file = f"{path}.json"
        is_gguf = True
    else:
        print(f"[TRELLIS2-DEBUG]   Local files not found, checking HuggingFace cache...", file=sys.stderr)
        # Parse HuggingFace path
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])

        # Check if cached in ComfyUI/models/trellis2
        models_dir = _get_trellis2_models_dir()
        # Normalize model_name for Windows paths (convert forward slash to OS separator)
        model_name_normalized = model_name.replace('/', os.sep)
        local_config = os.path.join(models_dir, f"{model_name_normalized}.json")
        local_weights = os.path.join(models_dir, f"{model_name_normalized}.safetensors")
        local_gguf_quant = os.path.join(models_dir, f"{model_name_normalized}_{gguf_quant}.gguf")
        local_gguf = os.path.join(models_dir, f"{model_name_normalized}.gguf")

        print(f"[TRELLIS2-DEBUG]   Checking local cache: {local_config}", file=sys.stderr)

        # Create subdirectories if needed
        os.makedirs(os.path.dirname(local_config), exist_ok=True)

        # Priority search based on enable_gguf flag
        found_local = False
        if os.path.exists(local_config):
            if enable_gguf:
                # When GGUF is enabled, only use local GGUF - don't fallback to safetensors
                # If no local GGUF, will proceed to download section
                if os.path.exists(local_gguf_quant):
                    print(f"[TRELLIS2-DEBUG]   Using cached quantized GGUF: {local_gguf_quant}", file=sys.stderr, flush=True)
                    config_file = local_config
                    model_file = local_gguf_quant
                    is_gguf = True
                    found_local = True
                elif os.path.exists(local_gguf):
                    print(f"[TRELLIS2-DEBUG]   Using cached generic GGUF: {local_gguf}", file=sys.stderr, flush=True)
                    config_file = local_config
                    model_file = local_gguf
                    is_gguf = True
                    found_local = True
                # No local GGUF found - will proceed to download
            else:
                # When GGUF is disabled, prefer safetensors but auto-detect GGUF if no safetensors
                if os.path.exists(local_weights):
                    print(f"[TRELLIS2-DEBUG]   Using cached Safetensors: {local_weights}", file=sys.stderr, flush=True)
                    config_file = local_config
                    model_file = local_weights
                    is_gguf = False
                    found_local = True
                elif os.path.exists(local_gguf_quant):
                    print(f"[TRELLIS2-DEBUG]   Auto-detected cached quantized GGUF: {local_gguf_quant}", file=sys.stderr, flush=True)
                    config_file = local_config
                    model_file = local_gguf_quant
                    is_gguf = True
                    found_local = True
                elif os.path.exists(local_gguf):
                    print(f"[TRELLIS2-DEBUG]   Auto-detected cached generic GGUF: {local_gguf}", file=sys.stderr, flush=True)
                    config_file = local_config
                    model_file = local_gguf
                    is_gguf = True
                    found_local = True
        else:
            print(f"[TRELLIS2-DEBUG]   Config not found at: {local_config}", file=sys.stderr)

        if found_local:
            pass # Already set config_file and model_file
        else:
            # Check if we can find local files with different preference
            # When enable_gguf=True but no GGUF exists, fallback to local safetensors
            if os.path.exists(local_config):
                if os.path.exists(local_weights):
                    print(f"[TRELLIS2]   Using local Safetensors (no GGUF available): {local_weights}", file=sys.stderr, flush=True)
                    config_file = local_config
                    model_file = local_weights
                    is_gguf = False
                    found_local = True
            
            if not found_local:
                # No local files found at all - raise error instead of downloading
                available_files = []
                if os.path.exists(local_config):
                    available_files.append(f"config: {local_config}")
                if os.path.exists(local_weights):
                    available_files.append(f"safetensors: {local_weights}")
                if os.path.exists(local_gguf_quant):
                    available_files.append(f"gguf_quant: {local_gguf_quant}")
                if os.path.exists(local_gguf):
                    available_files.append(f"gguf: {local_gguf}")
                
                raise FileNotFoundError(
                    f"[TRELLIS2] Model not found locally: {path}\n"
                    f"  Expected location: {models_dir}\n"
                    f"  Looking for: {model_name_normalized}.json + weights\n"
                    f"  Available: {available_files if available_files else 'None'}\n"
                    f"  Please download the model files manually to: {models_dir}"
                )

    with open(config_file, 'r') as f:
        config = json.load(f)

    # Auto-detect device: prefer CUDA, fallback to CPU
    if device is None:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if is_gguf:
        print(f"[TRELLIS2]   Loading GGUF checkpoint: {model_file}", file=sys.stderr, flush=True)
        sd, metadata = gguf_utils.load_gguf_checkpoint(model_file)
        
        # Patch config from metadata if needed
        if metadata:
            meta_map = {
                'trellis.attention.head_count': 'num_heads',
                'trellis.model.model_channels': 'model_channels',
                'trellis.model.num_blocks': 'num_blocks',
                'trellis.model.in_channels': 'in_channels',
                'trellis.model.out_channels': 'out_channels',
            }
            for k, v in meta_map.items():
                if k in metadata:
                    config['args'][v] = metadata[k]
                    print(f"[TRELLIS2]   Overriding {v} from GGUF: {metadata[k]}", file=sys.stderr)
    elif is_fp8:
        print(f"[TRELLIS2]   Loading FP8 Safetensors: {model_file}", file=sys.stderr, flush=True)
        raw_sd = load_file(model_file, device=device)
        
        # Dequantize FP8 weights using their scales
        # FP8 safetensors have weight_scale/input_scale alongside weight tensors
        sd = {}
        scales = {}
        import torch
        
        # First pass: collect scales
        for key, tensor in raw_sd.items():
            if key.endswith('_scale') or key.endswith('.weight_scale') or key.endswith('.input_scale'):
                scales[key] = tensor
            else:
                sd[key] = tensor
        
        # Second pass: dequantize FP8 weights using their scales
        fp8_count = 0
        for key, tensor in list(sd.items()):
            if gguf_utils.is_fp8_tensor(tensor):
                # Find corresponding scale
                scale_key = key.replace('.weight', '.weight_scale')
                if scale_key not in scales:
                    scale_key = key + '_scale'
                scale = scales.get(scale_key, None)
                
                # Dequantize FP8 -> bfloat16
                sd[key] = gguf_utils.dequantize_fp8(tensor, scale=scale, target_dtype=torch.bfloat16)
                fp8_count += 1
        
        print(f"[TRELLIS2]   Dequantized {fp8_count} FP8 tensors", file=sys.stderr, flush=True)
    else:
        sd = load_file(model_file, device=device)

    print(f"[TRELLIS2]   Building model: {config['name']}", file=sys.stderr, flush=True)
    model_cls = __getattr__(config['name'])
    
    # Merge arguments, prioritizing kwargs (which contains low_vram/fp16 settings)
    merged_args = {**config['args'], **kwargs}
    
    # Filter arguments based on the model's constructor signature
    import inspect
    sig = inspect.signature(model_cls.__init__)
    has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    
    final_args = {}
    for k, v in merged_args.items():
        if has_var_kwargs or k in sig.parameters:
            final_args[k] = v
        else:
            # Optionally log ignored arguments for debugging
            # print(f"[TRELLIS2-DEBUG]   Ignoring argument '{k}' for model '{config['name']}'", file=sys.stderr)
            pass

    model = model_cls(**final_args)
    model.to(device)  # Move empty model to GPU before loading weights
    
    print(f"[TRELLIS2]   Loading weights directly to {device}...", file=sys.stderr, flush=True)
    model.load_state_dict(sd, strict=False)

    # Register with disk offload manager if provided (only for safetensors usually)
    if disk_offload_manager is not None and not is_gguf:
        if model_key is None:
            raise ValueError(
                "model_key is required when disk_offload_manager is provided"
            )
        disk_offload_manager.register(model_key, model_file)

    return model

    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_flow import SLatFlowModel, ElasticSLatFlowModel
        
    from .sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
    from .sc_vaes.fdg_vae import FlexiDualGridVaeEncoder, FlexiDualGridVaeDecoder
