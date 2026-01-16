from typing import *
import gc
import sys
import torch
import torch.nn as nn
from .. import models
from ..utils.disk_offload import DiskOffloadManager


def _get_trellis2_models_dir():
    """Get the ComfyUI/models/trellis2 directory."""
    import os
    try:
        import folder_paths
        models_dir = os.path.join(folder_paths.models_dir, "trellis2")
    except ImportError:
        models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "models", "trellis2")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        disk_offload_manager: DiskOffloadManager = None,
    ):
        if models is None:
            return
        self.models = models
        self.disk_offload_manager = disk_offload_manager
        self.keep_model_loaded = True  # Default: keep models on GPU
        for model in self.models.values():
            if model is not None:  # Skip None placeholders (progressive loading)
                model.eval()

    @staticmethod
    def from_pretrained(
        path: str,
        models_to_load: list = None,
        enable_disk_offload: bool = False,
        enable_gguf: bool = False,
        gguf_quant: str = "Q8_0",
        enable_fp8: bool = False,
    ) -> "Pipeline":
        """
        Load a pretrained model.

        Args:
            path: Path to the model (local or HuggingFace repo)
            models_to_load: Optional list of model keys to load. If None, loads all models.
            enable_disk_offload: If True, models are NOT loaded upfront - they're loaded
                                 on-demand when first needed, then unloaded after use.
                                 This enables running on GPUs with limited VRAM.
            enable_gguf: Enable GGUF quantized model loading.
            gguf_quant: GGUF quantization type (e.g., Q8_0, Q4_K).
            enable_fp8: Enable FP8 scaled safetensors loading.
        """
        print(f"[TRELLIS2-DEBUG] Initializing Pipeline.from_pretrained(path={path}, enable_gguf={enable_gguf}, quant={gguf_quant}, enable_fp8={enable_fp8})", file=sys.stderr, flush=True)

        import os
        import json
        import shutil

        is_local = os.path.exists(f"{path}/pipeline.json")

        # Check for cached pipeline.json in ComfyUI/models/trellis2
        models_dir = _get_trellis2_models_dir()
        cached_config = os.path.join(models_dir, "pipeline.json")

        if is_local:
            print(f"[TRELLIS2] Loading pipeline config from local path...", file=sys.stderr, flush=True)
            config_file = f"{path}/pipeline.json"
        elif os.path.exists(cached_config):
            print(f"[TRELLIS2] Loading pipeline config from local cache...", file=sys.stderr, flush=True)
            config_file = cached_config
        else:
            from huggingface_hub import hf_hub_download
            print(f"[TRELLIS2] Downloading pipeline config from HuggingFace...", file=sys.stderr, flush=True)
            hf_config = hf_hub_download(path, "pipeline.json")
            # Cache it
            shutil.copy2(hf_config, cached_config)
            config_file = cached_config

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        # Create disk offload manager if enabled
        disk_offload_manager = DiskOffloadManager() if enable_disk_offload else None

        _models = {}
        # Filter to only load requested models
        model_items = [(k, v) for k, v in args['models'].items()
                       if models_to_load is None or k in models_to_load]
        total_models = len(model_items)

        if models_to_load:
            skipped = len(args['models']) - total_models
            if enable_disk_offload:
                print(f"[TRELLIS2] Registering {total_models} models for progressive loading (skipping {skipped} not needed)", file=sys.stderr, flush=True)
            else:
                print(f"[TRELLIS2] Loading {total_models} models (skipping {skipped} not needed for this resolution)", file=sys.stderr, flush=True)

        for i, (k, v) in enumerate(model_items, 1):
            # Check if v is already a full HuggingFace path (org/repo/file pattern)
            # Full paths have 3+ parts; relative paths like "ckpts/model" have only 2
            v_parts = v.split('/')
            if len(v_parts) >= 3 and not v.startswith('ckpts/'):
                # Already a full path (e.g., "microsoft/TRELLIS-image-large/ckpts/...")
                model_path = v
            else:
                # Relative path, prepend the base repo
                model_path = f"{path}/{v}"

            print(f"[TRELLIS2-DEBUG]   Processing model item {i}/{total_models}: {k} -> {model_path}", file=sys.stderr, flush=True)

            if enable_disk_offload:
                # PROGRESSIVE LOADING: Don't load model now, just ensure files are cached
                # and register path for on-demand loading later
                print(f"[TRELLIS2-DEBUG]   Registering {k} for on-demand loading...", file=sys.stderr, flush=True)
                weights_path = Pipeline._ensure_model_cached(model_path, models_dir, enable_gguf, gguf_quant)
                disk_offload_manager.register(k, weights_path)
                _models[k] = None  # Placeholder - will be loaded on-demand
                print(f"[TRELLIS2-DEBUG]   Registered {k} (path: {os.path.basename(weights_path)})", file=sys.stderr, flush=True)
            else:
                # IMMEDIATE LOADING: Load model to GPU now (original behavior)
                print(f"[TRELLIS2-DEBUG]   Loading {k} to GPU...", file=sys.stderr, flush=True)
                _models[k] = models.from_pretrained(
                    model_path,
                    disk_offload_manager=disk_offload_manager,
                    model_key=k,
                    enable_gguf=enable_gguf,
                    gguf_quant=gguf_quant,
                    enable_fp8=enable_fp8,
                )
                print(f"[TRELLIS2-DEBUG]   Loaded {k} to GPU", file=sys.stderr, flush=True)

        new_pipeline = Pipeline(_models, disk_offload_manager=disk_offload_manager)
        new_pipeline._pretrained_args = args
        new_pipeline.enable_gguf = enable_gguf
        new_pipeline.gguf_quant = gguf_quant
        new_pipeline.enable_fp8 = enable_fp8
        if enable_disk_offload:
            print(f"[TRELLIS2] All {total_models} models registered for progressive loading!", file=sys.stderr, flush=True)
        else:
            print(f"[TRELLIS2] All {total_models} models loaded!", file=sys.stderr, flush=True)
        return new_pipeline

    @staticmethod
    def _ensure_model_cached(model_path: str, models_dir: str, enable_gguf: bool = False, gguf_quant: str = "Q8_0") -> str:
        """
        Ensure model config and weights are cached locally.
        Returns the path to the safetensors or gguf file.

        This downloads files if needed but does NOT load them into GPU memory.
        """
        import os
        import shutil

        # Parse the path to determine if local or HuggingFace
        path_parts = model_path.split('/')

        # Check if it's a direct local path
        if os.path.exists(f"{model_path}.json"):
            if enable_gguf and os.path.exists(f"{model_path}_{gguf_quant}.gguf"):
                return f"{model_path}_{gguf_quant}.gguf"
            if enable_gguf and os.path.exists(f"{model_path}.gguf"):
                return f"{model_path}.gguf"
            if os.path.exists(f"{model_path}.safetensors"):
                return f"{model_path}.safetensors"

        # HuggingFace path
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        # Normalize path separators for this OS (Windows uses backslash)
        model_name = model_name.replace('/', os.sep)
        quant_model_name = f"{model_name}_{gguf_quant}"


        local_config = os.path.join(models_dir, f"{model_name}.json")
        local_gguf = os.path.join(models_dir, f"{model_name}.gguf")
        local_gguf_quant = os.path.join(models_dir, f"{quant_model_name}.gguf")
        local_safetensors = os.path.join(models_dir, f"{model_name}.safetensors")


        # Create subdirectories if needed
        os.makedirs(os.path.dirname(local_config), exist_ok=True)

        if enable_gguf and os.path.exists(local_gguf_quant):
            return local_gguf_quant
        if enable_gguf and os.path.exists(local_gguf):
            return local_gguf

        if os.path.exists(local_config):
            if enable_gguf and os.path.exists(local_gguf_quant): return local_gguf_quant
            if enable_gguf and os.path.exists(local_gguf): return local_gguf
            if os.path.exists(local_safetensors): return local_safetensors

        # Download from HuggingFace
        from huggingface_hub import hf_hub_download
        print(f"[TRELLIS2]   Downloading {model_name} config...", file=sys.stderr, flush=True)
        hf_config = hf_hub_download(repo_id, f"{model_name}.json")
        print(f"[TRELLIS2]   Downloading {model_name} weights (this may take a while)...", file=sys.stderr, flush=True)

        # Try to download GGUF first, then safetensors
        hf_weights = None
        weights_filename = None
        if enable_gguf:
            try:
                hf_weights = hf_hub_download(repo_id, f"{quant_model_name}.gguf")
                weights_filename = f"{quant_model_name}.gguf"
            except Exception:
                print(f"[TRELLIS2]   {quant_model_name}.gguf not found, trying {model_name}.gguf...", file=sys.stderr, flush=True)
                try:
                    hf_weights = hf_hub_download(repo_id, f"{model_name}.gguf")
                    weights_filename = f"{model_name}.gguf"
                except Exception:
                    print(f"[TRELLIS2]   {model_name}.gguf not found, trying {model_name}.safetensors...", file=sys.stderr, flush=True)
                    hf_weights = hf_hub_download(repo_id, f"{model_name}.safetensors")
                    weights_filename = f"{model_name}.safetensors"
        else: # Not enable_gguf, try safetensors first
            try:
                hf_weights = hf_hub_download(repo_id, f"{model_name}.safetensors")
                weights_filename = f"{model_name}.safetensors"
            except Exception:
                print(f"[TRELLIS2]   {model_name}.safetensors not found, trying {model_name}.gguf...", file=sys.stderr, flush=True)
                hf_weights = hf_hub_download(repo_id, f"{model_name}.gguf")
                weights_filename = f"{model_name}.gguf"


        # Copy to local models folder
        print(f"[TRELLIS2]   Caching to {models_dir}...", file=sys.stderr, flush=True)
        shutil.copy2(hf_config, local_config)
        shutil.copy2(hf_weights, os.path.join(models_dir, weights_filename))

        return os.path.join(models_dir, weights_filename)

    @property
    def device(self) -> torch.device:
        if hasattr(self, '_device'):
            return self._device
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                try:
                    return next(model.parameters()).device
                except StopIteration:
                    continue  # Model might be unloaded
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            if model is not None:
                model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))

    @property
    def low_vram(self) -> bool:
        return getattr(self, '_low_vram', False)

    @low_vram.setter
    def low_vram(self, value: bool) -> None:
        self._low_vram = value
        # Propagate to all loaded models
        for model in self.models.values():
            if model is not None and hasattr(model, 'low_vram'):
                model.low_vram = value

    def _load_model(self, model_key: str, device: torch.device = None) -> nn.Module:
        """
        Load a model to GPU - either move existing or load from disk.

        With progressive loading (disk_offload mode), models are loaded on-demand
        the first time they're needed, then unloaded after use to free VRAM.
        """
        if device is None:
            device = self.device

        model = self.models.get(model_key)

        # If model is None, load it from disk (first-time or after unload)
        if model is None and self.disk_offload_manager is not None:
            weights_path = self.disk_offload_manager.get_path(model_key)
            if weights_path:
                print(f"[TRELLIS2-DEBUG] _load_model({model_key}): Loading from disk cache", file=sys.stderr, flush=True)
                # Config is same path but with .json (remove .safetensors or .gguf)
                config_path = weights_path
                for sfx in ['.safetensors', '.gguf', '_Q8_0', '_Q6_K', '_Q5_K_M', '_Q5_K_S', '_Q4_K_M', '_Q4_K_S']:
                    config_path = config_path.replace(sfx, '')
                
                print(f"[TRELLIS2-DEBUG]   Config path: {config_path}", file=sys.stderr, flush=True)
                print(f"[TRELLIS2-DEBUG]   Weights path: {weights_path}", file=sys.stderr, flush=True)

                mem_before = torch.cuda.memory_allocated() / 1024**2
                print(f"[TRELLIS2] Loading {model_key} to {device}... (VRAM before: {mem_before:.0f} MB)", file=sys.stderr, flush=True)
                # Determine dtype and use_fp16 based on low_vram
                load_kwargs = {}
                if self.low_vram:
                    load_kwargs['use_fp16'] = True
                    load_kwargs['dtype'] = 'float16'

                model = models.from_pretrained(
                    config_path, 
                    device=str(device),
                    enable_gguf=getattr(self, 'enable_gguf', False),
                    gguf_quant=getattr(self, 'gguf_quant', 'Q8_0'),
                    enable_fp8=getattr(self, 'enable_fp8', False),
                    **load_kwargs
                )
                model.eval()
                # Apply low_vram setting if enabled
                if self.low_vram and hasattr(model, 'low_vram'):
                    print(f"[TRELLIS2-DEBUG]   Enabling low_vram for {model_key}", file=sys.stderr, flush=True)
                    model.low_vram = True
                self.models[model_key] = model
                # Enable activation checkpointing for memory reduction (cpu_offload or disk_offload mode)
                if not self.keep_model_loaded and hasattr(model, 'blocks'):
                    for block in model.blocks:
                        if hasattr(block, 'use_checkpoint'):
                            block.use_checkpoint = True
                    print(f"[TRELLIS2] {model_key} checkpointing enabled", file=sys.stderr, flush=True)
                mem_after = torch.cuda.memory_allocated() / 1024**2
                print(f"[TRELLIS2] {model_key} loaded (VRAM after: {mem_after:.0f} MB)", file=sys.stderr, flush=True)
        elif model is not None:
            # Model exists, just move to device if needed
            print(f"[TRELLIS2-DEBUG] _load_model({model_key}): Model already loaded, ensuring device {device}", file=sys.stderr, flush=True)
            model.to(device)
        else:
            print(f"[TRELLIS2-DEBUG] _load_model({model_key}): Error - model not found and no offload manager", file=sys.stderr, flush=True)

        return model

    def _unload_model(self, model_key: str) -> None:
        """
        Unload a model to free VRAM.

        With progressive loading, the model is deleted entirely and will be
        reloaded from disk the next time it's needed.
        """
        if self.keep_model_loaded:
            print(f"[TRELLIS2-DEBUG] _unload_model({model_key}): keep_model_loaded=True, skipping", file=sys.stderr, flush=True)
            return  # Keep model loaded, do nothing

        model = self.models.get(model_key)
        if model is not None:
            mem_before = torch.cuda.memory_allocated() / 1024**2
            reserved_before = torch.cuda.memory_reserved() / 1024**2
            print(f"[TRELLIS2] Unloading {model_key}... (allocated: {mem_before:.0f} MB, reserved: {reserved_before:.0f} MB)", file=sys.stderr, flush=True)
            # Delete the model entirely
            self.models[model_key] = None
            del model
            gc.collect()
            torch.cuda.empty_cache()
            mem_after = torch.cuda.memory_allocated() / 1024**2
            reserved_after = torch.cuda.memory_reserved() / 1024**2
            print(f"[TRELLIS2] {model_key} unloaded (allocated: {mem_after:.0f} MB, reserved: {reserved_after:.0f} MB)", file=sys.stderr, flush=True)
            print(f"[TRELLIS2-DEBUG] _unload_model({model_key}): VRAM reduction: {mem_before - mem_after:.0f} MB", file=sys.stderr, flush=True)
        else:
            print(f"[TRELLIS2-DEBUG] _unload_model({model_key}): Model already unloaded", file=sys.stderr, flush=True)
