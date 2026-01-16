from typing import *
from contextlib import contextmanager
import sys
import gc
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor, sparse_unbind, sparse_cat
from ..modules import image_feature_extractor
from ..representations import Mesh, MeshWithVoxel


class Trellis2ImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis2 image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        shape_slat_sampler (samplers.Sampler): The sampler for the structured latent.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        sparse_structure_sampler_params (dict): The parameters for the sparse structure sampler.
        shape_slat_sampler_params (dict): The parameters for the structured latent sampler.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model (Callable): The image conditioning model.
        rembg_model (Callable): The model for removing background.
        low_vram (bool): Whether to use low-VRAM mode.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        shape_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler: samplers.Sampler = None,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model: Callable = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
        default_pipeline_type: str = '1024_cascade',
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self.low_vram = low_vram
        self.default_pipeline_type = default_pipeline_type
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'

    @staticmethod
    def from_pretrained(
        path: str,
        models_to_load: list = None,
        enable_disk_offload: bool = False,
        enable_gguf: bool = False,
        gguf_quant: str = "Q8_0",
        enable_fp8: bool = False,
    ) -> "Trellis2ImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
            models_to_load: Optional list of model keys to load. If None, loads all models.
            enable_disk_offload: If True, enables disk-based model offloading for zero RAM usage.
            enable_gguf: Whether to use GGUF models.
            gguf_quant: GGUF quantization level.
            enable_fp8: Whether to use FP8 scaled safetensors.
        """
        pipeline = super(Trellis2ImageTo3DPipeline, Trellis2ImageTo3DPipeline).from_pretrained(
            path, models_to_load, enable_disk_offload=enable_disk_offload,
            enable_gguf=enable_gguf, gguf_quant=gguf_quant, enable_fp8=enable_fp8
        )
        new_pipeline = Trellis2ImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args
        print(f"[TRELLIS2-DEBUG] Trellis2ImageTo3DPipeline.from_pretrained: Assembling pipeline components", file=sys.stderr, flush=True)

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.shape_slat_sampler = getattr(samplers, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
        new_pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

        new_pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        new_pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

        new_pipeline.shape_slat_normalization = args['shape_slat_normalization']
        new_pipeline.tex_slat_normalization = args['tex_slat_normalization']

        # Only load image conditioning models when loading full pipeline (models_to_load is None)
        # When loading shape-only or texture-only, conditioning is provided externally via DinoV3 node
        if models_to_load is None:
            new_pipeline.image_cond_model = getattr(image_feature_extractor, args['image_cond_model']['name'])(**args['image_cond_model']['args'])
            new_pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])
        else:
            new_pipeline.image_cond_model = None
            new_pipeline.rembg_model = None

        new_pipeline.low_vram = args.get('low_vram', True)
        new_pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')
        new_pipeline.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        new_pipeline._device = 'cpu'

        return new_pipeline

    def to(self, device: torch.device) -> None:
        self._device = device
        # Only move models to device if keep_model_loaded is True
        # Otherwise, models will be loaded on-demand from disk
        if self.keep_model_loaded:
            super().to(device)
            if self.image_cond_model is not None:
                self.image_cond_model.to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            if self.low_vram:
                self.rembg_model.to(self.device)
            output = self.rembg_model(input)
            if self.low_vram:
                self.rembg_model.cpu()
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]], resolution: int, include_neg_cond: bool = True) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        self.image_cond_model.image_size = resolution
        if self.low_vram:
            self.image_cond_model.to(self.device)
        cond = self.image_cond_model(image)
        if self.low_vram:
            self.image_cond_model.cpu()
        if not include_neg_cond:
            return {'cond': cond}
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            resolution (int): The resolution of the sparse structure.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample sparse structure latent
        flow_model = self._load_model('sparse_structure_flow_model')
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling sparse structure",
        ).samples
        del noise, flow_model  # Free tensors and local model reference
        self._unload_model('sparse_structure_flow_model')

        # Decode sparse structure latent
        decoder = self._load_model('sparse_structure_decoder')
        decoded = decoder(z_s) > 0
        del z_s, decoder  # Free tensors and local model reference
        self._unload_model('sparse_structure_decoder')

        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
        del decoded  # Free decoded tensor
        gc.collect()
        torch.cuda.empty_cache()

        return coords

    def sample_shape_slat(
        self,
        cond: dict,
        model_key: str,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            model_key (str): The key of the flow model to use (e.g., 'shape_slat_flow_model_512').
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self._load_model(model_key)
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        del noise, flow_model
        self._unload_model(model_key)

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean

        return slat

    def sample_shape_slat_cascade(
        self,
        lr_cond: dict,
        cond: dict,
        model_key_lr: str,
        model_key: str,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning using cascade approach.

        Args:
            lr_cond (dict): The conditioning information for low-resolution.
            cond (dict): The conditioning information for high-resolution.
            model_key_lr (str): The key of the LR flow model (e.g., 'shape_slat_flow_model_512').
            model_key (str): The key of the HR flow model (e.g., 'shape_slat_flow_model_1024').
            lr_resolution (int): The low resolution.
            resolution (int): The target resolution.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
            max_num_tokens (int): Maximum number of tokens.
        """
        # LR pass
        import sys
        flow_model_lr = self._load_model(model_key_lr)
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        slat = self.shape_slat_sampler.sample(
            flow_model_lr,
            noise,
            **lr_cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat (LR)",
        ).samples
        del noise, flow_model_lr  # Free tensors and local model reference
        self._unload_model(model_key_lr)

        # Free LR conditioning and coords - no longer needed
        for k, v in lr_cond.items():
            if torch.is_tensor(v):
                lr_cond[k] = None
        del lr_cond, coords
        gc.collect()
        torch.cuda.empty_cache()
        mem_after_lr = torch.cuda.memory_allocated() / 1024**2
        print(f"[TRELLIS2] After LR cleanup: {mem_after_lr:.0f} MB", file=sys.stderr, flush=True)

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean

        # Upsample with batch-wise and spatial chunking if necessary
        decoder = self._load_model('shape_slat_decoder')
        decoder.low_vram = not self.keep_model_loaded
        
        slats = sparse_unbind(slat, dim=0)
        hr_coords_list = []
        for s in slats:
            if s.feats.shape[0] > 10000: # Lower threshold for upsample
                hc = self._spatial_chunked_call(decoder.upsample, s, upsample_times=4)
            else:
                hc = decoder.upsample(s, upsample_times=4)
            hr_coords_list.append(hc)
        hr_coords = torch.cat(hr_coords_list, dim=0)

        # Free LR slat and decoder - not used in HR pass
        del slats, slat, std, mean, decoder
        self._unload_model('shape_slat_decoder')

        hr_resolution = resolution
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens or hr_resolution == 1024:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128

        # Move conditioning to CPU to free GPU memory before loading HR model
        # This ensures maximum free VRAM when the 1024 model is loaded
        import sys
        mem_before_cleanup = torch.cuda.memory_allocated() / 1024**2
        print(f"[TRELLIS2] Before HR cleanup: {mem_before_cleanup:.0f} MB", file=sys.stderr, flush=True)

        cond_on_cpu = {}
        for k, v in cond.items():
            if torch.is_tensor(v):
                cond_on_cpu[k] = v.cpu()
            else:
                cond_on_cpu[k] = v
        del cond
        # Also move hr_coords to CPU temporarily
        hr_coords_cpu = hr_coords.cpu()
        coords_cpu = coords.cpu()
        del hr_coords, coords, quant_coords
        gc.collect()
        torch.cuda.empty_cache()

        mem_after_cleanup = torch.cuda.memory_allocated() / 1024**2
        print(f"[TRELLIS2] After HR cleanup: {mem_after_cleanup:.0f} MB (freed {mem_before_cleanup - mem_after_cleanup:.0f} MB)", file=sys.stderr, flush=True)

        # HR pass - Sample structured latent
        # Load the 1024 model with maximum free GPU memory
        flow_model = self._load_model(model_key)

        # Move conditioning and coords back to GPU
        cond = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in cond_on_cpu.items()}
        del cond_on_cpu
        coords = coords_cpu.to(self.device)
        del coords_cpu, hr_coords_cpu
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat (HR)",
        ).samples
        del noise, coords, cond, flow_model
        self._unload_model(model_key)

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean

        return slat, hr_resolution
    def _spatial_chunked_call(
        self,
        func: Callable,
        x: SparseTensor,
        padding: int = 2,
        **kwargs
    ) -> Any:
        """
        Call a function on a SparseTensor in spatial chunks to save VRAM.
        
        Args:
            func (Callable): The function to call.
            x (SparseTensor): The input sparse tensor (assumed batch_size=1).
            padding (int): The overlap padding between chunks.
            **kwargs: Additional arguments for the function.
        """
        coords = x.coords
        # Spatial coordinates are in columns 1, 2, 3
        spatial_coords = coords[:, 1:]
        
        # Determine split point (center of the grid)
        res = x.spatial_shape
        mid = [r // 2 for r in res]
        
        # 8 octants
        chunk_results = []
        
        for i in range(8):
            bx = (i >> 2) & 1
            by = (i >> 1) & 1
            bz = i & 1
            
            x_min = 0 if bx == 0 else mid[0]
            x_max = mid[0] if bx == 0 else res[0]
            y_min = 0 if by == 0 else mid[1]
            y_max = mid[1] if by == 0 else res[1]
            z_min = 0 if bz == 0 else mid[2]
            z_max = mid[2] if bz == 0 else res[2]
            
            # Filter voxels for this octant WITH padding
            mask_padded = (
                (spatial_coords[:, 0] >= x_min - padding) & (spatial_coords[:, 0] < x_max + padding) &
                (spatial_coords[:, 1] >= y_min - padding) & (spatial_coords[:, 1] < y_max + padding) &
                (spatial_coords[:, 2] >= z_min - padding) & (spatial_coords[:, 2] < z_max + padding)
            )
            
            if not mask_padded.any():
                chunk_results.append(None)
                continue
                
            # Extract chunk
            chunk_x = SparseTensor(
                feats=x.feats[mask_padded],
                coords=coords[mask_padded],
                shape=x.shape,
                scale=x._scale
            )
            
            # Chunk kwargs if they are SparseTensors
            chunk_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, SparseTensor):
                    sc = v.coords[:, 1:]
                    # Scale boundaries for this SparseTensor
                    sf = v.spatial_shape[0] / res[0]
                    m = (
                        (sc[:, 0] >= (x_min - padding) * sf) & (sc[:, 0] < (x_max + padding) * sf) &
                        (sc[:, 1] >= (y_min - padding) * sf) & (sc[:, 1] < (y_max + padding) * sf) &
                        (sc[:, 2] >= (z_min - padding) * sf) & (sc[:, 2] < (z_max + padding) * sf)
                    )
                    chunk_kwargs[k] = SparseTensor(feats=v.feats[m], coords=v.coords[m], shape=v.shape, scale=v._scale)
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], SparseTensor):
                    # List of SparseTensors (e.g. guide_subs)
                    cv_list = []
                    for cv in v:
                        sc = cv.coords[:, 1:]
                        sf = cv.spatial_shape[0] / res[0]
                        m = (
                            (sc[:, 0] >= (x_min - padding) * sf) & (sc[:, 0] < (x_max + padding) * sf) &
                            (sc[:, 1] >= (y_min - padding) * sf) & (sc[:, 1] < (y_max + padding) * sf) &
                            (sc[:, 2] >= (z_min - padding) * sf) & (sc[:, 2] < (z_max + padding) * sf)
                        )
                        cv_list.append(SparseTensor(feats=cv.feats[m], coords=cv.coords[m], shape=cv.shape, scale=cv._scale))
                    chunk_kwargs[k] = cv_list
                else:
                    chunk_kwargs[k] = v
            
            # Call function
            res_chunk = func(chunk_x, **chunk_kwargs)
            
            # Crop result back to original octant boundaries (no padding)
            if isinstance(res_chunk, SparseTensor):
                c = res_chunk.coords[:, 1:]
                mask_crop = (
                    (c[:, 0] >= x_min) & (c[:, 0] < x_max) &
                    (c[:, 1] >= y_min) & (c[:, 1] < y_max) &
                    (c[:, 2] >= z_min) & (c[:, 2] < z_max)
                )
                res_chunk = SparseTensor(
                    feats=res_chunk.feats[mask_crop],
                    coords=res_chunk.coords[mask_crop],
                    shape=res_chunk.shape,
                    scale=res_chunk._scale
                )
            elif isinstance(res_chunk, tuple):
                # Handle (SparseTensor, List[SparseTensor]) OR (List[Mesh], List[SparseTensor])
                h_chunk, subs_chunk = res_chunk
                if isinstance(h_chunk, SparseTensor):
                    c = h_chunk.coords[:, 1:]
                    mask_crop = (
                        (c[:, 0] >= x_min) & (c[:, 0] < x_max) &
                        (c[:, 1] >= y_min) & (c[:, 1] < y_max) &
                        (c[:, 2] >= z_min) & (c[:, 2] < z_max)
                    )
                    h_chunk = SparseTensor(
                        feats=h_chunk.feats[mask_crop],
                        coords=h_chunk.coords[mask_crop],
                        shape=h_chunk.shape,
                        scale=h_chunk._scale
                    )
                elif isinstance(h_chunk, list) and len(h_chunk) > 0 and hasattr(h_chunk[0], 'vertices'):
                    # List[Mesh] - we don't crop individual meshes yet, just collect them
                    pass
                
                # Crop subs
                cropped_subs = []
                for sub in subs_chunk:
                    sc = sub.coords[:, 1:]
                    sf = sub.spatial_shape[0] / res[0]
                    s_mask = (
                        (sc[:, 0] >= x_min * sf) & (sc[:, 0] < x_max * sf) &
                        (sc[:, 1] >= y_min * sf) & (sc[:, 1] < y_max * sf) &
                        (sc[:, 2] >= z_min * sf) & (sc[:, 2] < z_max * sf)
                    )
                    cropped_subs.append(SparseTensor(feats=sub.feats[s_mask], coords=sub.coords[s_mask], shape=sub.shape, scale=sub._scale))
                res_chunk = (h_chunk, cropped_subs)
            elif isinstance(res_chunk, torch.Tensor):
                c = res_chunk[:, 1:]
                mask_crop = (
                    (c[:, 0] >= x_min) & (c[:, 0] < x_max) &
                    (c[:, 1] >= y_min) & (c[:, 1] < y_max) &
                    (c[:, 2] >= z_min) & (c[:, 2] < z_max)
                )
                res_chunk = res_chunk[mask_crop]
                
            chunk_results.append(res_chunk)
            torch.cuda.empty_cache()
            
        valid_results = [r for r in chunk_results if r is not None]
        if not valid_results: return None
            
        if isinstance(valid_results[0], SparseTensor):
            # Custom merge for spatial chunks (preserve batch indices)
            merged_feats = torch.cat([r.feats for r in valid_results], dim=0)
            merged_coords = torch.cat([r.coords for r in valid_results], dim=0)
            return SparseTensor(
                feats=merged_feats, 
                coords=merged_coords, 
                shape=valid_results[0].shape, 
                scale=valid_results[0]._scale
            )
        elif isinstance(valid_results[0], tuple):
            # Merge (h, subs)
            h_list = [r[0] for r in valid_results]
            if isinstance(h_list[0], SparseTensor):
                merged_h_feats = torch.cat([r.feats for r in h_list], dim=0)
                merged_h_coords = torch.cat([r.coords for r in h_list], dim=0)
                merged_h = SparseTensor(
                    feats=merged_h_feats, 
                    coords=merged_h_coords, 
                    shape=h_list[0].shape, 
                    scale=h_list[0]._scale
                )
            elif isinstance(h_list[0], list):
                # Merge List[List[Mesh]] -> List[Mesh]
                # Since batch_size=1, each chunk returns [Mesh]
                all_v, all_f = [], []
                v_off = 0
                for ml in h_list:
                    for m in ml:
                        # Move to CPU immediately to save VRAM during merging
                        all_v.append(m.vertices.cpu())
                        all_f.append(m.faces.cpu() + v_off)
                        v_off += m.vertices.shape[0]
                merged_h = [Mesh(torch.cat(all_v, dim=0), torch.cat(all_f, dim=0))]
            
            num_levels = len(valid_results[0][1])
            merged_subs = []
            for level in range(num_levels):
                level_subs = [r[1][level] for r in valid_results]
                merged_sub_feats = torch.cat([r.feats for r in level_subs], dim=0)
                merged_sub_coords = torch.cat([r.coords for r in level_subs], dim=0)
                merged_subs.append(SparseTensor(
                    feats=merged_sub_feats, 
                    coords=merged_sub_coords, 
                    shape=level_subs[0].shape, 
                    scale=level_subs[0]._scale
                ))
            return merged_h, merged_subs
        elif isinstance(valid_results[0], torch.Tensor):
            return torch.cat(valid_results, dim=0)
        return valid_results

    def decode_shape_slat(
        self,
        slat: SparseTensor,
        resolution: int,
    ) -> Tuple[List[Mesh], List[SparseTensor]]:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.
            resolution (int): The resolution for decoding.

        Returns:
            List[Mesh]: The decoded meshes.
            List[SparseTensor]: The decoded substructures.
        """
        # Aggressive cleanup before loading decoder
        gc.collect()
        torch.cuda.empty_cache()
        
        decoder = self._load_model('shape_slat_decoder')
        decoder.set_resolution(resolution)
        decoder.low_vram = not self.keep_model_loaded
        
        # Batch-wise decoding to save VRAM
        slats = sparse_unbind(slat, dim=0)
        all_meshes = []
        all_subs = []
        
        for i, s in enumerate(slats):
            # Clear cache before each batch
            gc.collect()
            torch.cuda.empty_cache()
            
            # Decode single batch with spatial chunking if necessary
            # We use a threshold of 10k voxels to trigger spatial chunking for high-res
            if s.feats.shape[0] > 10000:
                print(f"[TRELLIS2-DEBUG] High density detected ({s.feats.shape[0]} voxels), using spatial chunking", file=sys.stderr)
                res = self._spatial_chunked_call(decoder, s, return_subs=True)
            else:
                res = decoder(s, return_subs=True)
                
            if isinstance(res, tuple):
                h, subs = res
            else:
                h, subs = res, []
            
            # Extract meshes from h
            # flexible_dual_grid_to_mesh is called inside decoder.forward in fdg_vae.py
            # But wait, our decoder.forward returns meshes if training=False
            # Let's check fdg_vae.py again.
            
            # Actually, FlexiDualGridVaeDecoder.forward returns meshes when not training.
            # So res is already List[Mesh] or (List[Mesh], List[SparseTensor])
            if isinstance(res, tuple):
                meshes, subs = res
            else:
                meshes, subs = res, []
            
            # Move meshes and subs to CPU to save VRAM
            all_meshes.extend([m.cpu() for m in meshes])
            all_subs.append([s.cpu() for s in subs])
            
            # Cleanup
            del meshes, subs, res, s
            gc.collect()
            torch.cuda.empty_cache()
            
        del decoder
        self._unload_model('shape_slat_decoder')
        
        # Merge substructures if they exist
        if all_subs and all_subs[0]:
            # all_subs is List[List[SparseTensor]]
            # We need to cat each level of substructures
            num_levels = len(all_subs[0])
            merged_subs = []
            for level in range(num_levels):
                level_subs = [s[level] for s in all_subs]
                merged_subs.append(sparse_cat(level_subs, dim=0))
            return all_meshes, merged_subs
        
        return all_meshes, []
    
    def sample_tex_slat(
        self,
        cond: dict,
        model_key: str,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            model_key (str): The key of the flow model to use (e.g., 'tex_slat_flow_model_1024').
            shape_slat (SparseTensor): The structured latent for shape
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat = (shape_slat - mean) / std

        flow_model = self._load_model(model_key)
        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        slat = self.tex_slat_sampler.sample(
            flow_model,
            noise,
            concat_cond=shape_slat,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling texture SLat",
        ).samples
        del noise, flow_model
        self._unload_model(model_key)

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean

        return slat

    def decode_tex_slat(
        self,
        slat: SparseTensor,
        subs: List[SparseTensor],
    ) -> SparseTensor:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.
            subs (List[SparseTensor]): The substructures from shape decoding.

        Returns:
            List[SparseTensor]: The decoded texture voxels
        """
        decoder = self._load_model('tex_slat_decoder')
        decoder.low_vram = not self.keep_model_loaded
        
        # Batch-wise decoding to save VRAM
        slats = sparse_unbind(slat, dim=0)
        # subs is List[SparseTensor], where each SparseTensor is batched
        # We need to unbind each guide sub as well
        unbound_subs = [sparse_unbind(sub, dim=0) for sub in subs]
        
        all_decoded = []
        for i, s in enumerate(slats):
            # Clear cache before each batch
            gc.collect()
            torch.cuda.empty_cache()
            
            # Get guide subs for this batch
            curr_subs = [us[i] for us in unbound_subs]
            
            # Decode single batch with spatial chunking if necessary
            if s.feats.shape[0] > 10000:
                print(f"[TRELLIS2-DEBUG] High density detected ({s.feats.shape[0]} voxels), using spatial chunking", file=sys.stderr)
                decoded = self._spatial_chunked_call(decoder, s, guide_subs=curr_subs) * 0.5 + 0.5
            else:
                decoded = decoder(s, guide_subs=curr_subs) * 0.5 + 0.5
            # Move to CPU to save VRAM
            all_decoded.append(decoded.cpu())
            
            # Cleanup
            del decoded, s, curr_subs
            gc.collect()
            torch.cuda.empty_cache()
            
        del decoder
        self._unload_model('tex_slat_decoder')
        
        # Merge results
        return sparse_cat(all_decoded, dim=0)
    
    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
    ) -> List[MeshWithVoxel]:
        """
        Decode the latent codes.

        Args:
            shape_slat (SparseTensor): The structured latent for shape.
            tex_slat (SparseTensor): The structured latent for texture.
            resolution (int): The resolution of the output.
        """
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)

        # Clear GPU cache before texture decoding
        torch.cuda.empty_cache()

        tex_voxels = self.decode_tex_slat(tex_slat, subs)

        # Delete subs immediately after use to free GPU memory
        del subs
        torch.cuda.empty_cache()

        out_mesh = []
        for m, v in zip(meshes, tex_voxels):
            m.fill_holes()
            out_mesh.append(
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin = [-0.5, -0.5, -0.5],
                    voxel_size = 1 / resolution,
                    coords = v.coords[:, 1:],
                    attrs = v.feats,
                    voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                    layout=self.pbr_attr_layout
                )
            )
        return out_mesh
    
    @torch.no_grad()
    def run_shape(
        self,
        cond: dict,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
    ) -> Tuple[List[Mesh], SparseTensor, int]:
        """
        Run shape generation only (no texture).

        Args:
            cond (dict): The conditioning dict with 'cond_512', 'cond_1024' (optional), 'neg_cond'.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            pipeline_type (str): The type of the pipeline. Options: '512', '1024', '1024_cascade', '1536_cascade'.
            max_num_tokens (int): The maximum number of tokens to use.

        Returns:
            Tuple of (meshes, shape_slat, resolution)
        """
        pipeline_type = pipeline_type or self.default_pipeline_type

        # Validate models
        if pipeline_type == '512':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
        elif pipeline_type == '1024':
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
        elif pipeline_type in ('1024_cascade', '1536_cascade'):
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")

        # Extract conditioning
        cond_512 = {'cond': cond['cond_512'], 'neg_cond': cond['neg_cond']}
        cond_1024 = {'cond': cond['cond_1024'], 'neg_cond': cond['neg_cond']} if 'cond_1024' in cond else None

        torch.manual_seed(seed)
        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
        coords = self.sample_sparse_structure(
            cond_512, ss_res,
            num_samples, sparse_structure_sampler_params
        )

        if pipeline_type == '512':
            shape_slat = self.sample_shape_slat(
                cond_512, 'shape_slat_flow_model_512',
                coords, shape_slat_sampler_params
            )
            res = 512
        elif pipeline_type == '1024':
            shape_slat = self.sample_shape_slat(
                cond_1024, 'shape_slat_flow_model_1024',
                coords, shape_slat_sampler_params
            )
            res = 1024
        elif pipeline_type == '1024_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                'shape_slat_flow_model_512', 'shape_slat_flow_model_1024',
                512, 1024,
                coords, shape_slat_sampler_params,
                max_num_tokens
            )
            del cond_512, cond_1024  # Free conditioning refs
        elif pipeline_type == '1536_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                'shape_slat_flow_model_512', 'shape_slat_flow_model_1024',
                512, 1536,
                coords, shape_slat_sampler_params,
                max_num_tokens
            )
            del cond_512, cond_1024  # Free conditioning refs

        # Free original cond dict references
        for k in list(cond.keys()):
            cond[k] = None
        gc.collect()
        torch.cuda.empty_cache()

        # Decode shape (keep subs for texture generation)
        meshes, subs = self.decode_shape_slat(shape_slat, res)

        torch.cuda.empty_cache()
        return meshes, shape_slat, subs, res

    @torch.no_grad()
    def run_texture(
        self,
        cond: dict,
        shape_slat: SparseTensor,
        resolution: int,
        seed: int = 42,
        tex_slat_sampler_params: dict = {},
        pipeline_type: Optional[str] = None,
    ) -> List[MeshWithVoxel]:
        """
        Run texture generation on existing shape.

        Args:
            cond (dict): The conditioning dict with 'cond_512', 'cond_1024' (optional), 'neg_cond'.
            shape_slat (SparseTensor): The shape latent from run_shape().
            resolution (int): The resolution from run_shape().
            seed (int): The random seed.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            pipeline_type (str): The type of the pipeline.

        Returns:
            List of MeshWithVoxel with PBR attributes.
        """
        pipeline_type = pipeline_type or self.default_pipeline_type

        # Validate texture models
        if pipeline_type == '512':
            assert 'tex_slat_flow_model_512' in self.models, "No 512 resolution texture SLat flow model found."
            tex_model_key = 'tex_slat_flow_model_512'
            tex_cond = {'cond': cond['cond_512'], 'neg_cond': cond['neg_cond']}
        else:
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
            tex_model_key = 'tex_slat_flow_model_1024'
            tex_cond = {'cond': cond['cond_1024'], 'neg_cond': cond['neg_cond']}

        torch.manual_seed(seed)

        # Sample texture latent
        tex_slat = self.sample_tex_slat(
            tex_cond, tex_model_key,
            shape_slat, tex_slat_sampler_params
        )

        # Clear GPU cache before decode
        torch.cuda.empty_cache()

        # Decode both shape and texture
        out_mesh = self.decode_latent(shape_slat, tex_slat, resolution)

        # Delete tex_slat after use
        del tex_slat
        torch.cuda.empty_cache()
        return out_mesh

    @torch.no_grad()
    def run_texture_with_subs(
        self,
        cond: dict,
        shape_slat: SparseTensor,
        subs: List[SparseTensor],
        meshes: List[Mesh],
        resolution: int,
        seed: int = 42,
        tex_slat_sampler_params: dict = {},
        pipeline_type: Optional[str] = None,
    ) -> List[MeshWithVoxel]:
        """
        Run texture generation using pre-computed subs (no shape re-decode).

        This is faster than run_texture() because it skips shape decoder inference.

        Args:
            cond (dict): The conditioning dict with 'cond_512', 'cond_1024' (optional), 'neg_cond'.
            shape_slat (SparseTensor): The shape latent from run_shape().
            subs (List[SparseTensor]): The substructures from run_shape().
            meshes (List[Mesh]): The meshes from run_shape().
            resolution (int): The resolution from run_shape().
            seed (int): The random seed.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            pipeline_type (str): The type of the pipeline.

        Returns:
            List of MeshWithVoxel with PBR attributes.
        """
        pipeline_type = pipeline_type or self.default_pipeline_type

        # Validate texture models
        if pipeline_type == '512':
            assert 'tex_slat_flow_model_512' in self.models, "No 512 resolution texture SLat flow model found."
            tex_model_key = 'tex_slat_flow_model_512'
            tex_cond = {'cond': cond['cond_512'], 'neg_cond': cond['neg_cond']}
        else:
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
            tex_model_key = 'tex_slat_flow_model_1024'
            tex_cond = {'cond': cond['cond_1024'], 'neg_cond': cond['neg_cond']}

        torch.manual_seed(seed)

        # Sample texture latent
        tex_slat = self.sample_tex_slat(
            tex_cond, tex_model_key,
            shape_slat, tex_slat_sampler_params
        )

        # Free shape_slat and cond - no longer needed after sampling
        del shape_slat, tex_cond, cond
        gc.collect()
        torch.cuda.empty_cache()

        # Decode texture using pre-computed subs (skip shape decoder!)
        tex_voxels = self.decode_tex_slat(tex_slat, subs)

        # Delete tex_slat after use
        del tex_slat
        torch.cuda.empty_cache()

        # Combine meshes with texture voxels
        out_mesh = []
        for m, v in zip(meshes, tex_voxels):
            m.fill_holes()
            out_mesh.append(
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin=[-0.5, -0.5, -0.5],
                    voxel_size=1 / resolution,
                    coords=v.coords[:, 1:],
                    attrs=v.feats,
                    voxel_shape=torch.Size([*v.shape, *v.spatial_shape]),
                    layout=self.pbr_attr_layout
                )
            )
        return out_mesh

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
    ) -> List[MeshWithVoxel]:
        """
        Run the full pipeline (shape + texture).

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            preprocess_image (bool): Whether to preprocess the image.
            return_latent (bool): Whether to return the latent codes.
            pipeline_type (str): The type of the pipeline. Options: '512', '1024', '1024_cascade', '1536_cascade'.
            max_num_tokens (int): The maximum number of tokens to use.
        """
        # Check pipeline type
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == '512':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_512' in self.models, "No 512 resolution texture SLat flow model found."
        elif pipeline_type == '1024':
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1024_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1536_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")

        if preprocess_image:
            image = self.preprocess_image(image)
        torch.manual_seed(seed)
        cond_512 = self.get_cond([image], 512)
        cond_1024 = self.get_cond([image], 1024) if pipeline_type != '512' else None
        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
        coords = self.sample_sparse_structure(
            cond_512, ss_res,
            num_samples, sparse_structure_sampler_params
        )
        if pipeline_type == '512':
            shape_slat = self.sample_shape_slat(
                cond_512, 'shape_slat_flow_model_512',
                coords, shape_slat_sampler_params
            )
            tex_slat = self.sample_tex_slat(
                cond_512, 'tex_slat_flow_model_512',
                shape_slat, tex_slat_sampler_params
            )
            res = 512
        elif pipeline_type == '1024':
            shape_slat = self.sample_shape_slat(
                cond_1024, 'shape_slat_flow_model_1024',
                coords, shape_slat_sampler_params
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, 'tex_slat_flow_model_1024',
                shape_slat, tex_slat_sampler_params
            )
            res = 1024
        elif pipeline_type == '1024_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                'shape_slat_flow_model_512', 'shape_slat_flow_model_1024',
                512, 1024,
                coords, shape_slat_sampler_params,
                max_num_tokens
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, 'tex_slat_flow_model_1024',
                shape_slat, tex_slat_sampler_params
            )
        elif pipeline_type == '1536_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                'shape_slat_flow_model_512', 'shape_slat_flow_model_1024',
                512, 1536,
                coords, shape_slat_sampler_params,
                max_num_tokens
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, 'tex_slat_flow_model_1024',
                shape_slat, tex_slat_sampler_params
            )
        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        else:
            return out_mesh

    # ==========================================================================
    # MULTI-IMAGE CONDITIONING (ported from TRELLIS v1)
    # ==========================================================================

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """
        Inject a sampler with multiple images as condition.

        Args:
            sampler_name (str): The name of the sampler to inject (e.g., 'sparse_structure_sampler').
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
            mode (str): 'stochastic' cycles through images per step, 'multidiffusion' averages predictions.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, '_old_inference_model', sampler._inference_model)

        # Sampler-specific kwargs that should NOT be passed to model
        SAMPLER_KWARGS = {'neg_cond', 'guidance_strength', 'guidance_interval', 'guidance_rescale'}

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images ({num_images}) is greater than "
                      f"number of steps ({num_steps}) for {sampler_name}. This may lead to some images "
                      f"not being used.\033[0m", file=sys.stderr)

            # Create index list that cycles through images
            cond_indices = (np.arange(num_steps) % num_images).tolist()
            # Cache for timestep -> image index to ensure consistency within a step
            step_cache = {}

            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                t_key = float(t)
                if t_key not in step_cache:
                    step_cache[t_key] = cond_indices.pop(0)
                
                cond_idx = step_cache[t_key]
                cond_i = cond[cond_idx]
                
                # Filter out sampler-specific kwargs only if they are NOT expected by the old method
                import inspect
                sig = inspect.signature(self._old_inference_model)
                model_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters or k not in SAMPLER_KWARGS}
                return self._old_inference_model(model, x_t, t, cond=cond_i, **model_kwargs)

        elif mode == 'multidiffusion':
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, guidance_strength, **kwargs):
                # Run model for each conditioning image with guidance_strength=1 (unguided pos)
                # This ensures GuidanceIntervalSamplerMixin and CFG mixin are respected
                preds = []
                for i in range(len(cond)):
                    pred = self._old_inference_model(model, x_t, t, cond[i], neg_cond=neg_cond, guidance_strength=1, **kwargs)
                    preds.append(pred)
                avg_pred_pos = sum(preds) / len(preds)

                # Run model once with guidance_strength=0 (unguided neg)
                pred_neg = self._old_inference_model(model, x_t, t, cond[0], neg_cond=neg_cond, guidance_strength=0, **kwargs)

                # Combine using standard CFG formula: pred = guidance_strength * pos + (1 - guidance_strength) * neg
                return guidance_strength * avg_pred_pos + (1 - guidance_strength) * pred_neg

        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'stochastic' or 'multidiffusion'.")

        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        try:
            yield
        finally:
            sampler._inference_model = sampler._old_inference_model
            delattr(sampler, '_old_inference_model')

    @torch.no_grad()
    def run_multi_image_shape(
        self,
        cond: dict,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ) -> Tuple[List[Mesh], SparseTensor, int]:
        """
        Run shape generation with multiple images as condition.

        Args:
            cond (dict): The conditioning dict with 'cond_512' [N, B, D], 'cond_1024' [N, B, D], 'neg_cond' [1, B, D].
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            pipeline_type (str): The type of the pipeline.
            max_num_tokens (int): The maximum number of tokens to use.
            mode (str): 'stochastic' or 'multidiffusion'.

        Returns:
            Tuple of (meshes, shape_slat, subs, resolution)
        """
        pipeline_type = pipeline_type or self.default_pipeline_type

        # Get number of images from conditioning
        num_images = cond['cond_512'].shape[0]
        print(f"[TRELLIS2] Multi-image shape generation with {num_images} images, mode={mode}", file=sys.stderr)

        # Extract conditioning - keep as [N, B, D] for multi-image
        # Use resolution-specific negative conditioning
        cond_512 = {'cond': cond['cond_512'], 'neg_cond': cond['neg_cond_512']}
        cond_1024 = {'cond': cond['cond_1024'], 'neg_cond': cond['neg_cond_1024']} if 'cond_1024' in cond else None

        torch.manual_seed(seed)
        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps', 12)

        # Sample sparse structure with multi-image injection
        with self.inject_sampler_multi_image('sparse_structure_sampler', num_images, ss_steps, mode=mode):
            coords = self.sample_sparse_structure(
                cond_512, ss_res,
                num_samples, sparse_structure_sampler_params
            )

        shape_steps = {**self.shape_slat_sampler_params, **shape_slat_sampler_params}.get('steps', 12)

        if pipeline_type == '512':
            with self.inject_sampler_multi_image('shape_slat_sampler', num_images, shape_steps, mode=mode):
                shape_slat = self.sample_shape_slat(
                    cond_512, 'shape_slat_flow_model_512',
                    coords, shape_slat_sampler_params
                )
            res = 512
        elif pipeline_type == '1024':
            with self.inject_sampler_multi_image('shape_slat_sampler', num_images, shape_steps, mode=mode):
                shape_slat = self.sample_shape_slat(
                    cond_1024, 'shape_slat_flow_model_1024',
                    coords, shape_slat_sampler_params
                )
            res = 1024
        elif pipeline_type in ('1024_cascade', '1536_cascade'):
            target_res = 1024 if pipeline_type == '1024_cascade' else 1536
            # For cascade, inject both LR and HR passes
            with self.inject_sampler_multi_image('shape_slat_sampler', num_images, shape_steps, mode=mode):
                shape_slat, res = self.sample_shape_slat_cascade(
                    cond_512, cond_1024,
                    'shape_slat_flow_model_512', 'shape_slat_flow_model_1024',
                    512, target_res,
                    coords, shape_slat_sampler_params,
                    max_num_tokens
                )
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")

        gc.collect()
        torch.cuda.empty_cache()

        # Decode shape
        meshes, subs = self.decode_shape_slat(shape_slat, res)

        torch.cuda.empty_cache()
        return meshes, shape_slat, subs, res

    @torch.no_grad()
    def run_multi_image_texture(
        self,
        cond: dict,
        shape_slat: SparseTensor,
        subs: List[SparseTensor],
        meshes: List[Mesh],
        resolution: int,
        seed: int = 42,
        tex_slat_sampler_params: dict = {},
        pipeline_type: Optional[str] = None,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ) -> List[MeshWithVoxel]:
        """
        Run texture generation with multiple images as condition.

        Args:
            cond (dict): The conditioning dict with 'cond_512' [N, B, D], 'cond_1024' [N, B, D], 'neg_cond' [1, B, D].
            shape_slat (SparseTensor): The shape latent from run_multi_image_shape().
            subs (List[SparseTensor]): The substructures from run_multi_image_shape().
            meshes (List[Mesh]): The meshes from run_multi_image_shape().
            resolution (int): The resolution from run_multi_image_shape().
            seed (int): The random seed.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            pipeline_type (str): The type of the pipeline.
            mode (str): 'stochastic' or 'multidiffusion'.

        Returns:
            List of MeshWithVoxel with PBR attributes.
        """
        pipeline_type = pipeline_type or self.default_pipeline_type

        # Get number of images from conditioning
        num_images = cond['cond_512'].shape[0]
        print(f"[TRELLIS2] Multi-image texture generation with {num_images} images, mode={mode}", file=sys.stderr)

        # Validate texture models
        if pipeline_type == '512':
            tex_model_key = 'tex_slat_flow_model_512'
            tex_cond = {'cond': cond['cond_512'], 'neg_cond': cond['neg_cond_512']}
        else:
            tex_model_key = 'tex_slat_flow_model_1024'
            tex_cond = {'cond': cond['cond_1024'], 'neg_cond': cond['neg_cond_1024']}

        torch.manual_seed(seed)
        tex_steps = {**self.tex_slat_sampler_params, **tex_slat_sampler_params}.get('steps', 12)

        # Sample texture latent with multi-image injection
        with self.inject_sampler_multi_image('tex_slat_sampler', num_images, tex_steps, mode=mode):
            tex_slat = self.sample_tex_slat(
                tex_cond, tex_model_key,
                shape_slat, tex_slat_sampler_params
            )

        del shape_slat, tex_cond
        gc.collect()
        torch.cuda.empty_cache()

        # Decode texture using pre-computed subs
        tex_voxels = self.decode_tex_slat(tex_slat, subs)

        del tex_slat
        torch.cuda.empty_cache()

        # Combine meshes with texture voxels
        out_mesh = []
        for m, v in zip(meshes, tex_voxels):
            m.fill_holes()
            out_mesh.append(
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin=[-0.5, -0.5, -0.5],
                    voxel_size=1 / resolution,
                    coords=v.coords[:, 1:],
                    attrs=v.feats,
                    voxel_shape=torch.Size([*v.shape, *v.spatial_shape]),
                    layout=self.pbr_attr_layout
                )
            )
        return out_mesh

