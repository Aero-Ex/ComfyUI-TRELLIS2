"""Multi-image conditioning nodes for TRELLIS.2 Image-to-3D generation.

Provides nodes that accept multiple images as input for improved 3D generation.
"""

try:
    from .utils.isolation import smart_isolated
except ImportError:
    from utils.isolation import smart_isolated


@smart_isolated(env="trellis2", import_paths=[".", ".."])
class Trellis2MultiImageConditioning:
    """Extract conditioning from multiple images for multi-view 3D generation."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_config": ("TRELLIS2_MODEL_CONFIG",),
                "images": ("IMAGE",),  # Batched images [N, H, W, C]
                "masks": ("MASK",),    # Batched masks [N, H, W]
                "include_1024": ("BOOLEAN", {"default": True}),
                "background_color": (["black", "gray", "white"], {"default": "black"}),
            },
        }

    RETURN_TYPES = ("TRELLIS2_MULTI_CONDITIONING", "IMAGE")
    RETURN_NAMES = ("conditioning", "preprocessed_images")
    FUNCTION = "get_conditioning"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Extract conditioning features from multiple images for multi-view 3D generation.

Use this node when you have multiple reference images of the same object from
different angles. The conditioning will be used by Multi-Image to Shape/Texture nodes.

Inputs:
- images: Batch of images (use Image Batch node to combine)
- masks: Batch of masks (use Mask Batch node to combine)

Supports 2-8 images for best results.
"""

    def get_conditioning(self, model_config, images, masks, include_1024=True, background_color="black"):
        from utils.stages import run_multi_conditioning

        conditioning, preprocessed_images = run_multi_conditioning(
            model_config=model_config,
            images=images,
            masks=masks,
            include_1024=include_1024,
            background_color=background_color,
        )

        return (conditioning, preprocessed_images)


@smart_isolated(env="trellis2", import_paths=[".", ".."], timeout=1800)
class Trellis2MultiImageToShape:
    """Generate 3D shape from multiple conditioning images using TRELLIS.2."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_config": ("TRELLIS2_MODEL_CONFIG",),
                "conditioning": ("TRELLIS2_MULTI_CONDITIONING",),
            },
            "optional": {
                "mode": (["stochastic", "multidiffusion"], {
                    "default": "stochastic",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "ss_guidance_strength": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "ss_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1}),
                "shape_guidance_strength": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "shape_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1}),
                "max_tokens": ("INT", {"default": 49152, "min": 16384, "max": 65536, "step": 4096}),
            }
        }

    RETURN_TYPES = ("TRELLIS2_SHAPE_RESULT", "TRIMESH")
    RETURN_NAMES = ("shape_result", "mesh")
    FUNCTION = "generate"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Generate 3D shape from multiple conditioning images.

Modes:
- stochastic: Uses a different image at each sampling step (faster, same VRAM)
- multidiffusion: Averages predictions from all images per step (highest quality, N× slower)

The model will incorporate features from all input views for more complete geometry.
"""

    def generate(
        self,
        model_config,
        conditioning,
        mode="stochastic",
        seed=0,
        ss_guidance_strength=7.5,
        ss_sampling_steps=12,
        shape_guidance_strength=7.5,
        shape_sampling_steps=12,
        max_tokens=49152,
    ):
        import trimesh as Trimesh
        from utils.stages import run_multi_image_shape_generation

        shape_result = run_multi_image_shape_generation(
            model_config=model_config,
            conditioning=conditioning,
            mode=mode,
            seed=seed,
            ss_guidance_strength=ss_guidance_strength,
            ss_sampling_steps=ss_sampling_steps,
            shape_guidance_strength=shape_guidance_strength,
            shape_sampling_steps=shape_sampling_steps,
            max_num_tokens=max_tokens,
        )

        # Create trimesh from vertices/faces
        tri_mesh = Trimesh.Trimesh(
            vertices=shape_result['mesh_vertices'],
            faces=shape_result['mesh_faces'],
            process=False
        )

        return (shape_result, tri_mesh)


@smart_isolated(env="trellis2", import_paths=[".", ".."], timeout=1800)
class Trellis2MultiImageToTexturedMesh:
    """Generate PBR textured mesh from shape using multiple conditioning images."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_config": ("TRELLIS2_MODEL_CONFIG",),
                "conditioning": ("TRELLIS2_MULTI_CONDITIONING",),
                "shape_result": ("TRELLIS2_SHAPE_RESULT",),
            },
            "optional": {
                "mode": (["stochastic", "multidiffusion"], {
                    "default": "stochastic",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "tex_guidance_strength": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "tex_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "TRELLIS2_VOXELGRID", "TRIMESH")
    RETURN_NAMES = ("trimesh", "voxelgrid", "pbr_pointcloud")
    FUNCTION = "generate"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Generate PBR textured mesh from shape using multiple conditioning images.

Uses the same multi-image mode as the shape generation to ensure consistency.
"""

    def generate(
        self,
        model_config,
        conditioning,
        shape_result,
        mode="stochastic",
        seed=0,
        tex_guidance_strength=7.5,
        tex_sampling_steps=12,
    ):
        import numpy as np
        import trimesh as Trimesh
        from utils.stages import run_multi_image_texture_generation

        texture_result = run_multi_image_texture_generation(
            model_config=model_config,
            conditioning=conditioning,
            shape_result=shape_result,
            mode=mode,
            seed=seed,
            tex_guidance_strength=tex_guidance_strength,
            tex_sampling_steps=tex_sampling_steps,
        )

        # Create trimesh from vertices/faces
        tri_mesh = Trimesh.Trimesh(
            vertices=texture_result['mesh_vertices'],
            faces=texture_result['mesh_faces'],
            process=False
        )

        # Create voxel grid dict for Rasterize PBR node
        voxel_grid = {
            'coords': texture_result['voxel_coords'],
            'attrs': texture_result['voxel_attrs'],
            'voxel_size': texture_result['voxel_size'],
            'layout': texture_result['pbr_layout'],
            'original_vertices': texture_result['original_vertices'],
            'original_faces': texture_result['original_faces'],
        }

        # Create debug point cloud
        coords = texture_result['voxel_coords']
        attrs = texture_result['voxel_attrs']
        voxel_size = texture_result['voxel_size']
        pbr_layout = texture_result['pbr_layout']

        # Convert voxel indices to world positions
        point_positions = coords * voxel_size

        # Apply Y-up to Z-up conversion
        point_positions[:, 1], point_positions[:, 2] = (
            point_positions[:, 2].copy(),
            -point_positions[:, 1].copy()
        )

        # Convert attrs from [-1, 1] to [0, 1]
        attrs_normalized = (attrs + 1.0) * 0.5

        # For trimesh.PointCloud colors, use base_color RGB + alpha
        base_color_slice = pbr_layout.get('base_color', slice(0, 3))
        alpha_slice = pbr_layout.get('alpha', slice(5, 6))

        colors_rgb = (attrs_normalized[:, base_color_slice] * 255).clip(0, 255).astype(np.uint8)
        colors_alpha = (attrs_normalized[:, alpha_slice] * 255).clip(0, 255).astype(np.uint8)
        colors_rgba = np.concatenate([colors_rgb, colors_alpha], axis=1)

        pointcloud = Trimesh.PointCloud(
            vertices=point_positions,
            colors=colors_rgba
        )

        return (tri_mesh, voxel_grid, pointcloud)


NODE_CLASS_MAPPINGS = {
    "Trellis2MultiImageConditioning": Trellis2MultiImageConditioning,
    "Trellis2MultiImageToShape": Trellis2MultiImageToShape,
    "Trellis2MultiImageToTexturedMesh": Trellis2MultiImageToTexturedMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis2MultiImageConditioning": "TRELLIS.2 Multi-Image Conditioning",
    "Trellis2MultiImageToShape": "TRELLIS.2 Multi-Image to Shape",
    "Trellis2MultiImageToTexturedMesh": "TRELLIS.2 Multi-Image to Textured Mesh",
}
