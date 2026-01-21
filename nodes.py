
import os
import folder_paths
import numpy as np
from .lighting_engine import LightForgeEngine
from .gemini_light_extractor import GeminiLightExtractor

class Load3DGS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ply_path": ("STRING", {"default": "input.ply"}),
            }
        }

    RETURN_TYPES = ("GS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "LightForge3DGS"

    def load_model(self, ply_path):
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"PLY file not found: {ply_path}")
        
        model_data = LightForgeEngine.load_ply(ply_path)
        return (model_data,)

class Relight3DGS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("GS_MODEL",),
                "azimuth": ("FLOAT", {"default": 180.0, "min": 0.0, "max": 360.0}),
                "elevation": ("FLOAT", {"default": 45.0, "min": 0.0, "max": 90.0}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0}),
                "ambient": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                "temperature": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0}),
                "cluster_mask": ("INT", {"default": -1, "min": -1, "max": 100}), # -1 = All
            },
            "optional": {
                "light_params": ("LIGHT_PARAMS",), # From Gemini
            }
        }

    RETURN_TYPES = ("GS_MODEL",)
    RETURN_NAMES = ("relit_model",)
    FUNCTION = "relight"
    CATEGORY = "LightForge3DGS"

    def relight(self, model, azimuth, elevation, intensity, ambient, temperature, cluster_mask=-1, light_params=None):
        # Override with light_params if provided
        if light_params:
            print(f"Using Gemini Light Params: {light_params}")
            azimuth = light_params.get("azimuth", azimuth)
            elevation = light_params.get("elevation", elevation)
            intensity = light_params.get("intensity", intensity)
            ambient = light_params.get("ambient", ambient)
            temperature = light_params.get("temperature", temperature)
            
        print(f"Relighting with: Az={azimuth}, El={elevation}, Int={intensity}, Amb={ambient}, Temp={temperature}, Cluster={cluster_mask}")
        
        new_model = LightForgeEngine.relight(model, azimuth, elevation, intensity, ambient, temperature, cluster_mask)
        return (new_model,)

class Segment3DGS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("GS_MODEL",),
                "n_clusters": ("INT", {"default": 10, "min": 2, "max": 50}),
            }
        }

    RETURN_TYPES = ("GS_MODEL",)
    RETURN_NAMES = ("segmented_model",)
    FUNCTION = "segment"
    CATEGORY = "LightForge3DGS"

    def segment(self, model, n_clusters):
        new_model = LightForgeEngine.segment(model, n_clusters)
        return (new_model,)

class Analyze3DGS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("GS_MODEL",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("analysis_text",)
    FUNCTION = "analyze"
    CATEGORY = "LightForge3DGS"

    def analyze(self, model):
        text = LightForgeEngine.analyze(model)
        return (text,)

class Save3DGS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("GS_MODEL",),
                "filename_prefix": ("STRING", {"default": "relit_model"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "save_model"
    OUTPUT_NODE = True
    CATEGORY = "LightForge3DGS"

    def save_model(self, model, filename_prefix):
        output_dir = folder_paths.get_output_directory()
        filename = f"{filename_prefix}.ply"
        output_path = os.path.join(output_dir, filename)
        
        LightForgeEngine.save_ply(model, output_path)
        
        return (output_path,)

class GeminiLightExtractorNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",), # ComfyUI Image Tensor
                "gemini_api_key": ("STRING", {"default": "", "multiline": False}),
                "model_name": (["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"], {"default": "gemini-2.0-flash-exp"}),
            }
        }

    RETURN_TYPES = ("LIGHT_PARAMS",)
    RETURN_NAMES = ("light_params",)
    FUNCTION = "extract_light"
    CATEGORY = "LightForge3DGS"

    def extract_light(self, image, gemini_api_key, model_name):
        # ComfyUI image is [B, H, W, C] Tensor, normalized 0-1.
        # Need to convert to PIL or bytes for Gemini.
        import torch
        from PIL import Image
        
        # Take first image in batch
        img_tensor = image[0] 
        i = 255. * img_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        extractor = GeminiLightExtractor(api_key=gemini_api_key, model_name=model_name)
        params = extractor.analyze(img)
        
        return (params,)


from .hdri_utils import HDRIAnalyzer

class LoadHDRI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hdri_path": ("STRING", {"default": "input.hdr"}),
            }
        }

    RETURN_TYPES = ("HDRI_PATH",)
    RETURN_NAMES = ("hdri_path",)
    FUNCTION = "load_hdri"
    CATEGORY = "LightForge3DGS"

    def load_hdri(self, hdri_path):
        if not os.path.exists(hdri_path):
            raise FileNotFoundError(f"HDRI file not found: {hdri_path}")
        return (hdri_path,)

class HDRILightExtractor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hdri_path": ("HDRI_PATH",),
            }
        }

    RETURN_TYPES = ("LIGHT_PARAMS",)
    RETURN_NAMES = ("light_params",)
    FUNCTION = "extract_light"
    CATEGORY = "LightForge3DGS"

    def extract_light(self, hdri_path):
        params = HDRIAnalyzer.load_and_analyze(hdri_path)
        if params is None:
            # Fallback
            params = {
                "azimuth": 180.0, "elevation": 45.0, "intensity": 1.0, "ambient": 0.2, "temperature": 0.0
            }
        return (params,)

class GeminiImageGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A sunset over the ocean, golden hour lighting"}),
                "gemini_api_key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "LightForge3DGS"

    def generate(self, prompt, gemini_api_key):
        import numpy as np
        import torch
        
        extractor = GeminiLightExtractor(api_key=gemini_api_key)
        pil_img = extractor.generate_image(prompt)
        
        if pil_img is None:
            # Return black placeholder or error
            return (torch.zeros((1, 512, 512, 3)),)
            
        # Convert PIL to Tensor [B, H, W, C]
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np)[None,]
        return (img_tensor,)

NODE_CLASS_MAPPINGS = {
    "Load3DGS": Load3DGS,
    "Relight3DGS": Relight3DGS,
    "Save3DGS": Save3DGS,
    "GeminiLightExtractor": GeminiLightExtractorNode,
    "LoadHDRI": LoadHDRI,
    "HDRILightExtractor": HDRILightExtractor,
    "GeminiImageGenerator": GeminiImageGenerator,
    "Segment3DGS": Segment3DGS,
    "Analyze3DGS": Analyze3DGS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load3DGS": "Load 3DGS (.ply)",
    "Relight3DGS": "Relight 3DGS (Physical)",
    "Save3DGS": "Save 3DGS (.ply)",
    "GeminiLightExtractor": "Gemini 3 Light Extractor",
    "LoadHDRI": "Load HDRI (.exr/.hdr)",
    "HDRILightExtractor": "HDRI Light Extractor",
    "GeminiImageGenerator": "Gemini 3 Image Generator (Nano Banana)",
    "Segment3DGS": "Segment 3DGS (K-Means)",
    "Analyze3DGS": "Analyze 3DGS Info"
}
