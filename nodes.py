
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
                "ply_path": ("STRING", {"default": ""}),
                "force_recompute_normals": ("BOOLEAN", {"default": False}),
                "normal_smoothing": ("INT", {"default": 30, "min": 5, "max": 100}),
            }
        }

    RETURN_TYPES = ("GS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "LightForge3DGS"

    def load_model(self, ply_path, force_recompute_normals, normal_smoothing):
        ply_path = ply_path.strip('"').strip("'")
        
        if not ply_path:
            raise ValueError("Please enter a path to a .ply file")
        
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"PLY file not found: {ply_path}")
        
        model_data = LightForgeEngine.load_ply(ply_path, force_recompute=force_recompute_normals, k=normal_smoothing)
        return (model_data,)

class Relight3DGS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("GS_MODEL",),
                "view_mode": (["Final", "Normals", "Light Map"], {"default": "Final"}),
                "azimuth": ("FLOAT", {"default": 180.0, "min": 0.0, "max": 360.0, "step": 1.0, "display": "slider"}),
                "elevation": ("FLOAT", {"default": 45.0, "min": 0.0, "max": 90.0, "step": 1.0, "display": "slider"}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01, "display": "slider"}),
                "ambient": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "temperature": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "global_brightness": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "display": "slider"}),
            },
            "optional": {
                "light_params": ("LIGHT_PARAMS",),
            }
        }

    RETURN_TYPES = ("GS_MODEL", "STRING")
    RETURN_NAMES = ("relit_model", "debug_info")
    FUNCTION = "relight"
    CATEGORY = "LightForge3DGS"

    def relight(self, model, view_mode, azimuth, elevation, intensity, ambient, temperature, global_brightness, light_params=None):
        # Override with light_params if provided
        if light_params:
            print(f"--- RELIGHT PARAMETERS (Gemini) ---")
            print(f"Original from Gemini: {light_params}")
            azimuth = light_params.get("azimuth", azimuth)
            elevation = light_params.get("elevation", elevation)
            intensity = light_params.get("intensity", intensity)
            ambient = light_params.get("ambient", ambient)
            temperature = light_params.get("temperature", temperature)
        else:
            print(f"--- RELIGHT PARAMETERS (Manual) ---")

        # Apply Global Brightness
        intensity *= global_brightness
        
        print(f"FINAL VALUES: Az={azimuth:.1f}, El={elevation:.1f}, Int={intensity:.2f}, Amb={ambient:.2f}, Temp={temperature:.1f}")
        
        new_model = LightForgeEngine.relight(model, azimuth, elevation, intensity, ambient, temperature, -1)
        
        # Check normals in debug
        normals = model['normals']
        if np.all(normals == 0):
            normals_status = "WARNING: Normals are all ZERO. Relighting will be flat."
        else:
            normals_status = f"Normals present (Shape: {normals.shape})"
            
        debug_text = f"Status: {normals_status}\n\nUsed Parameters:\nAzimuth: {azimuth:.1f}\nElevation: {elevation:.1f}\nIntensity: {intensity:.2f} (x{global_brightness})\nAmbient: {ambient:.2f}\nTemperature: {temperature:.1f}"
        
        return (new_model, debug_text)

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
                "model_name": (["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro"], {"default": "gemini-2.0-flash-exp"}),
            }
        }

    RETURN_TYPES = ("LIGHT_PARAMS", "STRING")
    RETURN_NAMES = ("light_params", "analysis_text")
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
        
        # Create descriptive text
        analysis_text = f"Gemini Analysis:\n"
        analysis_text += f"- Azimuth: {params['azimuth']}°\n"
        analysis_text += f"- Elevation: {params['elevation']}°\n"
        analysis_text += f"- Intensity: {params['intensity']}\n"
        analysis_text += f"- Ambient: {params['ambient']}\n"
        analysis_text += f"- Temperature: {params['temperature']}\n"
        
        return (params, analysis_text)







NODE_CLASS_MAPPINGS = {
    "Load3DGS": Load3DGS,
    "Relight3DGS": Relight3DGS,
    "Save3DGS": Save3DGS,
    "GeminiLightExtractor": GeminiLightExtractorNode,

    "Segment3DGS": Segment3DGS,
    "Analyze3DGS": Analyze3DGS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load3DGS": "Load 3DGS (.ply)",
    "Relight3DGS": "Relight 3DGS (Physical)",
    "Save3DGS": "Save 3DGS (.ply)",
    "GeminiLightExtractor": "Gemini 3 Light Extractor",

    "Segment3DGS": "Segment 3DGS (K-Means)",
    "Analyze3DGS": "Analyze 3DGS Info"
}
