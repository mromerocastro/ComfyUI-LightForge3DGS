
import os
import folder_paths
import numpy as np
import torch
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
            }
        }

    RETURN_TYPES = ("GS_MODEL", "STRING", "IMAGE")
    RETURN_NAMES = ("relit_model", "debug_info", "preview_image")
    FUNCTION = "relight_manual"
    CATEGORY = "LightForge3DGS"

    def relight_manual(self, model, view_mode, azimuth, elevation, intensity, ambient, temperature):
        # Direct pass to engine
        new_model = LightForgeEngine.relight(
            model, azimuth, elevation, intensity, ambient, temperature, 
            cluster_mask=-1, view_mode=view_mode
        )
        
        # Debug Info
        normals = model['normals']
        status = "Normals present" if not np.all(normals == 0) else "WARNING: Zero Normals"
        debug_text = f"Mode: Manual\nStatus: {status}\nParams:\nAz: {azimuth}\nEl: {elevation}\nInt: {intensity}\nAmb: {ambient}\nTemp: {temperature}"
        
        # Preview
        preview_np = LightForgeEngine.generate_preview(azimuth, elevation, intensity, ambient, temperature)
        preview_tensor = torch.from_numpy(preview_np).float().unsqueeze(0)

        return (new_model, debug_text, preview_tensor)

class Relight3DGS_Auto:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("GS_MODEL",),
                "light_params": ("LIGHT_PARAMS",),
                "view_mode": (["Final", "Normals", "Light Map"], {"default": "Final"}),
                # Optional overrides (modifiers) could be added here later if requested, 
                # but let's keep it clean as requested.
                "global_brightness": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("GS_MODEL", "STRING", "IMAGE")
    RETURN_NAMES = ("relit_model", "debug_info", "preview_image")
    FUNCTION = "relight_auto"
    CATEGORY = "LightForge3DGS"

    def relight_auto(self, model, light_params, view_mode, global_brightness):
        # Extract from Gemini params
        az = light_params.get("azimuth", 180.0)
        el = light_params.get("elevation", 45.0)
        intensity = light_params.get("intensity", 1.0) * global_brightness
        amb = light_params.get("ambient", 0.2)
        temp = light_params.get("temperature", 0.0)
        
        new_model = LightForgeEngine.relight(
            model, az, el, intensity, amb, temp, 
            cluster_mask=-1, view_mode=view_mode
        )
        
        normals = model['normals']
        status = "Normals present" if not np.all(normals == 0) else "WARNING: Zero Normals"
        debug_text = f"Mode: Auto (Gemini)\nStatus: {status}\nRaw Params:\n{light_params}\n\nUsed:\nAz: {az}\nEl: {el}\nInt: {intensity}\nAmb: {amb}\nTemp: {temp}"
        
        preview_np = LightForgeEngine.generate_preview(az, el, intensity, amb, temp)
        preview_tensor = torch.from_numpy(preview_np).float().unsqueeze(0)

        return (new_model, debug_text, preview_tensor)

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
    "Relight3DGS_Auto": Relight3DGS_Auto,
    "Save3DGS": Save3DGS,
    "GeminiLightExtractor": GeminiLightExtractorNode,

    "Segment3DGS": Segment3DGS,
    "Analyze3DGS": Analyze3DGS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load3DGS": "Load 3DGS (.ply)",
    "Relight3DGS": "Relight 3DGS (Physical)",
    "Relight3DGS_Auto": "Relight 3DGS (Auto / Gemini)",
    "Save3DGS": "Save 3DGS (.ply)",
    "GeminiLightExtractor": "Gemini 3 Light Extractor",

    "Segment3DGS": "Segment 3DGS (K-Means)",
    "Analyze3DGS": "Analyze 3DGS Info"
}
