
import os
import json
import logging
from google import genai
from google.genai import types
from PIL import Image

class GeminiLightExtractor:
    def __init__(self, api_key, model_name="gemini-2.0-flash-exp"):
        self.api_key = api_key
        # Handle model name mapping if needed, or pass directly
        self.model_name = model_name
        self.client = genai.Client(api_key=self.api_key)

    def analyze(self, image):
        """
        Analyzes the image and returns lighting parameters using Gemini.
        Image is a PIL Image.
        """
        print(f"Analyzing image with {self.model_name}...")
        
        if not self.api_key:
            print("ERROR: No API Key provided for Gemini.")
            return self._get_default_params()

        prompt = """
        Analyze this image to estimate the main light source parameters for physically based rendering.
        The goal is to relight a 3D scene to match the lighting in this image.
        
        Return a JSON object with the following keys and values:
        - azimuth: float (0-360 degrees). Direction of the light. 0 is North/Back, 90 is East/Right, 180 is South/Front, 270 is West/Left. 
        - elevation: float (0-90 degrees). 0 is horizon, 90 is zenith (directly overhead).
        - intensity: float (0.0 to 5.0). Brightness of the main light source. 1.0 is standard sun.
        - ambient: float (0.0 to 1.0). Ambient light level. 0.0 is pitch black shadows, 1.0 is fully lit shadows.
        - temperature: float (-1.0 to 1.0). Color temperature. -1.0 is cool/blue (morning/shade), 0.0 is neutral white, 1.0 is warm/orange (sunset/tungsten).
        
        Think step-by-step:
        1. Identify the brightest light source (sun, lamp, window).
        2. Estimate its direction relative to the camera view.
        3. Estimate the time of day or light type (warm/cool).
        4. Estimate contrast (ambient vs direct).
        """
        
        try:
            # Prepare the content
            # google-genai SDK handles PIL images directly in many versions, 
            # otherwise we might need to BytesIO it. The new SDK usually takes PIL.
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "azimuth": {"type": "NUMBER"},
                            "elevation": {"type": "NUMBER"},
                            "intensity": {"type": "NUMBER"},
                            "ambient": {"type": "NUMBER"},
                            "temperature": {"type": "NUMBER"},
                        },
                        "required": ["azimuth", "elevation", "intensity", "ambient", "temperature"]
                    }
                )
            )
            
            result_json = response.text
            print(f"Gemini Response: {result_json}")
            
            params = json.loads(result_json)
            return self._validate_params(params)
            
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_params()


    def generate_image(self, prompt, output_path="generated_ref.png"):
        """
        Generates an image using Gemini (Imagen 3 / Nano Banana).
        """
        print(f"Generating image for prompt: '{prompt}'...")
        if not self.api_key:
            print("ERROR: No API Key.")
            return None
            
        try:
            # Note: The exact method name for Imagen 3 in google-genai varies.
            # Assuming 'imagen-3.0-generate-001' or similar model id within generate_content 
            # or a specific client.models.generate_images if available.
            # As of late 2024/2025 SDK:
            
            # Using standard model name for image gen if 'gemini-2.0-flash' doesn't do it.
            # Assuming models/imagen-3.0-generate-001 or similar.
            image_model = "imagen-3.0-generate-001"
            
            response = self.client.models.generate_images(
                model=image_model,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio="1:1" # or "16:9"
                )
            )
            
            if response.generated_images:
                img_data = response.generated_images[0]
                # Save or return PIL
                if hasattr(img_data, 'image'):
                    img = img_data.image # PIL Image
                    return img
                elif hasattr(img_data, 'image_bytes'):
                    import io
                    return Image.open(io.BytesIO(img_data.image_bytes))
            
            print("No image returned.")
            return None

        except Exception as e:
            print(f"Error generating image: {e}")
            return None

    def _get_default_params(self):
        return {
            "azimuth": 180.0,
            "elevation": 45.0,
            "intensity": 1.0,
            "ambient": 0.2,
            "temperature": 0.0
        }
        
    def _validate_params(self, params):
        # Ensure types and ranges
        safe_params = self._get_default_params()
        for key in safe_params:
            if key in params:
                try:
                    val = float(params[key])
                    # precise clamping to logical ranges could happen here if strictly needed
                    if key == "azimuth": val = val % 360.0
                    if key == "elevation": val = max(0.0, min(90.0, val))
                    if key == "intensity": val = max(0.0, val)
                    if key == "ambient": val = max(0.0, min(1.0, val))
                    if key == "temperature": val = max(-1.0, min(1.0, val))
                    safe_params[key] = val
                except ValueError:
                    pass
        return safe_params
