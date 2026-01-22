
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
        
        IMPORTANT: You MUST provide realistic non-zero values. A typical outdoor scene has intensity around 1.0-2.0 and ambient around 0.15-0.4.
        
        Return a JSON object with the following keys and values:
        - azimuth: float (0-360 degrees). Direction of the light. 0 is North/Back, 90 is East/Right, 180 is South/Front, 270 is West/Left. 
        - elevation: float (0-90 degrees). 0 is horizon, 90 is zenith (directly overhead). Typical daylight is 30-60.
        - intensity: float (0.5 to 5.0). Brightness of the main light source. 1.0 is standard sun. NEVER use 0.
        - ambient: float (0.1 to 1.0). Ambient light level. 0.2 is typical outdoor, 0.4 is overcast. NEVER use 0.
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
            
            print("Sending request to Gemini...")
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
            print(f"Gemini RAW Response: {result_json}")
            
            if not result_json:
                 print("ERROR: Empty response from Gemini.")
                 return self._get_default_params()

            params = json.loads(result_json)
            return self._validate_params(params)
            
        except Exception as e:
            print(f"CRITICAL ERROR calling Gemini API: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_params()



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
                    # Clamp to logical ranges
                    if key == "azimuth": val = val % 360.0
                    if key == "elevation": val = max(0.0, min(90.0, val))
                    # CRITICAL: Enforce minimum intensity and ambient to prevent black output
                    if key == "intensity": val = max(0.5, min(5.0, val))
                    if key == "ambient": val = max(0.1, min(1.0, val))
                    if key == "temperature": val = max(-1.0, min(1.0, val))
                    safe_params[key] = val
                except ValueError:
                    pass
        return safe_params
