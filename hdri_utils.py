
import os
import numpy as np

class HDRIAnalyzer:
    @staticmethod
    def load_and_analyze(hdri_path):
        """
        Loads an HDRI file (EXR/HDR) and identifies the brightest light source.
        Returns light params dict.
        """
        try:
            import imageio.v3 as iio
        except ImportError:
            print("imageio not found. Please install imageio[pyav] or similar.")
            return None

        if not os.path.exists(hdri_path):
            raise FileNotFoundError(f"HDRI file not found: {hdri_path}")

        print(f"Loading HDRI: {hdri_path}")
        # Load image (H, W, C)
        # Using imageio to support .hdr/.exr
        try:
            img = iio.imread(hdri_path)
            # Ensure float32
            img = img.astype(np.float32)
        except Exception as e:
            print(f"Error loading HDRI: {e}")
            return None

        # Handle grayscale
        if len(img.shape) == 2:
            img = np.stack([img]*3, axis=-1)
            
        # 1. Find max intensity pixel
        # Luminance = 0.2126*R + 0.7152*G + 0.0722*B
        luminance = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
        
        # Max index
        max_idx = np.argmax(luminance)
        h, w = luminance.shape
        y_max, x_max = np.unravel_index(max_idx, (h, w))
        
        print(f"Brightest spot at ({x_max}, {y_max}) with luminance {luminance[y_max, x_max]}")
        
        # 2. Map (x, y) to Spherical Coordinates (Azimuth, Elevation)
        # Assuming Equirectangular projection
        # x goes from 0 to W (0 to 360 deg)
        # y goes from 0 to H (90 to -90 deg, usually top is +90)
        
        # Azimuth: 0 to 360 mapped to 0 to W?
        # Usually center is 0 deg or 180?
        # Let's assume standard equirectangular:
        # u = x / W, v = y / H
        # longitude (azimuth) = u * 360 - 180 (or 0 to 360)
        # latitude (elevation) = (0.5 - v) * 180 => 90 (top) to -90 (bottom)
        
        u = x_max / w
        v = y_max / h
        
        azimuth = u * 360.0
        # Phase shift: often we want 0 to be "forward" or "north". 
        # Standard: Center of image is forward.
        # But for 3DGS relighting "Directional Sun", we interpret Azimuth=0 as North (-Z).
        # We might need calibration. Let's output 0-360 directly from image left-to-right.
        
        elevation = (0.5 - v) * 180.0 
        # Elevation: 90 is top (y=0), -90 is bottom (y=H).
        # Note: In image coords, y=0 is top.
        # So v=0 => el=90. v=1 => el=-90. Correct.
        
        # 3. Intensity
        # Arbitrary scale based on max luminance.
        # Sun is usually super bright in HDR.
        intensity = 1.0 + np.log1p(luminance[y_max, x_max]) / 5.0
        intensity = min(intensity, 5.0) # Clamp
        
        # 4. Ambient
        # Average of entire map? Or lower percentile?
        ambient = np.mean(luminance) / (np.max(luminance) + 1e-6)
        ambient = np.clip(ambient * 5.0, 0.0, 1.0) # Bias up
        
        # 5. Temperature
        # Calculate avergae color of the brightest spot region
        # Take a window around max
        win = 5
        y0, y1 = max(0, y_max-win), min(h, y_max+win)
        x0, x1 = max(0, x_max-win), min(w, x_max+win)
        
        spot = img[y0:y1, x0:x1]
        avg_color = np.mean(spot, axis=(0,1))
        
        # Temp heuristic: R/B ratio
        r, g, b = avg_color
        # Warm: R > B. Cool: B > R.
        # Map to -1..1
        total = r+g+b+1e-6
        r_n, b_n = r/total, b/total
        temp = (r_n - b_n) * 2.0 # Scale
        temp = np.clip(temp, -1.0, 1.0) 
        
        params = {
            "azimuth": float(azimuth),
            "elevation": float(elevation),
            "intensity": float(intensity),
            "ambient": float(ambient),
            "temperature": float(temp)
        }
        return params

    @staticmethod
    def get_preview(hdri_path):
        """
        Returns a simplified LDR preview of the HDRI (tonemapped).
        """
        # Not implemented yet, assumes user loads HDRI directly to view it
        return None
