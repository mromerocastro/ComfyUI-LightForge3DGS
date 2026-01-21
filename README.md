# LightForge 3DGS ⚡

**AI-Powered Relighting for 3D Gaussian Splatting**

LightForge 3DGS is a ComfyUI Custom Node that allows you to physically relight 3D Gaussian Splatting models using light parameters extracted from reference images or HDRI maps, powered by **Google Gemini 3**.

Project for Google DeepMind Hackathon.

## Features
- **Gemini 3 Integration**: Analyze reference images to extract realistic lighting parameters (Azimuth, Elevation, Intensity, Temperature).
- **Nano Banana (Imagen 3)**: Generate reference lighting environments from text prompts directly in ComfyUI.
- **HDRI Support**: Extract dominant light sources from `.exr` and `.hdr` files.
- **Segmentation**: Spatially cluster the model to relight specific parts (e.g., separate objects).
- **Physical Relighting**: Modifies Spherical Harmonics (SH) coefficients of the 3DGS model to match the new lighting.
- **Normal Computation**: Automatically computes normals if missing from the PLY.

## Installation

1. Copy the `ComfyUI-LightForge3DGS` folder to your `ComfyUI/custom_nodes/` directory.
2. Install dependencies:
   ```bash
   pip install google-genai plyfile numpy open3d imageio[pyav] scikit-learn
   ```
3. Restart ComfyUI.

## Nodes

### 1. Load 3DGS (.ply)
Loads a Gaussian Splatting PLY file.
- **Input**: Path to `.ply`.
- **Output**: GS_MODEL.

### 2. Analyze 3DGS Info
Returns text statistics about the model.
- **Input**: GS_MODEL.
- **Output**: String.

### 3. Segment 3DGS (K-Means)
Segments the model into clusters.
- **Input**: GS_MODEL, n_clusters.
- **Output**: GS_MODEL (with labels).

### 4. Gemini 3 Light Extractor
Uses Gemini 3 Vision to analyze an image and estimate lighting.
- **Input**: Image (from Load Image or Generator), API Key.
- **Output**: LIGHT_PARAMS.

### 5. Gemini 3 Image Generator (Nano Banana)
Generates a reference image using Gemini/Imagen 3.
- **Input**: Prompt, API Key.
- **Output**: IMAGE.

### 6. HDRI Light Extractor
Analyzes an HDRI map to find the sun position.
- **Input**: HDRI Path.
- **Output**: LIGHT_PARAMS.

### 7. Relight 3DGS (Physical)
Applies the lighting parameters to the model. Can limit to specific cluster.
- **Input**: GS_MODEL, LIGHT_PARAMS (optional), Manual Overrides, Cluster Mask.
- **Output**: GS_MODEL (Relit).

### 8. Save 3DGS (.ply)
Saves the modified model.
- **Input**: GS_MODEL.
- **Output**: File Path.

## Example Workflows

Included in this folder are JSON files you can drag and drop into ComfyUI:

1.  **`workflow_text_to_relight.json`**: Uses Gemini (Nano Banana) to generate a reference image from text ("Cyberpunk city", "Sunset", etc.) and applies that lighting to your model.
2.  **`workflow_hdri_relight.json`**: Loads an HDRI file, extracts the sun position, and relights the model.

**How to use:**
1. Open ComfyUI.
2. Drag one of the `.json` files onto the canvas.
3. Update the `Load3DGS` node with the path to your `.ply` file.
4. (Optional) Add your Gemini API Key if using AI features.
5. Queue Prompt!
