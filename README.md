# LightForge 3DGS ⚡

**AI-Powered Relighting for 3D Gaussian Splatting**

LightForge 3DGS is a ComfyUI Custom Node that allows you to physically relight 3D Gaussian Splatting models using light parameters extracted from reference images or HDRI maps, powered by **Google Gemini 3**.

Project for Google DeepMind Hackathon.

## Features
- **Physical Relighting**: Modifies Spherical Harmonics (SH) coefficients of the 3DGS model to match the new lighting.
- **Robust Normal Computation**: Automatically computes normals if missing or invalid (using SciPy/PCA).
- **Gemini 3 Integration**: Analyze reference images to extract realistic lighting parameters.
- **Drag & Drop**: Load models easily from the ComfyUI input folder or absolute paths.

## Installation

1.  **Navigate to your ComfyUI custom nodes directory**:
    ```bash
    cd ComfyUI/custom_nodes/
    ```

2.  **Clone this repository**:
    ```bash
    git clone https://github.com/mromerocastro/ComfyUI-LightForge3DGS.git
    ```

3.  **Install Dependencies**:
    Go to the `ComfyUI-LightForge3DGS` folder and install the requirements.
    ```bash
    cd ComfyUI-LightForge3DGS
    pip install -r requirements.txt
    ```
    *Note: If you are using the ComfyUI Portable version, make sure to use the embedded python (e.g., `..\..\python_embeded\python.exe -m pip install -r requirements.txt`).*


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

### 5. Relight 3DGS (Physical)
Applies the lighting parameters to the model. 
**Behavior**:
- **Automatic**: Connect the `LIGHT_PARAMS` output from Gemini Extractor to the `light_params` input.
- **Manual**: Use the sliders (Azimuth, Elevation, etc.).
- **Visuals**: Check the **Preview Widget** (visual light) and the **Debug Info** output (text).

- **Input**: GS_MODEL, LIGHT_PARAMS (optional), Manual Overrides.
- **Output**: GS_MODEL (Relit), DEBUG_INFO (String).

### 6. Save 3DGS (.ply)
Saves the modified model.
- **Input**: GS_MODEL.
- **Output**: File Path.

## Example Workflows

Included in this folder is an example JSON file:

1.  **`workflow_ai_relight.json`**: (Create this yourself easily) `Load 3DGS` + `Load Image` nodes -> `Gemini Extractor` -> `Relight 3DGS`.

**How to use:**
1. Open ComfyUI.
2. Drag one of the `.json` files onto the canvas.
3. Update the `Load3DGS` node with the path to your `.ply` file.
4. (Optional) Add your Gemini API Key if using AI features.
5. Queue Prompt!
