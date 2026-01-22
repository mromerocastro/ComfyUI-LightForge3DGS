import numpy as np
import os

path = r"c:\Users\Marlon\Documents\DEVELOPMENT\ComfyUI-LightForge3DGS\base\segmentation_cache_with_normals.npz"

if os.path.exists(path):
    data = np.load(path)
    print(f"File: {os.path.basename(path)}")
    print("Keys found:", data.files)
    for key in data.files:
        print(f"- {key}: shape {data[key].shape}, dtype {data[key].dtype}")
else:
    print("File not found.")
