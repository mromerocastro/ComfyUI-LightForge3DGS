
import numpy as np

from plyfile import PlyData, PlyElement
import os

class LightForgeEngine:
    @staticmethod
    def load_ply(file_path, force_recompute=False, k=30):
        """
        Loads a 3DGS PLY file and returns a dictionary with numpy arrays.
        """
        print(f"Loading PLY: {file_path}")
        plydata = PlyData.read(file_path)
        vertex = plydata['vertex']
        
        # Extract properties
        positions = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
        
        # Extract SH DC (f_dc_0, f_dc_1, f_dc_2) - usually RGB-like
        try:
            colors_sh = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=-1)
        except:
            # Fallback if names are different or missing
            print("Warning: f_dc fields not found, trying RGB or setting zero.")
            colors_sh = np.zeros_like(positions)

        # Check for normals (nx, ny, nz)
        normals = None
        
        # If not forcing recompute, try loading from file
        if not force_recompute:
            if 'nx' in vertex and 'ny' in vertex and 'nz' in vertex:
                normals = np.stack([vertex['nx'], vertex['ny'], vertex['nz']], axis=-1)
                if np.all(normals == 0):
                   print("WARNING: PLY normals are ZERO. Recomputing...")
                   normals = None # Trigger recompute logic
            else:
                print("Normals not found in PLY.")

        # If data struct prepared, check for sidecar cache (only if not forcing)
        # Note: We build the dict later, but we need normals now or later.
        
        # ... Wait, the cache loading logic is currently AFTER dict creation. 
        # I need to restructure slightly to prioritize Cache -> File -> Compute.
        
        data = {
            'plydata': plydata, 
            'positions': positions,
            'normals': np.zeros_like(positions), # placeholder
            'colors_sh': colors_sh,
            'vertex_data': vertex.data
        }
        
        # Check Cache FIRST if not forcing
        loaded_from_cache = False
        if not force_recompute:
            base_dir = os.path.dirname(file_path)
            filename = os.path.basename(file_path)
            name_no_ext = os.path.splitext(filename)[0]
            candidates = [
                os.path.join(base_dir, f"{name_no_ext}.npz"),
                os.path.join(base_dir, "segmentation_cache_with_normals.npz"),
                os.path.join(base_dir, "segmentation_cache.npz")
            ]
            for npz_path in candidates:
                if os.path.exists(npz_path):
                    try:
                        print(f"Found cache file: {npz_path}")
                        cache = np.load(npz_path)
                        if len(cache['positions']) == len(positions):
                            print("✅ Cache matches PLY! Loading...")
                            if 'normals' in cache:
                                data['normals'] = cache['normals']
                                loaded_from_cache = True
                            if 'labels' in cache:
                                data['labels'] = cache['labels']
                            break
                    except Exception: pass
        
        # If no cache and (no file normals OR forcing), COMPUTE
        if not loaded_from_cache:
            if normals is not None and not force_recompute:
                data['normals'] = normals # Use file normals
            else:
                print(f"Computing normals (k={k})...")
                data['normals'] = LightForgeEngine.compute_normals(positions, k=k)

        return data

    @staticmethod
    def compute_normals(positions, k=30):
        """
        Computes normals using PCA via scikit-learn and numpy.
        """
        try:
            import scipy.spatial
        except ImportError:
            print("scipy not found. Cannot compute normals. Please install scipy.")
            return np.zeros_like(positions)

        print(f"Computing normals using PCA (via SciPy) k={k}...")
        
        # Find neighbors using SciPy KDTree (more standard in ComfyUI envs)
        tree = scipy.spatial.KDTree(positions)
        # query returns values, indices
        _, indices = tree.query(positions, k=k)
        
        # Gather neighborhoods: (N, k, 3)
        neighborhoods = positions[indices]
        
        # Center the neighborhoods
        means = np.mean(neighborhoods, axis=1, keepdims=True)
        centered = neighborhoods - means
        
        # Compute variances (N, 3, 3)
        # matrix multiplication of (3, k) * (k, 3) for each point
        covariances = np.matmul(centered.transpose(0, 2, 1), centered)
        
        # Eigen decomposition
        # eigh is for symmetric matrices. Returns eigenvalues (ascending) and eigenvectors.
        # The column v[:, i] is the eigenvector for w[i].
        # We want the eigenvector for the smallest eigenvalue (index 0).
        _, evecs = np.linalg.eigh(covariances)
        estimated_normals = evecs[:, :, 0]
        
        # Orientation consistency heuristic: Orient away from object centroid
        object_centroid = np.mean(positions, axis=0)
        view_dirs = positions - object_centroid
        
        # Dot product to check alignment
        dots = np.sum(estimated_normals * view_dirs, axis=1)
        
        # Flip where dot product is negative (pointing inwards)
        flip_mask = dots < 0
        estimated_normals[flip_mask] = -estimated_normals[flip_mask]
        
        return estimated_normals

    @staticmethod
    def sh_to_rgb(sh_dc):
        C0 = 0.28209479177387814
        rgb = sh_dc * C0 + 0.5
        return np.clip(rgb, 0, 1)

    @staticmethod
    def rgb_to_sh(rgb):
        C0 = 0.28209479177387814
        sh_dc = (rgb - 0.5) / C0
        return sh_dc

    @staticmethod
    def spherical_to_cartesian(azimuth, elevation):
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        x = np.cos(el_rad) * np.sin(az_rad)
        y = np.sin(el_rad) 
        z = np.cos(el_rad) * np.cos(az_rad)
        direction = np.array([x, y, z])
        return direction / np.linalg.norm(direction)


    @staticmethod
    def analyze(data):
        """
        Returns string analysis of the GS model.
        """
        pos = data['positions']
        normals = data['normals']
        colors = data['colors_sh']
        
        info = []
        info.append(f"Total Gaussians: {len(pos):,}")
        info.append(f"Bounding Box: Min {pos.min(axis=0)}, Max {pos.max(axis=0)}")
        info.append(f"Exposed Properties: {data['plydata']['vertex'].data.dtype.names}")
        
        if 'labels' in data:
            n_clusters = len(np.unique(data['labels']))
            info.append(f"Segmentation: {n_clusters} clusters found.")
            
        return "\n".join(info)

    @staticmethod
    def segment(data, n_clusters=7):
        """
        Segments gaussians using KMeans. Adds 'labels' to data.
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            print("scikit-learn not found. Skipping segmentation.")
            return data
            
        print(f"Segmenting {len(data['positions'])} gaussians into {n_clusters} clusters...")
        positions = data['positions']
        
        # Subsample for speed
        sample_size = min(50000, len(positions))
        indices = np.random.choice(len(positions), sample_size, replace=False)
        positions_sample = positions[indices]
        
        kmeans = KMeans(n_clusters=n_clusters, n_init=5)
        kmeans.fit(positions_sample)
        
        print("Predicting all labels...")
        labels = kmeans.predict(positions)
        
        new_data = data.copy()
        new_data['labels'] = labels
        return new_data

    @staticmethod
    def relight(data, azimuth, elevation, intensity, ambient, temp, cluster_mask=-1, view_mode="Final"):
        """
        Applies directional lighting to 3DGS data.
        If cluster_mask >= 0 and 'labels' in data, only relights that cluster.
        """
        normals = data['normals']
        colors_sh = data['colors_sh']
        labels = data.get('labels', None)
        
        # Calculate Sun Direction
        sun_dir = LightForgeEngine.spherical_to_cartesian(azimuth, elevation)
        
        # Calculate Lighting Factor (Lambertian)
        norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        norm_lengths = np.where(norm_lengths < 1e-6, 1.0, norm_lengths)
        normals_norm = normals / norm_lengths
        
        dot = np.dot(normals_norm, sun_dir)
        diffuse = np.maximum(0, dot)
        lighting_factor = ambient + (1.0 - ambient) * intensity * diffuse
        lighting_factor = np.clip(lighting_factor, 0, 2.0).reshape(-1, 1)
        
        # Create final lighting mask
        # Default: Apply to all (1.0)
        # If masking: Apply 1.0 to masked, and maybe neutral (1.0 for ambient?) to others? 
        # Or just don't change the others?
        # The equation above calculates a modification factor.
        # If we don't want to relight, the factor should be 1.0? 
        # No, the previous code replaces color. 
        # So we should calculate NEW colors for ALL, then blend based on mask.
        
        # Convert SH to RGB
        rgb = LightForgeEngine.sh_to_rgb(colors_sh)
        
        # Modulate RGB
        rgb_lit = rgb * lighting_factor
        
        # Apply Temperature
        if abs(temp) > 0.01:
            if temp > 0:  # Warm
                rgb_lit[:, 0] *= (1 + temp * 0.2)
                rgb_lit[:, 1] *= (1 + temp * 0.1)
                rgb_lit[:, 2] *= (1 - temp * 0.15)
            else:  # Cool
                rgb_lit[:, 0] *= (1 + temp * 0.15)
                rgb_lit[:, 1] *= (1 + temp * 0.1)
                rgb_lit[:, 2] *= (1 - temp * 0.2)
        
        # Helper to convert Normals to RGB (0..1)
        if view_mode == "Normals":
            # Map [-1, 1] -> [0, 1]
            rgb_vis = (normals_norm + 1.0) * 0.5
            new_sh_all = LightForgeEngine.rgb_to_sh(rgb_vis)
        
        elif view_mode == "Light Map":
            # Visualize lighting factor as grayscale
            # Normalize factor for visualization (approx)
            fact_vis = lighting_factor / max(intensity + 0.001, 1.0) 
            fact_vis = np.clip(fact_vis, 0, 1)
            rgb_vis = np.hstack([fact_vis, fact_vis, fact_vis])
            new_sh_all = LightForgeEngine.rgb_to_sh(rgb_vis)
            
        else:
            # Final Mode
            rgb_lit = np.clip(rgb_lit, 0, 1)
            new_sh_all = LightForgeEngine.rgb_to_sh(rgb_lit)
        
        # Apply Masking
        if cluster_mask >= 0 and labels is not None:
            print(f"Applying relighting ONLY to Cluster {cluster_mask}")
            # Create boolean mask
            mask = (labels == cluster_mask)
            # Expand mask for SH dims if needed, or index
            # colors_sh is [N, 3]
            # new_sh_all is [N, 3]
            
            final_sh = colors_sh.copy()
            final_sh[mask] = new_sh_all[mask]
        else:
            final_sh = new_sh_all
            
        # Update data copy
        new_data = data.copy()
        new_data['colors_sh'] = final_sh
        return new_data

    @staticmethod
    def save_ply(data, output_path):
        """
        Saves the modified 3DGS data to a PLY file.
        """
        original_vertex_data = data['vertex_data']
        new_sh = data['colors_sh']
        
        # Update vertex data
        # We need to copy to avoid modifying original if it's shared? 
        # But here we assume we are saving the 'data' object which is already modified or holder of modified data.
        # Actually vertex_data is a numpy structured array. We need to write into it.
        
        params_vertex_data = original_vertex_data.copy()
        
        params_vertex_data['f_dc_0'] = new_sh[:, 0]
        params_vertex_data['f_dc_1'] = new_sh[:, 1]
        params_vertex_data['f_dc_2'] = new_sh[:, 2]
        
        # Note: We are currently NOT saving computed normals back to file to keep file size same as original 
        # unless we explicitly want to add them. 
        # If the original didn't have normals, this saves without them (unless we add fields).
        # For safety/compatibility, let's just update colors.
        
        vertex_element = PlyElement.describe(params_vertex_data, 'vertex')
        PlyData([vertex_element], text=False).write(output_path)
        return output_path

    @staticmethod
    def generate_preview(azimuth, elevation, intensity, ambient, temperature, size=512):
        """
        Generates a preview image of a sphere with the applied lighting.
        Returns a numpy array (H, W, 3) normalized 0-1.
        """
        # Create a grid
        x = np.linspace(-1, 1, size)
        y = np.linspace(1, -1, size) # Flip Y to match image coords top-down
        xv, yv = np.meshgrid(x, y)
        
        # Define sphere mask: r^2 = x^2 + y^2
        r2 = xv**2 + yv**2
        mask = r2 <= 1.0
        
        # Calculate normals for the sphere: z = sqrt(1 - r^2)
        z = np.zeros_like(xv)
        z[mask] = np.sqrt(1.0 - r2[mask])
        
        # Normals grid: (Size, Size, 3)
        normals = np.zeros((size, size, 3))
        normals[:, :, 0] = xv
        normals[:, :, 1] = yv
        normals[:, :, 2] = z
        
        # Light Direction
        sun_dir = LightForgeEngine.spherical_to_cartesian(azimuth, elevation)
        
        # Flatten for calculation
        n_flat = normals.reshape(-1, 3)
        
        # Compute Dot Product
        dot = np.dot(n_flat, sun_dir)
        diffuse = np.maximum(0, dot)
        
        # Calculate Lighting Factor
        lighting_factor = ambient + (1.0 - ambient) * intensity * diffuse
        lighting_factor = lighting_factor.reshape(size, size, 1)
        
        # Base Color (White Sphere)
        rgb = np.ones((size, size, 3)) * lighting_factor
        
        # Apply Temperature (shared logic)
        if abs(temperature) > 0.01:
            if temperature > 0:  # Warm
                rgb[:, :, 0] *= (1 + temperature * 0.2)
                rgb[:, :, 1] *= (1 + temperature * 0.1)
                rgb[:, :, 2] *= (1 - temperature * 0.15)
            else:  # Cool
                rgb[:, :, 0] *= (1 + temperature * 0.15)
                rgb[:, :, 1] *= (1 + temperature * 0.1)
                rgb[:, :, 2] *= (1 - temperature * 0.2)
                
        # Mask background to black
        rgb[~mask] = 0.0
        
        # Clip to valid range
        rgb = np.clip(rgb, 0, 1)
        
        return rgb
