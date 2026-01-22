
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
        Computes normals using PCA with robust multi-strategy orientation.
        Includes validation, neighbor consistency, and degenerate normal handling.
        """
        try:
            import scipy.spatial
        except ImportError:
            print("scipy not found. Cannot compute normals. Please install scipy.")
            return np.zeros_like(positions)

        print(f"Computing normals using Enhanced PCA (k={k})...")
        eps = 1e-8  # Epsilon for numerical stability
        
        # Find neighbors using SciPy KDTree
        tree = scipy.spatial.KDTree(positions)
        _, indices = tree.query(positions, k=k)
        
        # Gather neighborhoods: (N, k, 3)
        neighborhoods = positions[indices]
        
        # Center the neighborhoods
        means = np.mean(neighborhoods, axis=1, keepdims=True)
        centered = neighborhoods - means
        
        # Compute covariances (N, 3, 3)
        covariances = np.matmul(centered.transpose(0, 2, 1), centered)
        
        # Eigen decomposition - smallest eigenvalue = normal direction
        eigenvalues, evecs = np.linalg.eigh(covariances)
        estimated_normals = evecs[:, :, 0]
        
        # === ROBUST NORMALIZATION ===
        # Ensure normals are unit length
        norm_lengths = np.linalg.norm(estimated_normals, axis=1, keepdims=True)
        norm_lengths = np.where(norm_lengths < eps, 1.0, norm_lengths)
        estimated_normals = estimated_normals / norm_lengths
        
        # Detect and mark degenerate normals (zero or NaN)
        degenerate_mask = (norm_lengths.squeeze() < eps) | np.isnan(estimated_normals).any(axis=1)
        if degenerate_mask.any():
            print(f"⚠️  Found {degenerate_mask.sum()} degenerate normals, setting to up vector")
            estimated_normals[degenerate_mask] = np.array([0.0, 1.0, 0.0])
        
        # === STRATEGY 1: Orient away from object centroid ===
        object_centroid = np.mean(positions, axis=0)
        view_dirs = positions - object_centroid
        view_dirs_norm = view_dirs / (np.linalg.norm(view_dirs, axis=1, keepdims=True) + eps)
        
        dots = np.sum(estimated_normals * view_dirs_norm, axis=1)
        flip_mask_centroid = dots < 0
        
        # === STRATEGY 2: Neighbor consistency voting ===
        # For each point, check if neighbors agree on orientation
        # Use k_vote smaller than k to avoid over-smoothing
        k_vote = min(10, k // 3)
        neighbor_normals = estimated_normals[indices[:, :k_vote]]
        
        # Compute average dot product with neighbors
        self_normals_expanded = estimated_normals[:, np.newaxis, :]
        neighbor_dots = np.sum(neighbor_normals * self_normals_expanded, axis=2)
        avg_neighbor_agreement = np.mean(neighbor_dots, axis=1)
        
        # If most neighbors disagree (avg < 0), flip
        flip_mask_neighbors = avg_neighbor_agreement < -0.1
        
        # === STRATEGY 3: Combined voting ===
        # Flip if BOTH strategies suggest flipping (more conservative)
        # OR if eigenvalue planarity is low (ambiguous cases)
        planarity = (eigenvalues[:, 1] - eigenvalues[:, 0]) / (eigenvalues[:, 2] + eps)
        low_confidence = planarity < 0.1
        
        # Combined flipping logic
        flip_mask = flip_mask_centroid.copy()
        flip_mask = flip_mask | (flip_mask_neighbors & low_confidence)
        
        estimated_normals[flip_mask] = -estimated_normals[flip_mask]
        
        # === FINAL VALIDATION ===
        # One more pass to ensure no degenerates slipped through
        final_lengths = np.linalg.norm(estimated_normals, axis=1)
        invalid_mask = (final_lengths < 0.9) | (final_lengths > 1.1) | np.isnan(estimated_normals).any(axis=1)
        if invalid_mask.any():
            print(f"⚠️  Correcting {invalid_mask.sum()} invalid normals in final pass")
            estimated_normals[invalid_mask] = np.array([0.0, 1.0, 0.0])
            norm_lengths = np.linalg.norm(estimated_normals, axis=1, keepdims=True)
            estimated_normals = estimated_normals / (norm_lengths + eps)
        
        # === DIAGNOSTICS ===
        flipped_pct = (flip_mask.sum() / len(flip_mask)) * 100
        degenerate_pct = (degenerate_mask.sum() / len(degenerate_mask)) * 100
        print(f"✓ Normals computed: {flipped_pct:.1f}% flipped, {degenerate_pct:.2f}% degenerate")
        
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
    def _ggx_specular(normals, light_dir, view_dir, roughness, specular_strength):
        """
        GGX/Cook-Torrance microfacet BRDF for more realistic specular highlights.
        Based on physically-based rendering (PBR) principles.
        """
        eps = 1e-8
        
        # Half vector
        half_vector = light_dir + view_dir
        half_vector = half_vector / (np.linalg.norm(half_vector) + eps)
        
        # Dot products
        n_dot_h = np.maximum(0, np.dot(normals, half_vector))
        n_dot_v = np.maximum(0, np.dot(normals, view_dir))
        n_dot_l = np.maximum(0, np.dot(normals, light_dir))
        
        # GGX Normal Distribution Function (D)
        alpha = roughness * roughness
        alpha2 = alpha * alpha
        denom = (n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0)
        D = alpha2 / (np.pi * denom * denom + eps)
        
        # Geometry term (simplified Schlick-GGX)
        k = alpha / 2.0
        G1_v = n_dot_v / (n_dot_v * (1.0 - k) + k + eps)
        G1_l = n_dot_l / (n_dot_l * (1.0 - k) + k + eps)
        G = G1_v * G1_l
        
        # Fresnel (simplified Schlick approximation, assume F0 = 0.04 for dielectrics)
        F0 = 0.04
        F = F0 + (1.0 - F0) * np.power(1.0 - n_dot_h, 5.0)
        
        # Cook-Torrance specular BRDF
        specular = (D * G * F) / (4.0 * n_dot_v * n_dot_l + eps)
        specular = specular * specular_strength
        
        return specular
    
    @staticmethod
    def relight(data, azimuth, elevation, intensity, ambient, temp, cluster_mask=-1, view_mode="Final", 
                specular_strength=0.0, shininess=30.0, two_sided=True, 
                roughness=0.5, sky_color=None, ground_color=None, wrap_lighting=0.0, shadow_strength=0.0):
        """
        Enhanced relighting with GGX BRDF, hemisphere lighting, and wrap lighting.
        Fixes dark spot artifacts and provides physically-based rendering quality.
        
        New parameters:
        - roughness: Surface roughness for GGX BRDF (0=smooth, 1=rough)
        - sky_color: RGB tuple for hemisphere sky color (None = use ambient)
        - ground_color: RGB tuple for hemisphere ground color (None = use ambient)
        - wrap_lighting: Wrap lighting amount for two-sided mode (0-1)
        - shadow_strength: Auto-shadowing strength (0-1)
        """
        eps = 1e-8
        normals = data['normals']
        colors_sh = data['colors_sh']
        labels = data.get('labels', None)
        positions = data.get('positions', None)
        
        # Calculate Sun Direction
        sun_dir = LightForgeEngine.spherical_to_cartesian(azimuth, elevation)
        
        # === ROBUST NORMAL VALIDATION ===
        norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        norm_lengths = np.where(norm_lengths < eps, 1.0, norm_lengths)
        normals_norm = normals / norm_lengths
        
        # Detect invalid normals
        invalid_mask = (norm_lengths.squeeze() < 0.5) | np.isnan(normals_norm).any(axis=1)
        if invalid_mask.any():
            print(f"⚠️  Found {invalid_mask.sum()} invalid normals during relighting, using up vector")
            normals_norm[invalid_mask] = np.array([0.0, 1.0, 0.0])
        
        # --- DIFFUSE (Lambertian with optional wrap) ---
        dot = np.dot(normals_norm, sun_dir)
        
        if two_sided:
            if wrap_lighting > 0.01:
                # Wrap lighting: smoother transition around edges
                diffuse = ((dot + wrap_lighting) / (1.0 + wrap_lighting))
                diffuse = np.maximum(0, diffuse)
            else:
                # Traditional two-sided with MINIMUM to prevent dark spots
                # Instead of 0.5 + 0.5 * dot (range 0-1), use 0.6 + 0.4 * dot (range 0.2-1.0)
                # This ensures even back-facing normals get at least 20% lighting
                diffuse = 0.6 + 0.4 * dot
        else:
            # Traditional one-sided lighting
            diffuse = np.maximum(0, dot)
        
        # --- SPECULAR (GGX or Blinn-Phong) ---
        view_dir = np.array([0.0, 0.0, 1.0])
        
        if specular_strength > 0.01:
            if roughness < 0.99:  # Use GGX for realistic materials
                specular = LightForgeEngine._ggx_specular(
                    normals_norm, sun_dir, view_dir, roughness, specular_strength
                )
            else:  # Fall back to Blinn-Phong for very rough surfaces
                half_vector = (sun_dir + view_dir)
                half_vector = half_vector / (np.linalg.norm(half_vector) + eps)
                spec_angle = np.dot(normals_norm, half_vector)
                specular = np.power(np.maximum(0, spec_angle), shininess) * specular_strength
        else:
            specular = np.zeros(len(normals_norm))
        
        # --- HEMISPHERE LIGHTING (GI Approximation) ---
        hemisphere_contribution = np.zeros((len(normals_norm), 3))
        
        if sky_color is not None or ground_color is not None:
            # Default colors if not provided
            if sky_color is None:
                sky_color = np.array([0.53, 0.81, 0.92])  # Light blue sky
            if ground_color is None:
                ground_color = np.array([0.24, 0.15, 0.09])  # Dark brown ground
            
            # Blend between sky and ground based on normal Y component
            normal_y = normals_norm[:, 1]  # Y component (-1 to 1)
            sky_factor = (normal_y + 1.0) * 0.5  # Convert to (0 to 1)
            sky_factor = np.clip(sky_factor, 0, 1).reshape(-1, 1)
            
            hemisphere_contribution = (sky_factor * sky_color + (1.0 - sky_factor) * ground_color) * ambient
        
        # --- AUTO-SHADOWING (Heuristic) ---
        shadow_factor = 1.0
        if shadow_strength > 0.01:
            # Areas facing away from light get more shadow
            shadow_from_orientation = np.clip(1.0 - dot, 0, 1)
            
            # Optional: Add height-based shadowing if positions available
            if positions is not None:
                # Normalize Y position to 0-1 range
                y_positions = positions[:, 1]
                y_min, y_max = y_positions.min(), y_positions.max()
                if y_max > y_min:
                    y_normalized = (y_positions - y_min) / (y_max - y_min + eps)
                    shadow_from_height = 1.0 - y_normalized * 0.3  # Lower areas get 30% more shadow
                else:
                    shadow_from_height = 1.0
            else:
                shadow_from_height = 1.0
            
            shadow_factor = 1.0 - (shadow_from_orientation * shadow_strength * 0.5 * shadow_from_height)
            shadow_factor = np.clip(shadow_factor, 0.1, 1.0)  # Never completely black
        
        # --- COMBINE LIGHTING ---
        # Minimum lighting floor to prevent completely black regions
        # Increased from 0.15 to 0.3 for more aggressive dark spot prevention
        min_lighting = max(ambient * 0.3, 0.15)  # At least 30% of ambient, minimum 0.15
        
        # Direct lighting (ambient + diffuse)
        direct_lighting = ambient + (1.0 - ambient) * intensity * diffuse * shadow_factor
        direct_lighting = np.clip(direct_lighting, min_lighting, 5.0).reshape(-1, 1)
        
        # Specular component
        specular_component = specular.reshape(-1, 1) * intensity
        
        # Convert SH to RGB
        rgb = LightForgeEngine.sh_to_rgb(colors_sh)
        
        # Combine: (Base * Direct) + Specular + Hemisphere
        if hemisphere_contribution.any():
            rgb_lit = (rgb * direct_lighting) + specular_component + hemisphere_contribution
        else:
            rgb_lit = (rgb * direct_lighting) + specular_component
        
        # --- TEMPERATURE (Color Tint) ---
        if abs(temp) > 0.01:
            if temp > 0:  # Warm
                rgb_lit[:, 0] *= (1 + temp * 0.2)
                rgb_lit[:, 1] *= (1 + temp * 0.1)
                rgb_lit[:, 2] *= (1 - temp * 0.15)
            else:  # Cool
                rgb_lit[:, 0] *= (1 + temp * 0.15)
                rgb_lit[:, 1] *= (1 + temp * 0.1)
                rgb_lit[:, 2] *= (1 - temp * 0.2)
        
        # --- VIEW MODES ---
        if view_mode == "Normals":
            rgb_vis = (normals_norm + 1.0) * 0.5
            new_sh_all = LightForgeEngine.rgb_to_sh(rgb_vis)
        
        elif view_mode == "Light Map":
            fact_vis = direct_lighting / max(intensity + 0.001, 1.0) 
            fact_vis = np.clip(fact_vis, 0, 1)
            rgb_vis = np.hstack([fact_vis, fact_vis, fact_vis])
            new_sh_all = LightForgeEngine.rgb_to_sh(rgb_vis)
        else:
            # Final - clip to valid range
            rgb_lit = np.clip(rgb_lit, 0, 1)
            new_sh_all = LightForgeEngine.rgb_to_sh(rgb_lit)
        
        # --- CLUSTER MASKING ---
        if cluster_mask >= 0 and labels is not None:
             print(f"Applying relighting ONLY to Cluster {cluster_mask}")
             mask = (labels == cluster_mask)
             final_sh = colors_sh.copy()
             final_sh[mask] = new_sh_all[mask]
        else:
             final_sh = new_sh_all
            
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
        
        params_vertex_data = original_vertex_data.copy()
        params_vertex_data['f_dc_0'] = new_sh[:, 0]
        params_vertex_data['f_dc_1'] = new_sh[:, 1]
        params_vertex_data['f_dc_2'] = new_sh[:, 2]
        
        vertex_element = PlyElement.describe(params_vertex_data, 'vertex')
        PlyData([vertex_element], text=False).write(output_path)
        return output_path

    @staticmethod
    def generate_preview(azimuth, elevation, intensity, ambient, temperature, specular_strength=0.0, shininess=30.0, size=512):
        """
        Generates a preview image of a sphere with the applied lighting (PBR).
        """
        x = np.linspace(-1, 1, size)
        y = np.linspace(1, -1, size)
        xv, yv = np.meshgrid(x, y)
        r2 = xv**2 + yv**2
        mask = r2 <= 1.0
        
        z = np.zeros_like(xv)
        z[mask] = np.sqrt(1.0 - r2[mask])
        
        normals = np.zeros((size, size, 3))
        normals[:, :, 0] = xv
        normals[:, :, 1] = yv
        normals[:, :, 2] = z
        
        sun_dir = LightForgeEngine.spherical_to_cartesian(azimuth, elevation)
        
        n_flat = normals.reshape(-1, 3)
        
        # Diffuse
        dot = np.dot(n_flat, sun_dir)
        diffuse = np.maximum(0, dot)
        lighting_factor = ambient + (1.0 - ambient) * intensity * diffuse
        
        # Specular
        view_dir = np.array([0.0, 0.0, 1.0])
        half_vector = (sun_dir + view_dir) / np.linalg.norm(sun_dir + view_dir)
        spec_angle = np.dot(n_flat, half_vector)
        specular = np.power(np.maximum(0, spec_angle), shininess) * specular_strength * intensity
        
        lighting_factor = lighting_factor.reshape(size, size, 1)
        specular = specular.reshape(size, size, 1)
        
        # Combine
        rgb = (np.ones((size, size, 3)) * lighting_factor) + specular
        
        # Temperature
        if abs(temperature) > 0.01:
            if temperature > 0:  # Warm
                rgb[:, :, 0] *= (1 + temperature * 0.2)
                rgb[:, :, 1] *= (1 + temperature * 0.1)
                rgb[:, :, 2] *= (1 - temperature * 0.15)
            else:  # Cool
                rgb[:, :, 0] *= (1 + temperature * 0.15)
                rgb[:, :, 1] *= (1 + temperature * 0.1)
                rgb[:, :, 2] *= (1 - temperature * 0.2)
                
        rgb[~mask] = 0.0
        rgb = np.clip(rgb, 0, 1)
        
        return rgb
