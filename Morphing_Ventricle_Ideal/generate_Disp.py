import numpy as np
import trimesh
from scipy.interpolate import Rbf
import os

# === USER SETTINGS ===
stl_folder = 'STL'                     # Folder with STL files
boundary_points_file = 'boundaryPoints.xyz'  # Patch points XYZ file
mesh_points_file = 'points'           # OpenFOAM full mesh points file (constant/polyMesh/points)
output_folder = 'pointDisplacementFiles'  # Where to write displacement folders
num_timesteps = 500                   # Number of STL frames (excluding 000)
patch_name = 'WALL'                   # OpenFOAM patch name to morph
timestep_interval = 0.002             # Output time interval in seconds

# === LOAD FULL MESH POINTS ===
def load_all_mesh_points(mesh_points_file):
    with open(mesh_points_file, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().isdigit():
            num_points = int(line.strip())
            start_idx = i + 2  # skip '(', then list starts
            break
    mesh_all_points = []
    for i in range(start_idx, start_idx + num_points):
        line = lines[i].strip().strip('()')
        coords = list(map(float, line.split()))
        mesh_all_points.append(coords)
    return np.array(mesh_all_points)

# Load mesh points
boundary_points = np.loadtxt(boundary_points_file)
all_points = load_all_mesh_points(mesh_points_file)

print(f'Loaded {len(boundary_points)} boundary points and {len(all_points)} total mesh points.')

# Load reference STL mesh
ref_stl_path = os.path.join(stl_folder, 'ventricle_000.stl')
mesh_ref = trimesh.load(ref_stl_path)
stl_ref_pts = mesh_ref.vertices

# Project boundary points onto STL for better RBF accuracy
print('Projecting boundary points onto STL reference...')
closest_boundary_pts, _, _ = trimesh.proximity.closest_point(mesh_ref, boundary_points)
print('Projection done.')

# Loop through all deformation steps and save every timestep_interval
for step_idx in range(1, num_timesteps + 1):
    i = step_idx  # STL frame index
    
    # Calculate time and round to avoid floating point artifacts
    time_val = round(step_idx * timestep_interval, 6)  # round to 6 decimal places
    # Convert to string and strip trailing zeros and '.' if any
    time_str = f'{time_val:.6f}'.rstrip('0').rstrip('.')

    stl_def_path = os.path.join(stl_folder, f'ventricle_{i:03d}.stl')
    if not os.path.exists(stl_def_path):
        print(f'Warning: {stl_def_path} not found, skipping.')
        continue

    print(f'[{step_idx}] Loading STL frame {i}...')
    mesh_def = trimesh.load(stl_def_path)
    stl_disp = mesh_def.vertices - stl_ref_pts

    print('Building RBF interpolators...')
    rbf_x = Rbf(stl_ref_pts[:, 0], stl_ref_pts[:, 1], stl_ref_pts[:, 2], stl_disp[:, 0], function='thin_plate')
    rbf_y = Rbf(stl_ref_pts[:, 0], stl_ref_pts[:, 1], stl_ref_pts[:, 2], stl_disp[:, 1], function='thin_plate')
    rbf_z = Rbf(stl_ref_pts[:, 0], stl_ref_pts[:, 1], stl_ref_pts[:, 2], stl_disp[:, 2], function='thin_plate')

    print('Interpolating displacement for full mesh...')
    disp_x = rbf_x(all_points[:, 0], all_points[:, 1], all_points[:, 2])
    disp_y = rbf_y(all_points[:, 0], all_points[:, 1], all_points[:, 2])
    disp_z = rbf_z(all_points[:, 0], all_points[:, 1], all_points[:, 2])
    full_displacement = np.vstack((disp_x, disp_y, disp_z)).T

    out_dir = os.path.join(output_folder)
    os.makedirs(out_dir, exist_ok=True)
    
    # NEW NAMING SCHEME: use integer frame index
    frame_index = step_idx
    fname = os.path.join(out_dir, f"pointDisplacement_{frame_index}")

    print(f'Writing pointDisplacement_{frame_index}...')
    with open(fname, 'w') as f:
        f.write(f'{len(full_displacement)}\n(\n')
        for d in full_displacement:
            f.write(f'({d[0]:.8f} {d[1]:.8f} {d[2]:.8f})\n')
        f.write(');\n\n')

    print(f'[{step_idx}] pointDisplacement written as frame {frame_index}.\n')

print('All time steps processed.')
