import numpy as np
import trimesh
from scipy.interpolate import Rbf
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# === USER SETTINGS ===
stl_folder = 'STL'
boundary_points_file = 'boundaryPoints.xyz'
mesh_points_file = 'points'
output_folder = 'pointDisplacementFiles'
num_timesteps = 500
patch_name = 'WALL'
timestep_interval = 0.002
n_workers = 4   # Number of parallel processes

# === LOAD DATA SHARED BY ALL FRAMES ===
def load_all_mesh_points(mesh_points_file):
    with open(mesh_points_file, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().isdigit():
            num_points = int(line.strip())
            start_idx = i + 2
            break
    mesh_all_points = []
    for i in range(start_idx, start_idx + num_points):
        line = lines[i].strip().strip('()')
        coords = list(map(float, line.split()))
        mesh_all_points.append(coords)
    return np.array(mesh_all_points)

boundary_points = np.loadtxt(boundary_points_file)
all_points = load_all_mesh_points(mesh_points_file)

ref_stl_path = os.path.join(stl_folder, 'ventricle_000.stl')
mesh_ref = trimesh.load(ref_stl_path)
stl_ref_pts = mesh_ref.vertices

# Project boundary points onto reference mesh
closest_boundary_pts, _, _ = trimesh.proximity.closest_point(mesh_ref, boundary_points)

# === FUNCTION TO PROCESS ONE FRAME ===
def process_frame(step_idx):
    import numpy as np
    import os
    import trimesh
    from scipy.interpolate import Rbf

    i = step_idx
    time_val = round(step_idx * timestep_interval, 6)
    time_str = f'{time_val:.6f}'.rstrip('0').rstrip('.')

    stl_def_path = os.path.join(stl_folder, f'ventricle_{i:03d}.stl')
    if not os.path.exists(stl_def_path):
        print(f'Warning: {stl_def_path} not found, skipping.')
        return

    print(f'[{step_idx}] Loading STL frame {i}...')
    mesh_def = trimesh.load(stl_def_path)
    stl_disp = mesh_def.vertices - stl_ref_pts

    print(f'[{step_idx}] Building RBF interpolators...')
    rbf_x = Rbf(stl_ref_pts[:,0], stl_ref_pts[:,1], stl_ref_pts[:,2], stl_disp[:,0], function='thin_plate')
    rbf_y = Rbf(stl_ref_pts[:,0], stl_ref_pts[:,1], stl_ref_pts[:,2], stl_disp[:,1], function='thin_plate')
    rbf_z = Rbf(stl_ref_pts[:,0], stl_ref_pts[:,1], stl_ref_pts[:,2], stl_disp[:,2], function='thin_plate')

    print(f'[{step_idx}] Interpolating displacement...')
    disp_x = rbf_x(all_points[:,0], all_points[:,1], all_points[:,2])
    disp_y = rbf_y(all_points[:,0], all_points[:,1], all_points[:,2])
    disp_z = rbf_z(all_points[:,0], all_points[:,1], all_points[:,2])
    full_displacement = np.vstack((disp_x, disp_y, disp_z)).T

    out_dir = os.path.join(output_folder)
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"pointDisplacement_{step_idx}")
    with open(fname, 'w') as f:
        f.write(f'{len(full_displacement)}\n(\n')
        for d in full_displacement:
            f.write(f'({d[0]:.8f} {d[1]:.8f} {d[2]:.8f})\n')
        f.write(');\n\n')

    print(f'[{step_idx}] Done.')
    return

# === RUN PARALLEL PROCESSING ===
if __name__ == "__main__":
    print(f"Processing {num_timesteps} frames in parallel with {n_workers} workers...")
    steps = list(range(1, num_timesteps + 1))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_frame, idx) for idx in steps]
        for future in as_completed(futures):
            # Optionally retrieve result or catch exceptions
            future.result()

    print("All frames processed.")
