# workspace_improved.py (Corrected with full Cosserat model alignment)
import numpy as np
import os, time, multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path if needed
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.read_config import load_config
# Correctly import from the 'cosserat' subdirectory
from src.cosserat.solver import solve_static_equilibrium
from src.cosserat.kinematics import forward_kinematics, discretize_robot

# ---------------- Worker Function for path tracing in 8D delta_l space ----------------
def trace_single_path_adaptive(task_args):
    path_id, delta_l_start, params, _, num_path_steps = task_args
    element_lengths, (n_pss, n_cms1, n_cms2) = discretize_robot(params)
    num_elements = len(element_lengths)

    dl_start = np.array(delta_l_start)
    q_guess = np.zeros((3, num_elements))
    points_on_path = []

    # solver options
    opts_rough = {'ftol':1e-4,'gtol':1e-4,'maxiter':500}
    opts_final = {'ftol':1e-9,'gtol':1e-8,'maxiter':2000}

    t_curr = 0.0
    t_target = 1.0
    # Start with a reasonably small step, the algorithm will increase it if possible
    t_step = 1.0 / (num_path_steps * 2) # Start with smaller steps
    prev_kappa_norm = np.linalg.norm(q_guess)
    successful_steps = 0

    pbar = tqdm(total=100, desc=f"Path {path_id}", leave=False)
    while t_curr < t_target - 1e-9:
        pbar.n = int(t_curr * 100)
        pbar.refresh()

        t_next = min(t_curr + t_step, t_target)
        dl_step = (1.0 - t_next) * dl_start

        res_coarse = solve_static_equilibrium(q_guess, dl_step, params, solver_options=opts_rough)
        k_coarse = res_coarse.get('kappas_solution')

        if k_coarse is None:
            t_step *= 0.5 # Halve step size
            if t_step < 1e-5:
                tqdm.write(f"Path {path_id}: Step size too small, giving up.")
                break
            continue

        # Acceptance test
        kappa_norm = np.linalg.norm(k_coarse)
        if prev_kappa_norm > 1e-6 and (kappa_norm / prev_kappa_norm > 5.0):
            t_step *= 0.5 # Halve step size on large kappa jump
            if t_step < 1e-5:
                tqdm.write(f"Path {path_id}: Step size too small after kappa jump, giving up.")
                break
            continue

        # Refine if it's the last step
        if abs(t_next - t_target) < 1e-9:
            res_fine = solve_static_equilibrium(k_coarse, dl_step, params, solver_options=opts_final)
            if res_fine.get('kappas_solution') is not None:
                k_coarse = res_fine['kappas_solution']

        # Accept step
        q_guess = k_coarse
        prev_kappa_norm = kappa_norm
        T, _ = forward_kinematics(q_guess, params)
        points_on_path.append({"pos": T[:3, 3].copy(), "delta_l": dl_step.copy(), "kappa_norm": prev_kappa_norm})
        
        t_curr = t_next
        successful_steps += 1
        # Increase step size if things are going well
        if successful_steps > 2:
            t_step = min(t_step * 1.5, 1.0 / num_path_steps) 
            successful_steps = 0
            
    pbar.n = 100
    pbar.refresh()
    pbar.close()
    return points_on_path

# ---------------- Boundary Generation (from fixed 8D actuations) ----------------
def run_workspace_analysis_homotopy(params, num_steps_per_path=20):
    delta_l_bounds = params.get('Bounds', {}).get('delta_l_bounds', [-0.12, 0.12])
    max_actuation = delta_l_bounds[1]
    
    # Define 8 boundary actuations (pulling each cable individually)
    boundary_actuations = []
    for i in range(8):
        vec = np.zeros(8)
        vec[i] = max_actuation
        boundary_actuations.append(vec)

    tasks = []
    for i, delta_l_boundary in enumerate(boundary_actuations):
        tasks.append((i, np.array(delta_l_boundary), params, None, num_steps_per_path))

    all_reachable_points_data = []
    num_cpus = multiprocessing.cpu_count()
    print(f"\n[Boundary Gen] Starting path tracing with {len(tasks)} paths on {max(1, num_cpus - 1)} processes.")

    with multiprocessing.Pool(processes=max(1, num_cpus - 1)) as pool:
        results_iterator = pool.imap_unordered(trace_single_path_adaptive, tasks)
        for path_points in tqdm(results_iterator, total=len(tasks), desc="Tracing Boundary Paths"):
            all_reachable_points_data.extend(path_points)
    return all_reachable_points_data

# ---------------- Interior Generation (from random 8D actuations) ----------------
def run_workspace_analysis_random_homotopy(params, num_paths=200, num_steps_per_path=10):
    delta_l_bounds = params.get('Bounds', {}).get('delta_l_bounds', [-0.12, 0.12])
    low, high = delta_l_bounds

    starting_actuations = [np.random.uniform(low, high, size=8) for _ in range(num_paths)]

    tasks = []
    for i, delta_l_start in enumerate(starting_actuations):
        tasks.append((i, delta_l_start, params, None, num_steps_per_path))

    all_reachable_points_data = []
    num_cpus = multiprocessing.cpu_count()
    print(f"\n[Interior Gen] Starting path tracing with {len(tasks)} paths on {max(1, num_cpus - 1)} processes.")

    with multiprocessing.Pool(processes=max(1, num_cpus - 1)) as pool:
        results_iterator = pool.imap_unordered(trace_single_path_adaptive, tasks)
        for path_points in tqdm(results_iterator, total=len(tasks), desc="Tracing Interior Paths"):
            all_reachable_points_data.extend(path_points)
    return all_reachable_points_data

# ---------------- plotting util ----------------
def plot_workspace_3_view(points, types, output_filename, title, slice_width=0.01):
    if points is None or len(points) < 1:
        print("Warning: Not enough points to plot workspace.")
        return

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_filename = os.path.join(project_root, output_filename)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    print(f"\nPlotting workspace 3-view (boundary+interior)... Saving to '{output_filename}'")
    fig = plt.figure(figsize=(14, 21))
    fig.suptitle(title, fontsize=16)

    ws_x, ws_y, ws_z = points[:, 0], points[:, 1], points[:, 2]
    types = np.array(types)
    boundary_mask = types == "boundary"
    interior_mask = types == "interior"

    def scatter_split(ax, x, y, types_subset, xlabel, ylabel, title):
        boundary_mask_subset = types_subset == "boundary"
        interior_mask_subset = types_subset == "interior"
        ax.scatter(x[interior_mask_subset], y[interior_mask_subset], c='b', marker='.', s=5, alpha=0.3, label='Interior')
        ax.scatter(x[boundary_mask_subset], y[boundary_mask_subset], c='r', marker='.', s=8, alpha=0.8, label='Boundary')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.axis('equal')
        ax.legend()

    ax3d = fig.add_subplot(3, 2, 1, projection='3d')
    ax3d.scatter(ws_x[interior_mask], ws_y[interior_mask], ws_z[interior_mask], c='b', marker='.', s=5, alpha=0.3, label='Interior')
    ax3d.scatter(ws_x[boundary_mask], ws_y[boundary_mask], ws_z[boundary_mask], c='r', marker='.', s=8, alpha=0.8, label='Boundary')
    ax3d.set_title('3D View'); ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
    ax3d.axis('equal')
    ax3d.legend()

    scatter_split(fig.add_subplot(3, 2, 2), ws_x, ws_y, types, 'X (m)', 'Y (m)', 'XY Plane')
    scatter_split(fig.add_subplot(3, 2, 3), ws_x, ws_z, types, 'X (m)', 'Z (m)', 'XZ Plane')
    scatter_split(fig.add_subplot(3, 2, 4), ws_y, ws_z, types, 'Y (m)', 'Z (m)', 'YZ Plane')

    slice_y0_mask = np.abs(ws_y) < slice_width
    scatter_split(fig.add_subplot(3, 2, 5), ws_x[slice_y0_mask], ws_z[slice_y0_mask], types[slice_y0_mask], 'X (m)', 'Z (m)', 'Y=0 Cross-Section')

    slice_x0_mask = np.abs(ws_x) < slice_width
    scatter_split(fig.add_subplot(3, 2, 6), ws_y[slice_x0_mask], ws_z[slice_x0_mask], types[slice_x0_mask], 'Y (m)', 'Z (m)', 'X=0 Cross-Section')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename, dpi=200)
    print(f"3-view plot saved to {os.path.abspath(output_filename)}")
    plt.close(fig)

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    start_time = time.time()
    
    cfg = load_config('config/config.json')

    # --- Generate boundary (homotopy) and interior (random_homotopy) ---
    num_boundary_steps = 20
    boundary_data = run_workspace_analysis_homotopy(params=cfg, num_steps_per_path=num_boundary_steps)
    for d in boundary_data:
        d["type"] = "boundary"

    # Use a reasonable number of paths for a verification run
    num_paths = 200
    num_steps = 10
    interior_data = run_workspace_analysis_random_homotopy(params=cfg, num_paths=num_paths, num_steps_per_path=num_steps)
    for d in interior_data:
        d["type"] = "interior"

    workspace_full_data = boundary_data + interior_data
    
    if workspace_full_data:
        # Remove duplicates before saving and plotting
        positions = np.array([d['pos'] for d in workspace_full_data])
        _, unique_indices = np.unique(positions.round(decimals=4), axis=0, return_index=True)
        
        workspace_points = positions[unique_indices]
        workspace_types = [workspace_full_data[i]['type'] for i in unique_indices]
        
        num_points = len(workspace_points)
        print(f"\nFound {num_points} unique points out of {len(workspace_full_data)} total generated points.")

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filename_suffix = f"hybrid_phys_{num_points}"
        title = f"Workspace (Physics-Based): {len(boundary_data)} boundary, {len(interior_data) - (len(workspace_full_data) - num_points)} interior"
        
        data_filename = os.path.join(project_root, 'plots', f'workspace_points_{filename_suffix}.npy')
        output_plot_filename = os.path.join(project_root, 'plots', f'workspace_plot_{filename_suffix}.png')

        unique_data_to_save = [workspace_full_data[i] for i in unique_indices]
        np.save(data_filename, np.array(unique_data_to_save, dtype=object), allow_pickle=True)
        print(f"Workspace data saved to {os.path.abspath(data_filename)}")

        plot_workspace_3_view(workspace_points, workspace_types, output_filename=output_plot_filename, title=title)
    else:
        print("No reachable points were generated.")
        
    end_time = time.time()
    print(f"\nTotal analysis and plotting time: {end_time - start_time:.2f} seconds")