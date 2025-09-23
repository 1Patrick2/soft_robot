import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import time
import multiprocessing
from tqdm import tqdm
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.read_config import load_config
from src.cosserat.solver import solve_static_equilibrium
from src.cosserat.kinematics import forward_kinematics, discretize_robot

# --- Worker Functions for Parallel Processing ---

def montecarlo_worker(task_args):
    """Worker function for parallel Monte Carlo sampling."""
    delta_l_sample, params, solver_opts, kappas_guess = task_args
    solve_result = solve_static_equilibrium(kappas_guess, delta_l_sample, params, solver_options=solver_opts)
    kappas_eq = solve_result["kappas_solution"]
    if kappas_eq is not None:
        T_final, _ = forward_kinematics(kappas_eq, params)
        return {"pos": T_final[:3, 3], "delta_l": delta_l_sample, "kappas_eq": kappas_eq}
    return None

def trace_single_path(task_args):
    """
    Worker function for parallel processing. Traces a single path from a start point to center.
    """
    path_id, delta_l_start, params, solver_opts, num_path_steps = task_args
    
    _, (n_pss, n_cms1, n_cms2) = discretize_robot(params)
    num_elements = n_pss + n_cms1 + n_cms2

    path = np.linspace(delta_l_start, np.zeros(8), num_path_steps)
    q_guess = np.zeros((3, num_elements))  # Reset guess for each new path
    points_on_path = []

    for delta_l_step in path:
        solve_result = solve_static_equilibrium(
            q_guess, delta_l_step, params, solver_options=solver_opts
        )
        kappas_eq = solve_result["kappas_solution"]

        if kappas_eq is not None:
            q_guess = kappas_eq  # Warm start for the next step
            T_final, _ = forward_kinematics(kappas_eq, params)
            points_on_path.append({
                "pos": T_final[:3, 3],
                "delta_l": delta_l_step,
                "kappas_eq": kappas_eq,
            })
        else:
            # If a step fails, stop tracing this path
            tqdm.write(f"Path {path_id} failed to converge, stopping trace for this path.")
            break
    return points_on_path

# --- Analysis Main Functions ---

def run_workspace_analysis_random_homotopy(num_paths=100, num_steps_per_path=5):
    """Generates a filled workspace using parallelized random short homotopy paths."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'config', 'config.json')
    params = load_config(config_path)

    delta_l_bounds = params.get('Bounds', {}).get('delta_l_bounds', [-0.05, 0.05])
    low, high = delta_l_bounds

    # --- Generate random starting actuations ---
    starting_actuations = [np.random.uniform(low, high, size=8) for _ in range(num_paths)]

    tasks = []
    solver_opts = {'ftol': 1e-4, 'gtol': 1e-4, 'maxiter': 500}
    for i, delta_l_start in enumerate(starting_actuations):
        tasks.append((i, delta_l_start, params, solver_opts, num_steps_per_path))

    all_reachable_points_data = []
    num_cpus = multiprocessing.cpu_count()
    print(f"[Random Homotopy] Starting path tracing with {len(tasks)} paths on {max(1, num_cpus - 1)} processes.")

    with multiprocessing.Pool(processes=max(1, num_cpus - 1)) as pool:
        results_iterator = pool.imap_unordered(trace_single_path, tasks)
        for path_points in tqdm(results_iterator, total=len(tasks), desc="Tracing Random Homotopy Paths"):
            all_reachable_points_data.extend(path_points)

    print("\n--- Analysis Complete ---")
    print(f"Generated {len(all_reachable_points_data)} points from {len(tasks)} paths.")
    return all_reachable_points_data

def run_workspace_analysis_montecarlo(num_samples=1000):
    """Generates a filled workspace using parallelized Monte Carlo sampling."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'config', 'config.json')
    params = load_config(config_path)

    delta_l_bounds = params.get('Bounds', {}).get('delta_l_bounds', [-0.05, 0.05])
    low, high = delta_l_bounds

    solver_opts = {'ftol': 1e-4, 'gtol': 1e-4, 'maxiter': 500}
    
    _, (n_pss, n_cms1, n_cms2) = discretize_robot(params)
    kappas_guess = np.zeros((3, n_pss + n_cms1 + n_cms2))

    delta_l_samples = [np.random.uniform(low, high, size=8) for _ in range(num_samples)]
    tasks = [(sample, params, solver_opts, kappas_guess) for sample in delta_l_samples]

    all_points = []
    num_cpus = multiprocessing.cpu_count()
    print(f"[Monte Carlo] Starting parallel sampling with {num_samples} samples on {max(1, num_cpus - 1)} processes.")

    with multiprocessing.Pool(processes=max(1, num_cpus - 1)) as pool:
        results_iterator = pool.imap_unordered(montecarlo_worker, tasks)
        for point_data in tqdm(results_iterator, total=len(tasks), desc="Running Monte Carlo Sampling"):
            if point_data is not None:
                all_points.append(point_data)
    
    return all_points

def run_workspace_analysis_homotopy(num_paths_per_set=4, num_steps_per_path=20):
    """Generates a shell workspace by tracing homotopy paths from fixed boundary configurations."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'config', 'config.json')
    params = load_config(config_path)

    if 'Cosserat' not in params:
        params['Cosserat'] = {'num_elements_pss': 10, 'num_elements_cms1': 5, 'num_elements_cms2': 5}

    delta_l_bounds = params.get('Bounds', {}).get('delta_l_bounds', [-0.05, 0.05])
    max_actuation = delta_l_bounds[1]
    m = max_actuation

    boundary_actuations = []
    for i in range(8):
        vec = np.zeros(8)
        vec[i] = m
        boundary_actuations.append(vec)
        vec[i] = -m
        boundary_actuations.append(vec)

    tasks = []
    solver_opts = {'ftol': 1e-4, 'gtol': 1e-4, 'maxiter': 500}
    for i, delta_l_boundary in enumerate(boundary_actuations):
        tasks.append((i, np.array(delta_l_boundary), params, solver_opts, num_steps_per_path))

    all_reachable_points_data = []
    num_cpus = multiprocessing.cpu_count()
    print(f"[Homotopy] Starting path tracing with {len(tasks)} paths on {max(1, num_cpus - 1)} processes.")

    with multiprocessing.Pool(processes=max(1, num_cpus - 1)) as pool:
        results_iterator = pool.imap_unordered(trace_single_path, tasks)
        for path_points in tqdm(results_iterator, total=len(tasks), desc="Tracing Homotopy Paths"):
            all_reachable_points_data.extend(path_points)

    print("\n--- Analysis Complete ---")
    print(f"Generated {len(all_reachable_points_data)} points from {len(tasks)} paths.")
    return all_reachable_points_data

def plot_workspace_3_view(points, output_filename, title, slice_width=0.01):
    if points is None or len(points) < 1:
        print("Warning: Not enough points to plot workspace.")
        return

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_filename = os.path.join(project_root, output_filename)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    print(f"\nPlotting workspace 3-view... Saving to '{output_filename}'")
    fig = plt.figure(figsize=(14, 21))
    fig.suptitle(title, fontsize=16)

    ws_x, ws_y, ws_z = points[:, 0], points[:, 1], points[:, 2]

    ax3d = fig.add_subplot(3, 2, 1, projection='3d')
    ax3d.scatter(ws_x, ws_y, ws_z, c='b', marker='.', s=5, alpha=0.4, label='Workspace')
    ax3d.set_title('3D View')
    ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
    ax3d.legend(); ax3d.axis('equal')

    ax_xy = fig.add_subplot(3, 2, 2); ax_xy.scatter(ws_x, ws_y, c='b', marker='.', s=5, alpha=0.4); ax_xy.set_title('XY Plane'); ax_xy.set_xlabel('X (m)'); ax_xy.set_ylabel('Y (m)'); ax_xy.grid(True); ax_xy.axis('equal')
    ax_xz = fig.add_subplot(3, 2, 3); ax_xz.scatter(ws_x, ws_z, c='b', marker='.', s=5, alpha=0.4); ax_xz.set_title('XZ Plane'); ax_xz.set_xlabel('X (m)'); ax_xz.set_ylabel('Z (m)'); ax_xz.grid(True); ax_xz.axis('equal')
    ax_yz = fig.add_subplot(3, 2, 4); ax_yz.scatter(ws_y, ws_z, c='b', marker='.', s=5, alpha=0.4); ax_yz.set_title('YZ Plane'); ax_yz.set_xlabel('Y (m)'); ax_yz.set_ylabel('Z (m)'); ax_yz.grid(True); ax_yz.axis('equal')
    ax_slice_y0 = fig.add_subplot(3, 2, 5); slice_y0_points = points[np.abs(points[:, 1]) < slice_width]; ax_slice_y0.scatter(slice_y0_points[:, 0], slice_y0_points[:, 2], c='m', marker='.', s=10, alpha=0.6); ax_slice_y0.set_title(f'Y=0 Cross-Section'); ax_slice_y0.set_xlabel('X (m)'); ax_slice_y0.set_ylabel('Z (m)'); ax_slice_y0.grid(True); ax_slice_y0.axis('equal')
    ax_slice_x0 = fig.add_subplot(3, 2, 6); slice_x0_points = points[np.abs(points[:, 0]) < slice_width]; ax_slice_x0.scatter(slice_x0_points[:, 1], slice_x0_points[:, 2], c='m', marker='.', s=10, alpha=0.6); ax_slice_x0.set_title(f'X=0 Cross-Section'); ax_slice_x0.set_xlabel('Y (m)'); ax_slice_x0.set_ylabel('Z (m)'); ax_slice_x0.grid(True); ax_slice_x0.axis('equal')

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
    
    # --- Mode Selection ---
    # mode = "homotopy"          # Generates the shell from fixed boundaries
    # mode = "montecarlo"        # Fills the interior with random cold starts (slow)
    mode = "random_homotopy"   # Fills the interior with random warm-start paths (fast & dense)

    if mode == "random_homotopy":
        num_paths = 100
        num_steps = 5
        workspace_full_data = run_workspace_analysis_random_homotopy(num_paths=num_paths, num_steps_per_path=num_steps)
        num_points = len(workspace_full_data)
        filename_suffix = f"random_homotopy_{num_points}"
        title = f"Workspace Analysis (Random Homotopy: {num_paths} paths, {num_steps} steps)"
    elif mode == "montecarlo":
        num_samples = 5000
        workspace_full_data = run_workspace_analysis_montecarlo(num_samples=num_samples)
        num_points = len(workspace_full_data)
        filename_suffix = f"montecarlo_{num_points}"
        title = f"Workspace Analysis (Monte Carlo: {num_samples} samples)"
    else: # homotopy
        num_steps = 20
        workspace_full_data = run_workspace_analysis_homotopy(num_steps_per_path=num_steps)
        num_points = len(workspace_full_data)
        filename_suffix = f"homotopy_{num_points}"
        title = f"Workspace Analysis (Homotopy: 20 paths, {num_steps} steps)"

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_filename = os.path.join(project_root, 'plots', f'workspace_points_{filename_suffix}.npy')
    output_plot_filename = f'plots/workspace_plot_{filename_suffix}.png'
    os.makedirs(os.path.dirname(data_filename), exist_ok=True)
    
    if workspace_full_data:
        save_data = np.array([d for d in workspace_full_data], dtype=object)
        np.save(data_filename, save_data, allow_pickle=True)
        print(f"Workspace data saved to {os.path.abspath(data_filename)}")
        
        workspace_points = np.array([d['pos'] for d in workspace_full_data])
        plot_workspace_3_view(workspace_points, output_filename=output_plot_filename, title=title)
    else:
        print("No reachable points found to plot.")

    end_time = time.time() 
    print(f"\nTotal analysis and plotting time: {end_time - start_time:.2f} seconds")
