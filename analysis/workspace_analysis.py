import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import time
import multiprocessing
from tqdm import tqdm
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.read_config import load_config
from src.solver import solve_static_equilibrium_diff4
from src.kinematics import forward_kinematics

def pretension_continuation(diff4_target, cfg):
    """
    Solves for the equilibrium configuration using a pretension continuation method.
    Starts with zero pretension and gradually ramps up to the target pretension,
    using the previous solution as a warm start.
    """
    q_guess = np.zeros(6)
    
    # Define a path for the pretension force
    target_pretension = cfg['Drive_Properties']['pretension_force_N']
    pretension_path = sorted(list(set([0.0, 0.1, 0.2, 0.3, 0.5, 1.0] + [target_pretension])))
    # Ensure the path does not exceed the target
    pretension_path = [p for p in pretension_path if p <= target_pretension]

    for p in pretension_path:
        cfg_local = deepcopy(cfg)
        cfg_local['Drive_Properties']['pretension_force_N'] = p
        res_dict = solve_static_equilibrium_diff4(q_guess, diff4_target, cfg_local)
        q_new = res_dict['q_solution']
        if q_new is None:
            return None  # Continuation failed
        q_guess = q_new
        
    # One final solve at the target pretension to be sure
    final_res_dict = solve_static_equilibrium_diff4(q_guess, diff4_target, cfg)
    return final_res_dict['q_solution']

def worker_solve_single_point(args):
    """
    Worker function for parallel processing. Solves for a single target control input.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'config', 'config.json')
    params = load_config(config_path)

    i, diff4_target = args
    
    # Use the new pretension continuation solver
    q_eq = pretension_continuation(diff4_target, params)
    
    if q_eq is not None and not np.any(np.isnan(q_eq)) and not np.any(np.isinf(q_eq)):
        try:
            # Use original params for final FK calculation
            T_final, _ = forward_kinematics(q_eq, params)
            return ("success", {
                "pos": T_final[:3, 3],
                "diff4": diff4_target,
                "q_eq": q_eq,
            })
        except Exception:
            return ("kin_fail", None)
    return ("solver_fail", None)

def run_workspace_analysis(num_samples=500, use_tqdm=True, parallel=True):
    """
    Runs the workspace analysis using Monte Carlo sampling and the pretension continuation solver.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'config', 'config.json')
    params = load_config(config_path)
    diff4_bounds = params.get('Bounds', {}).get('diff4_bounds', [-0.12, 0.12])

    tasks = []
    for i in range(num_samples):
        random_diff4 = np.random.uniform(diff4_bounds[0], diff4_bounds[1], size=4)
        tasks.append((i, random_diff4))
            
    reachable_points_data = []
    fail_counts = {'solver_fail': 0, 'kin_fail': 0}

    if parallel:
        num_cpus = multiprocessing.cpu_count()
        if use_tqdm:
            print(f"Detected {num_cpus} CPU cores. Using {max(1, num_cpus - 1)} processes.")
        with multiprocessing.Pool(processes=max(1, num_cpus - 1)) as pool:
            results_iterator = pool.imap_unordered(worker_solve_single_point, tasks)
            if use_tqdm:
                results_iterator = tqdm(results_iterator, total=num_samples, desc=f"Calculating Workspace ({num_samples} samples)")
            
            for status, result_dict in results_iterator:
                if status == "success":
                    reachable_points_data.append(result_dict)
                else:
                    fail_counts[status] += 1
    else:
        iterator = tasks
        if use_tqdm:
            iterator = tqdm(iterator, desc=f"Calculating Workspace ({num_samples} samples, Serial)")
        for task in iterator:
            status, result_dict = worker_solve_single_point(task)
            if status == "success":
                reachable_points_data.append(result_dict)
            else:
                fail_counts[status] += 1
    
    if use_tqdm:
        print("\n--- Analysis Complete ---")
        if num_samples > 0:
            success_rate = len(reachable_points_data) / num_samples * 100
            print(f"Success rate: {success_rate:.2f}% ({len(reachable_points_data)}/{num_samples})")
        print(f"Failure counts: Solver failed = {fail_counts['solver_fail']}, Kinematics failed = {fail_counts['kin_fail']}")

    return reachable_points_data

def plot_workspace_3_view(points, output_filename, slice_width=0.01):
    if points is None or len(points) < 1:
        print("Warning: Not enough points to plot workspace.")
        return

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_filename = os.path.join(project_root, output_filename)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    print(f"\nPlotting workspace 3-view... Saving to '{output_filename}'")
    fig = plt.figure(figsize=(14, 21))
    fig.suptitle('Workspace Analysis (Pretension Continuation)', fontsize=16)

    ws_x, ws_y, ws_z = points[:, 0], points[:, 1], points[:, 2]

    # ... (Plotting code is identical to before) ...
    ax3d = fig.add_subplot(3, 2, 1, projection='3d')
    ax3d.scatter(ws_x, ws_y, ws_z, c='c', marker='.', s=2, alpha=0.3, label='Workspace')
    ax3d.set_title('3D View')
    ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
    ax3d.legend(); ax3d.axis('equal')

    ax_xy = fig.add_subplot(3, 2, 2)
    ax_xy.scatter(ws_x, ws_y, c='c', marker='.', s=2, alpha=0.3)
    ax_xy.set_title('XY Plane Projection')
    ax_xy.set_xlabel('X (m)'); ax_xy.set_ylabel('Y (m)')
    ax_xy.grid(True); ax_xy.axis('equal')

    ax_xz = fig.add_subplot(3, 2, 3)
    ax_xz.scatter(ws_x, ws_z, c='c', marker='.', s=2, alpha=0.3)
    ax_xz.set_title('XZ Plane Projection')
    ax_xz.set_xlabel('X (m)'); ax_xz.set_ylabel('Z (m)')
    ax_xz.grid(True); ax_xz.axis('equal')

    ax_yz = fig.add_subplot(3, 2, 4)
    ax_yz.scatter(ws_y, ws_z, c='c', marker='.', s=2, alpha=0.3)
    ax_yz.set_title('YZ Plane Projection')
    ax_yz.set_xlabel('Y (m)'); ax_yz.set_ylabel('Z (m)')
    ax_yz.grid(True); ax_yz.axis('equal')

    ax_slice_y0 = fig.add_subplot(3, 2, 5)
    slice_y0_points = points[np.abs(points[:, 1]) < slice_width]
    if len(slice_y0_points) > 0:
        ax_slice_y0.scatter(slice_y0_points[:, 0], slice_y0_points[:, 2], c='m', marker='.', s=5, alpha=0.5)
    ax_slice_y0.set_title(f'Cross-Section at Y=0 (slice width {slice_width*2:.2f}m)')
    ax_slice_y0.set_xlabel('X (m)'); ax_slice_y0.set_ylabel('Z (m)')
    ax_slice_y0.grid(True); ax_slice_y0.axis('equal')

    ax_slice_x0 = fig.add_subplot(3, 2, 6)
    slice_x0_points = points[np.abs(points[:, 0]) < slice_width]
    if len(slice_x0_points) > 0:
        ax_slice_x0.scatter(slice_x0_points[:, 1], slice_x0_points[:, 2], c='m', marker='.', s=5, alpha=0.5)
    ax_slice_x0.set_title(f'Cross-Section at X=0 (slice width {slice_width*2:.2f}m)')
    ax_slice_x0.set_xlabel('Y (m)'); ax_slice_y0.set_ylabel('Z (m)')
    ax_slice_x0.grid(True); ax_slice_x0.axis('equal')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename, dpi=200)
    print(f"3-view plot saved to {os.path.abspath(output_filename)}")
    plt.close(fig)

if __name__ == '__main__':
    start_time = time.time()
    
    num_samples = 500
    print(f"--- Starting Workspace Analysis (Pretension Continuation, {num_samples} random samples) ---")
    
    workspace_full_data = run_workspace_analysis(num_samples=num_samples)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_filename = os.path.join(project_root, 'plots', f'workspace_points_pretension_continuation_{num_samples}.npy')
    output_plot_filename = f'plots/workspace_plot_pretension_continuation_{num_samples}.png'
    os.makedirs(os.path.dirname(data_filename), exist_ok=True)
    
    if workspace_full_data:
        save_data = np.array([d for d in workspace_full_data], dtype=object)
        np.save(data_filename, save_data, allow_pickle=True)
        print(f"Workspace data saved to {os.path.abspath(data_filename)}")
        
        workspace_points = np.array([d['pos'] for d in workspace_full_data])
        plot_workspace_3_view(workspace_points, output_filename=output_plot_filename)
    else:
        print("No reachable points found to plot.")

    end_time = time.time() 
    print(f"\nTotal analysis and plotting time: {end_time - start_time:.2f} seconds")
