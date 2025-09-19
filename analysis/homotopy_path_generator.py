import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import time
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.read_config import load_config
from src.cosserat.solver import solve_static_equilibrium
from src.cosserat.kinematics import forward_kinematics, discretize_robot

def plot_workspace_3_view(points, output_filename, slice_width=0.01):
    if points is None or len(points) < 1:
        print("Warning: Not enough points to plot workspace.")
        return

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_filename = os.path.join(project_root, output_filename)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    print(f"\nPlotting workspace 3-view... Saving to '{output_filename}'")
    fig = plt.figure(figsize=(14, 21))
    fig.suptitle('Workspace Analysis (Homotopy Path Generator)', fontsize=16)

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

def generate_homotopy_paths(num_path_steps=20):
    """
    Generates workspace points by tracing paths from the boundary to the center.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'config', 'config.json')
    params = load_config(config_path)

    if 'Cosserat' not in params:
        params['Cosserat'] = {'num_elements_pss': 10, 'num_elements_cms1': 5, 'num_elements_cms2': 5}

    _, (n_pss, n_cms1, n_cms2) = discretize_robot(params)
    num_elements = n_pss + n_cms1 + n_cms2

    delta_l_bounds = params.get('Bounds', {}).get('delta_l_bounds', [-0.05, 0.05])
    max_actuation = delta_l_bounds[1]

    # Define boundary actuation vectors (pulling each of the 8 cables individually)
    boundary_actuations = []
    for i in range(8):
        vec = np.zeros(8)
        vec[i] = max_actuation
        boundary_actuations.append(vec)

    all_reachable_points = []
    
    # Use the optimized solver options
    solver_opts = {
        'ftol': 1e-4,
        'gtol': 1e-4,
        'maxiter': 500
    }

    for i, delta_l_boundary in enumerate(tqdm(boundary_actuations, desc="Tracing Homotopy Paths")):
        path = np.linspace(delta_l_boundary, np.zeros(8), num_path_steps)
        q_guess = np.zeros((3, num_elements)) # Reset guess for each new path

        for delta_l_step in path:
            solve_result = solve_static_equilibrium(
                q_guess, delta_l_step, params, solver_options=solver_opts
            )
            kappas_eq = solve_result["kappas_solution"]

            if kappas_eq is not None:
                q_guess = kappas_eq  # Warm start for the next step
                T_final, _ = forward_kinematics(kappas_eq, params)
                all_reachable_points.append(T_final[:3, 3])
            else:
                # If a step fails, stop tracing this path
                tqdm.write(f"Path {i+1} failed to converge at step, moving to next path.")
                break
    
    return np.array(all_reachable_points)

if __name__ == '__main__':
    start_time = time.time()
    
    # Generate points using the homotopy method
    # 8 paths with 20 steps each = 160 points
    workspace_points = generate_homotopy_paths(num_path_steps=20)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_plot_filename = 'plots/workspace_plot_homotopy.png'
    
    if workspace_points.shape[0] > 0:
        plot_workspace_3_view(workspace_points, output_filename=output_plot_filename)
    else:
        print("No reachable points found to plot.")

    end_time = time.time() 
    print(f"\nTotal analysis and plotting time: {end_time - start_time:.2f} seconds")
