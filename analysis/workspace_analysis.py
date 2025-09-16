import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import time
import multiprocessing
from tqdm import tqdm
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.read_config import load_config
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import time
import multiprocessing
from tqdm import tqdm
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.read_config import load_config
# 修正: 导入新的 diff4 求解器
from src.solver import solve_static_equilibrium_diff4
from src.kinematics import forward_kinematics
# 修正: 导入 expand_diff4_to_motor8 用于后处理
from src.statics import calculate_total_potential_energy_disp_ctrl, calculate_drive_mapping, expand_diff4_to_motor8

def continuation_solve_diff4(q_guess, diff4, params, n_steps=30):
    """
    [IMPROVED as per test.md Step 4] Solves via parameter continuation (homotopy) on stiffness.
    Uses a smoother stiffness ramp and retries on failure.
    """
    orig_k_pss = params['Stiffness']['pss_total_equivalent_bending_stiffness']
    orig_k_cms = params['Stiffness']['cms_bending_stiffness']
    
    local_params = params.copy()
    local_params['Stiffness'] = params['Stiffness'].copy()

    try:
        q_current = q_guess
        for i in range(n_steps + 1):
            t = i / n_steps
            # Smoother stiffness ramp as per test.md
            stiffness_scale = 1e-6 + (t**2)
            
            local_params['Stiffness']['pss_total_equivalent_bending_stiffness'] = orig_k_pss * stiffness_scale
            local_params['Stiffness']['cms_bending_stiffness'] = orig_k_cms * stiffness_scale
            
            result_dict = solve_static_equilibrium_diff4(q_current, diff4, local_params)
            q_new = result_dict["q_solution"]
            
            # Retry logic as per test.md
            if q_new is None:
                for _ in range(5): # Retry 5 times
                    q_try = q_current + np.random.randn(6) * 1e-3 # Small perturbation
                    result_dict = solve_static_equilibrium_diff4(q_try, diff4, local_params)
                    q_new = result_dict["q_solution"]
                    if q_new is not None:
                        break
            
            if q_new is None:
                return None 
            q_current = q_new
        return q_current
    finally:
        params['Stiffness']['pss_total_equivalent_bending_stiffness'] = orig_k_pss
        params['Stiffness']['cms_bending_stiffness'] = orig_k_cms

def worker_solve_single_point(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'config', 'config.json')
    params = load_config(config_path)

    i, q0_6d = args
    np.random.seed(os.getpid() + int(time.time()))
    
    # 步骤3: 直接采样 diff4
    diff4_bounds = params.get('Bounds', {}).get('diff4_bounds', [-0.12, 0.12])
    random_diff4 = np.random.uniform(diff4_bounds[0], diff4_bounds[1], size=4)
    
    q_guess_rand = np.zeros(6)
    # Use a more conservative initial guess to improve stability
    q_guess_rand[::2] = np.random.uniform(-2.0, 2.0, 3)
    q_guess_rand[1::2] = np.random.uniform(-np.pi, np.pi, 3)
    
    # 调用新的 continuation 求解器
    q_eq = continuation_solve_diff4(q_guess_rand, random_diff4, params)
    
    if q_eq is not None and not np.any(np.isnan(q_eq)) and not np.any(np.isinf(q_eq)):
        try:
            # 为了计算能量和拉伸，需要将 diff4 展开
            delta_l_motor = expand_diff4_to_motor8(random_diff4, params)
            T_final, _ = forward_kinematics(q_eq, params)
            U_total = calculate_total_potential_energy_disp_ctrl(q_eq, delta_l_motor, params)
            delta_l_robot = calculate_drive_mapping(q_eq, params)
            stretch = delta_l_motor - delta_l_robot
            
            return ("success", {
                "pos": T_final[:3, 3],
                "diff4": random_diff4, # 保存 diff4
                "q_eq": q_eq,
                "U_total": U_total,
                "stretch": stretch
            })
        except Exception:
            return ("kin_fail", None)
    return ("solver_fail", None)

def run_monte_carlo_workspace_analysis(num_samples=4000, use_tqdm=True, parallel=True):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'config', 'config.json')
    params = load_config(config_path)
    q0_7d = np.array(params['Initial_State']['q0'])
    q0_6d = q0_7d[1:]

    tasks = [(i, q0_6d) for i in range(num_samples)]
    reachable_points_data = []
    fail_counts = {'solver_fail': 0, 'kin_fail': 0}

    if parallel:
        num_cpus = multiprocessing.cpu_count()
        if use_tqdm:
            print(f"Detected {num_cpus} CPU cores. Using {max(1, num_cpus - 1)} processes.")
        with multiprocessing.Pool(processes=max(1, num_cpus - 1)) as pool:
            results_iterator = pool.imap_unordered(worker_solve_single_point, tasks)
            if use_tqdm:
                results_iterator = tqdm(results_iterator, total=num_samples, desc="Calculating Workspace (Homotopy)")
            
            for status, result_dict in results_iterator:
                if status == "success":
                    reachable_points_data.append(result_dict)
                else:
                    fail_counts[status] += 1
    else:
        iterator = tasks
        if use_tqdm:
            iterator = tqdm(iterator, desc="Calculating Workspace (Homotopy, Serial)")
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

def plot_workspace_3_view(points, output_filename='plots/workspace_plot_homotopy.png', slice_width=0.01):
    if points is None or len(points) < 1:
        print("Warning: Not enough points to plot workspace.")
        return

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_filename = os.path.join(project_root, output_filename)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    print(f"\nPlotting workspace 3-view... Saving to '{output_filename}'")
    fig = plt.figure(figsize=(14, 21))
    fig.suptitle('Workspace Analysis (Homotopy Method)', fontsize=16)

    ws_x, ws_y, ws_z = points[:, 0], points[:, 1], points[:, 2]

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
    ax_slice_x0.set_xlabel('Y (m)'); ax_slice_x0.set_ylabel('Z (m)')
    ax_slice_x0.grid(True); ax_slice_x0.axis('equal')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename, dpi=200)
    print(f"3-view plot saved to {os.path.abspath(output_filename)}")
    plt.close(fig)


if __name__ == '__main__':
    start_time = time.time()
    
    print(f"--- Starting Monte Carlo Workspace Analysis (Homotopy Method) ---")
    
    workspace_full_data = run_monte_carlo_workspace_analysis(num_samples=1500)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_filename = os.path.join(project_root, 'plots', 'workspace_points_homotopy.npy')
    output_plot_filename = 'plots/workspace_plot_homotopy.png'
    os.makedirs(os.path.dirname(data_filename), exist_ok=True)
    
    save_data = np.array(workspace_full_data, dtype=object)
    np.save(data_filename, save_data, allow_pickle=True)
    print(f"Workspace data (homotopy) saved to {os.path.abspath(data_filename)}")
    
    if workspace_full_data:
        stretches = np.array([d['stretch'] for d in workspace_full_data])
        num_tight_cables = np.sum(stretches > 1e-6, axis=1)
        
        print("\n--- Stretch Diagnostics (Homotopy Method) ---")
        print(f"Avg. number of tight cables per successful point: {np.mean(num_tight_cables):.2f}")
        print(f"Max stretch observed: {np.max(stretches)*1000:.2f} mm")
        print(f"Avg. max stretch per point: {np.mean(np.max(stretches, axis=1))*1000:.2f} mm")

        workspace_points = np.array([d['pos'] for d in workspace_full_data])
        plot_workspace_3_view(workspace_points, output_filename=output_plot_filename)
    else:
        print("No reachable points found to plot.")

    end_time = time.time() 
    print(f"\nTotal analysis and plotting time: {end_time - start_time:.2f} seconds")