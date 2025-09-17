import numpy as np
import matplotlib.pyplot as plt
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
    q_guess = np.zeros(6)
    target_pretension = cfg['Drive_Properties']['pretension_force_N']
    pretension_path = sorted(list(set([0.0, 0.1, 0.2, 0.3] + [target_pretension])))
    pretension_path = [p for p in pretension_path if p <= target_pretension]
    if not pretension_path or pretension_path[-1] != target_pretension:
        pretension_path.append(target_pretension)

    for p in pretension_path:
        cfg_local = deepcopy(cfg)
        cfg_local['Drive_Properties']['pretension_force_N'] = p
        res_dict = solve_static_equilibrium_diff4(q_guess, diff4_target, cfg_local)
        q_new = res_dict['q_solution']
        if q_new is None:
            return None
        q_guess = q_new
    return q_guess

def worker_solve_for_stiffness(args):
    diff4_target, pss_stiffness = args
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'config', 'config.json')
    params = load_config(config_path)
    params['Stiffness']['pss_total_equivalent_bending_stiffness'] = pss_stiffness
    q_eq = pretension_continuation(diff4_target, params)
    if q_eq is not None:
        try:
            T_final, _ = forward_kinematics(q_eq, params)
            return T_final[:3, 3]
        except Exception:
            return None
    return None

def run_sensitivity_analysis():
    print("--- Starting PSS Stiffness Sensitivity Analysis ---")
    stiffness_values = np.linspace(15.0, 25.0, 10)
    num_samples_per_stiffness = 200
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'config', 'config.json')
    params = load_config(config_path)
    diff4_bounds = params.get('Bounds', {}).get('diff4_bounds', [-0.12, 0.12])
    results = []

    for stiffness in tqdm(stiffness_values, desc="Scanning PSS Stiffnesses"):
        tasks = []
        for _ in range(num_samples_per_stiffness):
            random_diff4 = np.random.uniform(diff4_bounds[0], diff4_bounds[1], size=4)
            tasks.append((random_diff4, stiffness))
        
        reachable_points = []
        with multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count() - 1)) as pool:
            point_results = list(tqdm(pool.imap(worker_solve_for_stiffness, tasks), total=len(tasks), desc=f"PSS Stiffness {stiffness:.2f}", leave=False))
            for p in point_results:
                if p is not None:
                    reachable_points.append(p)

        if not reachable_points:
            print(f"Warning: No reachable points found for stiffness = {stiffness:.4f}")
            results.append({'stiffness': stiffness, 'mean_z_rel': 0, 'max_z_rel': 0, 'mean_radius': 0, 'success_rate': 0})
            continue
            
        points_arr = np.array(reachable_points)
        base_z = params['Geometry']['PSS_initial_length']
        z_coords_rel = points_arr[:, 2] - base_z
        xy_radius = np.linalg.norm(points_arr[:, :2], axis=1)
        
        mean_z_rel = np.mean(z_coords_rel)
        max_z_rel = np.max(z_coords_rel)
        mean_radius = np.mean(xy_radius)
        success_rate = len(reachable_points) / num_samples_per_stiffness
        
        results.append({
            'stiffness': stiffness,
            'mean_z_rel': mean_z_rel,
            'max_z_rel': max_z_rel,
            'mean_radius': mean_radius,
            'success_rate': success_rate
        })
        
        print(f"Stiffness: {stiffness:.2f} -> Mean Z_rel: {mean_z_rel:.4f}, Max Z_rel: {max_z_rel:.4f}, Mean Radius: {mean_radius:.4f}, Success: {success_rate*100:.0f}%")

    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    fig.suptitle('PSS Bending Stiffness Sensitivity Analysis', fontsize=16)
    stiffness_axis = [r['stiffness'] for r in results]
    
    axes[0].plot(stiffness_axis, [r['mean_z_rel'] for r in results], 'bo-', label='Mean Z (relative to base)')
    axes[0].plot(stiffness_axis, [r['max_z_rel'] for r in results], 'ro-', label='Max Z (relative to base)')
    axes[0].set_ylabel('Z Coordinate (m)')
    axes[0].set_title('Workspace Height')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(stiffness_axis, [r['mean_radius'] for r in results], 'go-', label='Mean Radius')
    axes[1].set_ylabel('XY Radius (m)')
    axes[1].set_title('Workspace Radius')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(stiffness_axis, [r['success_rate'] for r in results], 'ko-', label='Success Rate')
    axes[2].set_ylabel('Success Rate')
    axes[2].set_title('Solver Success Rate')
    axes[2].set_xlabel('PSS Bending Stiffness (Nm^2/rad)')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = os.path.join(project_root, 'plots', 'sensitivity_analysis_pss_stiffness.png')
    plt.savefig(plot_filename)
    print(f"\nSensitivity analysis plot saved to {plot_filename}")
    plt.close(fig)

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    run_sensitivity_analysis()
