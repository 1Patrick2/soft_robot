# 诊断脚本C：内循环返回的解类型统计 (并行版)
import sys
import os
import multiprocessing
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cosserat.kinematics import discretize_robot, forward_kinematics
from src.cosserat.solver import solve_static_equilibrium
from src.utils.read_config import load_config

def diag_c_worker(task_args):
    """Worker for parallel diagnostic C script."""
    dl, params, kappas_guess = task_args
    res = solve_static_equilibrium(kappas_guess, dl, params)
    k_eq = res['kappas_solution']
    
    if k_eq is not None:
        kappa_norm = np.linalg.norm(k_eq)
        T, _ = forward_kinematics(k_eq, params)
        tip_z = T[2, 3]
        return (kappa_norm, tip_z)
    return None

def run_diag_c():
    print("Running Diagnostic C: Analyzing properties of successful solutions...")

    cfg = load_config('config/config.json')
    _, (n_pss, n_cms1, n_cms2) = discretize_robot(cfg)
    num_el = n_pss + n_cms1 + n_cms2

    try:
        succ_data = np.load('diagnostics/succ_dl.npy', allow_pickle=True)
    except FileNotFoundError:
        print("Error: diagnostics/succ_dl.npy not found. Please run Diagnostic A first.")
        return

    if len(succ_data) == 0:
        print("No successful solutions found in succ_dl.npy. Cannot run Diagnostic C.")
        return

    kappas_guess = np.zeros((3, num_el))
    tasks = [(s['dl'], cfg, kappas_guess) for s in succ_data]

    print(f"Re-solving for {len(succ_data)} successful samples to get solution properties (in parallel)...")
    
    kappa_norms = []
    tips_z = []
    num_cpus = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=max(1, num_cpus - 1)) as pool:
        results_iterator = pool.imap_unordered(diag_c_worker, tasks)
        for result in tqdm(results_iterator, total=len(tasks), desc="Analyzing Samples"):
            if result is not None:
                kappa_norm, tip_z = result
                kappa_norms.append(kappa_norm)
                tips_z.append(tip_z)

    if not kappa_norms: # Check if list is empty
        print("\n--- Diagnostic C Results ---")
        print("All re-solve attempts failed. Cannot compute statistics.")
        return

    print("\n--- Diagnostic C Results ---")
    print(f"kappa_norms: min={np.min(kappa_norms):.4f}, mean={np.mean(kappa_norms):.4f}, max={np.max(kappa_norms):.4f}")
    print(f"tip_z:       min={np.min(tips_z):.4f}, mean={np.mean(tips_z):.4f}, max={np.max(tips_z):.4f}")

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    run_diag_c()
