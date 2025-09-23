# 诊断脚本：统计成功率并保存成功样本 (并行版)
import sys
import os
import multiprocessing
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cosserat.kinematics import discretize_robot, forward_kinematics
from src.cosserat.solver import solve_static_equilibrium
from src.utils.read_config import load_config

def diag_worker(task_args):
    """Worker for parallel diagnostic script."""
    dl_sample, params, kappas_guess = task_args
    res = solve_static_equilibrium(kappas_guess, dl_sample, params)
    k_eq = res['kappas_solution']
    if k_eq is not None:
        T, _ = forward_kinematics(k_eq, params)
        return {'dl': dl_sample, 'pos': T[:3, 3]}
    return None

def run_diag_a():
    cfg = load_config('config/config.json')
    low, high = cfg.get('Bounds', {}).get('delta_l_bounds', [-0.05, 0.05])
    N = 500
    _, (n_pss, n_cms1, n_cms2) = discretize_robot(cfg)
    kappas_guess = np.zeros((3, n_pss + n_cms1 + n_cms2))

    # Generate all samples first
    delta_l_samples = [np.random.uniform(low, high, size=8) for _ in range(N)]
    tasks = [(sample, cfg, kappas_guess) for sample in delta_l_samples]

    print(f"Running Parallel Monte Carlo diagnostic with {N} samples...")
    succ_list = []
    num_cpus = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=max(1, num_cpus - 1)) as pool:
        results_iterator = pool.imap_unordered(diag_worker, tasks)
        for result in tqdm(results_iterator, total=N, desc="Running Diagnostic A"):
            if result is not None:
                succ_list.append(result)

    success = len(succ_list)
    print(f"MonteCarlo: {success}/{N} succeeded ({success/N*100:.1f}%)")
    np.save('diagnostics/succ_dl.npy', succ_list, allow_pickle=True)
    print("Saved successful samples to diagnostics/succ_dl.npy")

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    run_diag_a()