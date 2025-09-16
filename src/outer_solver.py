# src/outer_solver.py

import numpy as np
import pyswarms as ps
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares, minimize
import logging
import time
import sys
import os

# --- 核心依赖 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kinematics import forward_kinematics
from src.solver import solve_static_equilibrium_diff4 # 导入新的 diff4 求解器
from src.utils.read_config import load_config
from src.utils.drive_mapping import diff4_to_delta_l, delta_l_to_diff4

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =====================================================================
# === 核心工具函数
# =====================================================================
def calculate_pose_error(T_actual, T_target):
    pos_error = T_actual[:3, 3] - T_target[:3, 3]
    R_error = T_actual[:3, :3] @ T_target[:3, :3].T
    try:
        orientation_error = Rotation.from_matrix(R_error).as_rotvec()
    except ValueError:
        orientation_error = np.zeros(3)
    return np.concatenate([pos_error, orientation_error])

# =====================================================================
# === TRF 精炼求解器 (已适配 diff4)
# =====================================================================
def refine_ik_trf(initial_diff4, target_pose, config):
    logging.info("[TRF Solver] Starting refinement in 4D differential space...")
    # q_guess_cache = [np.zeros(6)] # [BUGFIX] Removed stateful cache
    last_solve_cache = {'diff4': None, 'q_eq': None, 'error_vec': None}

    trf_params = config.get('Solver', {}).get('trf', {})
    ftol, xtol, gtol = trf_params.get('ftol', 1e-6), trf_params.get('xtol', 1e-6), trf_params.get('gtol', 1e-6)
    max_nfev = trf_params.get('max_nfev', 1000)

    def solve_and_cache(diff4):
        if np.array_equal(diff4, last_solve_cache['diff4']):
            return last_solve_cache['q_eq'], last_solve_cache['error_vec']

        # [BUGFIX] Always use a fresh, stateless initial guess for TRF.
        q_guess = np.zeros(6)
        solve_result = solve_static_equilibrium_diff4(q_guess, diff4, config)
        q_eq = solve_result["q_solution"]
        
        if q_eq is None:
            last_solve_cache.update({'diff4': diff4, 'q_eq': None, 'error_vec': None})
            return None, None

        # q_guess_cache[0] = q_eq # [BUGFIX] Removed stateful cache update
        T_actual, _ = forward_kinematics(q_eq, config)
        
        pos_tol = config.get('Solver', {}).get('pos_tol_m', 0.01)
        ori_tol = config.get('Solver', {}).get('ori_tol_rad', np.deg2rad(5.0))
        error_vec = calculate_pose_error(T_actual, target_pose)
        error_vec[:3] /= pos_tol
        error_vec[3:] /= ori_tol
        
        last_solve_cache.update({'diff4': diff4, 'q_eq': q_eq, 'error_vec': error_vec})
        return q_eq, error_vec

    def residual(diff4):
        _, error_vec = solve_and_cache(diff4)
        if error_vec is None:
            return np.ones(6) * 1e3
        return error_vec

    jac_method = '2-point'

    bounds_config = config.get('Bounds', {}).get('diff4_bounds', [-0.12, 0.12])
    diff4_bounds = (np.full(4, bounds_config[0]), np.full(4, bounds_config[1]))

    res = least_squares(
        residual, initial_diff4, jac=jac_method, method='trf', 
        bounds=diff4_bounds, ftol=ftol, xtol=xtol, gtol=gtol,
        max_nfev=max_nfev, verbose=2
    )

    converged, final_diff4 = res.success, res.x
    details = {'reason': res.message, 'cost': res.cost, 'nfev': res.nfev, 'optimality': res.optimality}
    
    logging.info(f"✅ [TRF Solver] Refinement finished: {res.message}")
    return final_diff4, converged, details

# =====================================================================
# === 全局求解器 (P1.1: 增加 L-BFGS-B 预收敛)
# =====================================================================
def solve_ik_globally(target_pose, config):
    logging.info("[Relay Solver] Starting Phase 1: PSO Global Search with Phased Cost...")
    
    pso_params = config.get('Solver', {}).get('pso', {})
    stage_a_params = pso_params.get('stageA', {'n_particles': 64, 'iters': 50})
    weights = config.get('Solver', {}).get('weights', {})
    w_pos, w_ori, w_drive, w_reg = weights.get('w_pos', 100.0), weights.get('w_ori', 5.0), weights.get('w_drive', 1e-4), weights.get('w_reg', 1e-4)
    pos_scale = config.get('Solver', {}).get('pos_scale', 0.05)
    ori_scale = config.get('Solver', {}).get('ori_scale', 0.1)

    n_particles = stage_a_params['n_particles']
    q_cache = [np.zeros(6) for _ in range(n_particles)]

    # 使用闭包来追踪迭代次数
    def create_fitness_function(iters):
        pso_iter_count = 0

        def pso_fitness_func(diff4_swarm):
            nonlocal pso_iter_count
            costs = np.zeros(diff4_swarm.shape[0])
            for i in range(diff4_swarm.shape[0]):
                diff4_particle = diff4_swarm[i, :]
                
                solve_result = solve_static_equilibrium_diff4(q_cache[i], diff4_particle, config)
                q_eq = solve_result["q_solution"]
                
                if q_eq is None: 
                    costs[i] = 1e6
                else:
                    T_actual, _ = forward_kinematics(q_eq, config)
                    error_vec = calculate_pose_error(T_actual, target_pose)
                    pos_error = np.linalg.norm(error_vec[:3])
                    ori_error = np.linalg.norm(error_vec[3:])
                    
                    # 归一化误差
                    pos_error_n = pos_error / pos_scale
                    ori_error_n = ori_error / ori_scale

                    # 分阶段代价函数
                    if pso_iter_count < int(iters * 0.2):
                        # 早期只关注位置误差
                        costs[i] = pos_error_n
                    else:
                        # 后期加入所有项
                        drive_penalty = np.linalg.norm(diff4_particle)
                        reg_penalty = np.linalg.norm(q_eq)
                        costs[i] = (w_pos * pos_error_n) + (w_ori * ori_error_n) + (w_drive * drive_penalty) + (w_reg * reg_penalty)
                    
                    q_cache[i] = q_eq
            
            pso_iter_count += 1
            return costs
        return pso_fitness_func

    iters = stage_a_params['iters']
    pso_fitness_func = create_fitness_function(iters)

    bounds_config = config.get('Bounds', {}).get('diff4_bounds', [-0.12, 0.12])
    pso_bounds = (np.full(4, bounds_config[0]), np.full(4, bounds_config[1]))
    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
    optimizer = ps.single.GlobalBestPSO(n_particles=stage_a_params['n_particles'], dimensions=4, options=options, bounds=pso_bounds)
    
    _, pso_best_pos = optimizer.optimize(pso_fitness_func, iters=iters, verbose=True)
    logging.info("[Relay Solver] Phase 1 finished. PSO found a promising candidate.")

    logging.info("[Relay Solver] Starting Phase 2: Trust Region Refinement...")
    final_diff4, converged, details = refine_ik_trf(pso_best_pos, target_pose, config)

    final_delta_l = diff4_to_delta_l(final_diff4)
    solve_result_final = solve_static_equilibrium_diff4(np.zeros(6), final_diff4, config)
    final_q = solve_result_final["q_solution"]

    if final_q is None:
        final_error_mm = 9999.0
    else:
        T_final, _ = forward_kinematics(final_q, config)
        final_error_vec = calculate_pose_error(T_final, target_pose)
        final_error_mm = np.linalg.norm(final_error_vec[:3]) * 1000

    message = f"Relay solve finished. Converged: {converged}. Details: {details}"
    logging.info(message)

    return {'success': converged, 'delta_l': final_delta_l, 'q': final_q, 'error_mm': final_error_mm, 'message': message}

if __name__ == '__main__':
    print("--- 外循环求解器功能自检 (P1.1: L-BFGS-B 预收敛) ---")
    config = load_config('config/config.json')
    
    np.set_printoptions(precision=6, suppress=True)

    L_total = config['Geometry']['PSS_initial_length'] + config['Geometry']['CMS_proximal_length'] + config['Geometry']['CMS_distal_length']
    target_pose = np.array([[1, 0, 0, 0.05], [0, 1, 0, 0.05], [0, 0, 1, L_total * 0.7], [0, 0, 0, 1]])
    print(f"\n目标位姿:\n{target_pose}")

    print("\n--- 调用全局求解器 (solve_ik_globally) ---")
    start_time = time.time()
    result = solve_ik_globally(target_pose, config)
    end_time = time.time()
    print(f"--- 求解完成 (耗时: {end_time - start_time:.2f}s) ---")

    if result['q'] is None:
        print("求解失败，无法计算最终误差。")
    else:
        if result['success']:
            print("✅ 全局求解器报告收敛成功!")
        else:
            print(f"⚠️ 全局求解器未能完全收敛。详情: {result.get('message', 'N/A')}")

        print(f"  - 最终位置误差: {result['error_mm']:.4f} mm")
        # Old: T_final, _ = forward_kinematics(result['q'], config)
        T_final, _, _, _ = forward_kinematics(result['q'], config, return_all_transforms=True)
        final_error_vec = calculate_pose_error(T_final, target_pose)
        final_error_deg = np.rad2deg(np.linalg.norm(final_error_vec[3:]))
        print(f"  - 最终姿态误差: {final_error_deg:.4f} deg")
        print(f"  - 最终求解的电机位移 (m):\n    {result['delta_l']}")
        print(f"  - 最终求解的构型 q (6D):\n    {result['q']}")
