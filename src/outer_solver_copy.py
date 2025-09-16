# src/outer_solver.py

import sys
import os
import numpy as np
import pyswarms as ps
from scipy.spatial.transform import Rotation
import logging
import time
from scipy.optimize import least_squares

# --- 核心依赖 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kinematics import forward_kinematics
from src.solver import solve_static_equilibrium_disp_ctrl
from src.utils.read_config import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =====================================================================
# === 核心工具函数 (V-Final 物理重构版)
# =====================================================================

def calculate_pose_error(T_actual, T_target):
    """计算实际位姿与目标位姿之间的6D误差向量（位置+旋转向量）"""
    pos_error = T_actual[:3, 3] - T_target[:3, 3]
    R_error = T_actual[:3, :3] @ T_target[:3, :3].T
    try:
        orientation_error = Rotation.from_matrix(R_error).as_rotvec()
    except ValueError:
        # 矩阵非正交或接近奇异时，返回零误差
        orientation_error = np.zeros(3)
    return np.concatenate([pos_error, orientation_error])

def calculate_task_jacobian_numerical(delta_l, q_guess, config, target_pose, epsilon=1e-7):
    """
    [V-Final 物理重构版] 使用纯数值方法，计算最可靠的任务雅可比 J_task = dT/d(Δl_motor)
    """
    num_inputs = len(delta_l)
    num_outputs = 6
    J_numerical = np.zeros((num_outputs, num_inputs))

    # 计算基准点的平衡构型和位姿
    q_eq_base = solve_static_equilibrium_disp_ctrl(q_guess, delta_l, config)
    if q_eq_base is None:
        logging.warning("数值雅可比计算失败：基准点无法求解。")
        return None, None
    
    T_base, _ = forward_kinematics(q_eq_base, config)

    for i in range(num_inputs):
        # 使用中心差分以获得更高精度
        delta_l_plus = delta_l.copy(); delta_l_plus[i] += epsilon
        delta_l_minus = delta_l.copy(); delta_l_minus[i] -= epsilon

        # 正向扰动
        q_eq_plus = solve_static_equilibrium_disp_ctrl(q_eq_base, delta_l_plus, config)
        if q_eq_plus is None:
            logging.warning(f"数值雅可比计算失败 (+扰动, i={i})，内循环求解器未能收敛。")
            return None, q_eq_base
        T_plus, _ = forward_kinematics(q_eq_plus, config)
        pose_error_plus = calculate_pose_error(T_plus, target_pose)
        
        # 负向扰动
        q_eq_minus = solve_static_equilibrium_disp_ctrl(q_eq_base, delta_l_minus, config)
        if q_eq_minus is None:
            logging.warning(f"数值雅可比计算失败 (-扰动, i={i})，内循环求解器未能收敛。")
            return None, q_eq_base
        T_minus, _ = forward_kinematics(q_eq_minus, config)
        pose_error_minus = calculate_pose_error(T_minus, target_pose)
        
        # 计算差分
        column = (pose_error_plus - pose_error_minus) / (2 * epsilon)
        J_numerical[:, i] = column
        
    return J_numerical, q_eq_base

# =====================================================================
# === 终极求解器 (V-Final 物理重构版)
# =====================================================================

pso_best_solution_cache = {}

def solve_ik_pure_displacement(target_pose, config):
    """ [V-Final 物理重构版] 纯位移空间策略：PSO全局探索 + 数值雅可比精炼 """
    q0_7d = np.array(config['Initial_State']['q0'])
    q0_6d = q0_7d[1:]

    # --- Phase 1: PSO全局侦察 (带缓存) ---
    logging.info("[Phase 1] Starting PSO global search with solution caching...")
    pso_best_solution_cache.clear()

    def pso_fitness_func(delta_l_swarm):
        n_particles = delta_l_swarm.shape[0]
        errors = np.zeros(n_particles)
        for i in range(n_particles):
            delta_l_particle = delta_l_swarm[i, :]
            q_eq = solve_static_equilibrium_disp_ctrl(q0_6d, delta_l_particle, config)
            if q_eq is None: 
                errors[i] = 1e6; continue
            
            T_actual, _ = forward_kinematics(q_eq, config)
            error = np.linalg.norm(T_actual[:3, 3] - target_pose[:3, 3])
            errors[i] = error
            
            if 'best_error' not in pso_best_solution_cache or error < pso_best_solution_cache['best_error']:
                pso_best_solution_cache['best_error'] = error
                pso_best_solution_cache['q'] = q_eq
                pso_best_solution_cache['delta_l'] = delta_l_particle
        return errors

    pso_bounds = (np.full(8, 0.0), np.full(8, 0.05))
    optimizer = ps.single.GlobalBestPSO(n_particles=64, dimensions=8, options={'c1': 2.5, 'c2': 0.5, 'w': 0.9}, bounds=pso_bounds)
    pso_cost, _ = optimizer.optimize(pso_fitness_func, iters=50, verbose=False)
    
    # --- Phase 2: 从“无损缓存”中提取起点，进行高精度精炼 ---
    if 'q' not in pso_best_solution_cache:
        return {'success': False, 'delta_l': None, 'q': None, 'error': pso_cost, 'message': 'PSO failed to find any valid solution.'}

    q_base_camp = pso_best_solution_cache['q']
    dl_base_camp = pso_best_solution_cache['delta_l']
    initial_error = pso_best_solution_cache['best_error']

    logging.info(f"[Phase 2] Starting refinement from a perfectly matched base camp (error: {initial_error*1000:.4f} mm)...")

    refinement_cache = {'q': q_base_camp}

    def residual_and_jacobian_disp(delta_l):
        q_guess = refinement_cache['q']
        # [核心] 使用新的数值雅可比函数
        jacobian, q_eq = calculate_task_jacobian_numerical(delta_l, q_guess, config, target_pose)
        
        if jacobian is None or q_eq is None:
            return np.full(6, 1e6), np.zeros((6, 8))
        
        refinement_cache['q'] = q_eq
        T_actual, _ = forward_kinematics(q_eq, config)
        residual = calculate_pose_error(T_actual, target_pose)
        return residual, jacobian

    result = least_squares(
        lambda x: residual_and_jacobian_disp(x)[0], 
        dl_base_camp,
        jac=lambda x: residual_and_jacobian_disp(x)[1],
        method='trf', 
        bounds=(0, 0.05), 
        ftol=1e-9, xtol=1e-9, gtol=1e-9
    )

    # --- 结果处理 ---
    if result.success:
        final_dl = result.x
        final_q = refinement_cache['q']
        final_T, _ = forward_kinematics(final_q, config)
        final_error = np.linalg.norm(final_T[:3, 3] - target_pose[:3, 3])
        return {'success': True, 'delta_l': final_dl, 'q': final_q, 'error': final_error, 'message': 'Solver finished successfully.'}
    else:
        return {'success': False, 'delta_l': result.x, 'q': None, 'error': np.inf, 'message': f'least_squares refinement failed: {result.message}'}

if __name__ == '__main__':
    print("--- 外循环求解器功能自检 (V-Final 物理重构版) ---")
    config = load_config('config/config.json')
    np.set_printoptions(precision=6, suppress=True)

    L_total = config['Geometry']['PSS_initial_length'] + config['Geometry']['CMS_proximal_length'] + config['Geometry']['CMS_distal_length']
    target_pose = np.array([
        [1, 0, 0, 0.05],
        [0, 1, 0, 0.05],
        [0, 0, 1, L_total * 0.7],
        [0, 0, 0, 1]
    ])

    print(f"\n目标位姿:\n{target_pose}")

    print("\n--- 求解中 (使用纯数值雅可比)... ---")
    start_time = time.time()
    result = solve_ik_pure_displacement(target_pose, config)
    end_time = time.time()
    print(f"--- 求解完成 (耗时: {end_time - start_time:.2f}s) ---")
    if result['success']:
        print("✅ 终极求解器求解成功!")
        print(f"  - 最终欧氏距离误差: {result['error'] * 1000:.4f} mm")
        print(f"  - 最终求解的电机位移 (m):\n    {result['delta_l']}")
        print(f"  - 最终求解的构型 q (6D):\n    {result['q']}")
    else:
        print(f"❌ 终极求解器求解失败。详情: {result.get('message', 'N/A')}")














        # src/outer_solver.py

import sys
import os
import numpy as np
import pyswarms as ps
from scipy.spatial.transform import Rotation
import logging
import time
from scipy.optimize import minimize
from src.kinematics import forward_kinematics
from src.solver import solve_static_equilibrium_disp_ctrl
from src.utils.read_config import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =====================================================================
# === 核心工具函数 (无变化)
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
# === 武器库：(新增) 轻量级局部求解器
# =====================================================================
def solve_ik_locally(target_pose, config, delta_l_guess, q_guess):
    """
    [新增] 一个轻量级的、用于路径跟踪的局部IK求解器。
    它从一个高质量的初始猜测（来自上一个路径点）出发，进行快速精炼。
    """
    logging.info(f"[Local Solver] Refining from warm start...")

    # 直接进入精炼阶段，使用传入的猜测值
    dl_base_camp = delta_l_guess
    q_base_camp = q_guess

    refinement_cache = {'q': q_base_camp, 'eval_count': 0}
    def refinement_cost_function(delta_l):
        refinement_cache['eval_count'] += 1
        # Use the cached q as the guess for the next equilibrium solve
        q_eq = solve_static_equilibrium_disp_ctrl(refinement_cache['q'], delta_l, config)
        if q_eq is None: return 1e6
        
        # Cache the new solution for the next iteration
        refinement_cache['q'] = q_eq
        T_actual, _ = forward_kinematics(q_eq, config)
        error = np.linalg.norm(T_actual[:3, 3] - target_pose[:3, 3])
        return error

    # 使用固定的边界
    scipy_bounds = list(zip(np.full(8, 0.0), np.full(8, 0.15)))

    result = minimize(
        fun=refinement_cost_function,
        x0=dl_base_camp,
        method='L-BFGS-B',
        bounds=scipy_bounds,
        jac='3-point',
        # [关键] 使用更宽松的容差和更少的迭代次数，因为它已经是热启动
        options={'ftol': 1e-8, 'gtol': 1e-6, 'maxiter': 50}
    )

    # 结果处理
    final_dl = result.x
    # Use the last known good q from the cache
    final_q = solve_static_equilibrium_disp_ctrl(refinement_cache['q'], final_dl, config)
    if final_q is None: final_q = refinement_cache['q']
        
    final_T, _ = forward_kinematics(final_q, config)
    final_error = np.linalg.norm(final_T[:3, 3] - target_pose[:3, 3])
    
    return {'success': result.success, 'delta_l': final_dl, 'q': final_q, 'error': final_error, 'message': result.message}

# =====================================================================
# === 武器库：(重命名) 全局高精度求解器
# =====================================================================
pso_best_solution_cache = {}

def solve_ik_globally(target_pose, config):
    """
    (原 solve_ik_pure_displacement)
    用于单点的、不计代价的高精度全局求解。
    """
    q0_7d = np.array(config['Initial_State']['q0'])
    q0_6d = q0_7d[1:]

    # --- Phase 1: PSO 全局探索 ---
    logging.info("[Global Solver - Phase 1] Starting STRONG PSO global search...")
    pso_best_solution_cache.clear()

    def pso_fitness_func(delta_l_swarm):
        n_particles = delta_l_swarm.shape[0]
        errors = np.zeros(n_particles)
        # [V-Final.Memory] 继承上一个最优解作为起点，实现路径跟随
        q_guess_for_generation = pso_best_solution_cache.get('q', q0_6d)
        for i in range(n_particles):
            delta_l_particle = delta_l_swarm[i, :]
            # 使用继承来的q_guess进行求解
            q_eq = solve_static_equilibrium_disp_ctrl(q_guess_for_generation, delta_l_particle, config)
            if q_eq is None: 
                errors[i] = 1e6
                continue
            
            T_actual, _ = forward_kinematics(q_eq, config)
            error = np.linalg.norm(T_actual[:3, 3] - target_pose[:3, 3])
            errors[i] = error
            
            # 如果发现更优解，则更新缓存
            if 'best_error' not in pso_best_solution_cache or error < pso_best_solution_cache['best_error']:
                pso_best_solution_cache['best_error'] = error
                pso_best_solution_cache['q'] = q_eq
                pso_best_solution_cache['delta_l'] = delta_l_particle
        return errors

    n_particles = 128
    iters = 150
    options = {'c1': 0.8, 'c2': 0.2, 'w': 0.9}
    pso_bounds = (np.full(8, 0.0), np.full(8, 0.15))
    
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=8, options=options, bounds=pso_bounds)
    pso_cost, _ = optimizer.optimize(pso_fitness_func, iters=iters, verbose=True)
    
    # --- Phase 2: 从PSO最优解出发，使用 `minimize` 进行高精度精炼 ---
    if 'q' not in pso_best_solution_cache:
        best_pos_from_pso = optimizer.pos_history[-1][np.argmin(optimizer.cost_history[-1])]
        q_final = solve_static_equilibrium_disp_ctrl(q0_6d, best_pos_from_pso, config)
        if q_final is None: 
             return {'success': False, 'delta_l': best_pos_from_pso, 'q': None, 'error': pso_cost, 'message': 'PSO failed and could not find a valid final configuration.'}
        T_final, _ = forward_kinematics(q_final, config)
        final_error = np.linalg.norm(T_final[:3, 3] - target_pose[:3, 3])
        return {'success': False, 'delta_l': best_pos_from_pso, 'q': q_final, 'error': final_error, 'message': 'PSO failed to find any valid solution, returning best PSO particle.'}

    q_base_camp = pso_best_solution_cache['q']
    dl_base_camp = pso_best_solution_cache['delta_l']
    initial_error = pso_best_solution_cache['best_error']

    logging.info(f"[Global Solver - Phase 2] Starting MINIMIZE refinement from PSO best (error: {initial_error*1000:.4f} mm)...")

    refinement_cache = {'q': q_base_camp, 'eval_count': 0}
    def refinement_cost_function(delta_l):
        refinement_cache['eval_count'] += 1
        q_guess = refinement_cache['q']
        q_eq = solve_static_equilibrium_disp_ctrl(q_guess, delta_l, config)
        
        if q_eq is None:
            return 1e6
        
        refinement_cache['q'] = q_eq
        T_actual, _ = forward_kinematics(q_eq, config)
        error = np.linalg.norm(T_actual[:3, 3] - target_pose[:3, 3])
        
        if refinement_cache['eval_count'] % 10 == 0:
             logging.info(f"  Refinement eval #{refinement_cache['eval_count']}: error = {error*1000:.4f} mm")
        return error
    
    scipy_bounds = list(zip(pso_bounds[0], pso_bounds[1]))

    result = minimize(
        fun=refinement_cost_function,
        x0=dl_base_camp,
        method='L-BFGS-B',
        bounds=scipy_bounds,
        jac='3-point', 
        options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 200, 'eps': 1e-5}
    )

    # --- [诊断探针] ---
    logging.info(f"[DIAGNOSTIC] PSO found q_base_camp: {q_base_camp}")
    logging.info(f"[DIAGNOSTIC] Refinement result.x (final_dl): {result.x}")
    q_before_return = solve_static_equilibrium_disp_ctrl(refinement_cache['q'], result.x, config)
    logging.info(f"[DIAGNOSTIC] q calculated right before return: {q_before_return}")

    # --- 结果处理 ---
    final_dl = result.x
    final_q = solve_static_equilibrium_disp_ctrl(refinement_cache['q'], final_dl, config)
    if final_q is None:
        final_q = refinement_cache['q'] 
        
    final_T, _ = forward_kinematics(final_q, config)
    final_error = np.linalg.norm(final_T[:3, 3] - target_pose[:3, 3])

    if result.success:
        return {'success': True, 'delta_l': final_dl, 'q': final_q, 'error': final_error, 'message': f'Minimize refinement finished: {result.message}'}
    else:
        return {'success': False, 'delta_l': final_dl, 'q': final_q, 'error': final_error, 'message': f'Minimize refinement failed: {result.message}'}

if __name__ == '__main__':
    print("--- 外循环求解器功能自检 (V-Final.4: 分层策略版) ---")
    config = load_config('config/config.json')
    np.set_printoptions(precision=6, suppress=True)

    L_total = config['Geometry']['PSS_initial_length'] + config['Geometry']['CMS_proximal_length'] + config['Geometry']['CMS_distal_length']
    target_pose = np.array([
        [1, 0, 0, 0.05],
        [0, 1, 0, 0.05],
        [0, 0, 1, L_total * 0.7],
        [0, 0, 0, 1]
    ])

    print(f"\n目标位姿:\n{target_pose}")

    print("\n--- 调用全局求解器 (solve_ik_globally) ---")
    start_time = time.time()
    # In the self-test, we call the "heavy cannon"
    result = solve_ik_globally(target_pose, config)
    end_time = time.time()
    print(f"--- 求解完成 (耗时: {end_time - start_time:.2f}s) ---")
    
    if result['success']:
        print("✅ 全局求解器求解成功!")
    else:
        print(f"⚠️ 全局求解器未能完全收敛，但仍给出了最优结果。详情: {result.get('message', 'N/A')}")
        
    print(f"  - 最终欧氏距离误差: {result['error'] * 1000:.4f} mm")
    print(f"  - 最终求解的电机位移 (m):\n    {result['delta_l']}")
    print(f"  - 最终求解的构型 q (6D):\n    {result['q']}")
