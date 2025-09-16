# src/solver.py

import numpy as np
from scipy.optimize import minimize
import sys
import os
import logging

# --- 核心依赖 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.statics import (
    calculate_total_potential_energy_disp_ctrl,
    calculate_gradient_disp_ctrl
)
from src.utils.read_config import load_config
from src.nondimensionalizer import (
    get_characteristic_scales,
    q_to_nondimensional,
    q_from_nondimensional,
    gradient_to_nondimensional
)

# =====================================================================
# === 内循环求解器 (位移控) - [V-Final.Robust]
# =====================================================================

def solve_static_equilibrium_disp_ctrl(q_guess, delta_l_motor, params):
    """
    [V-Final.Robust] 核心求解器，包含多初值重启与回退机制。
    """
    scales = get_characteristic_scales(params)
    U_char = scales['U_char']
    hat_q_guess = q_to_nondimensional(q_guess, scales)

    def objective_function_hat(hat_q):
        q_phys = q_from_nondimensional(hat_q, scales)
        U_phys = calculate_total_potential_energy_disp_ctrl(q_phys, delta_l_motor, params)
        return U_phys / U_char

    def jacobian_function_hat(hat_q):
        q_phys = q_from_nondimensional(hat_q, scales)
        grad_phys = calculate_gradient_disp_ctrl(q_phys, delta_l_motor, params)
        return gradient_to_nondimensional(grad_phys, scales)

    # [Optimized] Read bounds and solver options from config
    solver_params = params.get('Solver', {})
    kappa_bound = solver_params.get('kappa_bound', 10.0)
    phi_bound = solver_params.get('phi_bound', np.pi)
    n_restarts = solver_params.get('restarts', 3)
    lbfgsb_opts = {
        'ftol': solver_params.get('ftol', 1e-8),
        'gtol': 1e-9, # [NEW] Add strict gradient tolerance
        'maxiter': solver_params.get('maxiter', 1000)
    }
    powell_opts = {'ftol': 1e-5, 'maxiter': 2500}

    bounds_phys = [(-kappa_bound, kappa_bound), (-phi_bound, phi_bound)] * 3
    hat_bounds = []
    for i in range(len(bounds_phys)):
        low, high = bounds_phys[i]
        if i % 2 == 0: # kappa
            hat_bounds.append((low * scales['L_char'], high * scales['L_char']))
        else: # phi
            hat_bounds.append((low, high))

    # --- 优化策略 ---
    # 1. 主力求解器: L-BFGS-B
    result = minimize(
        objective_function_hat, 
        hat_q_guess, 
        method='L-BFGS-B',
        jac=jacobian_function_hat, 
        bounds=hat_bounds,
        options=lbfgsb_opts
    )

    # 2. [Optimized] 多初值重启机制
    if not result.success:
        logging.warning(f"[Solver] L-BFGS-B failed on initial guess. Attempting {n_restarts} restarts.")
        for i in range(n_restarts):
            hat_q_perturbed = hat_q_guess + np.random.randn(len(hat_q_guess)) * 0.1 # 10% perturbation
            result_retry = minimize(
                objective_function_hat, 
                hat_q_perturbed, 
                method='L-BFGS-B',
                jac=jacobian_function_hat, 
                bounds=hat_bounds,
                options=lbfgsb_opts
            )
            if result_retry.success:
                logging.info(f"[Solver] L-BFGS-B succeeded on restart #{i+1}.")
                result = result_retry
                break
        else: # This else belongs to the for loop, executed if loop finishes without break
            logging.warning("[Solver] All L-BFGS-B restarts failed.")

    # 3. 备用求解器: Powell
    if not result.success:
        logging.warning(f"[Solver] Fallback to Powell optimizer.\nL-BFGS-B final result:\n{result}")
        # [关键加固] 使用L-BFGS-B失败时的'x'作为Powell的初值
        hat_q_start_powell = result.x if result.x is not None else hat_q_guess

        result = minimize(
            objective_function_hat,
            hat_q_start_powell,
            method='Powell',
            options=powell_opts
        )

    # 4. [新增] 终极备用求解器: SLSQP (带温启动)
    if not result.success:
        logging.warning(f"[Solver] Fallback to ULTIMATE optimizer: SLSQP...")
        hat_q_start_slsqp = result.x if result.x is not None else hat_q_guess
        result = minimize(
            objective_function_hat,
            hat_q_start_slsqp,
            method='SLSQP',
            jac=jacobian_function_hat, # SLSQP 也可以使用梯度信息
            bounds=hat_bounds,
            options={'ftol': 1e-7, 'maxiter': 2000}
        )

    # [Optimized] 扩展返回信息
    if result.success and not np.any(np.isnan(result.x)):
        q_solution_phys = q_from_nondimensional(result.x, scales)
        return {"q_solution": q_solution_phys, "result": result}
    else:
        logging.error(f"[Solver] All optimization strategies failed. Final result object:\n{result}")
        return {"q_solution": None, "result": result}


def solve_static_equilibrium_diff4(q_guess, diff4, params):
    """
    [NEW] Wrapper for the main solver that accepts a 4D differential drive input.
    """
    from src.statics import expand_diff4_to_motor8
    delta_l_motor = expand_diff4_to_motor8(diff4, params)
    return solve_static_equilibrium_disp_ctrl(q_guess, delta_l_motor, params)


# =====================================================================
# === 自检模块
# =====================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("--- 求解器思想实验 (零重力、零驱动) ---")
    config = load_config('config/config.json')
    np.set_printoptions(precision=6, suppress=True)

    config['Mass']['pss_kg'] = 0
    config['Mass']['cms_proximal_kg'] = 0
    config['Mass']['cms_distal_kg'] = 0
    print("已设置场景: 零重力, 零驱动电机位移")

    q_truth = np.zeros(6)
    delta_l_zero = np.zeros(8)
    q_guess_perturbed = np.full(6, 1e-6)
    print(f"初始猜测构型 (含扰动): {q_guess_perturbed}")

    solve_result = solve_static_equilibrium_disp_ctrl(q_guess_perturbed, delta_l_zero, config)
    q_eq_disp = solve_result["q_solution"]

    if q_eq_disp is not None:
        print(f"✅ 求解成功. 平衡构型 q_eq: {q_eq_disp}")
        if np.allclose(q_eq_disp, q_truth, atol=1e-3):
            print("  - ✅✅✅ 【最终验证通过】: 求解器成功收敛到理论真值附近！")
        else:
            print(f"  - ⚠️  【最终验证警告】: 求解器收敛，但与理论真值存在微小偏差。")
    else:
        print("❌ 求解失败.")

    print("\n" + "="*50)
    print("--- 求解器随机点压力测试 (有重力、随机驱动) ---")
    
    config_gravity = load_config('config/config.json')
    delta_l_motor_random = np.random.rand(8) * 0.002 
    q_guess_random = np.random.rand(6) * np.array([1, 2*np.pi, 1, 2*np.pi, 1, 2*np.pi]) - np.array([0.5, np.pi, 0.5, np.pi, 0.5, np.pi])

    print(f"测试条件: 有重力")
    print(f"  - 随机电机位移 Δl_motor: {delta_l_motor_random}")
    print(f"  - 随机初始猜测 q_guess: {q_guess_random}")

    solve_result_random = solve_static_equilibrium_disp_ctrl(q_guess_random, delta_l_motor_random, config_gravity)
    q_eq_random = solve_result_random["q_solution"]

    if q_eq_random is not None:
        print(f"✅ 【压力测试通过】: 求解器在随机点上成功收敛。\n")
        print(f"  - 平衡构型 q_eq: {q_eq_random}")
    else:
        print(f"❌ 【压力测试失败】: 求解器在随机点上未能收敛。")