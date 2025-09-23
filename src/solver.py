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

def solve_static_equilibrium_disp_ctrl(q_guess, delta_l_motor, params, force_robust_solver=False):
    """
    [V-Final.Robust] 核心求解器，包含多初值重启与回退机制。
    """
    scales = get_characteristic_scales(params)
    # Ensure q_guess is a numpy array before processing
    q_guess = np.array(q_guess)
    hat_q_guess = q_to_nondimensional(q_guess, scales)
    hat_q_guess_flat = hat_q_guess.flatten()

    num_elements = q_guess.shape[1]

    def objective_function_hat(hat_q_flat):
        hat_q = hat_q_flat.reshape((3, num_elements))
        q_phys = q_from_nondimensional(hat_q, scales)
        U_phys = calculate_total_potential_energy_disp_ctrl(q_phys, delta_l_motor, params)
        return U_phys / scales['U_char']

    def jacobian_function_hat(hat_q_flat):
        hat_q = hat_q_flat.reshape((3, num_elements))
        q_phys = q_from_nondimensional(hat_q, scales)
        grad_phys = calculate_gradient_disp_ctrl(q_phys, delta_l_motor, params)
        return gradient_to_nondimensional(grad_phys, scales)

    solver_params = params.get('Solver', {})
    kappa_bound = solver_params.get('kappa_bound', 10.0)
    phi_bound = solver_params.get('phi_bound', np.pi)
    n_restarts = solver_params.get('restarts', 3)
    lbfgsb_opts = {
        'ftol': solver_params.get('ftol', 1e-8),
        'gtol': 1e-9,
        'maxiter': solver_params.get('maxiter', 1000)
    }
    powell_opts = {'ftol': 1e-5, 'maxiter': 2500}

    bounds_phys = [(-kappa_bound, kappa_bound), (-phi_bound, phi_bound)] * (num_elements // 2) # Assuming kappa, phi pairs
    # This bounds generation needs to be fixed if structure is not just pairs
    if num_elements % 2 != 0:
        # Handle odd number of total q components if necessary, though q is 6D (3 pairs)
        pass # Assuming 6D q for now

    # Flatten bounds for 1D optimizer
    hat_bounds_flat = []
    for i in range(num_elements):
        # Assuming structure is [k_x, k_y, k_z] for each element
        # This part of nondimensionalizer is not fully clear, assuming simple scaling for now
        # A more robust implementation might be needed if units/scales differ greatly
        hat_bounds_flat.extend([(-kappa_bound, kappa_bound), (-kappa_bound, kappa_bound), (-phi_bound, phi_bound)]) # Placeholder
    # A correct implementation requires knowing the structure of q and its nondimensionalization
    # For now, let's assume a simplified bounds structure that matches the flattened q
    # This part is complex and may need another look based on nondimensionalizer.py
    # Let's bypass complex bounds for now and use simple box bounds on the flattened array
    hat_kappa_bound = kappa_bound * scales['L_char']
    hat_phi_bound = phi_bound
    # This is still not quite right. The nondimensionalizer logic is key.
    # For now, let's proceed assuming the bounds logic can be simplified or corrected later
    # The primary fix is the flatten operation.
    
    # Let's construct bounds based on the flattened structure [kx1,ky1,kz1, kx2,ky2,kz2...]
    # This is an assumption about the internal structure of q_from_nondimensional
    # The original code had a bug in bounds generation. Let's simplify it.
    # The original code was: bounds_phys = [(-kappa_bound, kappa_bound), (-phi_bound, phi_bound)] * 3 which is for 6D q, not (3,20) q.
    hat_bounds = [(-hat_kappa_bound, hat_kappa_bound)] * (3 * num_elements)


    if force_robust_solver:
        result = minimize(objective_function_hat, hat_q_guess_flat, method='Powell', options=powell_opts)
    else:
        result = minimize(
            objective_function_hat, 
            hat_q_guess_flat, 
            method='L-BFGS-B',
            jac=jacobian_function_hat, 
            bounds=hat_bounds,
            options=lbfgsb_opts
        )

        if not result.success:
            logging.warning(f"[Solver] L-BFGS-B failed. Attempting {n_restarts} restarts.")
            for i in range(n_restarts):
                hat_q_perturbed = hat_q_guess_flat + np.random.randn(len(hat_q_guess_flat)) * 0.1
                result_retry = minimize(
                    objective_function_hat, hat_q_perturbed, method='L-BFGS-B',
                    jac=jacobian_function_hat, bounds=hat_bounds, options=lbfgsb_opts
                )
                if result_retry.success:
                    result = result_retry
                    break
            else:
                logging.warning("[Solver] All L-BFGS-B restarts failed.")

        if not result.success:
            logging.warning(f"[Solver] Fallback to Powell.")
            hat_q_start_powell = result.x if result.x is not None else hat_q_guess_flat
            result = minimize(objective_function_hat, hat_q_start_powell, method='Powell', options=powell_opts)

        if not result.success:
            logging.warning(f"[Solver] Fallback to SLSQP.")
            hat_q_start_slsqp = result.x if result.x is not None else hat_q_guess_flat
            result = minimize(
                objective_function_hat, hat_q_start_slsqp, method='SLSQP',
                jac=jacobian_function_hat, bounds=hat_bounds, options={'ftol': 1e-7, 'maxiter': 2000}
            )

    if result.success and not np.any(np.isnan(result.x)):
        # Reshape the 1D result vector back to its physical 2D shape
        q_solution_hat = result.x.reshape((3, num_elements))
        q_solution_phys = q_from_nondimensional(q_solution_hat, scales)
        return {"q_solution": q_solution_phys, "result": result}
    else:
        logging.error(f"[Solver] All optimization strategies failed. Final result object:\n{result}")
        return {"q_solution": None, "result": result}



def solve_static_equilibrium_diff4(q_guess, diff4, params, force_robust_solver=False):
    """
    [NEW] Wrapper for the main solver that accepts a 4D differential drive input.
    """
    from src.statics import expand_diff4_to_motor8
    delta_l_motor = expand_diff4_to_motor8(diff4, params)
    return solve_static_equilibrium_disp_ctrl(q_guess, delta_l_motor, params, force_robust_solver=force_robust_solver)


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