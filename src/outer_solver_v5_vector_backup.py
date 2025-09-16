"""
本模块是求解逆运动学问题的主入口。

V5.0 版本更新:
- [方向三] 优化目标重构为(位姿误差+弹性力梯度)，实现物理一致性和数值稳定性。
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import logging
import time

# 本地项目模块导入
from src.kinematics import forward_kinematics, calculate_kinematic_jacobian_analytical
from src.statics import (
    calculate_hessian_analytical, 
    calculate_actuation_jacobian_analytical,
    calculate_elastic_gradient_analytical,
    calculate_elastic_hessian_analytical # 新增导入
)
from src.solver import solve_static_equilibrium
from src.utils.read_config import load_config
from src.nondimensionalizer import (
    get_characteristic_scales, 
    tau_from_nondimensional, tau_to_nondimensional
)

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_pose_error(T_actual, T_target):
    pos_error = T_actual[:3, 3] - T_target[:3, 3]
    R_error = T_actual[:3, :3] @ T_target[:3, :3].T
    orientation_error = Rotation.from_matrix(R_error).as_rotvec()
    return np.concatenate([pos_error, orientation_error])

def calculate_task_jacobian(tensions_si, q_guess_si, params_si):
    try:
        q_eq_si = solve_static_equilibrium(q_guess_si, tensions_si, params_si)
        H_si = calculate_hessian_analytical(q_eq_si, tensions_si, params_si)
        J_act_si = calculate_actuation_jacobian_analytical(q_eq_si, params_si)
        C_si = -J_act_si.T
        J_kin_si = calculate_kinematic_jacobian_analytical(q_eq_si, params_si)
        H_inv_si = np.linalg.inv(H_si)
        dq_dtau_si = -H_inv_si @ C_si
        J_task_si = J_kin_si @ dq_dtau_si
        return J_task_si
    except np.linalg.LinAlgError:
        logging.warning("Hessian矩阵是奇异的,无法计算任务雅可比.返回一个零矩阵.")
        return np.zeros((6, 8))

def residual_function_nondimensional(
    hat_tau_flat, T_target_si, q_guess_si, params_si, scales, 
    hat_previous_tau_flat, smoothing_weight, grad_energy_weight, pose_error_weight
):
    """
    [V5.0] 残差函数 - 优化目标: (位姿误差 + 弹性力梯度 + 平滑)
    """
    hat_tensions = {'tensions_short': hat_tau_flat[:4], 'tensions_long': hat_tau_flat[4:]}
    tensions_si = tau_from_nondimensional(hat_tensions, scales)
    q_eq_si = solve_static_equilibrium(q_guess_si, tensions_si, params_si)
    T_actual_si, _ = forward_kinematics(q_eq_si, params_si)
    
    # --- 1. 位姿误差项 (6D) ---
    error_vector_si = calculate_pose_error(T_actual_si, T_target_si)
    hat_error_vector = np.copy(error_vector_si)
    hat_error_vector[:3] /= scales['L_char']
    pose_error_term = hat_error_vector * pose_error_weight

    # --- 2. 平滑项 (8D) ---
    if smoothing_weight > 0 and hat_previous_tau_flat is not None:
        smoothing_term = np.sqrt(smoothing_weight) * (hat_tau_flat - hat_previous_tau_flat)
    else:
        smoothing_term = np.zeros_like(hat_tau_flat)

    # --- 3. 弹性力梯度项 (7D) ---
    if grad_energy_weight > 0:
        grad_E_si = calculate_elastic_gradient_analytical(q_eq_si, params_si)
        hat_grad_E = np.copy(grad_E_si)
        hat_grad_E[0] /= scales['F_char']
        hat_grad_E[1:] /= (scales['F_char'] * scales['L_char'])
        grad_energy_term = np.sqrt(grad_energy_weight) * hat_grad_E
    else:
        grad_energy_term = np.zeros(7)

    return np.concatenate([pose_error_term, smoothing_term, grad_energy_term])

def jacobian_function_nondimensional(
    hat_tau_flat, T_target_si, q_guess_si, params_si, scales,
    hat_previous_tau_flat, smoothing_weight, grad_energy_weight, pose_error_weight
):
    """
    [V5.0] 雅可比函数 - 优化目标: (位姿误差 + 弹性力梯度 + 平滑)
    """
    hat_tensions = {'tensions_short': hat_tau_flat[:4], 'tensions_long': hat_tau_flat[4:]}
    tensions_si = tau_from_nondimensional(hat_tensions, scales)
    q_eq_si = solve_static_equilibrium(q_guess_si, tensions_si, params_si)

    # --- 1. 位姿误差项的雅可比 ---
    J_task_si = calculate_task_jacobian(tensions_si, q_eq_si, params_si)
    F_char, L_char = scales['F_char'], scales['L_char']
    scaling_matrix_e = np.diag([1/L_char] * 3 + [1.0] * 3)
    J_pose_nondim = (scaling_matrix_e @ J_task_si) * F_char * pose_error_weight

    # --- 2. 平滑项的雅可比 ---
    if smoothing_weight > 0 and hat_previous_tau_flat is not None:
        J_smoothing = np.sqrt(smoothing_weight) * np.identity(8)
    else:
        J_smoothing = np.zeros((8, 8))

    # --- 3. 弹性力梯度项的雅可比 ---
    if grad_energy_weight > 0:
        H_total_si = calculate_hessian_analytical(q_eq_si, tensions_si, params_si)
        H_elastic_si = calculate_elastic_hessian_analytical(q_eq_si, params_si)
        C_si = -calculate_actuation_jacobian_analytical(q_eq_si, params_si).T
        try:
            H_inv_si = np.linalg.inv(H_total_si)
            dq_dtau_si = -H_inv_si @ C_si
            d_gradE_dtau_si = H_elastic_si @ dq_dtau_si
            
            # 无量纲化
            scaling_vec = np.concatenate([[1/F_char], np.full(6, 1/(F_char*L_char))])
            d_hat_gradE_dtau_si = d_gradE_dtau_si * scaling_vec[:, np.newaxis]
            d_hat_gradE_dhat_tau = d_hat_gradE_dtau_si * F_char

            J_grad_energy = np.sqrt(grad_energy_weight) * d_hat_gradE_dhat_tau
        except np.linalg.LinAlgError:
            J_grad_energy = np.zeros((7, 8))
    else:
        J_grad_energy = np.zeros((7, 8))

    return np.vstack([J_pose_nondim, J_smoothing, J_grad_energy])

def solve_inverse_kinematics(
    T_target, initial_tau_guess, q_guess, params, 
    previous_tau=None, smoothing_weight=5e-6, use_jacobian=False,
    grad_energy_weight=0.0, pose_error_weight=1.0
):
    """
    [V5.0] 最终版求解器
    """
    scales = get_characteristic_scales(params)
    hat_tau_guess = tau_to_nondimensional(initial_tau_guess, scales)
    hat_tau_guess_flat = np.concatenate([hat_tau_guess['tensions_short'], hat_tau_guess['tensions_long']])
    
    hat_previous_tau_flat = None
    if previous_tau is not None:
        hat_previous_tau = tau_to_nondimensional(previous_tau, scales)
        hat_previous_tau_flat = np.concatenate([hat_previous_tau['tensions_short'], hat_previous_tau['tensions_long']])

    bounds_si = ([0]*8, [100]*8)
    hat_bounds = (np.array(bounds_si[0]), np.array(bounds_si[1]) / scales['F_char'])

    jac_func = jacobian_function_nondimensional if use_jacobian else '2-point'
    
    result = least_squares(
        fun=residual_function_nondimensional,
        x0=hat_tau_guess_flat,
        jac=jac_func,
        args=(T_target, q_guess, params, scales, hat_previous_tau_flat, smoothing_weight, grad_energy_weight, pose_error_weight),
        bounds=hat_bounds,
        method='trf',
        verbose=0,
        ftol=1e-6, xtol=1e-6, gtol=1e-6
    )

    if result.success:
        hat_optimal_tau = {'tensions_short': result.x[:4], 'tensions_long': result.x[4:]}
        optimal_tau = tau_from_nondimensional(hat_optimal_tau, scales)
        result.x = np.concatenate([optimal_tau['tensions_short'], optimal_tau['tensions_long']])

    return result

def __main__():
    print("--- 外循环求解器模块功能自检 (V5.0) ---")
    config_params = load_config('config/config.json')
    q_guess_si = np.array(config_params['Initial_State']['q0'])
    np.set_printoptions(precision=4, suppress=True)

    tau_test_si = {'tensions_short': np.full(4, 5.0), 'tensions_long': np.full(4, 5.0)}
    
    print("\n--- 测试任务雅可比函数 ---")
    J_task = calculate_task_jacobian(tau_test_si, q_guess_si, config_params)
    print(f"计算得到的任务雅可比矩阵 (6x8):\n{J_task}")
    if J_task.shape == (6, 8):
        print("【测试通过】: 任务雅可比矩阵维度正确.")
    else:
        print("【测试失败】: 任务雅可比矩阵维度错误.")

if __name__ == '__main__':
    __main__()
