# verification/verify_jacobian.py

import numpy as np
import sys
import os
import logging
import time
from scipy.spatial.transform import Rotation

# --- 核心依赖 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.read_config import load_config
from src.kinematics import forward_kinematics, calculate_kinematic_jacobian_analytical
from src.statics import (
    calculate_actuation_jacobian_analytical,
    calculate_hessian_analytical_disp_ctrl,
    calculate_drive_mapping,
    calculate_hessian_disp_ctrl_high_performance
)
from src.solver import solve_static_equilibrium_disp_ctrl

def solve_static_equilibrium_disp_ctrl_robust(q_guess, delta_l_motor, params):
    """ A robust version of the solver for verification purposes. """
    from scipy.optimize import minimize
    
    def objective_function(q):
        # This is a simplified copy of the logic in solver.py
        # In a real scenario, you would refactor solver.py to allow method switching
        from src.statics import calculate_elastic_potential_energy, calculate_gravity_potential_energy, calculate_actuation_potential_energy_disp_ctrl
        U_e = calculate_elastic_potential_energy(q, params)
        U_g = calculate_gravity_potential_energy(q, params)
        U_a = calculate_actuation_potential_energy_disp_ctrl(q, delta_l_motor, params)
        return U_e + U_g + U_a

    def jacobian_function(q):
        from src.statics import calculate_elastic_gradient_analytical, calculate_gravity_gradient_analytical, calculate_actuation_jacobian_analytical, calculate_drive_mapping
        grad_e = calculate_elastic_gradient_analytical(q, params)
        grad_g = calculate_gravity_gradient_analytical(q, params)
        k_cable = params['Drive_Properties']['cable_stiffness']
        delta_l_short_robot, delta_l_long_robot = calculate_drive_mapping(q, params)
        delta_l_robot = np.concatenate([delta_l_short_robot, delta_l_long_robot])
        stretch = delta_l_motor - delta_l_robot
        beta = 1000
        d_smooth_max_zero_d_stretch = 0.5 * (1 + stretch / np.sqrt(stretch**2 + (1/beta)**2))
        stretch_tensioned = (stretch + np.sqrt(stretch**2 + (1/beta)**2)) / 2
        J_act = calculate_actuation_jacobian_analytical(q, params)
        grad_a = -k_cable * J_act.T @ (stretch_tensioned * d_smooth_max_zero_d_stretch)
        return grad_e + grad_g + grad_a

    bounds = [(-40, 40), (-np.inf, np.inf)] * 3
    result = minimize(
        objective_function, 
        q_guess, 
        method='SLSQP', 
        jac=jacobian_function, 
        bounds=bounds,
        options={'ftol': 1e-9, 'maxiter': 500}
    )
    if result.success and not np.any(np.isnan(result.x)):
        return result.x
    else:
        return None

def calculate_pose_error(T_actual, T_target):
    pos_error = T_actual[:3, 3] - T_target[:3, 3]
    R_error = T_actual[:3, :3] @ T_target[:3, :3].T
    try:
        orientation_error = Rotation.from_matrix(R_error).as_rotvec()
    except ValueError:
        orientation_error = np.zeros(3)
    return np.concatenate([pos_error, orientation_error])

def calculate_task_jacobian_numerical(delta_l, q_guess, config, target_pose, epsilon=1e-7):
    """ [V-Final 终极修复] 使用纯数值方法，计算最可靠的任务雅可比 """
    q_eq = solve_static_equilibrium_disp_ctrl_robust(q_guess, delta_l, config)
    if q_eq is None:
        return None, None

    T_base, _ = forward_kinematics(q_eq, config)
    base_pose_error = calculate_pose_error(T_base, target_pose)

    num_outputs = 6
    num_inputs = len(delta_l)
    J_numerical = np.zeros((num_outputs, num_inputs))

    for i in range(num_inputs):
        # 正向扰动
        delta_l_plus = delta_l.copy()
        delta_l_plus[i] += epsilon
        q_eq_plus = solve_static_equilibrium_disp_ctrl_robust(q_eq, delta_l_plus, config)
        if q_eq_plus is None:
            return None, q_eq
        T_plus, _ = forward_kinematics(q_eq_plus, config)
        pose_error_plus = calculate_pose_error(T_plus, target_pose)

        # 反向扰动
        delta_l_minus = delta_l.copy()
        delta_l_minus[i] -= epsilon
        q_eq_minus = solve_static_equilibrium_disp_ctrl_robust(q_eq, delta_l_minus, config)
        if q_eq_minus is None:
            return None, q_eq
        T_minus, _ = forward_kinematics(q_eq_minus, config)
        pose_error_minus = calculate_pose_error(T_minus, target_pose)

        # 中心差分计算twist向量的变化
        column = (pose_error_plus - pose_error_minus) / (2 * epsilon)
        J_numerical[:, i] = column

    return J_numerical, q_eq

def calculate_task_jacobian_disp_ctrl(delta_l, q_guess, params):
    """ [V-Final 终极修复] 计算纯位移控制下的任务雅可比 J_task = dT/d(dl) """
    q_eq = solve_static_equilibrium_disp_ctrl(q_guess, delta_l, params)
    if q_eq is None:
        return None, None
    
    # 1. 获取所有正确的“零件”
    # [核心] 使用最高性能的Hessian近似
    H_disp = calculate_hessian_disp_ctrl_high_performance(q_eq, delta_l, params)
    k_cable = params['Drive_Properties']['cable_stiffness']
    J_act = calculate_actuation_jacobian_analytical(q_eq, params)
    J_kin = calculate_kinematic_jacobian_analytical(q_eq, params)

    # 2. 计算混合偏导数矩阵 C = d(grad_U)/d(delta_l)
    delta_l_robot_short, delta_l_robot_long = calculate_drive_mapping(q_eq, params)
    delta_l_robot = np.concatenate([delta_l_robot_short, delta_l_robot_long])
    stretch = delta_l - delta_l_robot
    
    # [核心] 使用平滑的导数
    beta = 1000
    d_smooth_max_zero_d_stretch = 0.5 * (1 + stretch / np.sqrt(stretch**2 + (1/beta)**2))
    stretch_tensioned = (stretch + np.sqrt(stretch**2 + (1/beta)**2)) / 2
    C = -k_cable * J_act.T @ np.diag(stretch_tensioned * d_smooth_max_zero_d_stretch)

    # 3. 组装雅可比
    damping = 1e-6 * np.eye(H_disp.shape[0])
    try:
        H_inv = np.linalg.inv(H_disp + damping)
    except np.linalg.LinAlgError:
        H_inv = np.linalg.pinv(H_disp + damping)
    
    dq_ddl = -H_inv @ C
    J_task = J_kin @ dq_ddl
    return J_task, q_eq

def test_jacobian(q_test, delta_l_test, config, target_pose):
    """  计算解析雅可比和数值雅可比，并进行对比 """
    print("\n--- 测试点 ---")
    print(f"q_guess: {q_test}")
    print(f"delta_l: {delta_l_test}")
    print("\n--- 1. 计算解析 (或半解析) 雅可比 J_ana ---")
    start_time = time.time()
    try:
        J_ana, q_eq = calculate_task_jacobian_disp_ctrl(delta_l_test, q_test, config)
        if J_ana is None:
            print("❌ 解析雅可比计算失败，内循环求解器未能收敛。\n")
            return False, None, None
        print(f"J_ana 计算完成 (耗时: {time.time() - start_time:.4f}s)")
        print(J_ana)
    except Exception as e:
        print(f"ERROR: J_ana 计算失败: {e}")
        return False, None, None

    print("\n--- 2. 计算数值雅可比“真值” J_num ---")
    start_time = time.time()
    J_num, q_eq = calculate_task_jacobian_numerical(delta_l_test, q_test, config, target_pose)
    if J_num is None:
        print("❌ 数值雅可比计算失败，内循环求解器未能收敛。\n")
        return False, J_ana, None
    print(f"J_num 计算完成 (耗时: {time.time() - start_time:.4f}s)")
    print(J_num)
    print("\n--- 3. 最终审判 (对比) ---")
    J_diff = J_ana - J_num
    diff_norm = np.linalg.norm(J_diff)
    relative_error = diff_norm / (np.linalg.norm(J_num) + 1e-9)
    
    print(f"差值矩阵 (J_analytical - J_numerical):\n{J_diff}")
    print(f"\n差值范数: {diff_norm:.6e}")
    print(f"相对误差: {relative_error:.4%}")

    if relative_error < 1e-4:
        print("\n【✅✅✅ 测试通过】: 解析法与数值“真值”高度一致！")
        return True, J_ana, J_num
    else:
        print("\n【❌❌❌ 测试失败】: 解析法与数值“真值”存在显著差异！")
        return False, J_ana, J_num

def main():
    print("--- 任务空间雅可比 (Task Jacobian) 最终审判脚本 ---")
    config = load_config('config/config.json')
    np.set_printoptions(precision=6, suppress=True, linewidth=200)

    # 选取一个非零的、更具一般性的测试点
    q_test = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    dl_test = np.full(8, 0.01)
    L_total = config['Geometry']['PSS_initial_length'] + config['Geometry']['CMS_proximal_length'] + config['Geometry']['CMS_distal_length']
    target_pose = np.array([
        [1, 0, 0, 0.05],
        [0, 1, 0, 0.05],
        [0, 0, 1, L_total * 0.7],
        [0, 0, 0, 1]
    ])
    
    test_passed, J_ana, J_num = test_jacobian(q_test, dl_test, config, target_pose)

    if test_passed:
        print("\n【✅✅✅ 最终测试结果】解析雅可比正确。")
    else:
        print("\n【❌❌❌ 最终测试结果】解析雅可比计算存在错误，需要修正！")

if __name__ == '__main__':
    main()
