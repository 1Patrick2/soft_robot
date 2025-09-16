# verification/verify_task_jacobian.py

import numpy as np
import sys
import os
import logging

# --- 核心依赖 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.read_config import load_config
from src.kinematics import forward_kinematics
from src.solver import solve_static_equilibrium_disp_ctrl
from src.outer_solver import calculate_pose_error, calculate_task_jacobian_disp_ctrl

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def verify_task_jacobian():
    """
    通过将解析雅可比与高精度中心差分法计算的数值雅可比进行比较，
    来严格验证 `calculate_task_jacobian_disp_ctrl` 函数的正确性。
    """
    print("--- 任务空间雅可比 (J_task = dT/d(dl)) 验证脚本 ---")
    
    # 1. 设置测试环境
    config = load_config('config/config.json')
    q0_6d = np.array(config['Initial_State']['q0'])[1:]
    np.set_printoptions(precision=6, suppress=True)

    # 选择一个非零的、非对称的测试点，以避免特殊情况
    delta_l_test = np.array([0.02, 0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0]) # Larger, more directed test vector
    q_guess_test = np.random.rand(6) * 0.1 # Use a non-zero random initial guess

    # 目标位姿的具体值不影响雅可比的计算，但需要一个作为参考
    L_total = config['Geometry']['PSS_initial_length'] + config['Geometry']['CMS_proximal_length'] + config['Geometry']['CMS_distal_length']
    target_pose = np.array([
        [1, 0, 0, 0.01],
        [0, 1, 0, 0.02],
        [0, 0, 1, L_total * 0.9],
        [0, 0, 0, 1]
    ])

    print(f"\n测试点 (delta_l):\n{delta_l_test}")

    # 2. 计算解析雅可比
    print("\n--- 正在计算解析雅可比... ---")
    J_analytical, q_eq_base = calculate_task_jacobian_disp_ctrl(delta_l_test, q_guess_test, config)
    if J_analytical is None:
        print("❌ 解析雅可比计算失败，内循环求解器未能收敛。无法继续验证。\n")
        return
    print("✅ 解析雅可比计算成功。")

    # 3. 计算高精度数值雅可比
    print("\n--- 正在计算数值雅可比 (高精度中心差分)... ---")
    epsilon = 1e-7
    num_outputs = 6 # pose_error is 6D
    num_inputs = len(delta_l_test) # delta_l is 8D
    J_numerical = np.zeros((num_outputs, num_inputs))

    # 计算基准点的位姿误差
    T_base, _ = forward_kinematics(q_eq_base, config)
    base_pose_error = calculate_pose_error(T_base, target_pose)

    for i in range(num_inputs):
        # 正向扰动
        delta_l_plus = delta_l_test.copy()
        delta_l_plus[i] += epsilon
        q_eq_plus = solve_static_equilibrium_disp_ctrl(q_eq_base, delta_l_plus, config)
        if q_eq_plus is None:
            print(f"❌ 数值计算失败（正向扰动 i={i}），内循环求解器未能收敛。\n")
            return
        T_plus, _ = forward_kinematics(q_eq_plus, config)
        pose_error_plus = calculate_pose_error(T_plus, target_pose)

        # 反向扰动
        delta_l_minus = delta_l_test.copy()
        delta_l_minus[i] -= epsilon
        q_eq_minus = solve_static_equilibrium_disp_ctrl(q_eq_base, delta_l_minus, config)
        if q_eq_minus is None:
            print(f"❌ 数值计算失败（反向扰动 i={i}），内循环求解器未能收敛。\n")
            return
        T_minus, _ = forward_kinematics(q_eq_minus, config)
        pose_error_minus = calculate_pose_error(T_minus, target_pose)
        
        # 中心差分
        column = (pose_error_plus - pose_error_minus) / (2 * epsilon)
        J_numerical[:, i] = column
    
    print("✅ 数值雅可比计算成功。")

    # 4. 对比和结论
    print("\n--- 验证结果 ---")
    diff = J_analytical - J_numerical
    norm_diff = np.linalg.norm(diff)
    relative_norm_diff = norm_diff / np.linalg.norm(J_numerical)

    print(f"解析雅可比 (J_analytical):\n{J_analytical}")
    print(f"\n数值雅可比 (J_numerical):\n{J_numerical}")
    print(f"\n差值矩阵 (J_analytical - J_numerical):\n{diff}")
    print(f"\n差值范数 (||J_analytical - J_numerical||): {norm_diff:.6e}")
    print(f"相对差值范数: {relative_norm_diff:.6e}")

    # 容差可以设置得比较严格，因为我们的数值方法精度很高
    if relative_norm_diff < 1e-4:
        print("\n✅✅✅ [通过] 解析雅可比与数值雅可比完全一致！")
    else:
        print("\n❌❌❌ [失败] 解析雅可比与数值雅可比存在显著差异！")

if __name__ == '__main__':
    verify_task_jacobian()