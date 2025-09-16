# verification/debug_statics.py

import numpy as np
import sys
import os
import logging
import matplotlib.pyplot as plt

# --- 核心依赖 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.statics import (
    calculate_elastic_potential_energy,
    calculate_gravity_potential_energy,
    calculate_elastic_gradient_analytical,
    calculate_gravity_gradient_analytical,
    calculate_actuation_gradient_analytical, # <--- 修正：导入新的合并函数
    smooth_max_zero,
    calculate_drive_mapping
)
from src.utils.read_config import load_config
from scipy.optimize import approx_fprime

logging.basicConfig(level=logging.INFO, format='%(message)s')


def compare_gradients(name, analytical_grad, numerical_grad):
    diff_norm = np.linalg.norm(analytical_grad - numerical_grad)
    ref_norm = np.linalg.norm(numerical_grad)
    rel_error = diff_norm / (ref_norm + 1e-12)
    print(f"\n- {name} 梯度:")
    print(f"  - 解析法: {analytical_grad}")
    print(f"  - 数值法: {numerical_grad}")
    print(f"  - 相对误差: {rel_error:.6e}")
    if rel_error < 1e-5:
        print("  - ✅ 通过")
    else:
        print(f"  - ❌ 失败 (差值: {analytical_grad - numerical_grad})")
    return rel_error < 1e-5

if __name__ == '__main__':
    print("--- 静力学模块深度诊断 (debug_statics.py) ---")
    config = load_config('config/config.json')
    np.set_printoptions(precision=8, suppress=True)

    # --- 测试条件 ---
    delta_l_motor_test = np.array(
        [ 0.000004,  0.097598, -0.000016, -0.000006, -0.000014, -0.00001,   0.00001,  0.000009]
    )
    q_test_near_zero = np.full(6, 1e-5)

    print(f"\n--- 测试条件 ---")
    print(f"  - 测试构型 q: {q_test_near_zero}")
    print(f"  - 电机位移 Δl_motor: {delta_l_motor_test}")

    # --- 1. 分项梯度验证 ---
    print("\n--- 1. 分项梯度验证 (在 q_test_near_zero 点) ---")
    eps = 1e-8
    
    # --- 驱动能量 (Actuation) 合并测试 ---
    print("\n--- 驱动能量 (Actuation) 合并测试 ---")

    # 1.1 U_actuation (U_cable + U_pretension) 梯度
    grad_act_ana = calculate_actuation_gradient_analytical(q_test_near_zero, delta_l_motor_test, config)
    
    def actuation_energy_func(q):
        # 同时计算缆绳弹性和预紧力势能
        k_cable = config['Drive_Properties']['cable_stiffness']
        f_pre = config['Drive_Properties'].get('pretension_force_N', 0.0)
        delta_l_robot = calculate_drive_mapping(q, config)
        
        stretch = delta_l_motor_test - delta_l_robot
        U_cable = 0.5 * k_cable * np.sum(smooth_max_zero(stretch)**2)
        U_pretension = -f_pre * np.sum(delta_l_robot)
        
        return U_cable + U_pretension

    grad_act_num = approx_fprime(q_test_near_zero, actuation_energy_func, eps)
    compare_gradients("驱动总能量 (U_actuation)", grad_act_ana, grad_act_num)

    # --- 重力梯度测试 ---
    print("\n--- 重力能量 (Gravity) 测试 ---")
    grad_g_ana = calculate_gravity_gradient_analytical(q_test_near_zero, config)
    grad_g_num = approx_fprime(q_test_near_zero, lambda q: calculate_gravity_potential_energy(q, config), eps)
    compare_gradients("重力势能 (Gravity)", grad_g_ana, grad_g_num)
