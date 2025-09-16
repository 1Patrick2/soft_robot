# verification/verify_hessian_components.py

import numpy as np
import sys
import os

# [路径修正]
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.statics import (
    calculate_elastic_potential_energy,
    calculate_actuation_potential_energy,
    calculate_gravity_potential_energy,
    calculate_elastic_hessian_analytical,
    calculate_hessian_analytical # To extract H_A logic
)
from src.utils.read_config import load_config

def calculate_hessian_numerical_high_precision(q, energy_func, params, tensions=None, epsilon=1e-5):
    """
    高精度数值海森矩阵计算 (二阶中心差分)。
    """
    hessian = np.zeros((7, 7))
    for i in range(7):
        for j in range(i, 7): # 利用对称性，只计算上三角部分
            q_pp = q.copy(); q_pp[i] += epsilon; q_pp[j] += epsilon
            q_pm = q.copy(); q_pm[i] += epsilon; q_pm[j] -= epsilon
            q_mp = q.copy(); q_mp[i] -= epsilon; q_mp[j] += epsilon
            q_mm = q.copy(); q_mm[i] -= epsilon; q_mm[j] -= epsilon

            if tensions:
                U_pp = energy_func(q_pp, tensions, params)
                U_pm = energy_func(q_pm, tensions, params)
                U_mp = energy_func(q_mp, tensions, params)
                U_mm = energy_func(q_mm, tensions, params)
            else:
                U_pp = energy_func(q_pp, params)
                U_pm = energy_func(q_pm, params)
                U_mp = energy_func(q_mp, params)
                U_mm = energy_func(q_mm, params)

            hessian[i, j] = (U_pp - U_pm - U_mp + U_mm) / (4 * epsilon**2)
            if i != j:
                hessian[j, i] = hessian[i, j] # 对称填充
    return hessian

def get_analytical_H_A(q, tensions, params):
    """从 calculate_hessian_analytical 中提取 H_A 的计算逻辑。"""
    Lp, kp, phip, kc1, phic1, kc2, phic2 = q
    geo = params['Geometry']
    tau_short = tensions['tensions_short']
    tau_long = tensions['tensions_long']

    r_short, r_long = params['Geometry']['short_lines']['diameter_m'] / 2, params['Geometry']['long_lines']['diameter_m'] / 2
    alphas_short = np.deg2rad(geo['short_lines']['angles_deg'])
    alphas_long = np.deg2rad(geo['long_lines']['angles_deg'])
    Lc1 = geo['CMS_proximal_length']
    Lc2 = geo['CMS_distal_length']

    H_A = np.zeros((7, 7))
    cos_p_s = np.cos(phip - alphas_short)
    sin_p_s = np.sin(phip - alphas_short)
    cos_c1_s = np.cos(phic1 - alphas_short)
    sin_c1_s = np.sin(phic1 - alphas_short)
    cos_p_l = np.cos(phip - alphas_long)
    sin_p_l = np.sin(phip - alphas_long)
    cos_c1_l = np.cos(phic1 - alphas_long)
    sin_c1_l = np.sin(phic1 - alphas_long)
    cos_c2_l = np.cos(phic2 - alphas_long)
    sin_c2_l = np.sin(phic2 - alphas_long)

    H_A[0, 1] = - (np.sum(tau_short * r_short * cos_p_s) + np.sum(tau_long * r_long * cos_p_l))
    H_A[0, 2] = kp * (np.sum(tau_short * r_short * sin_p_s) + np.sum(tau_long * r_long * sin_p_l))
    H_A[1, 0] = H_A[0, 1]
    H_A[1, 2] = Lp * (np.sum(tau_short * r_short * sin_p_s) + np.sum(tau_long * r_long * sin_p_l))
    H_A[2, 0] = H_A[0, 2]
    H_A[2, 1] = H_A[1, 2]
    H_A[2, 2] = Lp * kp * (np.sum(tau_short * r_short * cos_p_s) + np.sum(tau_long * r_long * cos_p_l))
    H_A[3, 4] = Lc1 * (np.sum(tau_short * r_short * sin_c1_s) + np.sum(tau_long * r_long * sin_c1_l))
    H_A[4, 3] = H_A[3, 4]
    H_A[4, 4] = Lc1 * kc1 * (np.sum(tau_short * r_short * cos_c1_s) + np.sum(tau_long * r_long * cos_c1_l))
    H_A[5, 6] = Lc2 * np.sum(tau_long * r_long * sin_c2_l)
    H_A[6, 5] = H_A[5, 6]
    H_A[6, 6] = Lc2 * kc2 * np.sum(tau_long * r_long * cos_c2_l)
    return H_A

if __name__ == '__main__':
    config_params = load_config('config/config.json')
    np.set_printoptions(precision=8, suppress=True)

    q_test = np.random.rand(7) * 0.5 
    tensions_test = {
        'tensions_short': np.random.rand(4) * 20,
        'tensions_long': np.random.rand(4) * 20
    }
    
    print("--- Hessian Components High-Precision Verification ---")
    print(f"\nTest q: {q_test}")
    print(f"Test tensions: {tensions_test}")

    # 1. 弹性海森 H_E
    print("\n--- 1. Verifying Elastic Hessian (H_E) ---")
    H_E_analytical = calculate_elastic_hessian_analytical(q_test, config_params)
    H_E_numerical = calculate_hessian_numerical_high_precision(q_test, calculate_elastic_potential_energy, config_params)
    diff_E = np.linalg.norm(H_E_analytical - H_E_numerical)
    print(f"H_E Analytical:\n{H_E_analytical}")
    print(f"H_E Numerical:\n{H_E_numerical}")
    print(f"==> H_E Difference Norm: {diff_E:.8f}")
    if diff_E < 1e-5: print("==> H_E PASSED")
    else: print("==> H_E FAILED")

    # 2. 驱动海森 H_A
    print("\n--- 2. Verifying Actuation Hessian (H_A) ---")
    H_A_analytical = get_analytical_H_A(q_test, tensions_test, config_params)
    H_A_numerical = calculate_hessian_numerical_high_precision(q_test, calculate_actuation_potential_energy, config_params, tensions=tensions_test)
    diff_A = np.linalg.norm(H_A_analytical - H_A_numerical)
    print(f"H_A Analytical:\n{H_A_analytical}")
    print(f"H_A Numerical:\n{H_A_numerical}")
    print(f"==> H_A Difference Norm: {diff_A:.8f}")
    if diff_A < 1e-5: print("==> H_A PASSED")
    else: print("==> H_A FAILED")

    # 3. 重力海森 H_G
    print("\n--- 3. Verifying Gravity Hessian (H_G) ---")
    # 从主函数中提取H_G的计算方法（解析梯度差分法）
    from src.statics import calculate_gravity_gradient_analytical
    grad_g_base = calculate_gravity_gradient_analytical(q_test, config_params)
    H_G_analytical_grad_diff = np.zeros((7, 7))
    epsilon_grad = 1e-7
    for i in range(7):
        q_plus = q_test.copy(); q_plus[i] += epsilon_grad
        grad_g_plus = calculate_gravity_gradient_analytical(q_plus, config_params)
        H_G_analytical_grad_diff[:, i] = (grad_g_plus - grad_g_base) / epsilon_grad
    
    H_G_numerical = calculate_hessian_numerical_high_precision(q_test, calculate_gravity_potential_energy, config_params)
    diff_G = np.linalg.norm(H_G_analytical_grad_diff - H_G_numerical)
    print(f"H_G Analytical (from grad diff):\n{H_G_analytical_grad_diff}")
    print(f"H_G Numerical (from energy diff):\n{H_G_numerical}")
    print(f"==> H_G Difference Norm: {diff_G:.8f}")
    if diff_G < 1e-5: print("==> H_G PASSED")
    else: print("==> H_G FAILED")
