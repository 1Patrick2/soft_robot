# verification/verify_gravity_gradient.py

import numpy as np
import sys
import os
import logging

# --- 核心依赖 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kinematics import forward_kinematics, calculate_com_jacobians_analytical
from src.statics import calculate_gravity_potential_energy, calculate_gravity_gradient_analytical
from src.utils.read_config import load_config
from scipy.optimize import approx_fprime

logging.basicConfig(level=logging.INFO, format='%(message)s')


def numeric_grad_central(f, q, h=1e-6):
    """A more robust, manually implemented central difference gradient."""
    n = len(q)
    grad = np.zeros(n)
    for i in range(n):
        qp = q.copy(); qm = q.copy()
        qp[i] += h; qm[i] -= h
        fp = f(qp); fm = f(qm)
        grad[i] = (fp - fm) / (2*h)
    return grad


def calculate_com_jacobians_numerical(q, params, epsilon=1e-7):
    """Numerically compute the Jacobian of the center of mass for all segments."""
    J_com_num = {
        'pss': np.zeros((3, 6)),
        'cms1': np.zeros((3, 6)),
        'cms2': np.zeros((3, 6))
    }
    
    for i in range(6):
        h = epsilon * max(1.0, abs(q[i]))
        q_plus = q.copy(); q_plus[i] += h
        q_minus = q.copy(); q_minus[i] -= h

        _, com_pos_plus = forward_kinematics(q_plus, params)
        _, com_pos_minus = forward_kinematics(q_minus, params)

        for seg in ['pss', 'cms1', 'cms2']:
            J_com_num[seg][:, i] = (com_pos_plus[seg] - com_pos_minus[seg]) / (2 * h)
            
    return J_com_num

if __name__ == '__main__':
    print("--- 重力梯度及质心雅可比专项验证 ---")
    config = load_config('config/config.json')
    np.set_printoptions(precision=6, suppress=True, linewidth=150)

    q_test = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    print(f"\n--- 测试条件 ---")
    print(f"  - 测试构型 q: {q_test}")

    # --- 1. 验证质心雅可比 (J_com) ---
    print("\n--- 1. 质心雅可比 (J_com) 验证 ---")
    J_com_ana = calculate_com_jacobians_analytical(q_test, config)
    J_com_num = calculate_com_jacobians_numerical(q_test, config)
    
    has_error = False
    for seg in ['pss', 'cms1', 'cms2']:
        diff = np.linalg.norm(J_com_ana[seg] - J_com_num[seg])
        print(f"\n- J_com for segment '{seg}':")
        print(f"  - Analytical:\n{J_com_ana[seg]}")
        print(f"  - Numerical:\n{J_com_num[seg]}")
        print(f"  - Difference Norm: {diff:.6e}")
        if diff > 1e-5:
            print("  - ❌ 失败")
            has_error = True
        else:
            print("  - ✅ 通过")

    if has_error:
        print("\n[J_com 结论] ❌ 质心雅可比解析计算存在错误！这是重力梯度错误的直接原因。")
    else:
        print("\n[J_com 结论] ✅ 质心雅可比验证通过。")

    # --- 2. 验证重力梯度 ---
    print("\n--- 2. 重力梯度 (grad_g) 验证 ---")
    grad_g_ana = calculate_gravity_gradient_analytical(q_test, config)
    grad_g_num = numeric_grad_central(lambda q: calculate_gravity_potential_energy(q, config), q_test, h=1e-6)
    
    diff_norm = np.linalg.norm(grad_g_ana - grad_g_num)
    rel_error = diff_norm / (np.linalg.norm(grad_g_num) + 1e-12)
    
    print(f"\n- 重力梯度:")
    print(f"  - 解析法: {grad_g_ana}")
    print(f"  - 数值法: {grad_g_num}")
    print(f"  - 相对误差: {rel_error:.6e}")

    if rel_error < 1e-5:
        print("  - ✅ 结论: 重力梯度解析计算正确。")
    else:
        print("  - ❌ 结论: 重力梯度解析计算存在错误。")
