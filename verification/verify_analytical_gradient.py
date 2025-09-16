import numpy as np
import sys
import os
from scipy.optimize import approx_fprime

# --- 核心依赖 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.read_config import load_config
from src.statics import (
    calculate_gradient_disp_ctrl,
    calculate_total_potential_energy_disp_ctrl
)

def run_final_gradient_verification():
    """
    对最终的、完整的解析梯度函数进行最终的单元测试。
    """
    print("--- 最终完整解析梯度验证 ---")

    # 1. 加载配置并设置测试参数
    config_params = load_config('config/config.json')
    np.set_printoptions(precision=8, suppress=True)

    q_test = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    # 使用一个非零的电机位移来测试驱动项
    delta_l_motor_test = np.array([0.001, -0.001, 0.002, -0.002, 0.003, -0.003, 0.004, -0.004])

    print(f"\n--- 测试条件 ---")
    print(f"  - 测试构型 q: {q_test}")
    print(f"  - 电机位移 Δl: {delta_l_motor_test}")

    # 2. 计算我们自己实现的“全解析”梯度
    grad_implemented = calculate_gradient_disp_ctrl(q_test, delta_l_motor_test, config_params)

    # 3. 计算“真理”梯度 (总势能的纯数值梯度)
    total_energy_func = lambda q: calculate_total_potential_energy_disp_ctrl(q, delta_l_motor_test, config_params)
    grad_truth = approx_fprime(q_test, total_energy_func, 1e-8)

    # 4. 对比和分析
    print(f"\n--- 结果对比 ---")
    print(f"  - 我方全梯度: {grad_implemented}")
    print(f"  - 真理全梯度: {grad_truth}")

    diff_vector = grad_implemented - grad_truth
    diff_norm = np.linalg.norm(diff_vector)

    print(f"\n  - 差值范数: {diff_norm:.8f}")

    if diff_norm < 1e-5:
        print("\n  - ✅✅✅ 【最终测试通过】: 完整梯度函数正确！核心物理模型已完全重建！")
    else:
        print("\n  - ❌❌❌ 【最终测试失败】: 梯度计算存在最终错误！")
        print(f"  - 差值向量 (我方 - 真理): {diff_vector}")

if __name__ == '__main__':
    run_final_gradient_verification()