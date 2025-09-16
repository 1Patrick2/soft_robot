import numpy as np
import sys
import os
from scipy.optimize import approx_fprime

# --- 核心依赖 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.read_config import load_config
from src.statics import (
    calculate_total_potential_energy_disp_ctrl,
    calculate_gradient_disp_ctrl
)

def run_gradient_verification():
    """
    对我们实现的梯度函数进行最终的“真理”审计。
    通过对比我们自己的梯度函数和Scipy的高精度数值梯度，来验证总势能函数的正确性。
    """
    print("--- 梯度函数真理验证 (Gradient Truth Verification) ---")
    
    # 1. 加载配置并设置测试参数
    config_params = load_config('config/config.json')
    np.set_printoptions(precision=8, suppress=True)
    
    q_test = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    delta_l_motor_test = np.random.rand(8) * 0.002

    print(f"\n--- 测试条件 ---")
    print(f"  - 测试构型 q: {q_test}")
    print(f"  - 电机位移 Δl_motor: {delta_l_motor_test}")

    # 2. 计算我们自己实现的梯度
    grad_implemented = calculate_gradient_disp_ctrl(q_test, delta_l_motor_test, config_params)

    # 3. 定义用于有限差分法的基础能量函数
    #    这是我们要审判的核心目标
    energy_func_to_test = lambda q: calculate_total_potential_energy_disp_ctrl(q, delta_l_motor_test, config_params)

    # 4. 使用Scipy的有限差分法计算“真理”梯度
    #    使用一个与我们内部实现不同的epsilon，以确保独立性
    grad_truth = approx_fprime(q_test, energy_func_to_test, 1e-7)

    # 5. 对比和分析
    print(f"\n--- 结果对比 ---")
    print(f"  - 我方梯度: {grad_implemented}")
    print(f"  - 真理梯度: {grad_truth}")
    
    diff_vector = grad_implemented - grad_truth
    diff_norm = np.linalg.norm(diff_vector)

    print(f"\n  - 差值范数: {diff_norm:.8f}")

    if diff_norm < 1e-5:
        print("\n  - ✅✅✅ 【测试通过】: 梯度与真理完全一致！总势能函数大概率是正确的。")
    else:
        print("\n  - ❌❌❌ 【测试失败】: 梯度存在巨大差异！总势能函数的实现存在根本性错误！")
        print(f"  - 差值向量 (我方 - 真理): {diff_vector}")

if __name__ == '__main__':
    run_gradient_verification()