# verification/verify_elastic_hessian.py

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.statics import (
    calculate_elastic_potential_energy,
    calculate_elastic_hessian_analytical,
    calculate_hessian_numerical_high_precision
)
from src.utils.read_config import load_config

if __name__ == '__main__':
    print("--- 弹性势能海森矩阵 (Elastic Hessian) 专项验证脚本 ---")
    config = load_config('config/config.json')
    np.set_printoptions(precision=6, suppress=True, linewidth=200)

    q_test = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    print(f"测试构型 q: {q_test}\n")

    # 1. 计算解析Hessian
    H_ana = calculate_elastic_hessian_analytical(q_test, config)

    # 2. 计算数值Hessian
    def energy_wrapper(q, p): # 包装一下以匹配高精度数值计算器的接口
        return calculate_elastic_potential_energy(q, p)

    H_num = calculate_hessian_numerical_high_precision(q_test, energy_wrapper, config)

    # 3. 对比
    diff_norm = np.linalg.norm(H_ana - H_num)
    test_passed = diff_norm < 1e-6

    print(f"解析法 Hessian (H_E_ana):\n{H_ana}")
    print(f"\n数值法 Hessian (H_E_num):\n{H_num}")
    print(f"\n差值矩阵 (Ana - Num):\n{H_ana - H_num}")
    print(f"\n差值范数: {diff_norm:.8f}")

    print("\n--- 总结 ---")
    if test_passed:
        print("✅✅✅ 测试通过! 解析弹性Hessian与数值结果完全一致。")
    else:
        print("❌❌❌ 测试失败! 弹性Hessian的解析与数值实现不匹配。")
