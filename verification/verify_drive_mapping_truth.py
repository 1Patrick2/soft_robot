# verification/verify_drive_mapping_truth.py

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.statics import calculate_drive_mapping
from src.utils.read_config import load_config

def final_truth_verification():
    """
    对 drive_mapping 函数进行最终的、基于第一性原理的“外部审计”。
    """
    print("--- 核心物理模型 `calculate_drive_mapping` 最终真理审判 ---")
    
    config = load_config('config/config.json')
    np.set_printoptions(precision=8, suppress=True)

    # 1. 定义一个“无法出错”的简单世界
    # 只有PSS段在XZ平面内弯曲 (phi_p=0), 曲率为 1.0
    q_test = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    print(f"测试构型 q_test: {q_test}")

    # 2. 用“第一性原理”手动计算“标准答案”
    # 真理公式: delta_l = r * kappa * L
    L_p = config['Geometry']['PSS_initial_length']
    r_short = config['Geometry']['short_lines']['diameter_m'] / 2
    r_long = config['Geometry']['long_lines']['diameter_m'] / 2
    angles_s_rad = np.deg2rad(config['Geometry']['short_lines']['angles_deg'])
    angles_l_rad = np.deg2rad(config['Geometry']['long_lines']['angles_deg'])
    
    # 对于这个简单情况，我们代码中的模型（无论对错）都应该退化到这个真理
    # 注意：cos(angle - 0) = cos(angle)
    delta_l_short_truth = r_short * q_test[0] * L_p * np.cos(angles_s_rad)
    delta_l_long_truth = r_long * q_test[0] * L_p * np.cos(angles_l_rad)
    delta_l_truth = np.concatenate([delta_l_short_truth, delta_l_long_truth])
    print(f"理论真值 delta_l_truth: {delta_l_truth}")

    # 3. 调用我们代码中的函数进行计算
    delta_l_code_short, delta_l_code_long = calculate_drive_mapping(q_test, config)
    delta_l_code = np.concatenate([delta_l_code_short, delta_l_code_long])
    print(f"代码计算 delta_l_code:  {delta_l_code}")

    # 4. 进行最终对比
    is_correct = np.allclose(delta_l_code, delta_l_truth, atol=1e-9)
    
    print("\n--- 最终审计结果 ---")
    if is_correct:
        print("✅✅✅【真理验证通过】: `calculate_drive_mapping` 的核心实现正确！")
    else:
        print("❌❌❌【真理验证失败】: `calculate_drive_mapping` 的核心实现存在根本性错误！")
        diff = delta_l_code - delta_l_truth
        print(f"    - 差值 (code - truth): {diff}")

if __name__ == '__main__':
    final_truth_verification()