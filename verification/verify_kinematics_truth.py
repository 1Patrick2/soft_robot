# verification/verify_kinematics_truth.py

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kinematics import forward_kinematics
from src.utils.read_config import load_config

def final_kinematics_truth_verification():
    """
    对 forward_kinematics 函数进行最终的、基于第一性原理的“外部审计”。
    """
    print("--- 核心运动学模型 `forward_kinematics` 最终真理审判 ---")
    
    config = load_config('config/config.json')
    np.set_printoptions(precision=8, suppress=True)

    # 1. 定义一个“无法出错”的简单世界
    # 只有PSS段在XZ平面内弯曲 (phi_p=0), 曲率为 1.0. CMS段保持笔直。
    q_test = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    print(f"测试构型 q_test: {q_test}")

    # 2. 用“第一性原理”手动计算“标准答案”
    L_p = config['Geometry']['PSS_initial_length']
    L_c1 = config['Geometry']['CMS_proximal_length']
    L_c2 = config['Geometry']['CMS_distal_length']
    kappa_p = q_test[0]

    # 理论上，PSS段的变换矩阵
    c_kL, s_kL = np.cos(kappa_p * L_p), np.sin(kappa_p * L_p)
    T_pss_truth = np.array([
        [c_kL, 0, s_kL, (1 - c_kL) / kappa_p],
        [0, 1, 0, 0],
        [-s_kL, 0, c_kL, s_kL / kappa_p],
        [0, 0, 0, 1]
    ])

    # 理论上，CMS段是纯粹的Z轴平移
    T_cms1_truth = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,L_c1], [0,0,0,1]])
    T_cms2_truth = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,L_c2], [0,0,0,1]])

    # 最终的理论真值
    T_truth = T_pss_truth @ T_cms1_truth @ T_cms2_truth
    print(f"理论真值 T_truth:\n{T_truth}")

    # 3. 调用我们代码中的函数进行计算
    T_code, _ = forward_kinematics(q_test, config)
    print(f"代码计算 T_code:\n{T_code}")

    # 4. 进行最终对比
    is_correct = np.allclose(T_code, T_truth, atol=1e-9)
    
    print("\n--- 最终审计结果 ---")
    if is_correct:
        print("✅✅✅【真理验证通过】: `forward_kinematics` 的核心实现正确！")
    else:
        print("❌❌❌【真理验证失败】: `forward_kinematics` 的核心实现存在根本性错误！")
        diff = T_code - T_truth
        print(f"    - 差值矩阵 (code - truth):\n{diff}")

if __name__ == '__main__':
    final_kinematics_truth_verification()
