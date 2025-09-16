# verification/verify_com_jacobian.py

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kinematics import forward_kinematics, calculate_com_jacobians_analytical
from src.utils.read_config import load_config

def calculate_com_jacobians_numerical(q, params, epsilon=1e-7):
    """ 数值计算质心雅可比 """
    J_com_pss_num = np.zeros((3, 6))
    J_com_cms1_num = np.zeros((3, 6))
    J_com_cms2_num = np.zeros((3, 6))

    _, com_base = forward_kinematics(q, params)

    for i in range(6):
        q_plus = q.copy()
        q_plus[i] += epsilon
        _, com_plus = forward_kinematics(q_plus, params)
        
        J_com_pss_num[:, i] = (com_plus['pss'] - com_base['pss']) / epsilon
        J_com_cms1_num[:, i] = (com_plus['cms1'] - com_base['cms1']) / epsilon
        J_com_cms2_num[:, i] = (com_plus['cms2'] - com_base['cms2']) / epsilon
        
    return {'pss': J_com_pss_num, 'cms1': J_com_cms1_num, 'cms2': J_com_cms2_num}

if __name__ == '__main__':
    print("--- 质心雅可比 (COM Jacobians) 验证脚本 ---")
    config = load_config('config/config.json')
    np.set_printoptions(precision=6, suppress=True)

    q_test = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    print(f"测试构型 q: {q_test}\n")

    J_com_ana = calculate_com_jacobians_analytical(q_test, config)
    J_com_num = calculate_com_jacobians_numerical(q_test, config)

    all_passed = True
    for segment in ['pss', 'cms1', 'cms2']:
        diff_norm = np.linalg.norm(J_com_ana[segment] - J_com_num[segment])
        test_passed = diff_norm < 1e-6
        print(f"--- 验证段: {segment} ---")
        # print(f"解析法 J_com:\n{J_com_ana[segment]}")
        # print(f"数值法 J_com:\n{J_com_num[segment]}")
        print(f"差值范数: {diff_norm:.8f}")
        if test_passed:
            print(f"✅ {segment} PASSED\n")
        else:
            print(f"❌ {segment} FAILED\n")
            all_passed = False

    print("--- 总结 ---")
    if all_passed:
        print("✅✅✅ 所有质心雅可比测试通过!")
    else:
        print("❌❌❌ 质心雅可比计算存在Bug!")