import numpy as np
import sys
import os

# 将项目根目录添加到Python路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.read_config import load_config
from src.statics import (
    calculate_elastic_potential_energy,
    calculate_actuation_potential_energy,
    calculate_gravity_potential_energy,
    calculate_gradient_analytical
)

def __main__():
    """
    本脚本用于对核心物理模型进行最直接的、可解释的单元测试，
    以诊断为何求解器无法找到弯曲构型。
    """
    print("--- 物理模型诊断验证脚本 ---")
    
    # 1. 加载参数
    config_params = load_config('config/config.json')
    np.set_printoptions(precision=8, suppress=True)
    print("成功加载 'config.json' 中的参数ảng")

    # 2. 定义一个简单的、非笔直的测试构型
    # 假设只有PSS段在X-Z平面内有一个微小的弯曲
    q_test = np.array(config_params['Initial_State']['q0'])
    q_test[1] = 0.5  # kappa_p = 0.5 (一个合理的、非零的曲率)
    print(f"\n--- 测试构型 (q_test) ---\n{q_test}")

    # 3. 定义一组简单的、非对称的驱动力
    tensions_test = {
        'tensions_short': np.array([10.0, 0.0, 0.0, 0.0]),
        'tensions_long': np.array([10.0, 0.0, 0.0, 0.0])
    }
    print(f"\n--- 测试驱动力 (tensions_test) ---")
    for key, value in tensions_test.items():
        print(f"  - {key}: {value}")

    # 4. 分别计算三个核心势能项
    print("\n--- 势能分析 (在 q_test) ---")
    U_elastic = calculate_elastic_potential_energy(q_test, config_params)
    U_actuation = calculate_actuation_potential_energy(q_test, tensions_test, config_params)
    U_gravity = calculate_gravity_potential_energy(q_test, config_params)
    U_total = U_elastic + U_actuation + U_gravity

    print(f"  - 弹性势能 U_elastic: {U_elastic:.8f}")
    print(f"  - 驱动势能 U_actuation: {U_actuation:.8f}")
    print(f"  - 重力势能 U_gravity:   {U_gravity:.8f}")
    print("  -------------------------------------")
    print(f"  - 总势能 U_total:     {U_total:.8f}")

    # 5. 计算总势能的解析梯度
    grad_total = calculate_gradient_analytical(q_test, tensions_test, config_params)
    print("\n--- 梯度分析 (dU/dq at q_test) ---")
    print(f"  - 总梯度 grad_total: \n{grad_total}")

    # 6. 提出关键问题
    print("\n--- 核心诊断问题 ---")
    print("1. 在施加了不对称的10N驱动力后，U_actuation 是否为一个显著的负值？")
    print("   (如果是正值或接近于零，则驱动映射或能量计算可能存在符号错误)")
    print("2. 总梯度 grad_total 的各个分量（尤其是 dU/d_kappa) 是否为显著的非零值？")
    print("   (如果梯度很小，则说明系统在该点附近确实非常平坦，求解器难以移动)")
    print("3. 从物理直觉上看，这些能量和梯度的数值是否合理？")

if __name__ == '__main__':
    __main__()
