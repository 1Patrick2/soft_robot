import numpy as np
import sys
import os
import logging

# --- 核心依赖 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.read_config import load_config
from src.statics import (
    calculate_elastic_potential_energy,
    calculate_gravity_potential_energy,
    calculate_total_potential_energy_disp_ctrl,
    calculate_gradient_disp_ctrl,
    calculate_drive_mapping
)

def run_energy_analysis():
    """
    对总势能函数的各个组成部分进行剖析，以诊断为何求解器
    会从一个“真理”点被吸引到一个物理上错误的“陷阱”点。
    """
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    print("--- 能量地貌诊断脚本 ---")

    # 1. 加载配置并设置参数 (与一致性测试完全相同)
    config_params = load_config('config/config.json')
    np.set_printoptions(precision=8, suppress=True)

    q_truth = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    delta_l_truth = calculate_drive_mapping(q_truth, config_params)
    
    # 这是从上一个测试中得到的、求解器找到的错误解
    q_trap = np.array([-0.        , 0.299682, -0.        , 0.484323, -0.        , 0.641924])

    print(f"\n--- 分析点 ---")
    print(f"  - 真理点 q_truth: {q_truth}")
    print(f"  - 陷阱点 q_trap:  {q_trap}")
    print(f"  - 驱动输入 Δl:    {delta_l_truth}")

    # 2. 定义一个辅助函数来计算并打印所有能量分量和梯度范数
    def analyze_point(q, delta_l, point_name):
        U_e = calculate_elastic_potential_energy(q, config_params)
        U_g = calculate_gravity_potential_energy(q, config_params)
        U_total = calculate_total_potential_energy_disp_ctrl(q, delta_l, config_params)
        U_a_etc = U_total - U_e - U_g
        
        grad_total = calculate_gradient_disp_ctrl(q, delta_l, config_params)
        grad_norm = np.linalg.norm(grad_total)

        print(f"\n--- 能量与梯度 @ {point_name} ---")
        print(f"  - 弹性势能 (U_e):           {U_e:.8f}")
        print(f"  - 重力势能 (U_g):           {U_g:.8f}")
        print(f"  - 驱动及其他势能 (U_a_etc): {U_a_etc:.8f}")
        print(f"  ----------------------------------------")
        print(f"  - 总势能 (U_total):         {U_total:.8f}")
        print(f"  - 总梯度范数 ||∇U||:       {grad_norm:.8f}")

    # 3. 分别计算并打印两个点的能量
    analyze_point(q_truth, delta_l_truth, "真理点 (q_truth)")
    analyze_point(q_trap, delta_l_truth, "陷阱点 (q_trap)")

if __name__ == '__main__':
    run_energy_analysis()

