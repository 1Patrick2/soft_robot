import numpy as np
import sys
import os
import time

# --- 核心依赖 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.read_config import load_config
from src.kinematics import forward_kinematics
from src.statics import calculate_drive_mapping
from src.solver import solve_static_equilibrium_disp_ctrl # 导入我们要审判的核心函数

def run_inverse_validation():
    """
    执行决定性的“逆向验证”实验。
    直接给予内循环求解器完美的电机位移输入，看它能否反解出正确的机器人构型。
    这能将问题从“系统问题”精确定位到“内循环求解器问题”。
    """
    print("--- 内循环逆向验证 (Inverse Validation) ---")
    
    # 1. 加载配置
    config_params = load_config('config/config.json')
    np.set_printoptions(precision=6, suppress=True)
    print("✅ 1. 配置加载成功。")

    # =====================================================================
    # === 第一步：定义“真理"
    # =====================================================================
    
    # 定义一个与 final_truth_test.py 完全相同的“真理构型”
    q_truth = np.array([0.8, 0.5, 0.8, 1.0, 0.8, 1.5])
    print(f"\n--- STEP 1: 定义真理 ---")
    print(f"  - 真理构型 (q_truth):\n    {q_truth}")

    # 计算要维持这个构型所需的“真理驱动”
    delta_l_truth = calculate_drive_mapping(q_truth, config_params)
    print(f"  - 计算出的真理驱动 (delta_l_truth):\n    {delta_l_truth}")
    print("✅ 2. “真理”定义成功。")

    # =====================================================================
    # === 第二步：审判内循环求解器
    # =====================================================================
    print(f"\n--- STEP 2: 审判内循环求解器 ---")
    print("  - 以 q_guess=[0,0,0,0,0,0] 为起点, delta_l_motor=delta_l_truth 为输入...")
    
    # 定义一个带微小扰动的初始猜测，以帮助求解器“逃离” q=0 奇点
    q_guess_initial = np.random.rand(6) * 0.01
    
    start_time = time.time()
    # 直接调用内循环求解器
    q_found = solve_static_equilibrium_disp_ctrl(
        q_guess=q_guess_initial,
        delta_l_motor=delta_l_truth,
        params=config_params
    )
    end_time = time.time()
    
    print(f"  - 求解完成 (耗时: {end_time - start_time:.4f}s)")

    # =====================================================================
    # === 第三步：分析结果
    # =====================================================================
    print(f"\n--- STEP 3: 结果分析 ---")
    
    if q_found is not None:
        print(f"  - 求解器找到的构型 (q_found):\n    {q_found}")
        print(f"  - 我们期望的真理 (q_truth):\n    {q_truth}")
        
        # 计算差值
        diff_vector = q_found - q_truth
        diff_norm = np.linalg.norm(diff_vector)
        
        print(f"\n  - 构型向量差值范数: {diff_norm:.8f}")

        # 设定一个合理的容差
        if diff_norm < 1e-3:
            print("\n  - ✅✅✅ 【测试通过】✅✅✅")
            print("  - 结论: 内循环求解器工作正常！它能够在给定正确驱动时，精确找到对应的物理构型。问题可能出在外循环的PSO未能找到正确的驱动值。")
        else:
            print("\n  - ❌❌❌ 【测试失败】❌❌❌")
            print("  - 结论: 内循环求解器是问题的根源！即使给予完美的驱动输入，它也无法从q=0的起点逃逸，找到正确的能量平衡点。")
            print(f"  - 差值向量 (Found - Truth):\n    {diff_vector}")
    else:
        print("\n  - ❌❌❌ 【测试彻底失败】❌❌❌")
        print("  - 结论: 内循环求解器返回了 None，意味着它在计算过程中彻底崩溃或发散。")

if __name__ == '__main__':
    run_inverse_validation()