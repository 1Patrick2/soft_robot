import numpy as np
import sys
import os
import logging

# --- 核心依赖 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.read_config import load_config
from src.statics import calculate_drive_mapping
from src.solver import solve_static_equilibrium_disp_ctrl

def run_forward_backward_consistency_test():
    """
    对内循环求解器进行“正向-反向一致性”的决定性实验。
    该实验将验证，当给定一个已知构型所对应的精确驱动时，
    求解器能否从一个扰动点出发，稳定地收敛回到我们已知的真理构型。
    """
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    print("--- 内循环求解器正向-反向一致性测试 ---")

    # 1. 加载配置并设置测试参数
    config_params = load_config('config/config.json')
    np.set_printoptions(precision=6, suppress=True)

    # 2. 定义一个非平凡的“真理”构型 (q_truth)
    q_truth = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    # 3. [正向过程]：根据q_truth，计算出要维持该姿态所需的电机位移 (delta_l_truth)
    delta_l_truth = calculate_drive_mapping(q_truth, config_params)

    # 4. 定义一个受扰动的初始猜测点
    q_guess = q_truth + np.random.rand(6) * 0.1 # 在真理附近增加随机扰动

    print(f"\n--- 测试条件 ---")
    print(f"  - 真理构型 q_truth:      {q_truth}")
    print(f"  - 对应的电机位移 Δl_truth: {delta_l_truth}")
    print(f"  - 求解器初始猜测 q_guess:  {q_guess}")

    # 5. [反向过程]：调用求解器，看其能否从q_guess恢复到q_truth
    q_solution = solve_static_equilibrium_disp_ctrl(q_guess, delta_l_truth, config_params)

    # 6. 对比和分析
    print(f"\n--- 结果对比 ---")
    if q_solution is not None:
        print(f"  - 求解器返回构型 q_solution: {q_solution}")
        diff_vector = q_solution - q_truth
        diff_norm = np.linalg.norm(diff_vector)

        print(f"\n  - 误差范数 ||q_solution - q_truth||: {diff_norm:.6f}")

        if diff_norm < 1e-4:
            print("\n  - ✅✅✅ 【测试通过】: 求解器成功恢复了真理构型！内循环基本可靠。")
        else:
            print("\n  - ❌❌❌ 【测试失败】: 求解器收敛到了错误的位置！内循环存在根本性问题。")
            print(f"  - 差值向量 (q_solution - q_truth): {diff_vector}")
    else:
        print("\n  - ❌❌❌ 【测试失败】: 求解器未能收敛，返回了None。内循环极其不稳定。")

if __name__ == '__main__':
    run_forward_backward_consistency_test()
