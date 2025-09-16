import numpy as np
import sys
import os
import logging

# --- 核心依赖 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.read_config import load_config
from src.solver import solve_static_equilibrium_disp_ctrl

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def debug_solver_single_case():
    """
    建立一个最小化的、可稳定复现的失败案例，用于调试内循环求解器。
    """
    logging.info("--- 调试内循环求解器 (Debug Solver) ---")
    
    # 1. 加载我们最终的、物理正确的配置
    config_params = load_config('config/config.json')
    np.set_printoptions(precision=6, suppress=True)
    logging.info("✅ 1. 配置加载成功。")

    # 2. 定义一个固定的、非零的、有挑战性的驱动输入
    # 这个输入应该能让机器人产生一个明确的弯曲，而不是一个微不足道的移动
    delta_l_test = np.array([0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0])
    logging.info(f"✅ 2. 测试驱动向量 (delta_l_test): {delta_l_test}")

    # 3. 定义初始猜测点
    # 我们从“奇点”或其附近开始，因为这是最困难、最需要验证的场景
    q_guess_initial = np.full(6, 1e-6) # 使用一个极小的、确定性的非零向量
    logging.info(f"✅ 3. 初始构型猜测 (q_guess): {q_guess_initial}")

    # 4. 执行求解，并期待来自solver的详细回调日志
    logging.info("\n--- 开始调用求解器，观察内部迭代... ---")
    q_solution = solve_static_equilibrium_disp_ctrl(
        q_guess=q_guess_initial,
        delta_l_motor=delta_l_test,
        params=config_params
    )
    logging.info("--- 求解器调用结束 ---")

    # 5. 报告最终结果
    print("\n" + "="*50)
    print("              FINAL RESULT")
    print("="*50)
    if q_solution is not None:
        print(f"✅ 求解器返回了一个解: {q_solution}")
    else:
        print("❌ 求解器返回了 None，表示求解失败。")
    print("="*50)

if __name__ == '__main__':
    debug_solver_single_case()
