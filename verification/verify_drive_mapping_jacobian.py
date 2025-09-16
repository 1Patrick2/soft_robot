import numpy as np
import sys
import os

# --- 核心依赖 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.statics import calculate_actuation_jacobian_numerical
from src.statics import calculate_actuation_jacobian_numerical, calculate_actuation_jacobian_analytical
from src.utils.read_config import load_config

def verify_jacobian():
    """Compares the analytical and numerical drive mapping Jacobians."""
    print("--- 驱动雅可比 (J_act) 专项验证 ---")
    
    # --- 加载配置和测试参数 ---
    config = load_config('config/config.json')
    np.set_printoptions(precision=6, suppress=True, linewidth=150)
    q_test = np.array([0.1, np.pi/6, 0.2, -np.pi/4, 0.15, np.pi/3])
    
    print(f"\n测试构型 q: {q_test}")

    # --- 计算数值真值 ---
    print("\n计算数值雅可比 (真值)...")
    J_num = calculate_actuation_jacobian_numerical(q_test, config)
    print("数值雅可比 J_numerical:\n", J_num)

    # --- 计算解析值 (待实现) ---
    print("\n计算解析雅可比...")
    J_ana = calculate_actuation_jacobian_analytical(q_test, config)
    print("解析雅可比 J_analytical:\n", J_ana)
    
    # --- 对比 ---
    diff = J_ana - J_num
    diff_norm = np.linalg.norm(diff)
    ref_norm = np.linalg.norm(J_num)
    rel_error = diff_norm / (ref_norm + 1e-9)
    
    print(f"\n--- 结果对比 ---")
    print(f"  - 差值范数: {diff_norm:.6e}")
    print(f"  - 相对误差: {rel_error:.6e}")

    if rel_error < 1e-5:
        print("\n✅✅✅ 【验证通过】解析雅可比与数值真值一致！")
    else:
        print("\n❌❌❌ 【验证失败】解析雅可比存在错误！")
        print("差值矩阵 (Ana - Num):\n", diff)


if __name__ == '__main__':
    verify_jacobian()