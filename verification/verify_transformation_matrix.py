import numpy as np
import sys
import os

# 将项目根目录添加到Python路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kinematics import pcc_transformation

def main():
    """
    对核心运动学函数 pcc_transformation 进行最终的、最严格的手动验证。
    测试一个在Y-Z平面弯曲的简单构型 (phi=pi/2)，并与手动推导的理论真值进行对比。
    """
    print("---"" 核心变换矩阵 (pcc_transformation) 最终审查 ---")

    # 1. 定义测试参数
    L = 0.1
    kappa = 1.0
    phi = np.pi / 2
    print(f"\n测试参数: L={L}, kappa={kappa}, phi={phi:.4f} (pi/2)")

    # 2. 手动计算理论真值矩阵 (Expected Matrix)
    print("\n正在手动计算理论真值矩阵 T_expected...")
    c_phi = 0  # cos(pi/2)
    s_phi = 1  # sin(pi/2)
    kL = kappa * L
    c_kL = np.cos(kL)
    s_kL = np.sin(kL)

    T_expected = np.array([
        [c_phi**2 * (c_kL - 1) + 1,  s_phi * c_phi * (c_kL - 1), c_phi * s_kL, c_phi * (1 - c_kL) / kappa],
        [s_phi * c_phi * (c_kL - 1),  s_phi**2 * (c_kL - 1) + 1, s_phi * s_kL, s_phi * (1 - c_kL) / kappa],
        [-c_phi * s_kL,             -s_phi * s_kL,             c_kL,        s_kL / kappa],
        [0,                         0,                         0,           1]
    ])

    # 3. 调用函数计算实际值
    print("正在调用 pcc_transformation 函数计算实际矩阵 T_actual...")
    T_actual = pcc_transformation(kappa, phi, L)

    # 4. 对比结果
    print("\n---"" 结果对比 ---")
    np.set_printoptions(precision=8, suppress=True)

    print(f"\n理论真值矩阵 T_expected:")
    print(T_expected)
    print(f"\n函数计算矩阵 T_actual:")
    print(T_actual)

    diff = T_actual - T_expected
    diff_norm = np.linalg.norm(diff)

    print(f"\n差值矩阵范数 (Norm of Difference): {diff_norm:.10f}")

    # 最终结论
    print("\n---"" 最终结论 ---")
    if np.allclose(T_expected, T_actual, atol=1e-8):
        print("✅ [通过] 函数 `pcc_transformation` 与理论真值完全一致。")
        print("   核心运动学模型不存在问题。问题可能被锁定在静力学海森矩阵H中。")
    else:
        print("❌ [失败] 函数 `pcc_transformation` 与理论真值存在差异！已找到BUG！")
        print("   请仔细检查 T_actual 和 T_expected 矩阵中不一致的元素。")

if __name__ == "__main__":
    main()
